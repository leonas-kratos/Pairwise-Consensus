# -*- coding: utf-8 -*-
"""
PC-EKF V3 — UWB Realtime (Console only)
========================================
Đọc /dev/ttyACM0 @ 921600, in X Y ra console, ghi CSV.

Format input mỗi dòng:
  0x1001, 100
  0x1002, 200
  0x1003, 300
  0x1004, 400
"""

import sys, math, time, signal, csv, argparse
import numpy as np
from numba import njit
import re
from _robot_mixin import start_robot, stop_robot

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
SERIAL_PORT = '/dev/serial/by-id/usb-SEGGER_J-Link_000760192143-if00'
BAUD_RATE = 921600

#ANCHORS = np.array([
#    [4000, 8800],   # Anchor 0 — ID 0x1001
#    [0,    8800],   # Anchor 1 — ID 0x1002
#    [0,    0   ],   # Anchor 2 — ID 0x1003
#    [4000, 0   ],   # Anchor 3 — ID 0x1004
#], dtype=float)

ANCHORS = np.array([
    [0,       0],       # Anchor 0 — ID 0x1001
    [0,       4720],    # Anchor 1 — ID 0x1002
    [6300,    4720],    # Anchor 2 — ID 0x1003
    [6300,    0   ],    # Anchor 3 — ID 0x1004
], dtype=np.float64)


ANCHOR_IDS    = [0x1001, 0x1002, 0x1003, 0x1004]
ANCHOR_HEIGHT = 1400.0   # mm

N_ANCHORS  = 4
PCEKF_Q       = 0.01
PCEKF_R_BASE  = 300.0
PCEKF_R_SCALE = 10.0
LOG_FILE      = "realtime_log.csv"

# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY (Python — gọi lần init, không cần JIT)
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d):
    return math.sqrt(max(d * d - ANCHOR_HEIGHT * ANCHOR_HEIGHT, 0.0))

# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS — giữ nguyên logic, chỉ thêm @njit
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _wls_init_nb(z_ground, anchors):
    """
    WLS dùng tất cả C(N,2) cặp anchor thay vì cố định anchor 0.
    Robust hơn khi anchor 0 bị nhiễu lớn.
    """
    N = anchors.shape[0]
    n_pairs = N * (N - 1) // 2
    A   = np.empty((n_pairs, 2))
    bv  = np.empty(n_pairs)
    row = 0
    for i in range(N):
        xi = anchors[i, 0]; yi = anchors[i, 1]
        di = z_ground[i]
        if di < 1.0: di = 1.0
        for j in range(i + 1, N):
            xj = anchors[j, 0]; yj = anchors[j, 1]
            dj = z_ground[j]
            if dj < 1.0: dj = 1.0
            A[row, 0] = 2.0 * (xi - xj)
            A[row, 1] = 2.0 * (yi - yj)
            bv[row]   = (dj*dj - di*di) + (xi*xi - xj*xj) + (yi*yi - yj*yj)
            row += 1
    AtA = A.T @ A
    Atb = A.T @ bv
    det = AtA[0, 0]*AtA[1, 1] - AtA[0, 1]*AtA[1, 0]
    pos = np.empty(2)
    if abs(det) < 1e-12:
        s0 = 0.0; s1 = 0.0
        for k in range(N):
            s0 += anchors[k, 0]; s1 += anchors[k, 1]
        pos[0] = s0 / N; pos[1] = s1 / N
    else:
        pos[0] = (AtA[1, 1]*Atb[0] - AtA[0, 1]*Atb[1]) / det
        pos[1] = (AtA[0, 0]*Atb[1] - AtA[1, 0]*Atb[0]) / det
    return pos


@njit(cache=True)
def _h_obs_nb(x, y, anchors, anchor_h):
    """h(x): vector slant distances tới tất cả anchor."""
    N = anchors.shape[0]
    h = np.empty(N)
    for i in range(N):
        dx = x - anchors[i, 0]
        dy = y - anchors[i, 1]
        h[i] = math.sqrt(dx*dx + dy*dy + anchor_h*anchor_h)
    return h


@njit(cache=True)
def _jacobian_H_nb(x, y, anchors, anchor_h):
    """Jacobian H (N×2) của h(x) tại [x, y]."""
    N = anchors.shape[0]
    H = np.zeros((N, 2))
    for i in range(N):
        dx = x - anchors[i, 0]
        dy = y - anchors[i, 1]
        d  = math.sqrt(dx*dx + dy*dy + anchor_h*anchor_h)
        if d < 1e-6: d = 1e-6
        H[i, 0] = dx / d
        H[i, 1] = dy / d
    return H


@njit(cache=True)
def _t_kernel_nb(v, sigma=1.0):
    """t-distribution kernel (nu=4)."""
    nu = 4.0
    return (1.0 + v*v / (nu * sigma*sigma + 1e-9)) ** (-(nu + 1.0) / 2.0)


@njit(cache=True)
def _pc_scores_v3_nb(innov, S_diag):
    """PC scoring V3 — MAD Auto-Normalized (giữ nguyên logic)."""
    N = innov.shape[0]

    # std = innov / sqrt(S_diag)
    std = np.empty(N)
    for i in range(N):
        std[i] = innov[i] / math.sqrt(max(S_diag[i], 1e-9))

    # median + MAD
    # Numba không có np.median trực tiếp trên 1D — sort thủ công
    sorted_std = np.sort(std)
    if N % 2 == 0:
        med = 0.5 * (sorted_std[N//2 - 1] + sorted_std[N//2])
    else:
        med = sorted_std[N//2]

    abs_dev = np.empty(N)
    for i in range(N):
        abs_dev[i] = abs(std[i] - med)
    sorted_dev = np.sort(abs_dev)
    if N % 2 == 0:
        mad = 0.5 * (sorted_dev[N//2 - 1] + sorted_dev[N//2])
    else:
        mad = sorted_dev[N//2]

    norm = np.empty(N)
    for i in range(N):
        norm[i] = std[i] / (1.4826 * mad + 1e-9)

    scores = np.zeros(N)
    for i in range(N):
        s = 0.0
        cnt = 0
        for j in range(N):
            if j != i:
                s += _t_kernel_nb(norm[i] - norm[j])
                cnt += 1
        scores[i] = s / cnt if cnt > 0 else 1.0
    return scores


@njit(cache=True)
def _pcekf_step_nb(x, P, z_slant, anchors, anchor_h, Q, R_base, R_scale):
    """
    PC-EKF V3 step — giữ nguyên logic từ PCEKF_v3.step().

    x        : (2,) state [X, Y]
    P        : (2,2) covariance
    z_slant  : (N,) raw slant distances
    Returns  : x_new (2,), P_new (2,2), scores (N,)
    """
    N = anchors.shape[0]

    # Slant → ground
    z = np.empty(N)
    for i in range(N):
        g2 = z_slant[i]*z_slant[i] - anchor_h*anchor_h
        z[i] = math.sqrt(g2) if g2 > 0.0 else 0.0

    # Predict
    P = P + np.eye(2) * Q

    # h(x) và H
    h = _h_obs_nb(x[0], x[1], anchors, anchor_h)
    H = _jacobian_H_nb(x[0], x[1], anchors, anchor_h)

    innov = z - h

    # S_diag (chỉ lấy diagonal để tính PC scores)
    HP     = H @ P
    S_diag = np.empty(N)
    for i in range(N):
        S_diag[i] = HP[i, 0]*H[i, 0] + HP[i, 1]*H[i, 1] + R_base

    # PC scores → R matrix
    scores = _pc_scores_v3_nb(innov, S_diag)
    R = np.zeros((N, N))
    for i in range(N):
        R[i, i] = R_base * (1.0 + R_scale * (1.0 - scores[i]))

    # Full S = H P H^T + R, Kalman gain
    S   = HP @ H.T + R
    K   = P @ H.T @ np.linalg.inv(S)

    x_new = x + K @ innov
    P_new = (np.eye(2) - K @ H) @ P

    # Symmetrise & clamp
    P_new = 0.5 * (P_new + P_new.T)
    for i in range(2):
        if P_new[i, i] < 1e-9:
            P_new[i, i] = 1e-9

    return x_new, P_new, scores


def _warmup():
    """Pre-compile JIT kernels trước khi nhận dữ liệu thực."""
    print("[PC-EKF] Warming up JIT kernels...", end=" ", flush=True)
    _z  = np.array([1000.0, 1200.0, 800.0, 900.0])
    _zg = np.array([slant_to_ground(d) for d in _z])
    _wls_init_nb(_zg, ANCHORS)
    _x0 = np.array([3000.0, 2000.0])
    _P0 = np.eye(2) * 1e6
    _pcekf_step_nb(_x0, _P0, _z, ANCHORS, ANCHOR_HEIGHT,
                   PCEKF_Q, PCEKF_R_BASE, PCEKF_R_SCALE)
    print("OK")


# ══════════════════════════════════════════════════════════════════════
#  PC-EKF V3 CLASS — giữ nguyên interface, delegate xuống JIT kernels
# ══════════════════════════════════════════════════════════════════════
class PCEKF_v3:
    def __init__(self, q=PCEKF_Q, r_base=PCEKF_R_BASE, r_scale=PCEKF_R_SCALE):
        self.Q      = q
        self.R_base = r_base
        self.R_sc   = r_scale
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = np.array(x0, dtype=np.float64)
        self.P = np.eye(2) * 1e6

    def step(self, z_slant):
        self.x, self.P, scores = _pcekf_step_nb(
            self.x, self.P, np.asarray(z_slant, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT,
            self.Q, self.R_base, self.R_sc
        )
        return self.x.copy(), scores


def wls_init(z_ground):
    """Python wrapper cho _wls_init_nb (dùng khi init)."""
    return _wls_init_nb(np.asarray(z_ground, dtype=np.float64), ANCHORS)


# ══════════════════════════════════════════════════════════════════════
#  SERIAL PARSER
# ══════════════════════════════════════════════════════════════════════
class SerialParser:
    def __init__(self):
        self.buf = {}

    def feed(self, line: str):
        # bytes -> string nếu cần
        if isinstance(line, bytes):
            line = line.decode("ascii", errors="ignore")

        line = line.strip()

        if not line:
            return None

        # Tìm ID và khoảng cách
        # Ví dụ nhận được:
        # 0x1001,1698
        # 0x1001,1698 |
        # 0x1001 , 1698
        m = re.search(
            r'(0x[0-9A-Fa-f]+)\s*,\s*([0-9]+(?:\.[0-9]+)?)',
            line
        )

        if m is None:
            return None

        try:
            aid = int(m.group(1), 16)
            dist = float(m.group(2))
        except Exception:
            return None

        if aid not in ANCHOR_IDS:
            return None

        self.buf[ANCHOR_IDS.index(aid)] = dist

        # Đủ 4 anchor
        if len(self.buf) == N_ANCHORS:
            frame = np.array(
                [self.buf[i] for i in range(N_ANCHORS)],
                dtype=np.float64
            )
            self.buf.clear()
            return frame

        return None

# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run(port, baud, q, r_base, r_scale, log_file):
    _warmup()

    import serial
    ekf    = PCEKF_v3(q=q, r_base=r_base, r_scale=r_scale)
    parser = SerialParser()
    inited  = False
    frame_n = 0
    t0      = None

    print(f"Đang kết nối {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    print(f"OK. Đang chờ dữ liệu... (Ctrl+C để dừng)\n")
    start_robot()
    print(f"{'Time(s)':>8}  {'Frame':>6}  {'X(mm)':>9}  {'Y(mm)':>9}  "
          f"{'S0':>6}  {'S1':>6}  {'S2':>6}  {'S3':>6}  {'dt(ms)':>7}")
    print("-" * 78)

    with open(log_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s','frame','x_mm','y_mm',
                    'd0','d1','d2','d3',
                    'score0','score1','score2','score3'])

        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode('ascii', errors='ignore')
                frame = parser.feed(line)
                if frame is None:
                    continue

                if t0 is None:
                    t0 = time.perf_counter()

                # Khởi tạo EKF từ frame đầu tiên
                if not inited:
                    z_g = np.array([slant_to_ground(d) for d in frame])
                    x0  = wls_init(z_g)
                    ekf.init(x0)
                    inited = True
                    print(f"[EKF] init  x={x0[0]:.1f}  y={x0[1]:.1f}")

                t_step = time.perf_counter()
                pos, scores = ekf.step(frame)
                dt_ms = (time.perf_counter() - t_step) * 1000.0

                frame_n += 1
                t = time.perf_counter() - t0

                # In console
                print(f"{t:8.3f}  {frame_n:6d}  "
                      f"{pos[0]:9.1f}  {pos[1]:9.1f}  "
                      f"{scores[0]:6.3f}  {scores[1]:6.3f}  "
                      f"{scores[2]:6.3f}  {scores[3]:6.3f}  {dt_ms:7.3f}")

                # Ghi CSV
                w.writerow([f"{t:.4f}", frame_n,
                             f"{pos[0]:.2f}", f"{pos[1]:.2f}",
                             *[f"{d:.1f}" for d in frame],
                             *[f"{s:.4f}" for s in scores]])
                f.flush()

        except KeyboardInterrupt:
            pass
        finally:
            stop_robot()
            ser.close()
            print(f"\n[✓] Dừng. {frame_n} frames. Log → {log_file}")

# ══════════════════════════════════════════════════════════════════════
#  DEMO (không cần phần cứng)
# ══════════════════════════════════════════════════════════════════════
def demo(q, r_base, r_scale):
    _warmup()
    rng = np.random.default_rng(42)
    WAYPOINTS = np.array([[800,400],[3000,400],[3000,8000],[800,8000],[800,400]], dtype=float)
    segs = np.diff(WAYPOINTS, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = seg_len.sum()
    gt_pts = []
    for d in np.linspace(0, total, 500):
        idx  = np.clip(np.searchsorted(cum, d, 'right') - 1, 0, len(segs)-1)
        frac = (d - cum[idx]) / (seg_len[idx] + 1e-9)
        gt_pts.append(WAYPOINTS[idx] + frac * segs[idx])

    ekf = PCEKF_v3(q=q, r_base=r_base, r_scale=r_scale)
    start_robot()
    print(f"{'Step':>5}  {'X(mm)':>9}  {'Y(mm)':>9}  {'Err(mm)':>8}  {'dt(µs)':>7}  Scores")
    print("-" * 70)
    errors = []
    for i, gt in enumerate(gt_pts):
        nlos = np.zeros(N_ANCHORS)
        if 150 < i < 300:
            nlos[2] = rng.normal(300, 80)
        z = np.array([
            math.sqrt((gt[0]-ax)**2 + (gt[1]-ay)**2 + ANCHOR_HEIGHT**2)
            + rng.normal(0, 50) + nlos[j]
            for j, (ax, ay) in enumerate(ANCHORS)
        ])
        if i == 0:
            ekf.init(wls_init(np.array([slant_to_ground(d) for d in z])))
        t0_ = time.perf_counter()
        pos, scores = ekf.step(z)
        dt = (time.perf_counter() - t0_) * 1e6
        err = np.linalg.norm(pos - gt)
        errors.append(err)
        if i % 50 == 0 or i == len(gt_pts) - 1:
            print(f"{i:5d}  {pos[0]:9.1f}  {pos[1]:9.1f}  {err:8.1f}  {dt:7.1f}  {scores.round(3)}")

    errors = np.array(errors)
    print(f"\nRMSE={np.sqrt(np.mean(errors**2)):.1f}mm  "
          f"MAE={np.mean(errors):.1f}mm  P95={np.percentile(errors,95):.1f}mm")
    stop_robot()

# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="PC-EKF V3 Realtime")
    ap.add_argument("--port",    default=SERIAL_PORT)
    ap.add_argument("--baud",    type=int,   default=BAUD_RATE)
    ap.add_argument("--q",       type=float, default=PCEKF_Q)
    ap.add_argument("--r_base",  type=float, default=PCEKF_R_BASE)
    ap.add_argument("--r_scale", type=float, default=PCEKF_R_SCALE)
    ap.add_argument("--log",     default=LOG_FILE)
    ap.add_argument("--demo",    action="store_true")
    args = ap.parse_args()

    print("PC-EKF V3 — UWB Realtime (Numba JIT)")
    print(f"  Q={args.q}  R_base={args.r_base}  R_scale={args.r_scale}")
    print(f"  Anchors: {[hex(i) for i in ANCHOR_IDS]}")
    print(f"  Height : {ANCHOR_HEIGHT} mm\n")

    if args.demo:
        demo(args.q, args.r_base, args.r_scale)
    else:
        run(args.port, args.baud, args.q, args.r_base, args.r_scale, args.log)

if __name__ == "__main__":
    main()
