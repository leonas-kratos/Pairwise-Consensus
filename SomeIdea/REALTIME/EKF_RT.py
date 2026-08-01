# -*- coding: utf-8 -*-
"""
EKF_RT.py — Standard EKF Realtime (Numba JIT)
===============================================
Đọc /dev/serial/by-id/... @ 921600, in X Y ra console, ghi CSV.

Thuật toán: Extended Kalman Filter (EKF) chuẩn.
  - Observation model: slant distance tới 4 anchors
  - Jacobian H tính analytic (không cần sigma points)
  - Hot path hoàn toàn @njit(cache=True) → latency ~µs/step sau warm-up

Format input mỗi dòng:
  0x1001, 1234
  0x1002, 2345
  ...
"""

import sys, math, time, signal, csv, argparse, re
import numpy as np
import serial
from numba import njit

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
SERIAL_PORT   = '/dev/serial/by-id/usb-SEGGER_J-Link_000760192143-if00'
BAUD_RATE     = 921600

ANCHORS = np.array([
    [0,       0   ],   # Anchor 0 — ID 0x1001
    [0,       4720],   # Anchor 1 — ID 0x1002
    [6300,    4720],   # Anchor 2 — ID 0x1003
    [6300,    0   ],   # Anchor 3 — ID 0x1004
], dtype=np.float64)

ANCHOR_IDS    = [0x1001, 0x1002, 0x1003, 0x1004]
ANCHOR_HEIGHT = 1400.0   # mm
N_ANCHORS     = 4

EKF_Q     = 0.01
EKF_R     = 50.0
LOG_FILE  = "EKF_realtime_log.csv"

# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _wls_init_nb(z_ground, anchors, n):
    """WLS init từ tất cả C(N,2) cặp anchor — không dùng np.linalg.lstsq."""
    n_pairs = n * (n - 1) // 2
    A   = np.empty((n_pairs, 2))
    bv  = np.empty(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = anchors[i, 0]; yi = anchors[i, 1]
            xj = anchors[j, 0]; yj = anchors[j, 1]
            di = z_ground[i]; dj = z_ground[j]
            if di < 1.0: di = 1.0
            if dj < 1.0: dj = 1.0
            A[idx, 0] = 2.0 * (xi - xj)
            A[idx, 1] = 2.0 * (yi - yj)
            bv[idx]   = (dj*dj - di*di) + (xi*xi - xj*xj) + (yi*yi - yj*yj)
            idx += 1
    AtA = A.T @ A
    Atb = A.T @ bv
    det = AtA[0,0]*AtA[1,1] - AtA[0,1]*AtA[1,0]
    pos = np.empty(2)
    if abs(det) < 1e-12:
        s0 = 0.0; s1 = 0.0
        for k in range(n):
            s0 += anchors[k, 0]; s1 += anchors[k, 1]
        pos[0] = s0 / n; pos[1] = s1 / n
    else:
        pos[0] = (AtA[1,1]*Atb[0] - AtA[0,1]*Atb[1]) / det
        pos[1] = (AtA[0,0]*Atb[1] - AtA[1,0]*Atb[0]) / det
    return pos


@njit(cache=True)
def _ekf_step_nb(x, P, z_slant, anchors, anchor_h, Q, R):
    """
    EKF step hoàn toàn JIT.
    x       : (2,) state [X, Y]
    P       : (2,2) covariance
    z_slant : (N,) raw slant distances
    Returns : x_new, P_new (2,), (2,2)
    """
    N = anchors.shape[0]

    # 1. Convert slant → ground distance
    z = np.empty(N)
    for i in range(N):
        v = z_slant[i]*z_slant[i] - anchor_h*anchor_h
        z[i] = math.sqrt(v) if v > 0.0 else 0.0

    # 2. Predict (constant-position model)
    P = P + np.eye(2) * Q

    # 3. h(x) và Jacobian H
    h = np.empty(N)
    H = np.zeros((N, 2))
    for i in range(N):
        dx = x[0] - anchors[i, 0]
        dy = x[1] - anchors[i, 1]
        d  = math.sqrt(dx*dx + dy*dy + anchor_h*anchor_h)
        if d < 1e-6: d = 1e-6
        h[i]    = d
        H[i, 0] = dx / d
        H[i, 1] = dy / d

    # 4. Innovation
    innov = z - h

    # 5. S = H P H^T + R*I  (N×N)
    HP  = H @ P
    S   = HP @ H.T
    for i in range(N):
        S[i, i] += R

    # 6. Kalman gain K = P H^T S^{-1}
    PHt = P @ H.T      # 2×N
    # Invert S (4×4) — dùng Gauss-Jordan vì njit hỗ trợ np.linalg.inv
    S_inv = np.linalg.inv(S)
    K = PHt @ S_inv    # 2×N

    # 7. Update
    x_new = x + K @ innov
    P_new = (np.eye(2) - K @ H) @ P

    # Symmetrise
    P_new = 0.5 * (P_new + P_new.T)
    for i in range(2):
        if P_new[i, i] < 1e-9:
            P_new[i, i] = 1e-9

    return x_new, P_new


def _warmup():
    """Pre-compile JIT kernels trước khi nhận dữ liệu thực."""
    print("[EKF_RT] Warming up JIT kernels...", end=" ", flush=True)
    _z = np.array([1000.0, 1200.0, 800.0, 900.0])
    _zg = np.array([slant_to_ground_py(d) for d in _z])
    _wls_init_nb(_zg, ANCHORS, N_ANCHORS)
    _x0 = np.array([3000.0, 2000.0])
    _P0 = np.eye(2) * 1e6
    _ekf_step_nb(_x0, _P0, _z, ANCHORS, ANCHOR_HEIGHT, EKF_Q, EKF_R)
    print("OK")


def slant_to_ground_py(d):
    return math.sqrt(max(d*d - ANCHOR_HEIGHT*ANCHOR_HEIGHT, 0.0))

# ══════════════════════════════════════════════════════════════════════
#  SERIAL PARSER
# ══════════════════════════════════════════════════════════════════════
class SerialParser:
    def __init__(self):
        self.buf = {}

    def feed(self, line):
        if isinstance(line, bytes):
            line = line.decode("ascii", errors="ignore")
        line = line.strip()
        if not line:
            return None
        m = re.search(r'(0x[0-9A-Fa-f]+)\s*,\s*([0-9]+(?:\.[0-9]+)?)', line)
        if m is None:
            return None
        try:
            aid  = int(m.group(1), 16)
            dist = float(m.group(2))
        except Exception:
            return None
        if aid not in ANCHOR_IDS:
            return None
        self.buf[ANCHOR_IDS.index(aid)] = dist
        if len(self.buf) == N_ANCHORS:
            frame = np.array([self.buf[i] for i in range(N_ANCHORS)], dtype=np.float64)
            self.buf.clear()
            return frame
        return None

# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run(port, baud, q, r, log_file):
    _warmup()

    x      = None
    P      = np.eye(2) * 1e6
    parser = SerialParser()
    inited = False
    frame_n = 0
    t0      = None

    print(f"\nĐang kết nối {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    print(f"OK. Đang chờ dữ liệu... (Ctrl+C để dừng)\n")
    print(f"{'Time(s)':>8}  {'Frame':>6}  {'X(mm)':>9}  {'Y(mm)':>9}  {'dt(ms)':>7}")
    print("-" * 50)

    with open(log_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s','frame','x_mm','y_mm','d0','d1','d2','d3'])
        try:
            while True:
                raw   = ser.readline()
                if not raw:
                    continue
                frame = parser.feed(raw)
                if frame is None:
                    continue

                if t0 is None:
                    t0 = time.perf_counter()

                if not inited:
                    zg  = np.array([slant_to_ground_py(d) for d in frame])
                    x   = _wls_init_nb(zg, ANCHORS, N_ANCHORS)
                    P   = np.eye(2) * 1e6
                    inited = True
                    print(f"[EKF] init  x={x[0]:.1f}  y={x[1]:.1f}")

                t_step = time.perf_counter()
                x, P   = _ekf_step_nb(x, P, frame, ANCHORS, ANCHOR_HEIGHT, q, r)
                dt_ms  = (time.perf_counter() - t_step) * 1000.0

                frame_n += 1
                t = time.perf_counter() - t0

                print(f"{t:8.3f}  {frame_n:6d}  {x[0]:9.1f}  {x[1]:9.1f}  {dt_ms:7.3f}")
                w.writerow([f"{t:.4f}", frame_n,
                             f"{x[0]:.2f}", f"{x[1]:.2f}",
                             *[f"{d:.1f}" for d in frame]])
                f.flush()

        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
            print(f"\n[✓] Dừng. {frame_n} frames. Log → {log_file}")


# ══════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════
def demo(q, r):
    _warmup()
    rng = np.random.default_rng(42)
    WAYPOINTS = np.array([[800,400],[3000,400],[3000,8000],[800,8000],[800,400]], dtype=float)
    segs = np.diff(WAYPOINTS, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    cum  = np.concatenate([[0], np.cumsum(seg_len)])
    total = seg_len.sum()
    gt_pts = []
    for d in np.linspace(0, total, 500):
        idx  = int(np.clip(np.searchsorted(cum, d, 'right') - 1, 0, len(segs)-1))
        frac = (d - cum[idx]) / (seg_len[idx] + 1e-9)
        gt_pts.append(WAYPOINTS[idx] + frac * segs[idx])

    x = None; P = np.eye(2) * 1e6; errors = []
    print(f"{'Step':>5}  {'X(mm)':>9}  {'Y(mm)':>9}  {'Err(mm)':>8}  dt(µs)")
    print("-" * 50)

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
            zg = np.array([slant_to_ground_py(d) for d in z])
            x  = _wls_init_nb(zg, ANCHORS, N_ANCHORS)
            P  = np.eye(2) * 1e6

        t0_ = time.perf_counter()
        x, P = _ekf_step_nb(x, P, z, ANCHORS, ANCHOR_HEIGHT, q, r)
        dt   = (time.perf_counter() - t0_) * 1e6

        err = math.sqrt((x[0]-gt[0])**2 + (x[1]-gt[1])**2)
        errors.append(err)
        print(f"{i:5d}  {x[0]:9.1f}  {x[1]:9.1f}  {err:8.1f}  {dt:6.1f}")
        time.sleep(0.005)

    errors = np.array(errors)
    print(f"\nRMSE={np.sqrt(np.mean(errors**2)):.1f}mm  "
          f"MAE={np.mean(errors):.1f}mm  P95={np.percentile(errors,95):.1f}mm")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="EKF Realtime — Standard EKF + Numba JIT")
    ap.add_argument("--port",  default=SERIAL_PORT)
    ap.add_argument("--baud",  type=int,   default=BAUD_RATE)
    ap.add_argument("--q",     type=float, default=EKF_Q)
    ap.add_argument("--r",     type=float, default=EKF_R)
    ap.add_argument("--log",   default=LOG_FILE)
    ap.add_argument("--demo",  action="store_true")
    args = ap.parse_args()

    print("EKF_RT — Standard Extended Kalman Filter (Numba JIT)")
    print(f"  Q={args.q}  R={args.r}")
    print(f"  Anchors: {[hex(i) for i in ANCHOR_IDS]}")
    print(f"  Height : {ANCHOR_HEIGHT} mm\n")

    if args.demo:
        demo(args.q, args.r)
    else:
        run(args.port, args.baud, args.q, args.r, args.log)

if __name__ == "__main__":
    main()
