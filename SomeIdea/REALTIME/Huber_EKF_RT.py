# -*- coding: utf-8 -*-
"""
Huber_EKF_RT.py — Huber-EKF Realtime (Numba JIT)
===============================================
Đọc /dev/serial/by-id/... @ 921600, in X Y ra console, ghi CSV.

Thuật toán: EKF + Huber IRLS M-estimator (reweight R theo residual chuẩn hóa).

Format input mỗi dòng:
  0x1001, 1234
  ...
"""

import sys, math, time, csv, argparse, re
import numpy as np
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
ANCHOR_HEIGHT = 1400.0
N_ANCHORS     = 4


Q = 0.01
R = 50.0
HUBER_DELTA   = 20.0
HUBER_MAXITER = 5
LOG_FILE      = "Huber_EKF_realtime_log.csv"

# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _ls_position_nb(distances, anchors):
    """LS position closed-form 2×2 (anchor-0 baseline)."""
    x0 = anchors[0, 0]; y0 = anchors[0, 1]
    d0 = distances[0]
    if d0 < 1.0: d0 = 1.0
    N = anchors.shape[0]
    A = np.empty((N - 1, 2))
    b_vec = np.empty(N - 1)
    for i in range(1, N):
        xi = anchors[i, 0]; yi = anchors[i, 1]
        di = distances[i]
        if di < 1.0: di = 1.0
        A[i-1, 0] = 2.0 * (xi - x0)
        A[i-1, 1] = 2.0 * (yi - y0)
        b_vec[i-1] = (d0*d0 - di*di) - (x0*x0 - xi*xi) - (y0*y0 - yi*yi)
    AtA = A.T @ A
    Atb = A.T @ b_vec
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


def slant_to_ground_py(d):
    return math.sqrt(max(d*d - ANCHOR_HEIGHT*ANCHOR_HEIGHT, 0.0))


@njit(cache=True)
def _make_spd_nb(P, eps=1e-9):
    P = 0.5 * (P + P.T)
    for i in range(P.shape[0]):
        P[i, i] += eps
        if P[i, i] < eps:
            P[i, i] = eps
    return P


@njit(cache=True)
def _ekf_h_H_nb(x, anchors, anchor_h):
    N = anchors.shape[0]
    h = np.empty(N)
    H = np.zeros((N, 2))
    for i in range(N):
        dx = x[0] - anchors[i, 0]
        dy = x[1] - anchors[i, 1]
        d  = math.sqrt(dx*dx + dy*dy + anchor_h*anchor_h)
        if d < 1e-6: d = 1e-6
        h[i] = d
        H[i, 0] = dx / d
        H[i, 1] = dy / d
    return h, H


@njit(cache=True)
def _huber_ekf_step_nb(x, P, z_slant, anchors, anchor_h, Q, R_base, delta, maxiter):
    N = anchors.shape[0]
    P = P + np.eye(2) * Q
    h, H = _ekf_h_H_nb(x, anchors, anchor_h)
    innov = z_slant - h
    R_diag = np.full(N, R_base)
    for _ in range(maxiter):
        HP = H @ P
        S_diag = np.empty(N)
        for i in range(N):
            s = 0.0
            for k in range(2):
                row_k = 0.0
                for j in range(2):
                    row_k += H[i, j] * P[j, k]
                s += row_k * H[i, k]
            S_diag[i] = s + R_diag[i]
            if S_diag[i] < 1e-9:
                S_diag[i] = 1e-9
        for i in range(N):
            r_sc = innov[i] / math.sqrt(S_diag[i])
            ar = abs(r_sc)
            if ar <= delta:
                w = 1.0
            else:
                w = delta / (ar + 1e-9)
            if w < 1e-4:
                w = 1e-4
            R_diag[i] = R_base / w
    S = (H @ P) @ H.T
    for i in range(N):
        S[i, i] += R_diag[i]
    K = (P @ H.T) @ np.linalg.inv(S)
    x_new = x + K @ innov
    P_new = _make_spd_nb((np.eye(2) - K @ H) @ P)
    return x_new, P_new


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


def _warmup():
    print("[Huber_EKF_RT] Warming up JIT kernels...", end=" ", flush=True)
    _z = np.array([1000.0, 1200.0, 800.0, 900.0])
    _zg = np.array([slant_to_ground_py(d) for d in _z])
    _ls_position_nb(_zg, ANCHORS)
    _x0 = np.array([3000.0, 2000.0]); _P0 = np.eye(2) * 1e6
    _huber_ekf_step_nb(_x0, _P0, _z, ANCHORS, ANCHOR_HEIGHT, Q, R, HUBER_DELTA, HUBER_MAXITER)

    print("OK")


def _step(x, P, z, extra, delta=None, maxiter=None):
    if delta is None: delta = HUBER_DELTA
    if maxiter is None: maxiter = HUBER_MAXITER
    x, P = _huber_ekf_step_nb(x, P, z, ANCHORS, ANCHOR_HEIGHT, Q, R, delta, maxiter)
    return x, P, extra



def run(port, baud, log_file, delta=None, maxiter=None):
    _warmup()
    x = None
    P = np.eye(2) * 1e6
    extra = None
    parser = SerialParser()
    inited = False
    frame_n = 0
    t0 = None

    import serial
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
                raw = ser.readline()
                if not raw:
                    continue
                frame = parser.feed(raw)
                if frame is None:
                    continue
                if t0 is None:
                    t0 = time.perf_counter()
                if not inited:
                    zg = np.array([slant_to_ground_py(d) for d in frame])
                    x  = _ls_position_nb(zg, ANCHORS)
                    P  = np.eye(2) * 1e6
                    inited = True
                    print(f"[Huber_EKF] init  x={x[0]:.1f}  y={x[1]:.1f}")

                t_step = time.perf_counter()
                x, P, extra = _step(x, P, frame, extra, delta, maxiter)
                dt_ms = (time.perf_counter() - t_step) * 1000.0
                frame_n += 1
                t = time.perf_counter() - t0
                print(f"{t:8.3f}  {frame_n:6d}  {x[0]:9.1f}  {x[1]:9.1f}  {dt_ms:7.3f}")
                w.writerow([f"{t:.4f}", frame_n, f"{x[0]:.2f}", f"{x[1]:.2f}",
                             *[f"{d:.1f}" for d in frame]])
                f.flush()
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
            print(f"\n[✓] Dừng. {frame_n} frames. Log → {log_file}")


def demo(delta=None, maxiter=None):
    _warmup()
    rng = np.random.default_rng(42)
    WAYPOINTS = np.array([[800,400],[3000,400],[3000,8000],[800,8000],[800,400]], dtype=float)
    segs = np.diff(WAYPOINTS, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = seg_len.sum()
    gt_pts = []
    for d in np.linspace(0, total, 500):
        idx  = int(np.clip(np.searchsorted(cum, d, 'right') - 1, 0, len(segs)-1))
        frac = (d - cum[idx]) / (seg_len[idx] + 1e-9)
        gt_pts.append(WAYPOINTS[idx] + frac * segs[idx])

    x = None; P = np.eye(2) * 1e6; errors = []; extra = None
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
            x  = _ls_position_nb(zg, ANCHORS)
            P  = np.eye(2) * 1e6
        t0_ = time.perf_counter()
        x, P, extra = _step(x, P, z, extra, delta, maxiter)
        dt = (time.perf_counter() - t0_) * 1e6
        err = math.sqrt((x[0]-gt[0])**2 + (x[1]-gt[1])**2)
        errors.append(err)
        if i % 50 == 0 or i == len(gt_pts) - 1:
            print(f"{i:5d}  {x[0]:9.1f}  {x[1]:9.1f}  {err:8.1f}  {dt:6.1f}")
    errors = np.array(errors)
    print(f"\nRMSE={np.sqrt(np.mean(errors**2)):.1f}mm  "
          f"MAE={np.mean(errors):.1f}mm  P95={np.percentile(errors,95):.1f}mm")


def main():
    ap = argparse.ArgumentParser(description="Huber_EKF Realtime — Numba JIT")
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=BAUD_RATE)
    ap.add_argument("--log", default=LOG_FILE)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--delta", type=float, default=HUBER_DELTA)
    ap.add_argument("--maxiter", type=int, default=HUBER_MAXITER)
    args = ap.parse_args()
    print("Huber_EKF_RT — Huber IRLS EKF (Numba JIT)")
    print(f"  Q={Q}  R={R}  (hardcoded)  delta={args.delta}  maxiter={args.maxiter}")
    print(f"  Anchors: {[hex(i) for i in ANCHOR_IDS]}")
    print(f"  Height : {ANCHOR_HEIGHT} mm\n")
    if args.demo:
        demo(args.delta, args.maxiter)
    else:
        run(args.port, args.baud, args.log, args.delta, args.maxiter)

if __name__ == "__main__":
    main()
