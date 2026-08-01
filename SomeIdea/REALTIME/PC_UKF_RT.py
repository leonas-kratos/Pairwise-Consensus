# -*- coding: utf-8 -*-
"""
PC_UKF_RT.py — PC-UKF-v3 Realtime (Numba JIT)
===============================================
Đọc /dev/serial/by-id/... @ 921600, in X Y ra console, ghi CSV.

Thuật toán: UKF + Pairwise Consensus V3 (MAD auto-normalized T-kernel → adaptive R).

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
PC_R_SCALE = 2.0    # from sota PCUKFv3
LOG_FILE   = "PC_UKF_realtime_log.csv"
TAG = "PC_UKF"


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


UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

def _ukf_weights(n, alpha=UKF_ALPHA, beta=UKF_BETA, kappa=UKF_KAPPA):
    lam = alpha**2 * (n + kappa) - n
    c   = n + lam
    Wm  = np.full(2 * n + 1, 0.5 / c)
    Wc  = np.full(2 * n + 1, 0.5 / c)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return Wm.astype(np.float64), Wc.astype(np.float64), float(c)

WM, WC, C_SPREAD = _ukf_weights(2)


@njit(cache=True)
def _ukf_moments_nb(x, P, Wm, Wc, c, anchors, anchor_h):
    n = 2
    N = anchors.shape[0]
    n_sig = 2 * n + 1
    P_spd = _make_spd_nb(P.copy(), 1e-6)
    S = np.linalg.cholesky(c * P_spd)
    pts = np.empty((n_sig, n))
    pts[0, 0] = x[0]; pts[0, 1] = x[1]
    for i in range(n):
        pts[i + 1, 0]     = x[0] + S[0, i]
        pts[i + 1, 1]     = x[1] + S[1, i]
        pts[n + i + 1, 0] = x[0] - S[0, i]
        pts[n + i + 1, 1] = x[1] - S[1, i]
    Z = np.empty((n_sig, N))
    for i in range(n_sig):
        for j in range(N):
            dx = pts[i, 0] - anchors[j, 0]
            dy = pts[i, 1] - anchors[j, 1]
            Z[i, j] = math.sqrt(dx*dx + dy*dy + anchor_h*anchor_h)
    z_hat = np.zeros(N)
    for i in range(n_sig):
        for j in range(N):
            z_hat[j] += Wm[i] * Z[i, j]
    Pzz = np.zeros((N, N))
    Pxz = np.zeros((n, N))
    for i in range(n_sig):
        for a in range(N):
            dz_a = Z[i, a] - z_hat[a]
            for b in range(N):
                Pzz[a, b] += Wc[i] * dz_a * (Z[i, b] - z_hat[b])
            for a2 in range(n):
                Pxz[a2, a] += Wc[i] * (pts[i, a2] - x[a2]) * dz_a
    return z_hat, Pzz, Pxz


@njit(cache=True)
def _pc_scores_v3_nb(innovations, S_diag, N):
    std_innov = np.empty(N)
    for i in range(N):
        denom = S_diag[i]
        if denom < 1e-9: denom = 1e-9
        std_innov[i] = innovations[i] / (denom ** 0.5)
    tmp = std_innov.copy(); tmp.sort()
    if N % 2 == 0:
        med = (tmp[N//2 - 1] + tmp[N//2]) * 0.5
    else:
        med = tmp[N//2]
    abs_dev = np.empty(N)
    for i in range(N):
        abs_dev[i] = abs(std_innov[i] - med)
    abs_dev.sort()
    if N % 2 == 0:
        mad = (abs_dev[N//2 - 1] + abs_dev[N//2]) * 0.5
    else:
        mad = abs_dev[N//2]
    sigma_hat = 1.4826 * mad + 1e-9
    normed = np.empty(N)
    for i in range(N):
        normed[i] = std_innov[i] / sigma_hat
    nu = 4.0; eps = 1e-9
    scores = np.empty(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if j != i:
                d = normed[i] - normed[j]
                s += (1.0 + d*d / (nu * 1.0 + eps)) ** (-(nu + 1.0) * 0.5)
        scores[i] = s / (N - 1)
    return scores


@njit(cache=True)
def _pc_ukf_step_nb(x, P, z_slant, anchors, anchor_h, Q, r_base, r_scale, Wm, Wc, c):
    N = anchors.shape[0]
    x_pred = x.copy()
    P_pred = P + np.eye(2) * Q
    z_hat, Pzz_no_R, Pxz = _ukf_moments_nb(x_pred, P_pred, Wm, Wc, c, anchors, anchor_h)
    innov = z_slant - z_hat
    S_diag = np.empty(N)
    for i in range(N):
        S_diag[i] = Pzz_no_R[i, i] + r_base
    scores = _pc_scores_v3_nb(innov, S_diag, N)
    Pzz = Pzz_no_R.copy()
    for i in range(N):
        Pzz[i, i] += r_base * (1.0 + r_scale * (1.0 - scores[i]))
    K = Pxz @ np.linalg.inv(Pzz)
    x_new = x_pred + K @ innov
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


def _make_state(**kw):
    return None

def _step(x, P, z, state, r_scale=None):
    if r_scale is None: r_scale = PC_R_SCALE
    x, P = _pc_ukf_step_nb(x, P, z, ANCHORS, ANCHOR_HEIGHT, Q, R, r_scale, WM, WC, C_SPREAD)
    return x, P, state

def _warmup():
    print("[PC_UKF_RT] Warming up JIT kernels...", end=" ", flush=True)
    _z = np.array([1000.0, 1200.0, 800.0, 900.0])
    _ls_position_nb(np.array([slant_to_ground_py(d) for d in _z]), ANCHORS)
    _pc_ukf_step_nb(np.array([3000.0,2000.0]), np.eye(2)*1e6, _z, ANCHORS, ANCHOR_HEIGHT,
                    Q, R, PC_R_SCALE, WM, WC, C_SPREAD)
    print("OK")


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


def run(port, baud, log_file, **kw):
    _warmup()
    state = _make_state(**kw)
    x = None
    P = np.eye(2) * 1e6
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
                    print(f"[{TAG}] init  x={x[0]:.1f}  y={x[1]:.1f}")

                t_step = time.perf_counter()
                x, P, state = _step(x, P, frame, state, **kw)
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


def demo(**kw):
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

    x = None; P = np.eye(2) * 1e6; errors = []; state = _make_state(**kw)
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
        x, P, state = _step(x, P, z, state, **kw)
        dt = (time.perf_counter() - t0_) * 1e6
        err = math.sqrt((x[0]-gt[0])**2 + (x[1]-gt[1])**2)
        errors.append(err)
        if i % 50 == 0 or i == len(gt_pts) - 1:
            print(f"{i:5d}  {x[0]:9.1f}  {x[1]:9.1f}  {err:8.1f}  {dt:6.1f}")
    errors = np.array(errors)
    print(f"\nRMSE={np.sqrt(np.mean(errors**2)):.1f}mm  "
          f"MAE={np.mean(errors):.1f}mm  P95={np.percentile(errors,95):.1f}mm")


def main():
    ap = argparse.ArgumentParser(description="PC_UKF Realtime — Numba JIT")
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=BAUD_RATE)
    ap.add_argument("--log", default=LOG_FILE)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--r-scale", type=float, default=PC_R_SCALE)
    args = ap.parse_args()
    print("PC_UKF_RT — Pairwise Consensus UKF v3 (Numba JIT)")
    print(f"  Q={Q}  R={R}  (hardcoded)  r_scale={args.r_scale}")
    print(f"  Anchors: {[hex(i) for i in ANCHOR_IDS]}")
    print(f"  Height : {ANCHOR_HEIGHT} mm\n")
    kw = dict(r_scale=args.r_scale)
    if args.demo: demo(**kw)
    else: run(args.port, args.baud, args.log, **kw)

if __name__ == "__main__":
    main()
