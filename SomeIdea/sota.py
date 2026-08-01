# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — V20: Robust & Adaptive KF
===================================================
CÁC PHƯƠNG PHÁP:
  1. Raw + LS        — baseline
  2. UKF              — Unscented Kalman Filter chuẩn
  3. Huber-UKF        — IRLS Huber M-estimator
  4. MCC-UKF          — Maximum Correntropy Criterion UKF
  5. PC-UKF-v3        — MAD Auto-Normalized, Sigma-Free (V3)
                        · Innovation từ UKF predict step (sigma points)
                        · MAD normalization tự động → không cần tune sigma
                        · Pairwise T-kernel với sigma=1.0 cố định
                        · Cố định Q=0.01, R=50 — chỉ tune r_scale
  6. GUKF             — Gaussian-smoothed UKF

Format file .txt: timestamp, d0_slant, d1_slant, d2_slant, d3_slant, v_mm_s, gz_dps
"""

import os
import glob
import math
import time
import warnings
import itertools
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import wilcoxon
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
ANCHORS = np.array([
    [8800, 0   ],   # A1
    [8800, 4000],   # A2
    [0,    4000],   # A3
    [0,    0   ],   # A4
], dtype=float)

WAYPOINTS = np.array([
    [400.0,  3200.0],   # A
    [400.0,  1000.0],   # B
    [8000.0, 1000.0],   # C
    [8000.0, 3200.0],   # D
    [400.0,  3200.0],   # A
], dtype=float)

ANCHOR_HEIGHT = 1400.0

SPEED      = 200.0
GT_SPACING = 5.0
N_ANCHORS  = 4

# ─── Standard UKF ────────────────────────────────────────────────────
UKF_STD_Q      = 0.01
UKF_STD_R      = 50.0

# ─── Huber-UKF ───────────────────────────────────────────────────────
HUBER_Q       = 0.01
HUBER_R       = 50.0
HUBER_DELTA   = 20.0
HUBER_MAXITER = 5

# ─── MCC-UKF ─────────────────────────────────────────────────────────
MCC_Q          = 0.01
MCC_R          = 50.0
MCC_KERNEL_BW  = 1700.0
MCC_MAXITER    = 5

# ─── PC-UKF-v3 (MAD Auto-Normalized, Sigma-Free) ─────────────────────
# Q và R cố định; chỉ tune r_scale
PCUKF_Q        = 0.01
PCUKF_R_BASE   = 50.0
PCUKF_R_SCALE  = 2.0    # param duy nhất cần tune

# ─── GUKF ────────────────────────────────────────────────────────────
GUKF_Q      = 0.01
GUKF_R      = 50.0
GUKF_SIGMA  = 5.0
GUKF_N_HALF = 4

# ─── UKF common ──────────────────────────────────────────────────────
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

DO_GRID_SEARCH = True
DATA_DIR = "./data"
SAVE_DIR = "./outputs_sota"

# ─── Motion sanity check ─────────────────────────────────────────────
MOTION_RATIO_MIN = 0.85


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY & GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d_slant):
    return math.sqrt(max(d_slant**2 - ANCHOR_HEIGHT**2, 0.0))


def build_ground_truth(waypoints=WAYPOINTS, speed=SPEED, spacing=GT_SPACING):
    segs       = np.diff(waypoints, axis=0)
    seg_len    = np.linalg.norm(segs, axis=1)
    total_dist = seg_len.sum()
    n_points   = max(int(total_dist / spacing), 2)
    cum_dist   = np.concatenate([[0], np.cumsum(seg_len)])
    query_dist = np.linspace(0, total_dist, n_points)
    gt = np.zeros((n_points, 2))
    for i, d in enumerate(query_dist):
        idx  = np.clip(np.searchsorted(cum_dist, d, 'right') - 1, 0, len(segs) - 1)
        frac = (d - cum_dist[idx]) / (seg_len[idx] + 1e-9)
        gt[i] = waypoints[idx] + frac * segs[idx]
    return gt, total_dist / speed


def nearest_gt_error(pos_xy, gt_xy):
    tree = cKDTree(gt_xy)
    errors, _ = tree.query(pos_xy)
    return errors


def LS_position(distances, anchors=ANCHORS, weights=None):
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi - x0), 2*(yi - y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
        w.append(weights[i] if weights is not None else 1.0)
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    W  = np.diag(w)
    try:
        pos, *_ = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _ls_position_nb(distances, anchors):
    """Giải LS position (closed-form 2×2, không dùng np.linalg.inv)."""
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
        pos[0] = np.nan; pos[1] = np.nan
    else:
        pos[0] = (AtA[1, 1]*Atb[0] - AtA[0, 1]*Atb[1]) / det
        pos[1] = (AtA[0, 0]*Atb[1] - AtA[1, 0]*Atb[0]) / det
    return pos


@njit(cache=True)
def _ls_file_nb(dist_mat, anchors):
    """Batch LS cho toàn bộ file: dist_mat (T, N) → pos (T, 2)."""
    T = dist_mat.shape[0]
    pos = np.empty((T, 2))
    for t in range(T):
        p = _ls_position_nb(dist_mat[t], anchors)
        pos[t, 0] = p[0]; pos[t, 1] = p[1]
    return pos


@njit(cache=True)
def _pc_scores_v3_nb(innovations, Pzz_diag, N, r_base):
    """
    PC scoring V3 — MAD Auto-Normalized, Sigma-Free — Numba JIT.
    innovations : (N,)  — innovation vector
    Pzz_diag    : (N,)  — diagonal of Pzz
    N           : int   — number of anchors
    r_base      : float — base measurement noise

    Bước 1: geo-normalize
    Bước 2: MAD normalize (sort-based, không dùng np.median)
    Bước 3: T-kernel pairwise (nu=4, sigma=1 cố định)
    """
    # Bước 1: geo-normalize
    std_innov = np.empty(N)
    for i in range(N):
        denom = Pzz_diag[i]
        if denom < 1e-9: denom = 1e-9
        std_innov[i] = innovations[i] / (denom ** 0.5)

    # Bước 2: MAD normalize (sort-based, Numba-compatible)
    tmp = std_innov.copy(); tmp.sort()
    if N % 2 == 0:
        med = (tmp[N//2 - 1] + tmp[N//2]) * 0.5
    else:
        med = tmp[N//2]
    abs_dev = np.empty(N)
    for i in range(N): abs_dev[i] = abs(std_innov[i] - med)
    abs_dev.sort()
    if N % 2 == 0:
        mad = (abs_dev[N//2 - 1] + abs_dev[N//2]) * 0.5
    else:
        mad = abs_dev[N//2]
    sigma_hat = 1.4826 * mad + 1e-9
    normed = np.empty(N)
    for i in range(N): normed[i] = std_innov[i] / sigma_hat

    # Bước 3: T-kernel pairwise (nu=4, sigma=1 cố định)
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
def _make_spd_nb(P, eps=1e-9):
    P = 0.5 * (P + P.T)
    n = P.shape[0]
    for i in range(n):
        P[i, i] += eps
        if P[i, i] < eps:
            P[i, i] = eps
    return P


@njit(cache=True)
def _ukf_moments_nb(x, P, Wm, Wc, c, anchors, anchor_h):
    """z_hat, Pzz (no R), Pxz từ sigma points — Numba JIT."""
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
def _ukf_step_nb(x, P, z_raw, anchors, anchor_h, Q, R, Wm, Wc, c):
    N = anchors.shape[0]
    x_pred = x.copy()
    P_pred = P + np.eye(2) * Q
    z_hat, Pzz, Pxz = _ukf_moments_nb(x_pred, P_pred, Wm, Wc, c, anchors, anchor_h)
    for i in range(N):
        Pzz[i, i] += R
    innov = z_raw - z_hat
    K = Pxz @ np.linalg.inv(Pzz)
    x_new = x_pred + K @ innov
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


@njit(cache=True)
def _huber_ukf_step_nb(x, P, z_raw, anchors, anchor_h, Q, R_base, delta, maxiter, Wm, Wc, c):
    N = anchors.shape[0]
    x_pred = x.copy()
    P_pred = P + np.eye(2) * Q
    z_hat, Pzz_no_R, Pxz = _ukf_moments_nb(x_pred, P_pred, Wm, Wc, c, anchors, anchor_h)
    R_diag = np.full(N, R_base)
    for _ in range(maxiter):
        for i in range(N):
            pzz_ii = Pzz_no_R[i, i] + R_diag[i]
            if pzz_ii < 1e-9:
                pzz_ii = 1e-9
            r_sc = (z_raw[i] - z_hat[i]) / math.sqrt(pzz_ii)
            ar = abs(r_sc)
            w = 1.0 if ar <= delta else delta / (ar + 1e-9)
            if w < 1e-4:
                w = 1e-4
            R_diag[i] = R_base / w
    Pzz = Pzz_no_R.copy()
    for i in range(N):
        Pzz[i, i] += R_diag[i]
    K = Pxz @ np.linalg.inv(Pzz)
    x_new = x_pred + K @ (z_raw - z_hat)
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


@njit(cache=True)
def _mcc_ukf_step_nb(x, P, z_raw, anchors, anchor_h, Q, R_base, kernel_bw, maxiter, Wm, Wc, c):
    N = anchors.shape[0]
    x_pred = x.copy()
    P_pred = P + np.eye(2) * Q
    z_hat, Pzz_no_R, Pxz = _ukf_moments_nb(x_pred, P_pred, Wm, Wc, c, anchors, anchor_h)
    x_cur = x_pred.copy()
    R_diag = np.full(N, R_base)
    K = np.zeros((2, N))
    innov = z_raw - z_hat
    for _ in range(maxiter):
        for i in range(N):
            w = math.exp(-0.5 * innov[i]*innov[i] / (kernel_bw*kernel_bw + 1e-9))
            if w < 1e-4:
                w = 1e-4
            R_diag[i] = R_base / w
        Pzz = Pzz_no_R.copy()
        for i in range(N):
            Pzz[i, i] += R_diag[i]
        K = Pxz @ np.linalg.inv(Pzz)
        x_new = x_pred + K @ innov
        dx0 = x_new[0] - x_cur[0]
        dx1 = x_new[1] - x_cur[1]
        if math.sqrt(dx0*dx0 + dx1*dx1) < 1e-3:
            x_cur = x_new
            break
        x_cur = x_new
    Pzz = Pzz_no_R.copy()
    for i in range(N):
        Pzz[i, i] += R_diag[i]
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_cur, P_new


@njit(cache=True)
def _pc_ukf_step_nb(x, P, z_raw, anchors, anchor_h, Q, r_base, r_scale, Wm, Wc, c):
    N = anchors.shape[0]
    x_pred = x.copy()
    P_pred = P + np.eye(2) * Q
    z_hat, Pzz_no_R, Pxz = _ukf_moments_nb(x_pred, P_pred, Wm, Wc, c, anchors, anchor_h)
    innov = z_raw - z_hat
    S_diag = np.empty(N)
    for i in range(N):
        S_diag[i] = Pzz_no_R[i, i] + r_base
    scores = _pc_scores_v3_nb(innov, S_diag, N, r_base)
    Pzz = Pzz_no_R.copy()
    for i in range(N):
        Pzz[i, i] += r_base * (1.0 + r_scale * (1.0 - scores[i]))
    K = Pxz @ np.linalg.inv(Pzz)
    x_new = x_pred + K @ innov
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


@njit(cache=True)
def _g_ukf_step_nb(x, P, z_raw, anchors, anchor_h, Q, R, buf, buf_len, kernel, win, Wm, Wc, c):
    N = anchors.shape[0]
    if buf_len < win:
        for j in range(N):
            buf[buf_len, j] = z_raw[j]
        buf_len += 1
    else:
        for i in range(win - 1):
            for j in range(N):
                buf[i, j] = buf[i + 1, j]
        for j in range(N):
            buf[win - 1, j] = z_raw[j]
    z_s = np.zeros(N)
    if buf_len < win:
        wsum = 0.0
        for i in range(buf_len):
            w = kernel[win - buf_len + i]
            wsum += w
            for j in range(N):
                z_s[j] += w * buf[i, j]
        for j in range(N):
            z_s[j] /= (wsum + 1e-18)
    else:
        for i in range(win):
            for j in range(N):
                z_s[j] += kernel[i] * buf[i, j]
    x_pred = x.copy()
    P_pred = P + np.eye(2) * Q
    z_hat, Pzz, Pxz = _ukf_moments_nb(x_pred, P_pred, Wm, Wc, c, anchors, anchor_h)
    for i in range(N):
        Pzz[i, i] += R
    K = Pxz @ np.linalg.inv(Pzz)
    x_new = x_pred + K @ (z_s - z_hat)
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new, buf, buf_len


def _warmup_numba():
    """Pre-compile tất cả @njit kernels tại import time."""
    _d = np.array([1000.0, 1200.0, 800.0, 900.0])
    _a = ANCHORS.copy()
    _ls_position_nb(_d, _a)
    _ls_file_nb(np.stack([_d]), _a)
    _innov = np.array([10.0, -5.0, 200.0, 3.0])
    _pzz   = np.array([60.0, 60.0, 60.0, 60.0])
    _pc_scores_v3_nb(_innov, _pzz, 4, 50.0)
    Wm, Wc, c = ukf_weights(2)
    _x0 = np.array([2000.0, 2000.0])
    _P0 = np.eye(2) * 1e6
    _ukf_step_nb(_x0, _P0, _d, _a, ANCHOR_HEIGHT, 0.01, 50.0, Wm, Wc, c)
    _huber_ukf_step_nb(_x0, _P0, _d, _a, ANCHOR_HEIGHT, 0.01, 50.0, 20.0, 5, Wm, Wc, c)
    _mcc_ukf_step_nb(_x0, _P0, _d, _a, ANCHOR_HEIGHT, 0.01, 50.0, 1700.0, 5, Wm, Wc, c)
    _pc_ukf_step_nb(_x0, _P0, _d, _a, ANCHOR_HEIGHT, 0.01, 50.0, 2.0, Wm, Wc, c)
    _ker = _make_gaussian_kernel(4, 5.0)
    _buf = np.zeros((9, 4))
    _g_ukf_step_nb(_x0, _P0, _d, _a, ANCHOR_HEIGHT, 0.01, 50.0, _buf, 0, _ker, 9, Wm, Wc, c)


# NOTE: ukf_weights / _make_gaussian_kernel defined below — warmup deferred to after them.


# ══════════════════════════════════════════════════════════════════════
#  MOTION SANITY CHECK
# ══════════════════════════════════════════════════════════════════════
def compute_trajectory_displacement(pos_xy):
    if len(pos_xy) < 2:
        return 0.0
    diffs = np.diff(pos_xy, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def is_trajectory_collapsed(pos_xy, expected_path_length,
                             ratio_min=MOTION_RATIO_MIN):
    if len(pos_xy) < 2:
        return True
    total_disp = compute_trajectory_displacement(pos_xy)
    if total_disp < ratio_min * expected_path_length:
        return True
    bbox_x = pos_xy[:, 0].max() - pos_xy[:, 0].min()
    bbox_y = pos_xy[:, 1].max() - pos_xy[:, 1].min()
    bbox_diag = math.sqrt(bbox_x**2 + bbox_y**2)
    if bbox_diag < 0.05 * expected_path_length:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
#  OBSERVATION MODEL & UKF UTILITIES
# ══════════════════════════════════════════════════════════════════════
def h_obs(state, anchors=ANCHORS):
    x, y = state[0], state[1]
    return np.array([
        math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
        for ax, ay in anchors
    ])


def ukf_weights(n, alpha=UKF_ALPHA, beta=UKF_BETA, kappa=UKF_KAPPA):
    lam = alpha**2 * (n + kappa) - n
    c   = n + lam
    Wm  = np.full(2*n + 1, 0.5 / c)
    Wc  = np.full(2*n + 1, 0.5 / c)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)
    return Wm.astype(np.float64), Wc.astype(np.float64), float(c)


def sigma_points(x, P, c):
    """Python wrapper — giữ API; hot path dùng _ukf_moments_nb."""
    n = len(x)
    try:
        S = np.linalg.cholesky(c * P)
    except np.linalg.LinAlgError:
        S = np.linalg.cholesky(c * (P + np.eye(n) * 1e-6))
    pts = np.zeros((2*n + 1, n))
    pts[0] = x
    for i in range(n):
        pts[i + 1]     = x + S[:, i]
        pts[n + i + 1] = x - S[:, i]
    return pts


def ukf_measurement_moments(x_pred, P_pred, Wm, Wc, c):
    return _ukf_moments_nb(
        np.asarray(x_pred, dtype=np.float64),
        np.asarray(P_pred, dtype=np.float64),
        np.asarray(Wm, dtype=np.float64),
        np.asarray(Wc, dtype=np.float64),
        float(c), ANCHORS, ANCHOR_HEIGHT)


def make_spd(P, eps=1e-9):
    return _make_spd_nb(np.asarray(P, dtype=np.float64).copy(), eps)


def default_init(dist_raw_0):
    pos = LS_position(dist_raw_0)
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


def _make_gaussian_kernel(n_half, sigma):
    idx = np.arange(-n_half, n_half + 1, dtype=float)
    h   = np.exp(-0.5 * idx**2 / (sigma**2 + 1e-12))
    return (h / h.sum()).astype(np.float64)


# Warmup sau khi ukf_weights / _make_gaussian_kernel đã có
_warmup_numba()


# ══════════════════════════════════════════════════════════════════════
#  1. STANDARD UKF  (JIT hot path)
# ══════════════════════════════════════════════════════════════════════
class StandardUKF:
    def __init__(self, q=UKF_STD_Q, r=UKF_STD_R):
        self.n = 2
        self.q = float(q)
        self.r = float(r)
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(self.n, dtype=np.float64) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        self.x, self.P = _ukf_step_nb(
            self.x, self.P, np.asarray(z_raw, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r,
            self.Wm, self.Wc, self.c)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  3. HUBER-UKF  (JIT hot path)
# ══════════════════════════════════════════════════════════════════════
class HuberUKF:
    def __init__(self, q=HUBER_Q, r=HUBER_R,
                 delta=HUBER_DELTA, maxiter=HUBER_MAXITER):
        self.n = 2
        self.q = float(q)
        self.r = float(r)
        self.delta = float(delta)
        self.maxiter = int(maxiter)
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(self.n, dtype=np.float64) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        self.x, self.P = _huber_ukf_step_nb(
            self.x, self.P, np.asarray(z_raw, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r,
            self.delta, self.maxiter, self.Wm, self.Wc, self.c)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  4. MCC-UKF  (JIT hot path)
# ══════════════════════════════════════════════════════════════════════
class MCCUKF:
    def __init__(self, q=MCC_Q, r=MCC_R,
                 kernel_bw=MCC_KERNEL_BW, maxiter=MCC_MAXITER):
        self.n = 2
        self.q = float(q)
        self.r = float(r)
        self.kernel_bw = float(kernel_bw)
        self.maxiter = int(maxiter)
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(self.n, dtype=np.float64) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        self.x, self.P = _mcc_ukf_step_nb(
            self.x, self.P, np.asarray(z_raw, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r,
            self.kernel_bw, self.maxiter, self.Wm, self.Wc, self.c)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  5. PC-UKF-v3 — MAD Auto-Normalized, Sigma-Free  (JIT hot path)
# ══════════════════════════════════════════════════════════════════════
def _t_kernel_v3(diff_vec, sigma=1.0):
    """T-distribution kernel với sigma=1.0 cố định."""
    nu  = 4.0
    eps = 1e-9
    return (1.0 + diff_vec**2 / (nu * sigma**2 + eps)) ** (-(nu + 1.0) / 2.0)


def pc_scores_v3(innovations, Pzz_diag):
    """Python wrapper → _pc_scores_v3_nb (cùng logic)."""
    return _pc_scores_v3_nb(
        np.asarray(innovations, dtype=np.float64),
        np.asarray(Pzz_diag, dtype=np.float64),
        len(innovations), 50.0)


class PCUKFv3:
    """
    PC-UKF V3: UKF-2D với PC scoring MAD auto-normalized (JIT).

    Cố định: Q=0.01, R_base=50
    Chỉ tune: r_scale
    """
    def __init__(self, q=PCUKF_Q, r_base=PCUKF_R_BASE, r_scale=PCUKF_R_SCALE):
        self.n = 2
        self.q = float(q)
        self.R_base = float(r_base)
        self.R_scale = float(r_scale)
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(self.n, dtype=np.float64) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        self.x, self.P = _pc_ukf_step_nb(
            self.x, self.P, np.asarray(z_raw, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.R_base, self.R_scale,
            self.Wm, self.Wc, self.c)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  6. GUKF — Gaussian Unscented Kalman Filter  (JIT hot path)
# ══════════════════════════════════════════════════════════════════════
class GUKF:
    def __init__(self, q=GUKF_Q, r=GUKF_R,
                 sigma=GUKF_SIGMA, n_half=GUKF_N_HALF):
        self.n = 2
        self.q = float(q)
        self.r = float(r)
        self.kernel = _make_gaussian_kernel(n_half, sigma)
        self.n_half = int(n_half)
        self.win = 2 * self.n_half + 1
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None
        self._buf = np.zeros((self.win, N_ANCHORS), dtype=np.float64)
        self._buf_len = 0

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(self.n, dtype=np.float64) * 1e6
        self._buf = np.zeros((self.win, N_ANCHORS), dtype=np.float64)
        self._buf_len = 0

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        self.x, self.P, self._buf, self._buf_len = _g_ukf_step_nb(
            self.x, self.P, np.asarray(z_raw, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r,
            self._buf, self._buf_len, self.kernel, self.win,
            self.Wm, self.Wc, self.c)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  FILTER WRAPPERS
# ══════════════════════════════════════════════════════════════════════
def run_filter(filt_class, dist_raw, **kwargs):
    T    = len(dist_raw)
    filt = filt_class(**kwargs)
    pos  = np.full((T, 2), np.nan)
    x0   = default_init(dist_raw[0])
    filt.init(x0)
    for t in range(T):
        pos[t] = filt.step(dist_raw[t])
    return pos


# ══════════════════════════════════════════════════════════════════════
#  PARSE FILE
# ══════════════════════════════════════════════════════════════════════
def parse_file(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                d_slant  = [float(parts[i + 1]) for i in range(4)]
                d_ground = [slant_to_ground(d) for d in d_slant]
                rows.append(d_ground)
            except ValueError:
                continue
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def safe_len(path):
    data = parse_file(path)
    return len(data) if data is not None else 0


# ══════════════════════════════════════════════════════════════════════
#  METHODS TABLE
# ══════════════════════════════════════════════════════════════════════
METHODS = {
    "Raw+LS"     : None,
    "UKF"        : (StandardUKF, dict(q=UKF_STD_Q,  r=UKF_STD_R)),
    "Huber-UKF"  : (HuberUKF,   dict(q=HUBER_Q,     r=HUBER_R,    delta=HUBER_DELTA)),
    "MCC-UKF"    : (MCCUKF,     dict(q=MCC_Q,       r=MCC_R,      kernel_bw=MCC_KERNEL_BW)),
    "PC-UKF-v3"  : (PCUKFv3,    dict(q=PCUKF_Q,     r_base=PCUKF_R_BASE,
                                      r_scale=PCUKF_R_SCALE)),
    "GUKF"       : (GUKF,        dict(q=GUKF_Q,      r=GUKF_R,
                                      sigma=GUKF_SIGMA, n_half=GUKF_N_HALF)),
}

COLORS = {
    "Raw+LS"     : '#9E9E9E',
    "UKF"        : '#00BCD4',
    "Huber-UKF"  : '#E91E63',
    "MCC-UKF"    : '#FF9800',
    "PC-UKF-v3"  : '#9C27B0',
    "GUKF"       : '#4CAF50',
}


# ══════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════
def evaluate_files(file_paths, gt_xy, expected_path_length):
    all_errors    = {m: [] for m in METHODS}
    all_pos       = {m: [] for m in METHODS}
    per_file_rmse = {m: [] for m in METHODS}
    collapse_warn = {m: [] for m in METHODS}
    timing        = {m: [] for m in METHODS if m != "Raw+LS"}

    hdr = f"{'File':<18s}"
    for m in METHODS:
        hdr += f" {m:>12s}"
    print("\n" + "═" * 85)
    print("  PER-FILE RMSE (mm)   [⚠ = trajectory collapsed / static]")
    print("═" * 85)
    print(hdr)
    print("─" * 85)

    for path in file_paths:
        dist_raw = parse_file(path)
        if dist_raw is None:
            continue
        T   = len(dist_raw)
        row = f"{os.path.basename(path):<18s}"

        for name, method in METHODS.items():
            if method is None:
                # Raw+LS: JIT batch — nhanh hơn list comprehension Python
                pos = _ls_file_nb(dist_raw.astype(np.float64), ANCHORS)
            else:
                filt_cls, kwargs = method
                t0  = time.perf_counter()
                pos = run_filter(filt_cls, dist_raw, **kwargs)
                timing[name].append(time.perf_counter() - t0)

            valid = pos[~np.any(np.isnan(pos), axis=1)]

            collapsed = is_trajectory_collapsed(valid, expected_path_length)
            if collapsed:
                collapse_warn[name].append(os.path.basename(path))

            errs  = nearest_gt_error(valid, gt_xy)
            rmse  = np.sqrt(np.mean(errs**2)) if len(errs) > 0 else float('nan')
            flag  = "⚠" if collapsed else " "
            row  += f" {flag}{rmse:>10.1f}"
            all_errors[name].extend(errs)
            all_pos[name].extend(valid)
            per_file_rmse[name].append(rmse)

        print(row)

    print("─" * 85)

    print("\n  TIMING (ms/sample):")
    for name, times in timing.items():
        if not times:
            continue
        total_t = sum(times)
        n_samp  = sum(safe_len(p) for p in file_paths[:len(times)])
        if n_samp > 0:
            print(f"    {name:<14s}: {total_t * 1000 / n_samp:.4f} ms/sample")

    print("\n  COLLAPSE WARNINGS (trajectory đứng yên):")
    any_warn = False
    for name, files in collapse_warn.items():
        if files:
            any_warn = True
            print(f"    ⚠  {name:<14s}: {len(files)} file(s) → {', '.join(files[:5])}"
                  + (" ..." if len(files) > 5 else ""))
    if not any_warn:
        print("    ✅ Không có trajectory nào bị collapse.")

    return (
        {m: np.array(v) for m, v in all_errors.items()},
        {m: np.array(v) for m, v in all_pos.items()},
        collapse_warn,
        per_file_rmse,
    )


def compute_metrics(errors, pos_xy=None, expected_path_length=None):
    if len(errors) == 0:
        return {k: float('nan') for k in
                ['mae', 'rmse', 'cep50', 'cep90', 'p95', 'max',
                 'motion_ratio', 'collapsed']}
    base = {
        'mae'  : float(np.mean(errors)),
        'rmse' : float(np.sqrt(np.mean(errors**2))),
        'cep50': float(np.percentile(errors, 50)),
        'cep90': float(np.percentile(errors, 90)),
        'p95'  : float(np.percentile(errors, 95)),
        'max'  : float(np.max(errors)),
    }
    if pos_xy is not None and expected_path_length is not None and len(pos_xy) > 1:
        disp  = compute_trajectory_displacement(pos_xy)
        ratio = disp / (expected_path_length + 1e-9)
        base['motion_ratio'] = round(ratio, 3)
        base['collapsed']    = is_trajectory_collapsed(pos_xy, expected_path_length)
    else:
        base['motion_ratio'] = float('nan')
        base['collapsed']    = False
    return base


# ══════════════════════════════════════════════════════════════════════
#  GRID SEARCH
# ══════════════════════════════════════════════════════════════════════
COLLAPSE_PENALTY = 1e9


def _eval_rmse_single(filt_class, kwargs, file_paths, gt_xy, expected_path_length):
    pos_all     = []
    n_collapsed = 0
    n_files     = 0
    for path in file_paths:
        d = parse_file(path)
        if d is None:
            continue
        n_files += 1
        p = run_filter(filt_class, d, **kwargs)
        valid = p[~np.any(np.isnan(p), axis=1)]
        if is_trajectory_collapsed(valid, expected_path_length):
            n_collapsed += 1
        else:
            pos_all.extend(valid)
    if n_files == 0:
        return float('inf')
    collapse_rate = n_collapsed / n_files
    if collapse_rate > 0.30:
        return COLLAPSE_PENALTY + collapse_rate
    if not pos_all:
        return float('inf')
    errs = nearest_gt_error(np.array(pos_all), gt_xy)
    rmse = float(np.sqrt(np.mean(errs**2)))
    rmse += n_collapsed * 500.0
    return rmse


def grid_search_huber(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'    : [50.0],
        'delta': [1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search Huber-UKF: {len(combos)} combinations...")
    best_rmse = float('inf')
    best = dict(q=HUBER_Q, r=HUBER_R, delta=HUBER_DELTA)
    for combo in combos:
        params = dict(zip(keys, combo))
        params['q'] = HUBER_Q
        rmse = _eval_rmse_single(HuberUKF, params, file_paths, gt_xy, expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best Huber-UKF: RMSE={best_rmse:.1f}mm{tag} | r={best['r']} delta={best['delta']}")
    return best


def grid_search_mcc(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'        : [50.0],
        'kernel_bw': [100.0, 200.0, 500.0, 1100.0, 1200.0, 1300.0, 1500.0, 1700.0, 2000.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search MCC-UKF: {len(combos)} combinations...")
    best_rmse = float('inf')
    best = dict(q=MCC_Q, r=MCC_R, kernel_bw=MCC_KERNEL_BW)
    for combo in combos:
        params = dict(zip(keys, combo))
        params['q'] = MCC_Q
        rmse = _eval_rmse_single(MCCUKF, params, file_paths, gt_xy, expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best MCC-UKF: RMSE={best_rmse:.1f}mm{tag} | r={best['r']} kernel_bw={best['kernel_bw']}")
    return best


def grid_search_pcukf_v3(file_paths, gt_xy, expected_path_length):
    """
    Grid search PC-UKF-v3:
    - Q=0.01 cố định, R_base=50 cố định
    - Chỉ tune r_scale (param duy nhất của V3)
    """
    r_scale_grid = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    print(f"\n  Grid search PC-UKF-v3 (Q=0.01 cố định, R=50 cố định): "
          f"{len(r_scale_grid)} combinations (chỉ tune r_scale)...")
    best_rmse = float('inf')
    best = dict(q=PCUKF_Q, r_base=PCUKF_R_BASE, r_scale=PCUKF_R_SCALE)
    for r_scale in r_scale_grid:
        params = dict(q=PCUKF_Q, r_base=PCUKF_R_BASE, r_scale=r_scale)
        rmse = _eval_rmse_single(PCUKFv3, params, file_paths, gt_xy, expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best PC-UKF-v3: RMSE={best_rmse:.1f}mm{tag} | "
          f"Q={best['q']} (fixed) R_base={best['r_base']} (fixed) r_scale={best['r_scale']}")
    return best


def grid_search_gukf(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'      : [50.0],
        'sigma'  : [0.5, 1.0, 2.0, 3.0, 5.0],
        'n_half' : [1, 2, 3, 4],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search GUKF: {len(combos)} combinations...")
    best_rmse = float('inf')
    best = dict(q=GUKF_Q, r=GUKF_R, sigma=GUKF_SIGMA, n_half=GUKF_N_HALF)
    for combo in combos:
        params = dict(zip(keys, combo))
        params['q'] = GUKF_Q
        rmse = _eval_rmse_single(GUKF, params, file_paths, gt_xy, expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best GUKF: RMSE={best_rmse:.1f}mm{tag} | r={best['r']} sigma={best['sigma']} n_half={best['n_half']}")
    return best


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_cdf(errors_dict, collapse_warn, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, errors in errors_dict.items():
        if len(errors) == 0:
            continue
        s    = np.sort(errors)
        cdf  = np.arange(1, len(s) + 1) / len(s)
        rmse = np.sqrt(np.mean(errors**2))
        c    = COLORS.get(label, 'gray')
        lw   = 2.5 if label not in ("Raw+LS",) else 1.5
        ls   = '-'  if label not in ("Raw+LS",) else '--'
        n_col = len(collapse_warn.get(label, []))
        warn_tag = f" ⚠{n_col}" if n_col > 0 else ""
        ax.plot(s, cdf, lw=lw, color=c, ls=ls,
                label=f"{label}{warn_tag}  RMSE={rmse:.1f}mm")
        ax.axvline(rmse, color=c, ls=':', lw=0.8, alpha=0.4)
    ax.set_xlabel("Position Error (mm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] CDF → {save_path}")
    plt.close()


def plot_trajectories(positions_dict, gt_xy, collapse_warn, save_path):
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'k--', lw=2.5, label='Ground Truth', alpha=0.5, zorder=5)
    for label, pos in positions_dict.items():
        color = COLORS.get(label, 'gray')
        if len(pos) == 0:
            continue
        errs = nearest_gt_error(pos, gt_xy)
        rmse = np.sqrt(np.mean(errs**2))
        n_col = len(collapse_warn.get(label, []))
        warn = f" ⚠{n_col}" if n_col > 0 else ""
        lw = 2.0 if label != "Raw+LS" else 1.2
        ls = '-'  if label != "Raw+LS" else '--'
        ax.plot(pos[:, 0], pos[:, 1],
                color=color, lw=lw, ls=ls, alpha=0.75,
                label=f"{label}{warn}  RMSE={rmse:.1f}mm")
    for nm, pt in zip(["A", "B", "C", "D"], WAYPOINTS[:4]):
        ax.scatter(*pt, s=90, color='black', zorder=10)
        ax.annotate(nm, pt, textcoords="offset points",
                    xytext=(6, 4), fontsize=13, fontweight='bold')
    for j, (ax_, ay_) in enumerate(ANCHORS):
        ax.scatter(ax_, ay_, s=100, marker='s', color='red', zorder=10)
        ax.annotate(f"A{j+1}", (ax_, ay_), textcoords="offset points",
                    xytext=(5, 5), fontsize=10, color='red')
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.set_aspect('equal')
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Trajectories → {save_path}")
    plt.close()


def plot_bar(metrics_dict, save_path):
    methods      = list(metrics_dict.keys())
    metric_names = ['rmse', 'mae', 'cep50', 'p95']
    labels_disp  = ['RMSE', 'MAE', 'CEP50', 'P95']
    fig, axes    = plt.subplots(1, 4, figsize=(24, 6))
    for ax, mname, mlabel in zip(axes, metric_names, labels_disp):
        vals    = [metrics_dict[m][mname] for m in methods]
        colors  = [COLORS.get(m, 'gray') for m in methods]
        hatches = ['//' if metrics_dict[m].get('collapsed', False) else '' for m in methods]
        bars    = ax.bar(range(len(methods)), vals, color=colors,
                         alpha=0.85, edgecolor='white', lw=1.5)
        for bar, val, hatch, m in zip(bars, vals, hatches, methods):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor('darkred')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-', '\n') for m in methods],
                           fontsize=9, ha='center')
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  V20: Robust & Adaptive UKF — UWB Indoor Positioning")
    print("=" * 70)
    print("  1. Raw+LS       — Weighted Least Squares (baseline)")
    print("  2. UKF           — Unscented Kalman Filter chuẩn")
    print("  3. Huber-UKF     — IRLS với Huber M-estimator")
    print("  4. MCC-UKF       — Maximum Correntropy Criterion UKF")
    print("  5. PC-UKF-v3     — MAD Auto-Normalized, Sigma-Free")
    print("                     · Q=0.01 cố định, R=50 cố định")
    print("                     · Chỉ tune r_scale")
    print("  6. GUKF          — Gaussian-smoothed UKF")
    print("=" * 70)
    print(f"\n  Motion collapse detection: ratio_min={MOTION_RATIO_MIN:.0%}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"\n[!] Không tìm thấy file .txt trong '{DATA_DIR}/'")
        return
    print(f"\n  Tìm thấy {n} file(s) trong '{DATA_DIR}/'")

    np.random.seed(42)
    perm       = np.random.permutation(n)
    eval_files = [all_files[i] for i in perm]

    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    expected_path_length = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"  Path={expected_path_length:.0f}mm  Time={total_time:.1f}s  GT={len(gt_xy)} points")

    # ── Grid search ────────────────────────────────────────────────────
    if DO_GRID_SEARCH:
        print("\n  [Grid Search] Tuning tất cả filter (với motion penalty)...")

        best_huber = grid_search_huber(eval_files, gt_xy, expected_path_length)
        best_mcc   = grid_search_mcc(eval_files, gt_xy, expected_path_length)
        best_pc    = grid_search_pcukf_v3(eval_files, gt_xy, expected_path_length)
        best_gukf  = grid_search_gukf(eval_files, gt_xy, expected_path_length)

        METHODS["Huber-UKF"] = (HuberUKF, best_huber)
        METHODS["MCC-UKF"]   = (MCCUKF,   best_mcc)
        METHODS["PC-UKF-v3"] = (PCUKFv3,  best_pc)
        METHODS["GUKF"]      = (GUKF,      best_gukf)

    # ── Evaluate ──────────────────────────────────────────────────────
    errors, positions, collapse_warn, per_file_rmse = evaluate_files(
        eval_files, gt_xy, expected_path_length)

    # ── Summary table ─────────────────────────────────────────────────
    metrics = {}
    print(f"\n{'═' * 105}")
    print(f"  SUMMARY TABLE (mm)   — ⚠ = trajectory collapsed (motion_ratio < {MOTION_RATIO_MIN:.0%})")
    print(f"{'═' * 105}")
    print(f"  {'Method':<14s} {'RMSE mean±std':>18s} {'MAE':>7s} {'CEP50':>7s} {'CEP90':>7s} "
          f"{'P95':>7s} {'MAX':>7s} {'MotionR':>8s} {'Status'}")
    print(f"  {'─' * 87}")

    for label, errs in errors.items():
        if len(errs) == 0:
            continue
        pos_arr = positions.get(label, np.empty((0, 2)))
        m = compute_metrics(errs, pos_arr, expected_path_length)
        metrics[label] = m

        pf_vals = [v for v in per_file_rmse.get(label, []) if not math.isnan(v)]
        if len(pf_vals) >= 2:
            pf_mean = np.mean(pf_vals)
            pf_std  = np.std(pf_vals, ddof=1)
            std_str = f"{pf_mean:.1f} ± {pf_std:.1f}"
        elif len(pf_vals) == 1:
            std_str = f"{pf_vals[0]:.1f} ± N/A"
        else:
            std_str = "N/A"

        n_col  = len(collapse_warn.get(label, []))
        status = f"⚠ {n_col} file(s) collapsed" if n_col > 0 else "✅ OK"
        mr_str = (f"{m['motion_ratio']:.2f}"
                  if not math.isnan(m.get('motion_ratio', float('nan'))) else "N/A")
        print(f"  {label:<14s} {std_str:>18s} {m['mae']:>7.1f}"
              f" {m['cep50']:>7.1f} {m['cep90']:>7.1f} {m['p95']:>7.1f} {m['max']:>7.1f}"
              f"  {mr_str:>8s}  {status}")

    # ── Wilcoxon ──────────────────────────────────────────────────────
    print(f"\n  Wilcoxon tests (two-sided, vs PC-UKF-v3):")
    ref_err = errors.get("PC-UKF-v3", np.array([]))
    for name, errs in errors.items():
        if name == "PC-UKF-v3" or len(errs) == 0 or len(ref_err) == 0:
            continue
        N = min(len(errs), len(ref_err))
        if N > 20:
            try:
                _, p = wilcoxon(errs[:N], ref_err[:N])
                sym  = '✅ p<0.05' if p < 0.05 else '⚠️  ns'
                print(f"    {name:<14s} vs PC-UKF-v3: p={p:.4f}  {sym}")
            except ValueError:
                pass

    # ── Plots ─────────────────────────────────────────────────────────
    plot_cdf(errors, collapse_warn, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(positions, gt_xy, collapse_warn,
                      os.path.join(SAVE_DIR, 'trajectories.png'))
    plot_bar(metrics, os.path.join(SAVE_DIR, 'bar_comparison.png'))

    for label, errs in errors.items():
        safe = label.lower().replace('+', '_').replace('-', '_').replace(' ', '_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errs)

    print(f"\n[✓] Kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
