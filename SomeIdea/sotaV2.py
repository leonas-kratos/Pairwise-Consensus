# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — SOTA V2
=================================
Tất cả filter dùng R=50, Q=0.01 cố định.
Chỉ tune tham số đặc trưng của từng phương pháp.
Toàn bộ hot-path dùng Numba @njit.

CÁC PHƯƠNG PHÁP:
  1.  Raw + LS          — baseline
  2.  UKF               — Unscented Kalman Filter chuẩn (không tune)
  3.  Huber-UKF         — IRLS Huber M-estimator          | tune: delta
  4.  MCC-UKF           — Maximum Correntropy Criterion    | tune: kernel_bw
  5.  PC-UKF-v4         — Gaussian Pairwise Consistency    | tune: r_scale
  6.  Robust-UKF        — Chi2 gating + R inflate          | tune: chi2_gate, r_inflate

Format file .txt: timestamp, d0_slant, d1_slant, d2_slant, d3_slant, v_mm_s, gz_dps
"""

import os, glob, math, time, warnings, itertools
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════
#  CONFIG — R=50, Q=0.01 TOÀN CỤC
# ══════════════════════════════════════════════════════════════════════
ANCHORS = np.array([
    [8800, 0   ],
    [8800, 4000],
    [0,    4000],
    [0,    0   ],
], dtype=float)

WAYPOINTS = np.array([
    [400.0,  3200.0],
    [400.0,  1000.0],
    [8000.0, 1000.0],
    [8000.0, 3200.0],
    [400.0,  3200.0],
], dtype=float)

ANCHOR_HEIGHT   = 1400.0
SPEED           = 200.0
GT_SPACING      = 5.0
N_ANCHORS       = 4

# ── Cố định cho tất cả filter ────────────────────────────────────────
FIXED_Q  = 0.01
FIXED_R  = 50.0

# ── UKF sigma-point params ────────────────────────────────────────────
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

DO_GRID_SEARCH  = True
DATA_DIR        = "./data"
SAVE_DIR        = "./outputs_sotaV2"
MOTION_RATIO_MIN = 0.95
COLLAPSE_PENALTY = 1e9


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY & GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d):
    return math.sqrt(max(d**2 - ANCHOR_HEIGHT**2, 0.0))


def build_ground_truth(waypoints=WAYPOINTS, speed=SPEED, spacing=GT_SPACING):
    segs      = np.diff(waypoints, axis=0)
    seg_len   = np.linalg.norm(segs, axis=1)
    total     = seg_len.sum()
    n_pts     = max(int(total / spacing), 2)
    cum       = np.concatenate([[0], np.cumsum(seg_len)])
    query     = np.linspace(0, total, n_pts)
    gt        = np.zeros((n_pts, 2))
    for i, d in enumerate(query):
        idx     = np.clip(np.searchsorted(cum, d, 'right') - 1, 0, len(segs)-1)
        frac    = (d - cum[idx]) / (seg_len[idx] + 1e-9)
        gt[i]   = waypoints[idx] + frac * segs[idx]
    return gt, total / speed


def nearest_gt_error(pos_xy, gt_xy):
    tree = cKDTree(gt_xy)
    errs, _ = tree.query(pos_xy)
    return errs


def LS_position(distances, anchors=ANCHORS):
    x0, y0 = anchors[0]
    d0 = max(distances[0], 1.0)
    rows, b = [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di = max(distances[i], 1.0)
        rows.append([2*(xi-x0), 2*(yi-y0)])
        b.append((d0**2-di**2) - (x0**2-xi**2) - (y0**2-yi**2))
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    try:
        pos, *_ = np.linalg.lstsq(A.T@A, A.T@bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


def compute_trajectory_displacement(pos_xy):
    if len(pos_xy) < 2: return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos_xy, axis=0), axis=1)))


def is_trajectory_collapsed(pos_xy, expected_path_length, ratio_min=MOTION_RATIO_MIN):
    if len(pos_xy) < 2: return True
    disp = compute_trajectory_displacement(pos_xy)
    if disp < ratio_min * expected_path_length: return True
    bx = pos_xy[:,0].max() - pos_xy[:,0].min()
    by = pos_xy[:,1].max() - pos_xy[:,1].min()
    if math.sqrt(bx**2 + by**2) < 0.05 * expected_path_length: return True
    return False


def default_init(dist_raw_0):
    pos = LS_position(dist_raw_0)
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])



def ukf_weights(n, alpha=UKF_ALPHA, beta=UKF_BETA, kappa=UKF_KAPPA):
    lam  = alpha**2 * (n + kappa) - n
    c    = n + lam
    Wm   = np.full(2*n+1, 0.5/c)
    Wc   = np.full(2*n+1, 0.5/c)
    Wm[0] = lam/c
    Wc[0] = lam/c + (1 - alpha**2 + beta)
    return Wm.astype(np.float64), Wc.astype(np.float64), float(c)


# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS — SHARED
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _ls_position_nb(distances, anchors):
    x0 = anchors[0,0]; y0 = anchors[0,1]
    d0 = distances[0]; d0 = d0 if d0>=1.0 else 1.0
    N  = anchors.shape[0]
    A  = np.empty((N-1, 2)); bv = np.empty(N-1)
    for i in range(1, N):
        xi = anchors[i,0]; yi = anchors[i,1]
        di = distances[i]; di = di if di>=1.0 else 1.0
        A[i-1,0] = 2.0*(xi-x0); A[i-1,1] = 2.0*(yi-y0)
        bv[i-1]  = (d0*d0-di*di) - (x0*x0-xi*xi) - (y0*y0-yi*yi)
    AtA = A.T@A; Atb = A.T@bv
    det = AtA[0,0]*AtA[1,1] - AtA[0,1]*AtA[1,0]
    pos = np.empty(2)
    if abs(det) < 1e-12:
        pos[0] = np.nan; pos[1] = np.nan
    else:
        pos[0] = (AtA[1,1]*Atb[0] - AtA[0,1]*Atb[1]) / det
        pos[1] = (AtA[0,0]*Atb[1] - AtA[1,0]*Atb[0]) / det
    return pos


@njit(cache=True)
def _ls_file_nb(dist_mat, anchors):
    T   = dist_mat.shape[0]
    pos = np.empty((T, 2))
    for t in range(T):
        p = _ls_position_nb(dist_mat[t], anchors)
        pos[t,0] = p[0]; pos[t,1] = p[1]
    return pos


@njit(cache=True)
def _make_spd_nb(P, eps=1e-9):
    P = 0.5*(P + P.T)
    n = P.shape[0]
    for i in range(n):
        P[i,i] += eps
        if P[i,i] < eps: P[i,i] = eps
    return P


@njit(cache=True)
def _ukf_moments_nb(x, P, Wm, Wc, c, anchors, anchor_h):
    """Sigma-point moments: z_hat, Pzz (no R), Pxz."""
    n     = 2; N = anchors.shape[0]; ns = 2*n+1
    P_spd = _make_spd_nb(P.copy(), 1e-6)
    S     = np.linalg.cholesky(c * P_spd)
    pts   = np.empty((ns, n))
    pts[0,0] = x[0]; pts[0,1] = x[1]
    for i in range(n):
        pts[i+1,0]   = x[0]+S[0,i]; pts[i+1,1]   = x[1]+S[1,i]
        pts[n+i+1,0] = x[0]-S[0,i]; pts[n+i+1,1] = x[1]-S[1,i]
    Z = np.empty((ns, N))
    for i in range(ns):
        for j in range(N):
            dx = pts[i,0]-anchors[j,0]; dy = pts[i,1]-anchors[j,1]
            Z[i,j] = math.sqrt(dx*dx + dy*dy + anchor_h*anchor_h)
    z_hat = np.zeros(N)
    for i in range(ns):
        for j in range(N): z_hat[j] += Wm[i]*Z[i,j]
    Pzz = np.zeros((N,N)); Pxz = np.zeros((n,N))
    for i in range(ns):
        for a in range(N):
            dza = Z[i,a]-z_hat[a]
            for b in range(N): Pzz[a,b] += Wc[i]*dza*(Z[i,b]-z_hat[b])
            for a2 in range(n): Pxz[a2,a] += Wc[i]*(pts[i,a2]-x[a2])*dza
    return z_hat, Pzz, Pxz


# ── 1. Standard UKF ───────────────────────────────────────────────────
@njit(cache=True)
def _ukf_step_nb(x, P, z, anchors, anchor_h, Q, R, Wm, Wc, c):
    N = anchors.shape[0]
    P_pred = P + np.eye(2)*Q
    z_hat, Pzz, Pxz = _ukf_moments_nb(x, P_pred, Wm, Wc, c, anchors, anchor_h)
    for i in range(N): Pzz[i,i] += R
    K = Pxz @ np.linalg.inv(Pzz)
    x_new = x + K @ (z - z_hat)
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


# ── 2. Huber-UKF ──────────────────────────────────────────────────────
@njit(cache=True)
def _huber_ukf_step_nb(x, P, z, anchors, anchor_h, Q, R, delta, maxiter, Wm, Wc, c):
    N = anchors.shape[0]
    P_pred = P + np.eye(2)*Q
    z_hat, Pzz0, Pxz = _ukf_moments_nb(x, P_pred, Wm, Wc, c, anchors, anchor_h)
    R_diag = np.full(N, R)
    innov  = z - z_hat
    for _ in range(maxiter):
        for i in range(N):
            s  = Pzz0[i,i] + R_diag[i]; s = s if s>1e-9 else 1e-9
            rs = innov[i] / math.sqrt(s)
            ar = abs(rs)
            w  = 1.0 if ar <= delta else delta/(ar+1e-9)
            if w < 1e-4: w = 1e-4
            R_diag[i] = R / w
    Pzz = Pzz0.copy()
    for i in range(N): Pzz[i,i] += R_diag[i]
    K     = Pxz @ np.linalg.inv(Pzz)
    x_new = x + K @ innov
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


# ── 3. MCC-UKF ────────────────────────────────────────────────────────
@njit(cache=True)
def _mcc_ukf_step_nb(x, P, z, anchors, anchor_h, Q, R, bw, maxiter, Wm, Wc, c):
    N = anchors.shape[0]
    P_pred = P + np.eye(2)*Q
    z_hat, Pzz0, Pxz = _ukf_moments_nb(x, P_pred, Wm, Wc, c, anchors, anchor_h)
    R_diag = np.full(N, R); innov = z - z_hat; x_cur = x.copy(); K = np.zeros((2,N))
    for _ in range(maxiter):
        for i in range(N):
            w = math.exp(-0.5*innov[i]*innov[i]/(bw*bw+1e-9))
            if w < 1e-4: w = 1e-4
            R_diag[i] = R/w
        Pzz = Pzz0.copy()
        for i in range(N): Pzz[i,i] += R_diag[i]
        K = Pxz @ np.linalg.inv(Pzz)
        x_new = x + K @ innov
        dx = x_new[0]-x_cur[0]; dy = x_new[1]-x_cur[1]
        if math.sqrt(dx*dx+dy*dy) < 1e-3: x_cur = x_new; break
        x_cur = x_new
    Pzz = Pzz0.copy()
    for i in range(N): Pzz[i,i] += R_diag[i]
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_cur, P_new


# ── 4. PC-UKF-v4 (Gaussian pairwise) ─────────────────────────────────
@njit(cache=True)
def _pc_scores_gauss_nb(innov, Pzz_diag, N):
    """
    Pairwise consistency scoring với giả định Gaussian (giống RUKF).
    Chuẩn hóa: nm[i] = innov[i] / sqrt(S[i,i])  ~ N(0,1) nếu inlier.
    Score[i] = mean over j≠i của pdf_gauss(nm[i] - nm[j])
             = mean exp(-0.5*(nm[i]-nm[j])²/2)   (variance=2 vì diff 2 N(0,1))
    Anchor nào "xa" đám đông theo N(0,1) → score thấp → R inflate nhiều.
    """
    nm = np.empty(N)
    for i in range(N):
        s = Pzz_diag[i]; s = s if s > 1e-9 else 1e-9
        nm[i] = innov[i] / math.sqrt(s)   # chuẩn hóa theo S, giả định ~N(0,1)
    scores = np.empty(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if j != i:
                d = nm[i] - nm[j]
                s += math.exp(-0.25 * d * d)   # N(0,2): var=2 → -d²/(2*2)
        scores[i] = s / (N - 1)
    return scores


@njit(cache=True)
def _pc_ukf_step_nb(x, P, z, anchors, anchor_h, Q, R, r_scale, Wm, Wc, c):
    N = anchors.shape[0]
    P_pred = P + np.eye(2)*Q
    z_hat, Pzz0, Pxz = _ukf_moments_nb(x, P_pred, Wm, Wc, c, anchors, anchor_h)
    innov  = z - z_hat
    S_diag = np.empty(N)
    for i in range(N): S_diag[i] = Pzz0[i,i] + R
    scores = _pc_scores_gauss_nb(innov, S_diag, N)
    Pzz    = Pzz0.copy()
    for i in range(N): Pzz[i,i] += R*(1.0 + r_scale*(1.0-scores[i]))
    K     = Pxz @ np.linalg.inv(Pzz)
    x_new = x + K @ innov
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


# ── 5. Robust-UKF (Chi2 gating + R inflate) ──────────────────────────
@njit(cache=True)
def _robust_ukf_step_nb(x, P, z, anchors, anchor_h, Q, R, chi2_gate, r_inflate,
                        Wm, Wc, c):
    """
    Robust-UKF: kiểm tra Mahalanobis distance; nếu vượt ngưỡng thì inflate R.
    Outlier per-anchor: inflate R của anchor đó thêm r_inflate lần.
    Tune: chi2_gate (threshold DOF=1), r_inflate (inflate factor).
    """
    N = anchors.shape[0]
    P_pred = P + np.eye(2)*Q
    z_hat, Pzz0, Pxz = _ukf_moments_nb(x, P_pred, Wm, Wc, c, anchors, anchor_h)
    innov  = z - z_hat
    R_diag = np.full(N, R)
    for i in range(N):
        s_ii = Pzz0[i,i] + R
        s_ii = s_ii if s_ii > 1e-9 else 1e-9
        md2_i = innov[i]*innov[i] / s_ii
        if md2_i > chi2_gate:
            R_diag[i] = R * r_inflate
    Pzz = Pzz0.copy()
    for i in range(N): Pzz[i,i] += R_diag[i]
    K     = Pxz @ np.linalg.inv(Pzz)
    x_new = x + K @ innov
    P_new = _make_spd_nb(P_pred - K @ Pzz @ K.T)
    return x_new, P_new


# ══════════════════════════════════════════════════════════════════════
#  WARMUP — pre-compile tất cả JIT kernels
# ══════════════════════════════════════════════════════════════════════
def _warmup_numba():
    d  = np.array([9000.0, 9200.0, 5000.0, 5100.0])
    a  = ANCHORS.copy()
    _ls_position_nb(d, a)
    _ls_file_nb(np.stack([d]), a)
    Wm, Wc, c = ukf_weights(2)
    x0 = np.array([2000.0, 2000.0]); P0 = np.eye(2)*1e6
    _ukf_step_nb(x0, P0, d, a, ANCHOR_HEIGHT, FIXED_Q, FIXED_R, Wm, Wc, c)
    _huber_ukf_step_nb(x0, P0, d, a, ANCHOR_HEIGHT, FIXED_Q, FIXED_R, 1.5, 5, Wm, Wc, c)
    _mcc_ukf_step_nb(x0, P0, d, a, ANCHOR_HEIGHT, FIXED_Q, FIXED_R, 1000.0, 5, Wm, Wc, c)
    _pc_ukf_step_nb(x0, P0, d, a, ANCHOR_HEIGHT, FIXED_Q, FIXED_R, 5.0, Wm, Wc, c)
    _robust_ukf_step_nb(x0, P0, d, a, ANCHOR_HEIGHT, FIXED_Q, FIXED_R, 3.84, 10.0, Wm, Wc, c)
    print("  [JIT] All kernels compiled ✓")


# ══════════════════════════════════════════════════════════════════════
#  FILTER CLASSES
# ══════════════════════════════════════════════════════════════════════
class StandardUKF:
    def __init__(self, q=FIXED_Q, r=FIXED_R):
        self.q = float(q); self.r = float(r)
        self.Wm, self.Wc, self.c = ukf_weights(2)
        self.x = None; self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(2, dtype=np.float64)*1e6

    def step(self, z):
        if self.x is None: return np.full(2, np.nan)
        self.x, self.P = _ukf_step_nb(
            self.x, self.P, np.asarray(z, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r, self.Wm, self.Wc, self.c)
        return self.x.copy()


class HuberUKF:
    def __init__(self, q=FIXED_Q, r=FIXED_R, delta=1.5, maxiter=5):
        self.q = float(q); self.r = float(r)
        self.delta = float(delta); self.maxiter = int(maxiter)
        self.Wm, self.Wc, self.c = ukf_weights(2)
        self.x = None; self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(2, dtype=np.float64)*1e6

    def step(self, z):
        if self.x is None: return np.full(2, np.nan)
        self.x, self.P = _huber_ukf_step_nb(
            self.x, self.P, np.asarray(z, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r,
            self.delta, self.maxiter, self.Wm, self.Wc, self.c)
        return self.x.copy()


class MCCUKF:
    def __init__(self, q=FIXED_Q, r=FIXED_R, kernel_bw=1000.0, maxiter=5):
        self.q = float(q); self.r = float(r)
        self.bw = float(kernel_bw); self.maxiter = int(maxiter)
        self.Wm, self.Wc, self.c = ukf_weights(2)
        self.x = None; self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(2, dtype=np.float64)*1e6

    def step(self, z):
        if self.x is None: return np.full(2, np.nan)
        self.x, self.P = _mcc_ukf_step_nb(
            self.x, self.P, np.asarray(z, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r,
            self.bw, self.maxiter, self.Wm, self.Wc, self.c)
        return self.x.copy()


class PCUKFv3:
    def __init__(self, q=FIXED_Q, r_base=FIXED_R, r_scale=5.0):
        self.q = float(q); self.r = float(r_base); self.r_scale = float(r_scale)
        self.Wm, self.Wc, self.c = ukf_weights(2)
        self.x = None; self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(2, dtype=np.float64)*1e6

    def step(self, z):
        if self.x is None: return np.full(2, np.nan)
        self.x, self.P = _pc_ukf_step_nb(
            self.x, self.P, np.asarray(z, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r, self.r_scale,
            self.Wm, self.Wc, self.c)
        return self.x.copy()


class RobustUKF:
    """Chi2 per-anchor gating + R inflate khi phát hiện outlier."""
    def __init__(self, q=FIXED_Q, r=FIXED_R, chi2_gate=3.84, r_inflate=10.0):
        self.q = float(q); self.r = float(r)
        self.gate = float(chi2_gate); self.inflate = float(r_inflate)
        self.Wm, self.Wc, self.c = ukf_weights(2)
        self.x = None; self.P = None

    def init(self, x0):
        self.x = x0.astype(np.float64).copy()
        self.P = np.eye(2, dtype=np.float64)*1e6

    def step(self, z):
        if self.x is None: return np.full(2, np.nan)
        self.x, self.P = _robust_ukf_step_nb(
            self.x, self.P, np.asarray(z, dtype=np.float64),
            ANCHORS, ANCHOR_HEIGHT, self.q, self.r,
            self.gate, self.inflate, self.Wm, self.Wc, self.c)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  METHODS TABLE — R=50 Q=0.01 cố định, chỉ tune param đặc trưng
# ══════════════════════════════════════════════════════════════════════
METHODS = {
    "Raw+LS"     : None,
    "UKF"        : (StandardUKF, dict(q=FIXED_Q, r=FIXED_R)),
    "Huber-UKF"  : (HuberUKF,   dict(q=FIXED_Q, r=FIXED_R, delta=1.5)),
    "MCC-UKF"    : (MCCUKF,     dict(q=FIXED_Q, r=FIXED_R, kernel_bw=1000.0)),
    "PC-UKF-v4"  : (PCUKFv3,    dict(q=FIXED_Q, r_base=FIXED_R, r_scale=5.0)),
    "Robust-UKF" : (RobustUKF,  dict(q=FIXED_Q, r=FIXED_R, chi2_gate=3.84, r_inflate=10.0)),
}

COLORS = {
    "Raw+LS"     : '#9E9E9E',
    "UKF"        : '#00BCD4',
    "Huber-UKF"  : '#E91E63',
    "MCC-UKF"    : '#FF9800',
    "PC-UKF-v4"  : '#9C27B0',
    "Robust-UKF" : '#009688',
}


# ══════════════════════════════════════════════════════════════════════
#  PARSE FILE
# ══════════════════════════════════════════════════════════════════════
def parse_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) < 7: continue
            try:
                d_sl = [float(parts[i+1]) for i in range(4)]
                rows.append([slant_to_ground(d) for d in d_sl])
            except ValueError:
                continue
    return np.array(rows, dtype=np.float64) if rows else None


def safe_len(path):
    d = parse_file(path)
    return len(d) if d is not None else 0


def run_filter(filt_cls, dist_raw, **kw):
    filt = filt_cls(**kw)
    T    = len(dist_raw)
    pos  = np.full((T, 2), np.nan)
    filt.init(default_init(dist_raw[0]))
    for t in range(T):
        pos[t] = filt.step(dist_raw[t])
    return pos


# ══════════════════════════════════════════════════════════════════════
#  GRID SEARCH — tune chỉ param đặc trưng, R=50 Q=0.01 cố định
# ══════════════════════════════════════════════════════════════════════
def _eval(filt_cls, kw, files, gt_xy, path_len):
    pos_all = []; n_col = 0; n_f = 0
    for p in files:
        d = parse_file(p)
        if d is None: continue
        n_f += 1
        pos = run_filter(filt_cls, d, **kw)
        v   = pos[~np.any(np.isnan(pos), axis=1)]
        if is_trajectory_collapsed(v, path_len): n_col += 1
        else: pos_all.extend(v)
    if n_f == 0: return float('inf')
    if n_col/n_f > 0.30: return COLLAPSE_PENALTY + n_col/n_f
    if not pos_all: return float('inf')
    errs = nearest_gt_error(np.array(pos_all), gt_xy)
    return float(np.sqrt(np.mean(errs**2))) + n_col*500.0


def _grid(name, filt_cls, base_kw, tune_grid, files, gt_xy, path_len):
    keys   = list(tune_grid.keys())
    combos = list(itertools.product(*[tune_grid[k] for k in keys]))
    print(f"\n  Grid {name}: {len(combos)} combos | tune {keys} | R={FIXED_R} Q={FIXED_Q} fixed")
    best_r = float('inf'); best_kw = base_kw.copy()
    for combo in combos:
        kw = base_kw.copy()
        kw.update(dict(zip(keys, combo)))
        r  = _eval(filt_cls, kw, files, gt_xy, path_len)
        if r < best_r: best_r = r; best_kw = kw.copy()
    tag = " [⚠COLLAPSED]" if best_r >= COLLAPSE_PENALTY else ""
    tuned = {k: best_kw[k] for k in keys}
    print(f"  Best {name}: RMSE={best_r:.1f}mm{tag} | {tuned}")
    return best_kw


def run_grid_search(files, gt_xy, path_len):
    bkw = {}
    # Huber — tune delta
    bkw["Huber-UKF"] = _grid("Huber-UKF", HuberUKF,
        dict(q=FIXED_Q, r=FIXED_R, delta=1.5),
        {'delta': [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0]},
        files, gt_xy, path_len)

    # MCC — tune kernel_bw
    bkw["MCC-UKF"] = _grid("MCC-UKF", MCCUKF,
        dict(q=FIXED_Q, r=FIXED_R, kernel_bw=1000.0),
        {'kernel_bw': [100.0, 200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 3000.0]},
        files, gt_xy, path_len)

    # PC-UKF-v4 — tune r_scale
    bkw["PC-UKF-v4"] = _grid("PC-UKF-v4", PCUKFv3,
        dict(q=FIXED_Q, r_base=FIXED_R, r_scale=5.0),
        {'r_scale': [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]},
        files, gt_xy, path_len)

    # Robust-UKF — tune chi2_gate, r_inflate
    bkw["Robust-UKF"] = _grid("Robust-UKF", RobustUKF,
        dict(q=FIXED_Q, r=FIXED_R, chi2_gate=3.84, r_inflate=10.0),
        {'chi2_gate': [1.0, 2.0, 3.84, 6.63, 10.0],
         'r_inflate': [5.0, 10.0, 20.0, 50.0, 100.0]},
        files, gt_xy, path_len)

    return bkw


# ══════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════
def evaluate_files(files, gt_xy, path_len):
    all_err  = {m: [] for m in METHODS}
    all_pos  = {m: [] for m in METHODS}
    pf_rmse  = {m: [] for m in METHODS}
    col_warn = {m: [] for m in METHODS}
    timing   = {m: [] for m in METHODS if m != "Raw+LS"}

    W = max(len(m) for m in METHODS) + 2
    hdr = f"{'File':<20s}" + "".join(f" {m:>{W}s}" for m in METHODS)
    sep = "═" * (20 + len(METHODS)*(W+1))
    print(f"\n{sep}")
    print("  PER-FILE RMSE (mm)")
    print(sep); print(hdr); print("─"*len(hdr))

    for path in files:
        d = parse_file(path)
        if d is None: continue
        row = f"{os.path.basename(path):<20s}"
        for name, method in METHODS.items():
            if method is None:
                pos = _ls_file_nb(d.astype(np.float64), ANCHORS)
            else:
                cls, kw = method
                t0  = time.perf_counter()
                pos = run_filter(cls, d, **kw)
                timing[name].append(time.perf_counter() - t0)
            valid = pos[~np.any(np.isnan(pos), axis=1)]
            col   = is_trajectory_collapsed(valid, path_len)
            if col: col_warn[name].append(os.path.basename(path))
            errs  = nearest_gt_error(valid, gt_xy)
            rmse  = float(np.sqrt(np.mean(errs**2))) if len(errs) else float('nan')
            flag  = "⚠" if col else " "
            row  += f" {flag}{rmse:>{W-1}.1f}"
            all_err[name].extend(errs)
            all_pos[name].extend(valid)
            pf_rmse[name].append(rmse)
        print(row)
    print("─"*len(hdr))

    print("\n  TIMING (ms/sample):")
    for name, ts in timing.items():
        if not ts: continue
        n_s = sum(safe_len(p) for p in files[:len(ts)])
        if n_s > 0:
            print(f"    {name:<14s}: {sum(ts)*1000/n_s:.4f} ms/sample")

    print("\n  COLLAPSE WARNINGS:")
    any_w = False
    for name, fs in col_warn.items():
        if fs:
            any_w = True
            print(f"    ⚠ {name:<14s}: {', '.join(fs[:5])}")
    if not any_w:
        print("    ✅ Không có trajectory nào bị collapse.")

    return (
        {m: np.array(v) for m, v in all_err.items()},
        {m: np.array(v) for m, v in all_pos.items()},
        col_warn, pf_rmse,
    )


def compute_metrics(errs, pos_xy=None, path_len=None):
    if len(errs) == 0:
        return {k: float('nan') for k in
                ['mae','rmse','cep50','cep90','p95','max','motion_ratio','collapsed']}
    m = {
        'mae'  : float(np.mean(errs)),
        'rmse' : float(np.sqrt(np.mean(errs**2))),
        'cep50': float(np.percentile(errs, 50)),
        'cep90': float(np.percentile(errs, 90)),
        'p95'  : float(np.percentile(errs, 95)),
        'max'  : float(np.max(errs)),
    }
    if pos_xy is not None and path_len and len(pos_xy) > 1:
        disp = compute_trajectory_displacement(pos_xy)
        m['motion_ratio'] = round(disp/(path_len+1e-9), 3)
        m['collapsed']    = is_trajectory_collapsed(pos_xy, path_len)
    else:
        m['motion_ratio'] = float('nan'); m['collapsed'] = False
    return m


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_cdf(errors, col_warn, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, errs in errors.items():
        if len(errs) == 0: continue
        color = COLORS.get(name, 'gray')
        lw = 2.5 if name not in ("Raw+LS",) else 1.2
        ls = '--' if name == "Raw+LS" else '-'
        n_col = len(col_warn.get(name, []))
        warn  = f" ⚠{n_col}" if n_col else ""
        rmse  = float(np.sqrt(np.mean(errs**2)))
        xs = np.sort(errs); ys = np.arange(1, len(xs)+1)/len(xs)
        ax.plot(xs, ys, color=color, lw=lw, ls=ls,
                label=f"{name}{warn} (RMSE={rmse:.1f}mm)")
    ax.set_xlabel("Positioning Error (mm)"); ax.set_ylabel("CDF")
    ax.legend(fontsize=7, loc='lower right'); ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] CDF → {save_path}"); plt.close()


def plot_trajectories(positions, gt_xy, col_warn, save_path):
    fig, ax = plt.subplots(figsize=(13, 15))
    ax.plot(gt_xy[:,0], gt_xy[:,1], 'k--', lw=2.5, label='Ground Truth', alpha=0.5, zorder=5)
    for label, pos in positions.items():
        if len(pos) == 0: continue
        color = COLORS.get(label, 'gray')
        errs  = nearest_gt_error(pos, gt_xy)
        rmse  = np.sqrt(np.mean(errs**2))
        n_col = len(col_warn.get(label, []))
        warn  = f" ⚠{n_col}" if n_col else ""
        lw = 2.0 if label != "Raw+LS" else 1.0
        ax.plot(pos[:,0], pos[:,1], color=color, lw=lw,
                ls=('-' if label != "Raw+LS" else '--'),
                alpha=0.7, label=f"{label}{warn}  {rmse:.1f}mm")
    for nm, pt in zip(["A","B","C","D"], WAYPOINTS[:4]):
        ax.scatter(*pt, s=90, color='black', zorder=10)
        ax.annotate(nm, pt, xytext=(6,4), textcoords="offset points",
                    fontsize=13, fontweight='bold')
    for j, (ax_, ay_) in enumerate(ANCHORS):
        ax.scatter(ax_, ay_, s=100, marker='s', color='red', zorder=10)
        ax.annotate(f"A{j+1}", (ax_, ay_), xytext=(5,5),
                    textcoords="offset points", fontsize=9, color='red')
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.legend(fontsize=7, loc='best'); ax.set_aspect('equal')
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Trajectories → {save_path}"); plt.close()


def plot_bar(metrics, save_path):
    methods = list(metrics.keys())
    mnames  = ['rmse','mae','cep50','p95']
    mlabels = ['RMSE','MAE','CEP50','P95']
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    for ax, mn, ml in zip(axes, mnames, mlabels):
        vals    = [metrics[m][mn] for m in methods]
        colors  = [COLORS.get(m,'gray') for m in methods]
        bars    = ax.bar(range(len(methods)), vals, color=colors, alpha=0.85,
                         edgecolor='white', lw=1.5)
        for bar, val, m in zip(bars, vals, methods):
            if metrics[m].get('collapsed', False):
                bar.set_hatch('//'); bar.set_edgecolor('darkred')
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-','\n') for m in methods], fontsize=8)
        ax.set_ylabel("mm"); ax.set_title(ml, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}"); plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("="*72)
    print("  SOTA V2 — UWB Indoor Positioning (R=50 Q=0.01 fixed)")
    print("="*72)
    print("  Filter            | Tune param")
    print("  ─────────────────────────────────────────────")
    print("  UKF               | (không tune — baseline)")
    print("  Huber-UKF         | delta")
    print("  MCC-UKF           | kernel_bw")
    print("  PC-UKF-v4         | r_scale  [Gaussian pairwise]")
    print("  Robust-UKF        | chi2_gate, r_inflate")
    print("="*72)
    print(f"  Motion collapse: ratio_min={MOTION_RATIO_MIN:.0%}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"\n[!] Không tìm thấy file .txt trong '{DATA_DIR}/'"); return
    print(f"\n  Tìm thấy {n} file(s) trong '{DATA_DIR}/'")

    np.random.seed(42)
    files = [all_files[i] for i in np.random.permutation(n)]

    gt_xy, total_time = build_ground_truth()
    path_len = float(np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum())
    print(f"  Path={path_len:.0f}mm  Time={total_time:.1f}s  GT={len(gt_xy)} pts")

    # ── JIT warmup ─────────────────────────────────────────────────────
    _warmup_numba()

    # ── Grid search ────────────────────────────────────────────────────
    if DO_GRID_SEARCH:
        print("\n  [Grid Search] R=50 Q=0.01 cố định, tune param đặc trưng...")
        best_kw = run_grid_search(files, gt_xy, path_len)
        for name, kw in best_kw.items():
            cls = METHODS[name][0]
            METHODS[name] = (cls, kw)

    # ── Evaluate ──────────────────────────────────────────────────────
    errs, positions, col_warn, pf_rmse = evaluate_files(files, gt_xy, path_len)

    # ── Summary ───────────────────────────────────────────────────────
    metrics = {}
    W = max(len(m) for m in METHODS)
    print(f"\n{'═'*110}")
    print(f"  SUMMARY TABLE (mm)   — ⚠ = collapsed (motion_ratio < {MOTION_RATIO_MIN:.0%})")
    print(f"{'═'*110}")
    print(f"  {'Method':<{W+2}s} {'RMSE mean±std':>20s} {'MAE':>7s} "
          f"{'CEP50':>7s} {'CEP90':>7s} {'P95':>7s} {'MAX':>8s} {'MotionR':>8s} {'Status'}")
    print(f"  {'─'*100}")

    for label, e in errs.items():
        if len(e) == 0: continue
        pos_arr = positions.get(label, np.empty((0,2)))
        m = compute_metrics(e, pos_arr, path_len); metrics[label] = m
        pf = [v for v in pf_rmse.get(label,[]) if not math.isnan(v)]
        std_s = (f"{np.mean(pf):.1f} ± {np.std(pf,ddof=1):.1f}"
                 if len(pf)>=2 else (f"{pf[0]:.1f} ± N/A" if pf else "N/A"))
        n_col  = len(col_warn.get(label, []))
        status = f"⚠ {n_col} file(s)" if n_col else "✅ OK"
        mr     = (f"{m['motion_ratio']:.2f}"
                  if not math.isnan(m.get('motion_ratio', float('nan'))) else "N/A")
        print(f"  {label:<{W+2}s} {std_s:>20s} {m['mae']:>7.1f} "
              f"{m['cep50']:>7.1f} {m['cep90']:>7.1f} {m['p95']:>7.1f} {m['max']:>8.1f} "
              f"{mr:>8s}  {status}")

    # ── Wilcoxon vs PC-UKF-v4 ─────────────────────────────────────────
    print(f"\n  Wilcoxon (two-sided, vs PC-UKF-v4):")
    ref = errs.get("PC-UKF-v4", np.array([]))
    for name, e in errs.items():
        if name == "PC-UKF-v4" or len(e)==0 or len(ref)==0: continue
        N = min(len(e), len(ref))
        if N > 20:
            try:
                _, p = wilcoxon(e[:N], ref[:N])
                sym  = '✅ p<0.05' if p < 0.05 else '⚠️  ns'
                print(f"    {name:<{W+2}s} vs PC-UKF-v4: p={p:.4f}  {sym}")
            except ValueError:
                pass

    # ── Plots ─────────────────────────────────────────────────────────
    plot_cdf(errs, col_warn, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(positions, gt_xy, col_warn,
                      os.path.join(SAVE_DIR, 'trajectories.png'))
    plot_bar(metrics, os.path.join(SAVE_DIR, 'bar_comparison.png'))

    for label, e in errs.items():
        safe = label.lower().replace('+','_').replace('-','_').replace(' ','_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), e)

    print(f"\n[✓] Kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
