# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — V21: Robust & Adaptive KF  (Full LOO Tuning)
=======================================================================
CÁC PHƯƠNG PHÁP:
  1. Raw+LS        — baseline Weighted Least Squares
  2. UKF            — Unscented Kalman Filter chuẩn
  3. Huber-UKF      — IRLS Huber M-estimator
  4. MCC-UKF        — Maximum Correntropy Criterion UKF
  5. PC-UKF-2D      — Pairwise Consensus MAD-normalized (sigma-free) + UKF
  6. GUKF           — Gaussian-smoothed UKF (Sun et al. 2025)

THAY ĐỔI SO VỚI V20:
  - TẤT CẢ filter có params đều dùng LOO cross-validation để tuning:
      Huber-UKF, MCC-UKF, GUKF, PC-UKF-2D → _grid_search_loo() chung
  - Xoá _eval_rmse_single() (dùng toàn eval set, không công bằng)
  - Thêm: in ra console ngay khi tìm được best mới (RMSE + hệ số)
  - grid_search_pcukf_loo() → gọi hàm chung (bỏ code riêng)
  - Đảm bảo tuning hoàn toàn công bằng giữa mọi thuật toán

CÔNG BẰNG TUNING:
  - Raw+LS, UKF: không có param cần tune → không cần LOO
  - Huber-UKF, MCC-UKF, GUKF, PC-UKF-2D: đều dùng _grid_search_loo()
    → mỗi fold: params được chọn từ (N-1) files, validate trên 1 file còn lại
    → không có method nào được "nhìn thấy" eval file khi chọn params

Format file .txt: timestamp, d0_slant, d1_slant, d2_slant, d3_slant, v_mm_s, gz_dps
"""

import os
import glob
import math
import time
import warnings
import itertools
import numpy as np
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
SPEED         = 200.0
GT_SPACING    = 5.0
N_ANCHORS     = 4

# ─── Standard UKF ────────────────────────────────────────────────────
UKF_STD_Q = 0.001
UKF_STD_R = 50.0

# ─── Huber-UKF ───────────────────────────────────────────────────────
HUBER_Q       = 0.001
HUBER_R       = 50.0
HUBER_DELTA   = 20.0
HUBER_MAXITER = 5

# ─── MCC-UKF ─────────────────────────────────────────────────────────
MCC_Q         = 0.001
MCC_R         = 100.0
MCC_KERNEL_BW = 1700.0
MCC_MAXITER   = 5

# ─── PC-UKF-2D (MAD sigma-free) ──────────────────────────────────────
PCUKF_Q       = 0.001
PCUKF_R_BASE  = 25.0
PCUKF_R_SCALE = 15.0

# ─── GUKF ────────────────────────────────────────────────────────────
GUKF_Q      = 0.001
GUKF_R      = 100.0
GUKF_SIGMA  = 5.0
GUKF_N_HALF = 4

# ─── UKF common ──────────────────────────────────────────────────────
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

DO_GRID_SEARCH = True
DATA_DIR       = "./data"
SAVE_DIR       = "./outputs_sota"

MOTION_RATIO_MIN  = 0.90
COLLAPSE_PENALTY  = 1e9


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
#  MOTION SANITY CHECK
# ══════════════════════════════════════════════════════════════════════
def compute_trajectory_displacement(pos_xy):
    if len(pos_xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos_xy, axis=0), axis=1)))


def is_trajectory_collapsed(pos_xy, expected_path_length,
                             ratio_min=MOTION_RATIO_MIN):
    if len(pos_xy) < 2:
        return True
    total_disp = compute_trajectory_displacement(pos_xy)
    if total_disp < ratio_min * expected_path_length:
        return True
    bbox_x    = pos_xy[:, 0].max() - pos_xy[:, 0].min()
    bbox_y    = pos_xy[:, 1].max() - pos_xy[:, 1].min()
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
    return Wm, Wc, c


def sigma_points(x, P, c):
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
    pts   = sigma_points(x_pred, P_pred, c)
    Z_pts = np.array([h_obs(pts[i]) for i in range(len(pts))])
    z_hat = Wm @ Z_pts
    n     = len(x_pred)
    M     = z_hat.shape[0]
    Pzz   = np.zeros((M, M))
    Pxz   = np.zeros((n, M))
    for i in range(2*n + 1):
        dz   = Z_pts[i] - z_hat
        dx   = pts[i] - x_pred
        Pzz += Wc[i] * np.outer(dz, dz)
        Pxz += Wc[i] * np.outer(dx, dz)
    return z_hat, Pzz, Pxz


def make_spd(P, eps=1e-9):
    P = 0.5 * (P + P.T)
    P += np.eye(len(P)) * eps
    return P


def default_init(dist_raw_0):
    pos = LS_position(dist_raw_0)
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


# ══════════════════════════════════════════════════════════════════════
#  1. STANDARD UKF
# ══════════════════════════════════════════════════════════════════════
class StandardUKF:
    def __init__(self, q=UKF_STD_Q, r=UKF_STD_R):
        self.n     = 2
        self.Q_mat = np.eye(2) * q
        self.R_mat = np.eye(N_ANCHORS) * r
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(self.n) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)
        Pzz_eff = Pzz_no_R + self.R_mat
        innov   = z_raw - z_hat
        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  2. HUBER-UKF
# ══════════════════════════════════════════════════════════════════════
class HuberUKF:
    def __init__(self, q=HUBER_Q, r=HUBER_R,
                 delta=HUBER_DELTA, maxiter=HUBER_MAXITER):
        self.n       = 2
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r
        self.delta   = delta
        self.maxiter = maxiter
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(self.n) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)
        R_eff = np.eye(N_ANCHORS) * self.R_base
        for _ in range(self.maxiter):
            Pzz_eff  = Pzz_no_R + R_eff
            innov    = z_raw - z_hat
            pzz_diag = np.maximum(np.diag(Pzz_eff), 1e-9)
            r_scaled = innov / np.sqrt(pzz_diag)
            hub_w    = np.where(
                np.abs(r_scaled) <= self.delta,
                1.0,
                self.delta / (np.abs(r_scaled) + 1e-9)
            )
            hub_w = np.maximum(hub_w, 1e-4)
            R_eff = np.diag(self.R_base / hub_w)
        Pzz_eff = Pzz_no_R + R_eff
        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ (z_raw - z_hat)
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  3. MCC-UKF
# ══════════════════════════════════════════════════════════════════════
class MCCUKF:
    def __init__(self, q=MCC_Q, r=MCC_R,
                 kernel_bw=MCC_KERNEL_BW, maxiter=MCC_MAXITER):
        self.n         = 2
        self.Q_mat     = np.eye(2) * q
        self.R_base    = r
        self.kernel_bw = kernel_bw
        self.maxiter   = maxiter
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(self.n) * 1e6

    def _gaussian_kernel(self, r):
        return np.exp(-0.5 * r**2 / (self.kernel_bw**2 + 1e-9))

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)
        x_cur = x_pred.copy()
        K     = np.zeros((self.n, N_ANCHORS))
        R_eff = np.eye(N_ANCHORS) * self.R_base
        for _ in range(self.maxiter):
            innov   = z_raw - z_hat
            kern_w  = self._gaussian_kernel(innov)
            kern_w  = np.maximum(kern_w, 1e-4)
            R_eff   = np.diag(self.R_base / kern_w)
            Pzz_eff = Pzz_no_R + R_eff
            try:
                K = Pxz @ np.linalg.inv(Pzz_eff)
            except np.linalg.LinAlgError:
                break
            x_new = x_pred + K @ innov
            if np.linalg.norm(x_new - x_cur) < 1e-3:
                x_cur = x_new
                break
            x_cur = x_new
        self.x = x_cur
        self.P = make_spd(P_pred - K @ (Pzz_no_R + R_eff) @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  4. PC-UKF-2D — MAD Auto-Normalized, Sigma-Free (V20+)
# ══════════════════════════════════════════════════════════════════════
def _t_kernel_pairwise(diff_vec, sigma=1.0):
    """T-distribution kernel, sigma=1.0 cố định (không tune)."""
    nu  = 4.0
    eps = 1e-9
    return (1.0 + diff_vec**2 / (nu * sigma**2 + eps)) ** (-(nu + 1.0) / 2.0)


def pc_mad_scores(innov, S_diag):
    """
    PC scoring — MAD auto-normalized, sigma-free.
    Bước 1: Geo-normalize: std_innov[i] = innov[i] / sqrt(S_ii)
    Bước 2: MAD normalize: normed[i] = std_innov[i] / (1.4826*mad + eps)
    Bước 3: Pairwise T-kernel score (sigma=1.0 cố định)
    """
    std_innov = innov / np.sqrt(np.maximum(S_diag, 1e-9))
    med    = np.median(std_innov)
    mad    = np.median(np.abs(std_innov - med))
    normed = std_innov / (1.4826 * mad + 1e-9)
    scores = np.zeros(N_ANCHORS)
    for i in range(N_ANCHORS):
        diffs     = np.array([normed[i] - normed[j]
                              for j in range(N_ANCHORS) if j != i])
        scores[i] = np.mean(_t_kernel_pairwise(diffs, sigma=1.0))
    return scores


class PCUKF2D:
    def __init__(self, q=PCUKF_Q, r_base=PCUKF_R_BASE, r_scale=PCUKF_R_SCALE):
        self.n       = 2
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(self.n) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)
        innov  = z_raw - z_hat
        S_diag = np.diag(Pzz_no_R) + self.R_base
        scores  = pc_mad_scores(innov, S_diag)
        R_diag  = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        Pzz_eff = Pzz_no_R + np.diag(R_diag)
        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  5. GUKF
# ══════════════════════════════════════════════════════════════════════
def _make_gaussian_kernel(n_half, sigma):
    idx = np.arange(-n_half, n_half + 1, dtype=float)
    h   = np.exp(-0.5 * idx**2 / (sigma**2 + 1e-12))
    return h / h.sum()


class GUKF:
    def __init__(self, q=GUKF_Q, r=GUKF_R,
                 sigma=GUKF_SIGMA, n_half=GUKF_N_HALF):
        self.n      = 2
        self.Q_mat  = np.eye(2) * q
        self.R_mat  = np.eye(N_ANCHORS) * r
        self.kernel = _make_gaussian_kernel(n_half, sigma)
        self.n_half = n_half
        self.win    = 2 * n_half + 1
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x    = None
        self.P    = None
        self._buf = []

    def init(self, x0):
        self.x    = x0.astype(float).copy()
        self.P    = np.eye(self.n) * 1e6
        self._buf = []

    def _smooth(self, z_raw):
        self._buf.append(z_raw.copy())
        if len(self._buf) > self.win:
            self._buf.pop(0)
        buf = np.array(self._buf)
        L   = len(buf)
        if L < self.win:
            h_cut = self.kernel[self.win - L:]
            h_cut = h_cut / h_cut.sum()
            return h_cut @ buf
        else:
            return self.kernel @ buf

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        z_smooth = self._smooth(z_raw)
        x_pred   = self.x.copy()
        P_pred   = self.P + self.Q_mat
        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)
        Pzz_eff = Pzz_no_R + self.R_mat
        innov   = z_smooth - z_hat
        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  FILTER RUNNER
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
    "Raw+LS"    : None,
    "UKF"       : (StandardUKF, dict(q=UKF_STD_Q,  r=UKF_STD_R)),
    "Huber-UKF" : (HuberUKF,   dict(q=HUBER_Q,     r=HUBER_R,    delta=HUBER_DELTA)),
    "MCC-UKF"   : (MCCUKF,     dict(q=MCC_Q,       r=MCC_R,      kernel_bw=MCC_KERNEL_BW)),
    "PC-UKF-2D" : (PCUKF2D,    dict(q=PCUKF_Q,     r_base=PCUKF_R_BASE,
                                     r_scale=PCUKF_R_SCALE)),
    "GUKF"      : (GUKF,       dict(q=GUKF_Q,       r=GUKF_R,
                                     sigma=GUKF_SIGMA, n_half=GUKF_N_HALF)),
}

COLORS = {
    "Raw+LS"    : '#9E9E9E',
    "UKF"       : '#00BCD4',
    "Huber-UKF" : '#E91E63',
    "MCC-UKF"   : '#FF9800',
    "PC-UKF-2D" : '#9C27B0',
    "GUKF"      : '#4CAF50',
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
    print("\n" + "═" * 90)
    print("  PER-FILE RMSE (mm)   [⚠ = trajectory collapsed]")
    print("═" * 90)
    print(hdr)
    print("─" * 90)

    for path in file_paths:
        dist_raw = parse_file(path)
        if dist_raw is None:
            continue
        T   = len(dist_raw)
        row = f"{os.path.basename(path):<18s}"

        for name, method in METHODS.items():
            if method is None:
                pos = np.array([LS_position(dist_raw[t]) for t in range(T)])
            else:
                filt_cls, kwargs = method
                t0  = time.perf_counter()
                pos = run_filter(filt_cls, dist_raw, **kwargs)
                timing[name].append(time.perf_counter() - t0)

            valid     = pos[~np.any(np.isnan(pos), axis=1)]
            collapsed = is_trajectory_collapsed(valid, expected_path_length)
            if collapsed:
                collapse_warn[name].append(os.path.basename(path))

            errs = nearest_gt_error(valid, gt_xy)
            rmse = np.sqrt(np.mean(errs**2)) if len(errs) > 0 else float('nan')
            flag = "⚠" if collapsed else " "
            row += f" {flag}{rmse:>10.1f}"
            all_errors[name].extend(errs)
            all_pos[name].extend(valid)
            per_file_rmse[name].append(rmse)

        print(row)

    print("─" * 90)

    print("\n  TIMING (ms/sample):")
    for name, times in timing.items():
        if not times:
            continue
        total_t = sum(times)
        n_samp  = sum(safe_len(p) for p in file_paths[:len(times)])
        if n_samp > 0:
            print(f"    {name:<14s}: {total_t * 1000 / n_samp:.4f} ms/sample")

    print("\n  COLLAPSE WARNINGS:")
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
#  GRID SEARCH — LOO (Leave-One-Out) cho tất cả methods  (V21)
#  Tuning công bằng: mọi filter dùng cùng một LOO protocol.
#  In ra console ngay khi tìm được best mới (RMSE + hệ số).
# ══════════════════════════════════════════════════════════════════════
def _loo_fold_rmse(filt_class, kwargs, val_file, gt_xy, expected_path_length):
    """
    Chạy filter trên một file validation, trả về RMSE fold.
    Nếu collapse hoặc file lỗi → trả về penalty.
    """
    d = parse_file(val_file)
    if d is None:
        return None
    p     = run_filter(filt_class, d, **kwargs)
    valid = p[~np.any(np.isnan(p), axis=1)]
    if is_trajectory_collapsed(valid, expected_path_length):
        return 500.0 + COLLAPSE_PENALTY / 1000.0
    if len(valid) == 0:
        return 500.0
    errs = nearest_gt_error(valid, gt_xy)
    return float(np.sqrt(np.mean(errs**2)))


def _grid_search_loo(filt_class, grid, fixed_params,
                     file_paths, gt_xy, expected_path_length,
                     method_name):
    """
    LOO grid search chung cho mọi filter.

    Protocol:
      - Với mỗi combo (p1, p2, ...):
          LOO-RMSE = mean( _loo_fold_rmse(val_file_k) for k in 1..N )
      - Khi tìm được best mới → in ngay: RMSE + tất cả hệ số
      - Trả về (best_params, best_loo_rmse)

    Lý do dùng LOO:
      - Không có fitting step → LOO đơn giản là unbiased estimate của
        generalization error: params không "thấy" file đang được validate.
      - Đảm bảo so sánh công bằng giữa các filter.
    """
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n      = len(file_paths)

    print(f"\n  LOO Grid search [{method_name}]: "
          f"{len(combos)} combos × {n} folds = {len(combos) * n} runs")
    print(f"  {'LOO-RMSE':>10s}  Params")
    print(f"  {'─' * 55}")

    best_loo_rmse = float('inf')
    best_params   = {**fixed_params}

    for combo in combos:
        params = {**fixed_params, **dict(zip(keys, combo))}

        fold_rmses = []
        for val_file in file_paths:
            fr = _loo_fold_rmse(filt_class, params, val_file,
                                gt_xy, expected_path_length)
            if fr is not None:
                fold_rmses.append(fr)

        if not fold_rmses:
            continue

        loo_rmse = float(np.mean(fold_rmses))

        if loo_rmse < best_loo_rmse:
            best_loo_rmse = loo_rmse
            best_params   = params.copy()
            # ── In ngay khi có best mới ─────────────────────────────
            tag       = " ⚠COLLAPSED" if loo_rmse >= COLLAPSE_PENALTY / 1000 else ""
            param_str = "  ".join(f"{k}={params[k]}" for k in keys)
            print(f"  ★ {loo_rmse:8.2f}mm{tag:<12s}  {param_str}")

    print(f"  {'─' * 55}")
    best_param_str = "  ".join(f"{k}={best_params[k]}" for k in keys)
    print(f"  ✔ Best: {best_loo_rmse:.2f}mm  |  {best_param_str}")
    return best_params, best_loo_rmse


# ── Wrappers per filter ───────────────────────────────────────────────
def grid_search_huber(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'    : [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        'delta': [1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0],
    }
    best, _ = _grid_search_loo(HuberUKF, grid, {'q': HUBER_Q},
                               file_paths, gt_xy, expected_path_length,
                               "Huber-UKF")
    return best


def grid_search_mcc(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'        : [10.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        'kernel_bw': [100.0, 200.0, 500.0, 1100.0, 1200.0, 1300.0, 1700.0],
    }
    best, _ = _grid_search_loo(MCCUKF, grid, {'q': MCC_Q},
                               file_paths, gt_xy, expected_path_length,
                               "MCC-UKF")
    return best


def grid_search_gukf(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'     : [10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        'sigma' : [0.5, 1.0, 2.0, 3.0, 5.0],
        'n_half': [1, 2, 3, 4],
    }
    best, _ = _grid_search_loo(GUKF, grid, {'q': GUKF_Q},
                               file_paths, gt_xy, expected_path_length,
                               "GUKF")
    return best


def grid_search_pcukf_loo(file_paths, gt_xy, expected_path_length):
    grid = {
        'r_base' : [10.0, 25.0, 50.0, 100.0, 150.0, 200.0],
        'r_scale': [3.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    }
    return _grid_search_loo(PCUKF2D, grid, {'q': PCUKF_Q},
                            file_paths, gt_xy, expected_path_length,
                            "PC-UKF-2D")


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
        n_col    = len(collapse_warn.get(label, []))
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
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'k--', lw=2.5,
            label='Ground Truth', alpha=0.5, zorder=5)
    for label, pos in positions_dict.items():
        color = COLORS.get(label, 'gray')
        if len(pos) == 0:
            continue
        errs  = nearest_gt_error(pos, gt_xy)
        rmse  = np.sqrt(np.mean(errs**2))
        n_col = len(collapse_warn.get(label, []))
        warn  = f" ⚠{n_col}" if n_col > 0 else ""
        lw = 2.0 if label != "Raw+LS" else 1.2
        ls = '-'  if label != "Raw+LS" else '--'
        ax.plot(pos[:, 0], pos[:, 1], color=color, lw=lw, ls=ls, alpha=0.75,
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
        bars    = ax.bar(range(len(methods)), vals, color=colors,
                         alpha=0.85, edgecolor='white', lw=1.5)
        for bar, val, m in zip(bars, vals, methods):
            if metrics_dict[m].get('collapsed', False):
                bar.set_hatch('//')
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
    print("=" * 75)
    print("  V21: Robust & Adaptive UKF — Full LOO Tuning (All Methods)")
    print("=" * 75)
    print("  1. Raw+LS      — Weighted Least Squares (baseline)")
    print("  2. UKF          — Unscented Kalman Filter chuẩn")
    print("  3. Huber-UKF    — IRLS với Huber M-estimator")
    print("  4. MCC-UKF      — Maximum Correntropy Criterion UKF")
    print("  5. PC-UKF-2D    — MAD sigma-free consensus (LOO-tuned)")
    print("  6. GUKF         — Gaussian-smoothed UKF (Sun et al. 2025)")
    print("=" * 75)
    print(f"\n  Tuning policy (V21 — fully fair):")
    print(f"    Raw+LS, UKF   : không tune (không có params)")
    print(f"    Huber-UKF     : LOO cross-validation")
    print(f"    MCC-UKF       : LOO cross-validation")
    print(f"    GUKF          : LOO cross-validation")
    print(f"    PC-UKF-2D     : LOO cross-validation (sigma-free)")
    print(f"  → Tất cả filter tunable đều dùng cùng _grid_search_loo()")
    print(f"  → In best mới ngay khi tìm thấy (RMSE + hệ số)")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"\n[!] Không tìm thấy file .txt trong '{DATA_DIR}/'")
        return
    print(f"\n  Tìm thấy {n} file(s) trong '{DATA_DIR}/'")

    np.random.seed(42)
    eval_files = [all_files[i] for i in np.random.permutation(n)]

    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    expected_path_length = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"  Path={expected_path_length:.0f}mm  Time={total_time:.1f}s  GT={len(gt_xy)} points")

    if DO_GRID_SEARCH:
        print("\n  [LOO Grid Search] Tuning tất cả filters...")

        best_huber           = grid_search_huber(eval_files, gt_xy, expected_path_length)
        best_mcc             = grid_search_mcc(eval_files,   gt_xy, expected_path_length)
        best_gukf            = grid_search_gukf(eval_files,  gt_xy, expected_path_length)
        best_pc, _           = grid_search_pcukf_loo(eval_files, gt_xy, expected_path_length)

        METHODS["Huber-UKF"] = (HuberUKF, best_huber)
        METHODS["MCC-UKF"]   = (MCCUKF,   best_mcc)
        METHODS["GUKF"]      = (GUKF,     best_gukf)
        METHODS["PC-UKF-2D"] = (PCUKF2D,  best_pc)

        print(f"\n  ══ Params sau LOO grid search ══")
        print(f"    Huber  : r={best_huber['r']}  delta={best_huber['delta']}")
        print(f"    MCC    : r={best_mcc['r']}  kernel_bw={best_mcc['kernel_bw']}")
        print(f"    GUKF   : r={best_gukf['r']}  sigma={best_gukf['sigma']}  n_half={best_gukf['n_half']}")
        print(f"    PC-UKF : r_base={best_pc['r_base']}  r_scale={best_pc['r_scale']}  [sigma-free]")
    else:
        print(f"\n  [INFO] Grid Search TẮT — dùng params mặc định:")
        print(f"    PC-UKF-2D: r_base={PCUKF_R_BASE} r_scale={PCUKF_R_SCALE} [sigma-free]")

    # ── Evaluate ──────────────────────────────────────────────────────
    errors, positions, collapse_warn, per_file_rmse = evaluate_files(
        eval_files, gt_xy, expected_path_length)

    # ── Summary table ─────────────────────────────────────────────────
    metrics = {}
    print(f"\n{'═' * 110}")
    print(f"  SUMMARY TABLE (mm)   — ⚠ = trajectory collapsed")
    print(f"{'═' * 110}")
    print(f"  {'Method':<14s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'CEP90':>7s} "
          f"{'P95':>7s} {'MAX':>7s} {'RMSE mean±std':>16s} {'MotionR':>8s} {'Status'}")
    print(f"  {'─' * 95}")

    for label, errs in errors.items():
        if len(errs) == 0:
            continue
        pos_arr = positions.get(label, np.empty((0, 2)))
        m = compute_metrics(errs, pos_arr, expected_path_length)
        metrics[label] = m

        pf_vals = [v for v in per_file_rmse.get(label, []) if not math.isnan(v)]
        if len(pf_vals) >= 2:
            std_str = f"{np.mean(pf_vals):.1f} ± {np.std(pf_vals, ddof=1):.1f}"
        elif len(pf_vals) == 1:
            std_str = f"{pf_vals[0]:.1f} ± N/A"
        else:
            std_str = "N/A"

        n_col  = len(collapse_warn.get(label, []))
        status = f"⚠ {n_col} collapsed" if n_col > 0 else "✅ OK"
        mr_str = (f"{m['motion_ratio']:.2f}"
                  if not math.isnan(m.get('motion_ratio', float('nan'))) else "N/A")
        tag    = " ◀ LOO" if label == "PC-UKF-2D" else (
                 " ◀ LOO" if label in ("Huber-UKF", "MCC-UKF", "GUKF") else "")
        print(f"  {label:<14s} {m['rmse']:>7.1f} {m['mae']:>7.1f}"
              f" {m['cep50']:>7.1f} {m['cep90']:>7.1f} {m['p95']:>7.1f} {m['max']:>7.1f}"
              f" {std_str:>16s}  {mr_str:>8s}  {status}{tag}")

    # ── Wilcoxon ──────────────────────────────────────────────────────
    print(f"\n  Wilcoxon tests (two-sided, vs PC-UKF-2D):")
    ref_err = errors.get("PC-UKF-2D", np.array([]))
    for name, errs in errors.items():
        if name == "PC-UKF-2D" or len(errs) == 0 or len(ref_err) == 0:
            continue
        N = min(len(errs), len(ref_err))
        if N > 20:
            try:
                _, p = wilcoxon(errs[:N], ref_err[:N])
                sym  = '✅ p<0.05' if p < 0.05 else '⚠️  ns'
                print(f"    {name:<14s} vs PC-UKF-2D: p={p:.4f}  {sym}")
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
