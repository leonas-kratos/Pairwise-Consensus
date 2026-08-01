# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — V20: Robust & Adaptive KF
===================================================
CÁC PHƯƠNG PHÁP:
  1. Raw + LS        — baseline
  2. UKF              — Unscented Kalman Filter chuẩn
  3. Huber-UKF        — IRLS Huber M-estimator
  4. MCC-UKF          — Maximum Correntropy Criterion UKF
  5. PC-UKF-2D        — Pairwise Consensus V3 (MAD auto-normalized, sigma-free)
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
    [4000, 8800],
    [0,    8800],
    [0,    0   ],
    [4000, 0   ],
], dtype=float)

ANCHOR_HEIGHT = 1400.0

WAYPOINTS = np.array([
    [800.0,  400.0],    # A
    [3000.0, 400.0],    # B
    [3000.0, 8000.0],   # C
    [800.0,  8000.0],   # D
    [800.0,  400.0],    # A (khép kín)
], dtype=float)

SPEED      = 200.0
GT_SPACING = 5.0
N_ANCHORS  = 4

# ─── Standard UKF ────────────────────────────────────────────────────
UKF_STD_Q = 0.001
UKF_STD_R = 100.0

# ─── Huber-UKF ───────────────────────────────────────────────────────
HUBER_Q       = 0.001
HUBER_R       = 50.0
HUBER_DELTA   = 20.0
HUBER_MAXITER = 5

# ─── MCC-UKF ─────────────────────────────────────────────────────────
MCC_Q         = 0.001
MCC_R         = 50.0
MCC_KERNEL_BW = 1700.0
MCC_MAXITER   = 5

# ─── PC-UKF-2D ───────────────────────────────────────────────────────
PCUKF_Q      = 0.001
PCUKF_R_BASE = 50.0
PCUKF_R_SCALE = 10.0
PCUKF_SIGMA  = 1.0

# ─── GUKF (Gaussian-smoothed UKF) ────────────────────────────────────
GUKF_Q      = 0.001
GUKF_R      = 50.0
GUKF_SIGMA  = 5.0
GUKF_N_HALF = 4

# ─── UKF common ──────────────────────────────────────────────────────
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

# ─── Grid search flags ───────────────────────────────────────────────
# Bật/tắt grid search trong nhánh fallback (chỉ có 1 file)
DO_GRID_SEARCH     = False
# Bật/tắt grid search trong LOO (≥ 2 file).
# Khi False → dùng default params, không mất thời gian tune.
DO_LOO_GRID_SEARCH = True

DATA_DIR = "./data"
SAVE_DIR = "./outputs_LOO"

# ─── Motion sanity check ─────────────────────────────────────────────
MOTION_RATIO_MIN = 1.00

# ─── Shared fixed params cho grid search — fair comparison ───────────
FIXED_Q = 0.001   # process noise — tất cả method dùng chung
FIXED_R = 100.0    # base measurement noise — tất cả method dùng chung

COLLAPSE_PENALTY = 1e9


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
    diffs = np.diff(pos_xy, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


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
#  3. HUBER-UKF
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
        K     = np.zeros((self.n, N_ANCHORS))

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
#  4. MCC-UKF
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
#  5. PC-UKF-2D  (V3 — MAD Auto-Normalized, Sigma-Free)
# ══════════════════════════════════════════════════════════════════════
def _pc_scores_v3(innovations, Pzz_diag):
    """
    PC Scoring V3 — MAD auto-normalized, sigma-free.

    1. Geo-normalize:  std_innov[i] = ν[i] / sqrt(Pzz_diag[i])  ≈ N(0,1) LOS
    2. MAD normalize:  normed[i] = std_innov[i] / (1.4826·MAD)
       → tự động scale, không cần tune sigma
    3. T-kernel (σ=1 cố định): score[i] = mean_j≠i T(normed[i]-normed[j])
    """
    std_innov = innovations / np.sqrt(np.maximum(Pzz_diag, 1e-9))
    med    = np.median(std_innov)
    mad    = np.median(np.abs(std_innov - med))
    normed = std_innov / (1.4826 * mad + 1e-9)

    nu  = 4.0
    scores = np.zeros(N_ANCHORS)
    for i in range(N_ANCHORS):
        diffs = np.array([normed[i] - normed[j]
                          for j in range(N_ANCHORS) if j != i])
        scores[i] = np.mean(
            (1.0 + diffs**2 / (nu + 1e-9)) ** (-(nu + 1.0) / 2.0)
        )
    return scores


class PCUKF2D:
    """
    PC-UKF V3: innovation từ UKF predict step, MAD auto-normalized.
    Tham số đặc trưng duy nhất cần tune: r_scale.
    Q và r_base cố định (= FIXED_Q, FIXED_R).
    """
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

    def _z_hat_and_Pzz_diag(self):
        """Tính z_hat và Pzz diagonal từ sigma points (không update state)."""
        pts   = sigma_points(self.x, self.P, self.c)
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*self.n + 1)])
        z_hat = self.Wm @ Z_pts
        Pzz_diag = np.full(N_ANCHORS, self.R_base)
        for i in range(2*self.n + 1):
            dz = Z_pts[i] - z_hat
            Pzz_diag += self.Wc[i] * dz**2
        return z_hat, Pzz_diag

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)

        # 1. Predict
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        self.x = x_pred
        self.P = P_pred

        # 2. z_hat + Pzz_diag từ sigma points
        z_hat, Pzz_diag = self._z_hat_and_Pzz_diag()

        # 3. PC scoring V3 (MAD auto-normalized)
        innov  = z_raw - z_hat
        scores = _pc_scores_v3(innov, Pzz_diag)
        R_diag = self.R_base * (1.0 + self.R_scale * (1.0 - scores))

        # 4. UKF update với R adaptive (full Pzz matrix)
        pts   = sigma_points(x_pred, P_pred, self.c)
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*self.n + 1)])
        z_hat = self.Wm @ Z_pts

        R   = np.diag(R_diag)
        Pzz = R.copy()
        Pxz = np.zeros((self.n, N_ANCHORS))
        for i in range(2*self.n + 1):
            dz   = Z_pts[i] - z_hat
            dx   = pts[i] - x_pred
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()

        self.x = x_pred + K @ (z_raw - z_hat)
        self.P = make_spd(P_pred - K @ Pzz @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  6. GUKF — Gaussian Unscented Kalman Filter
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

        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat

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
#  METHODS TABLE
# ══════════════════════════════════════════════════════════════════════
METHODS = {
    "Raw+LS"    : None,
    "UKF"       : (StandardUKF, dict(q=UKF_STD_Q,  r=UKF_STD_R)),
    "Huber-UKF" : (HuberUKF,   dict(q=HUBER_Q,     r=HUBER_R,    delta=HUBER_DELTA)),
    "MCC-UKF"   : (MCCUKF,     dict(q=MCC_Q,       r=MCC_R,      kernel_bw=MCC_KERNEL_BW)),
    "PC-UKF-2D" : (PCUKF2D,    dict(q=PCUKF_Q,     r_base=PCUKF_R_BASE,
                                     r_scale=PCUKF_R_SCALE)),
    "GUKF"      : (GUKF,        dict(q=GUKF_Q,      r=GUKF_R,
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
#  EVALUATE (fallback — không LOO)
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
    print("\n" + "═" * 80)
    print("  PER-FILE RMSE (mm)   [⚠ = trajectory collapsed / static]")
    print("═" * 80)
    print(hdr)
    print("─" * 80)

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

    print("─" * 80)

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
            print(f"    ⚠  {name:<14s}: {len(files)} file(s) → "
                  f"{', '.join(files[:5])}"
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
def _eval_rmse_single(filt_class, kwargs, file_paths, gt_xy,
                      expected_path_length):
    pos_all     = []
    n_collapsed = 0
    n_files     = 0

    for path in file_paths:
        d = parse_file(path)
        if d is None:
            continue
        n_files += 1
        p     = run_filter(filt_class, d, **kwargs)
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
    """
    Tune duy nhất delta — tham số đặc thù của Huber.
    10 điểm log-spaced trên [0.1, 100].
    """
    deltas = np.logspace(-1, 2, 10).tolist()   # ~0.1, 0.22, 0.46 … 100
    print(f"\n  Grid search Huber-UKF: {len(deltas)} points "
          f"[q={FIXED_Q}, r={FIXED_R} fixed]")
    best_rmse = float('inf')
    best      = dict(q=FIXED_Q, r=FIXED_R, delta=HUBER_DELTA)
    for delta in deltas:
        params = dict(q=FIXED_Q, r=FIXED_R, delta=delta)
        rmse   = _eval_rmse_single(HuberUKF, params, file_paths, gt_xy,
                                    expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best      = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best Huber-UKF: RMSE={best_rmse:.1f}mm{tag} | "
          f"delta={best['delta']:.3f}")
    return best


def grid_search_mcc(file_paths, gt_xy, expected_path_length):
    """
    Tune duy nhất kernel_bw — tham số đặc thù của MCC.
    10 điểm log-spaced trên [100, 5000].
    """
    bws = np.logspace(2, np.log10(5000), 10).tolist()
    print(f"\n  Grid search MCC-UKF: {len(bws)} points "
          f"[q={FIXED_Q}, r={FIXED_R} fixed]")
    best_rmse = float('inf')
    best      = dict(q=FIXED_Q, r=FIXED_R, kernel_bw=MCC_KERNEL_BW)
    for bw in bws:
        params = dict(q=FIXED_Q, r=FIXED_R, kernel_bw=bw)
        rmse   = _eval_rmse_single(MCCUKF, params, file_paths, gt_xy,
                                    expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best      = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best MCC-UKF: RMSE={best_rmse:.1f}mm{tag} | "
          f"kernel_bw={best['kernel_bw']:.1f}")
    return best


def grid_search_pcukf(file_paths, gt_xy, expected_path_length):
    """
    PC-UKF V3 — chỉ tune r_scale (tham số đặc trưng duy nhất).
    Q=FIXED_Q, r_base=FIXED_R cố định. sigma loại bỏ hoàn toàn (MAD auto).
    10 điểm log-spaced trên [1, 100].
    """
    r_scales = np.logspace(0, 2, 10).tolist()   # 1 → 100
    print(f"\n  Grid search PC-UKF-2D (V3, sigma-free): {len(r_scales)} points "
          f"[q={FIXED_Q}, r_base={FIXED_R} fixed]")
    best_rmse = float('inf')
    best      = dict(q=FIXED_Q, r_base=FIXED_R, r_scale=PCUKF_R_SCALE)
    for r_scale in r_scales:
        params = dict(q=FIXED_Q, r_base=FIXED_R, r_scale=r_scale)
        rmse   = _eval_rmse_single(PCUKF2D, params, file_paths, gt_xy,
                                    expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best      = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best PC-UKF-2D: RMSE={best_rmse:.1f}mm{tag} | "
          f"r_scale={best['r_scale']:.3f}")
    return best


def grid_search_gukf(file_paths, gt_xy, expected_path_length):
    """
    Tune sigma và n_half — tham số đặc thù của Gaussian kernel.
    8×8 = 64 combinations; sigma log-spaced [0.3, 20], n_half linear.
    """
    grid = {
        'sigma'  : np.logspace(np.log10(0.3), np.log10(20), 8).tolist(),
        'n_half' : [1, 2, 3, 4, 5, 6, 8, 10],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search GUKF: {len(combos)} combinations "
          f"[q={FIXED_Q}, r={FIXED_R} fixed]")
    best_rmse = float('inf')
    best      = dict(q=FIXED_Q, r=FIXED_R,
                     sigma=GUKF_SIGMA, n_half=GUKF_N_HALF)
    for combo in combos:
        params        = dict(zip(keys, combo))
        params['q']   = FIXED_Q
        params['r']   = FIXED_R
        rmse = _eval_rmse_single(GUKF, params, file_paths, gt_xy,
                                  expected_path_length)
        if rmse < best_rmse:
            best_rmse = rmse
            best      = params.copy()
    tag = " [⚠ COLLAPSED]" if best_rmse >= COLLAPSE_PENALTY else ""
    print(f"  Best GUKF: RMSE={best_rmse:.1f}mm{tag} | "
          f"sigma={best['sigma']:.3f} n_half={best['n_half']}")
    return best


def _default_params():
    """Trả về params mặc định khi không chạy grid search."""
    return (
        dict(q=FIXED_Q, r=FIXED_R, delta=HUBER_DELTA),
        dict(q=FIXED_Q, r=FIXED_R, kernel_bw=MCC_KERNEL_BW),
        dict(q=FIXED_Q, r_base=FIXED_R, r_scale=PCUKF_R_SCALE),
        dict(q=FIXED_Q, r=FIXED_R, sigma=GUKF_SIGMA, n_half=GUKF_N_HALF),
    )


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
        lw   = 2.5 if label != "Raw+LS" else 1.5
        ls   = '-'  if label != "Raw+LS" else '--'
        n_col    = len(collapse_warn.get(label, []))
        warn_tag = f" ⚠{n_col}" if n_col > 0 else ""
        ax.plot(s, cdf, lw=lw, color=c, ls=ls,
                label=f"{label}{warn_tag}  RMSE={rmse:.1f}mm")
        ax.axvline(rmse, color=c, ls=':', lw=0.8, alpha=0.4)
    ax.set_xlabel("Position Error (mm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("CDF",                 fontsize=13, fontweight='bold')
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
    ax.plot(gt_xy[:, 0], gt_xy[:, 1],
            'k--', lw=2.5, label='Ground Truth', alpha=0.5, zorder=5)

    for label, pos in positions_dict.items():
        color = COLORS.get(label, 'gray')
        if len(pos) == 0:
            continue
        errs  = nearest_gt_error(pos, gt_xy)
        rmse  = np.sqrt(np.mean(errs**2))
        n_col = len(collapse_warn.get(label, []))
        warn  = f" ⚠{n_col}" if n_col > 0 else ""
        lw    = 2.0 if label != "Raw+LS" else 1.2
        ls    = '-'  if label != "Raw+LS" else '--'
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
        hatches = ['//' if metrics_dict[m].get('collapsed', False) else ''
                   for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors,
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
#  LEAVE-ONE-OUT CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
def loo_tune_and_eval(all_files, gt_xy, expected_path_length):
    """
    Leave-One-Out cross-validation:
      - Với mỗi fold i: (optionally) tune trên (n-1) file, test trên file i
      - Raw+LS, UKF: params cố định, không tune
      - Huber, MCC, PC, GUKF: tune khi DO_LOO_GRID_SEARCH=True,
                               ngược lại dùng default params

    Trả về errors aggregate trên toàn bộ dataset (unbiased).
    """
    n = len(all_files)
    gs_label = "with grid search" if DO_LOO_GRID_SEARCH else "default params (grid search OFF)"
    print(f"\n{'═'*70}")
    print(f"  LEAVE-ONE-OUT CV — {n} folds  [{gs_label}]")
    print(f"{'═'*70}")

    loo_errors    = {m: [] for m in METHODS}
    loo_pos       = {m: [] for m in METHODS}
    per_file_rmse = {m: [] for m in METHODS}
    fold_params   = []

    hdr = f"  {'File':<20s}"
    for m in METHODS:
        hdr += f" {m:>12s}"
    print(hdr)
    print("  " + "─" * (20 + 13 * len(METHODS)))

    for i, test_file in enumerate(all_files):
        tune_files = [f for j, f in enumerate(all_files) if j != i]
        fname      = os.path.basename(test_file)

        # ── Tune (hoặc dùng default) ───────────────────────────────────
        if DO_LOO_GRID_SEARCH:
            best_huber = grid_search_huber(tune_files, gt_xy, expected_path_length)
            best_mcc   = grid_search_mcc  (tune_files, gt_xy, expected_path_length)
            best_pc    = grid_search_pcukf(tune_files, gt_xy, expected_path_length)
            best_gukf  = grid_search_gukf (tune_files, gt_xy, expected_path_length)
        else:
            best_huber, best_mcc, best_pc, best_gukf = _default_params()

        fold_params.append({
            'fold': i, 'test': fname,
            'huber': best_huber, 'mcc': best_mcc,
            'pc': best_pc, 'gukf': best_gukf,
        })

        # ── Methods dict với params của fold này ───────────────────────
        methods_fold = {
            "Raw+LS"    : None,
            "UKF"       : (StandardUKF, dict(q=UKF_STD_Q, r=UKF_STD_R)),
            "Huber-UKF" : (HuberUKF,   best_huber),
            "MCC-UKF"   : (MCCUKF,     best_mcc),
            "PC-UKF-2D" : (PCUKF2D,    best_pc),
            "GUKF"      : (GUKF,        best_gukf),
        }

        # ── Evaluate trên test file ────────────────────────────────────
        dist_raw = parse_file(test_file)
        if dist_raw is None:
            print(f"  [!] Không đọc được {fname}, bỏ qua.")
            continue

        T   = len(dist_raw)
        row = f"  {fname:<20s}"

        for name, method in methods_fold.items():
            if method is None:
                pos = np.array([LS_position(dist_raw[t]) for t in range(T)])
            else:
                filt_cls, kwargs = method
                pos = run_filter(filt_cls, dist_raw, **kwargs)

            valid = pos[~np.any(np.isnan(pos), axis=1)]
            errs  = nearest_gt_error(valid, gt_xy)
            rmse  = float(np.sqrt(np.mean(errs**2))) if len(errs) > 0 else float('nan')

            row += f" {rmse:>12.1f}"
            loo_errors[name].extend(errs)
            loo_pos[name].extend(valid)
            per_file_rmse[name].append(rmse)

        print(row)

    # ── In bảng params per fold ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  PC-UKF params per fold (verify LOO):")
    print(f"  {'Fold':<6s} {'Test file':<20s} "
          f"{'R_base':>8s} {'R_scale':>8s}")
    print(f"  {'─'*46}")
    for p in fold_params:
        pc = p['pc']
        print(f"  {p['fold']+1:<6d} {p['test']:<20s} "
              f"{pc.get('r_base', FIXED_R):>8.1f} "
              f"{pc.get('r_scale', PCUKF_R_SCALE):>8.3f}")

    return (
        {m: np.array(v) for m, v in loo_errors.items()},
        {m: np.array(v) for m, v in loo_pos.items()},
        per_file_rmse,
    )


# ══════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════
def print_summary(errors, positions, per_file_rmse,
                  collapse_warn, expected_path_length, mode_label):
    metrics = {}
    print(f"\n{'═'*105}")
    print(f"  SUMMARY TABLE — {mode_label} (mm)")
    print(f"{'═'*105}")
    print(f"  {'Method':<14s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} "
          f"{'CEP90':>7s} {'P95':>7s} {'MAX':>7s} "
          f"{'mean±std':>14s} {'MotionR':>8s} {'Status'}")
    print(f"  {'─'*95}")

    for label, errs in errors.items():
        if len(errs) == 0:
            continue
        pos_arr = positions.get(label, np.empty((0, 2)))
        m = compute_metrics(errs, pos_arr, expected_path_length)
        metrics[label] = m

        pf_vals = [v for v in per_file_rmse.get(label, [])
                   if not math.isnan(v)]
        if len(pf_vals) >= 2:
            std_str = f"{np.mean(pf_vals):.1f}±{np.std(pf_vals, ddof=1):.1f}"
        elif len(pf_vals) == 1:
            std_str = f"{pf_vals[0]:.1f}±N/A"
        else:
            std_str = "N/A"

        n_col  = len(collapse_warn.get(label, []))
        status = f"⚠ {n_col} collapsed" if n_col > 0 else "✅ OK"
        mr_str = (f"{m['motion_ratio']:.2f}"
                  if not math.isnan(m.get('motion_ratio', float('nan')))
                  else "N/A")
        print(f"  {label:<14s} {m['rmse']:>7.1f} {m['mae']:>7.1f}"
              f" {m['cep50']:>7.1f} {m['cep90']:>7.1f}"
              f" {m['p95']:>7.1f} {m['max']:>7.1f}"
              f" {std_str:>14s} {mr_str:>8s}  {status}")

    # Wilcoxon
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

    return metrics


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  V20: Robust & Adaptive UKF + LOO-CV")
    print("=" * 70)
    print("  1. Raw+LS       — Least Squares baseline")
    print("  2. UKF          — Unscented Kalman Filter")
    print("  3. Huber-UKF    — IRLS Huber M-estimator")
    print("  4. MCC-UKF      — Maximum Correntropy Criterion UKF")
    print("  5. PC-UKF-2D    — Pairwise Consensus UKF (proposed)")
    print("  6. GUKF         — Gaussian-smoothed UKF")
    print(f"  DO_GRID_SEARCH     = {DO_GRID_SEARCH}")
    print(f"  DO_LOO_GRID_SEARCH = {DO_LOO_GRID_SEARCH}")
    print("=" * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"\n[!] Không tìm thấy file .txt trong '{DATA_DIR}/'")
        return
    print(f"\n  Tìm thấy {n} file(s) trong '{DATA_DIR}/'")

    np.random.seed(42)

    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    expected_path_length = np.linalg.norm(
        np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"  Path={expected_path_length:.0f}mm  "
          f"Time={total_time:.1f}s  GT={len(gt_xy)} points")

    if n >= 2:
        # ── LOO ─────────────────────────────────────────────────────────
        mode_label = f"LOO-CV ({n} folds)"
        errors, positions, per_file_rmse = loo_tune_and_eval(
            all_files, gt_xy, expected_path_length)

        # Collapse warn từ aggregate positions
        collapse_warn = {m: [] for m in METHODS}
        for name, pos_arr in positions.items():
            if len(pos_arr) > 0 and is_trajectory_collapsed(
                    pos_arr, expected_path_length):
                collapse_warn[name].append("aggregate")

    else:
        # ── Fallback: chỉ có 1 file ──────────────────────────────────────
        mode_label = "Grid Search (1 file — chưa đủ mẫu cho LOO)"
        print(f"\n[!] Chỉ có {n} file — LOO cần ≥ 2 file.")
        print("    → Dùng grid search nếu DO_GRID_SEARCH=True, "
              "thu thêm mẫu để chạy LOO.")

        if DO_GRID_SEARCH:
            print("\n  [Grid Search] Tuning...")
            best_huber, best_mcc, best_pc, best_gukf = (
                grid_search_huber(all_files, gt_xy, expected_path_length),
                grid_search_mcc  (all_files, gt_xy, expected_path_length),
                grid_search_pcukf(all_files, gt_xy, expected_path_length),
                grid_search_gukf (all_files, gt_xy, expected_path_length),
            )
            METHODS["Huber-UKF"] = (HuberUKF, best_huber)
            METHODS["MCC-UKF"]   = (MCCUKF,   best_mcc)
            METHODS["PC-UKF-2D"] = (PCUKF2D,  best_pc)
            METHODS["GUKF"]      = (GUKF,      best_gukf)

        errors, positions, collapse_warn, per_file_rmse = evaluate_files(
            all_files, gt_xy, expected_path_length)

    # ── Summary + Wilcoxon ─────────────────────────────────────────────
    metrics = print_summary(errors, positions, per_file_rmse,
                            collapse_warn, expected_path_length, mode_label)

    # ── Plots ──────────────────────────────────────────────────────────
    suffix = '_loo' if n >= 2 else '_grid'
    plot_cdf(errors, collapse_warn,
             os.path.join(SAVE_DIR, f'cdf{suffix}.png'))
    plot_trajectories(positions, gt_xy, collapse_warn,
                      os.path.join(SAVE_DIR, f'trajectories{suffix}.png'))
    plot_bar(metrics,
             os.path.join(SAVE_DIR, f'bar{suffix}.png'))

    for label, errs in errors.items():
        safe = (label.lower()
                .replace('+', '_').replace('-', '_').replace(' ', '_'))
        np.save(os.path.join(SAVE_DIR, f'errors{suffix}_{safe}.npy'), errs)

    print(f"\n[✓] Kết quả lưu tại '{SAVE_DIR}/'")
    if n >= 2:
        gs_info = ("với grid search per fold"
                   if DO_LOO_GRID_SEARCH
                   else "với default params (DO_LOO_GRID_SEARCH=False)")
        print(f"\n  → Paper claim:")
        print(f"    'Hyperparameters were selected via leave-one-out")
        print(f"     cross-validation on {n} independent runs ({gs_info}).'")
    else:
        print(f"\n  → Thu thêm mẫu NLOS để chạy LOO.")


if __name__ == "__main__":
    main()
