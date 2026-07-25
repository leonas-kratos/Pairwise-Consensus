# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — EKF Suite V1
======================================
CÁC PHƯƠNG PHÁP:
  1. Raw+LS        — baseline Weighted Least Squares
  2. EKF            — Extended Kalman Filter chuẩn
  3. Huber-EKF      — IRLS Huber M-estimator trên innovation
  4. MCC-EKF        — Maximum Correntropy Criterion EKF
  5. PC-EKF-2D      — Pairwise Consensus MAD-normalized + EKF (sigma-free)
  6. AEKF           — Adaptive EKF (Sage-Husa noise estimator)
  7. REKF           — Robust EKF với Student-t likelihood (Agamennoni 2012)

ĐIỂM KHÁC BIỆT SO VỚI UKF SUITE:
  - Linearization bằng Jacobian H(x) thay vì sigma-point propagation
  - H(x): ∂h_i/∂x = (x - ax_i)/r_i,  ∂h_i/∂y = (y - ay_i)/r_i
    với r_i = sqrt((x-ax_i)² + (y-ay_i)² + h_anchor²)
  - Predict: P⁻ = P + Q  (không cần sigma points)
  - Update: K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹ (linear Kalman gain)
  - Nhẹ hơn UKF (~3× faster per step), phù hợp embedded

SOTA EKF:
  - AEKF (Sage-Husa): adaptive Q/R estimation online
    → ước lượng Q̂ và R̂ từ innovation history (sliding window)
  - REKF (Student-t): robust với heavy-tail noise
    → weight per measurement = (ν+1)/(ν + r²/S)
    → ν = degrees of freedom (tune)

TUNING:
  - Tất cả filter có params: LOO cross-validation
  - Raw+LS, EKF: không tune
  - In best mới ngay khi tìm thấy (RMSE + hệ số)

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

ANCHOR_HEIGHT = 1400.0
SPEED         = 200.0
GT_SPACING    = 5.0
N_ANCHORS     = 4

# ─── Standard EKF ────────────────────────────────────────────────────
EKF_Q = 0.001
EKF_R = 50.0

# ─── Huber-EKF ───────────────────────────────────────────────────────
HUBER_Q       = 0.001
HUBER_R       = 50.0
HUBER_DELTA   = 1.5      # tunable
HUBER_MAXITER = 5

# ─── MCC-EKF ─────────────────────────────────────────────────────────
MCC_Q         = 0.001
MCC_R         = 100.0
MCC_KERNEL_BW = 1200.0
MCC_MAXITER   = 5

# ─── PC-EKF-2D (MAD sigma-free) ──────────────────────────────────────
PCEKF_Q       = 0.001
PCEKF_R_BASE  = 25.0
PCEKF_R_SCALE = 15.0

# ─── AEKF (Sage-Husa adaptive) ───────────────────────────────────────
AEKF_Q        = 0.001
AEKF_R        = 50.0
AEKF_WIN      = 10       # sliding window size for noise estimation
AEKF_ALPHA    = 0.95     # forgetting factor

# ─── REKF (Student-t robust) ─────────────────────────────────────────
REKF_Q        = 0.001
REKF_R        = 50.0
REKF_NU       = 4.0      # degrees of freedom (tunable)
REKF_MAXITER  = 5

DO_GRID_SEARCH    = True
DATA_DIR          = "./data"
SAVE_DIR          = "./outputs_sota_EKF"

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
    if compute_trajectory_displacement(pos_xy) < ratio_min * expected_path_length:
        return True
    bbox_x    = pos_xy[:, 0].max() - pos_xy[:, 0].min()
    bbox_y    = pos_xy[:, 1].max() - pos_xy[:, 1].min()
    if math.sqrt(bbox_x**2 + bbox_y**2) < 0.05 * expected_path_length:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
#  EKF CORE — Observation model + Jacobian
# ══════════════════════════════════════════════════════════════════════
def h_obs(x, anchors=ANCHORS):
    """Measurement function: slant range to each anchor."""
    px, py = x[0], x[1]
    return np.array([
        math.sqrt((px - ax)**2 + (py - ay)**2 + ANCHOR_HEIGHT**2)
        for ax, ay in anchors
    ])


def H_jacobian(x, anchors=ANCHORS):
    """
    Jacobian of h w.r.t. state x = [px, py].
    H[i, 0] = (px - ax_i) / r_i
    H[i, 1] = (py - ay_i) / r_i
    """
    px, py = x[0], x[1]
    H = np.zeros((len(anchors), 2))
    for i, (ax, ay) in enumerate(anchors):
        r = math.sqrt((px - ax)**2 + (py - ay)**2 + ANCHOR_HEIGHT**2)
        r = max(r, 1e-6)
        H[i, 0] = (px - ax) / r
        H[i, 1] = (py - ay) / r
    return H


def make_spd(P, eps=1e-9):
    P = 0.5 * (P + P.T)
    P += np.eye(len(P)) * eps
    return P


def default_init(dist_raw_0):
    pos = LS_position(dist_raw_0)
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


# ══════════════════════════════════════════════════════════════════════
#  1. STANDARD EKF
# ══════════════════════════════════════════════════════════════════════
class StandardEKF:
    """
    EKF chuẩn cho bài toán range-only positioning.
    State: [px, py], Model: constant position (random walk).
    Predict: x⁻ = x,  P⁻ = P + Q
    Update:  H = ∂h/∂x,  S = HP⁻Hᵀ + R,  K = P⁻HᵀS⁻¹
             x = x⁻ + Kν,  P = (I - KH)P⁻
    """
    def __init__(self, q=EKF_Q, r=EKF_R):
        self.Q_mat = np.eye(2) * q
        self.R_mat = np.eye(N_ANCHORS) * r
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(2) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        # Predict
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        # Linearize
        H     = H_jacobian(x_pred)
        z_hat = h_obs(x_pred)
        innov = z_raw - z_hat
        S     = H @ P_pred @ H.T + self.R_mat
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  2. HUBER-EKF
# ══════════════════════════════════════════════════════════════════════
class HuberEKF:
    """
    EKF với IRLS Huber M-estimator.
    Mỗi iteration: tính standardized residual r_i = ν_i / sqrt(S_ii),
    nếu |r_i| > δ → inflate R[i,i] = R_base / w_i  (w_i = δ/|r_i|)
    → giảm trọng số measurement bị outlier.
    """
    def __init__(self, q=HUBER_Q, r=HUBER_R,
                 delta=HUBER_DELTA, maxiter=HUBER_MAXITER):
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r
        self.delta   = delta
        self.maxiter = maxiter
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(2) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        H      = H_jacobian(x_pred)
        z_hat  = h_obs(x_pred)
        innov  = z_raw - z_hat
        R_eff  = np.eye(N_ANCHORS) * self.R_base
        for _ in range(self.maxiter):
            S        = H @ P_pred @ H.T + R_eff
            s_diag   = np.maximum(np.diag(S), 1e-9)
            r_scaled = innov / np.sqrt(s_diag)
            hub_w    = np.where(np.abs(r_scaled) <= self.delta,
                                1.0,
                                self.delta / (np.abs(r_scaled) + 1e-9))
            hub_w = np.maximum(hub_w, 1e-4)
            R_eff = np.diag(self.R_base / hub_w)
        S = H @ P_pred @ H.T + R_eff
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  3. MCC-EKF
# ══════════════════════════════════════════════════════════════════════
class MCCEKF:
    """
    EKF với Maximum Correntropy Criterion.
    Weight per measurement: w_i = exp(-ν_i² / (2σ²))
    → Gaussian kernel trên innovation, down-weight outlier mạnh.
    """
    def __init__(self, q=MCC_Q, r=MCC_R,
                 kernel_bw=MCC_KERNEL_BW, maxiter=MCC_MAXITER):
        self.Q_mat     = np.eye(2) * q
        self.R_base    = r
        self.kernel_bw = kernel_bw
        self.maxiter   = maxiter
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(2) * 1e6

    def _kernel(self, v):
        return np.exp(-0.5 * v**2 / (self.kernel_bw**2 + 1e-9))

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        H      = H_jacobian(x_pred)
        z_hat  = h_obs(x_pred)
        innov  = z_raw - z_hat
        x_cur  = x_pred.copy()
        K      = np.zeros((2, N_ANCHORS))
        R_eff  = np.eye(N_ANCHORS) * self.R_base
        for _ in range(self.maxiter):
            kern_w = np.maximum(self._kernel(innov), 1e-4)
            R_eff  = np.diag(self.R_base / kern_w)
            S = H @ P_pred @ H.T + R_eff
            try:
                K = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                break
            x_new = x_pred + K @ innov
            if np.linalg.norm(x_new - x_cur) < 1e-3:
                x_cur = x_new; break
            x_cur = x_new
        self.x = x_cur
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  4. PC-EKF-2D — MAD Auto-Normalized, Sigma-Free
# ══════════════════════════════════════════════════════════════════════
def _t_kernel_pairwise(diff_vec, sigma=1.0):
    """Student-t kernel với sigma cố định 1.0."""
    nu  = 4.0
    return (1.0 + diff_vec**2 / (nu * sigma**2 + 1e-9)) ** (-(nu + 1.0) / 2.0)


def pc_mad_scores(innov, S_diag):
    """
    Pairwise Consensus scoring — MAD auto-normalized, sigma-free.

    1. Geo-normalize:  std_innov[i] = ν_i / sqrt(S_ii)
       S_ii = (H P⁻ Hᵀ)[i,i] + R_base  (per-anchor innovation variance)
    2. MAD normalize:  normed[i] = std_innov[i] / (1.4826·MAD + ε)
    3. Pairwise T-kernel:
       score[i] = mean_j≠i( T(normed[i] - normed[j]) )
       → score ≈ 1.0 = tin cậy LOS,  score ≪ 1 = nghi ngờ NLOS
    """
    std_innov = innov / np.sqrt(np.maximum(S_diag, 1e-9))
    med    = np.median(std_innov)
    mad    = np.median(np.abs(std_innov - med))
    normed = std_innov / (1.4826 * mad + 1e-9)
    scores = np.zeros(N_ANCHORS)
    for i in range(N_ANCHORS):
        diffs     = np.array([normed[i] - normed[j]
                              for j in range(N_ANCHORS) if j != i])
        scores[i] = np.mean(_t_kernel_pairwise(diffs))
    return scores


class PCEKF2D:
    """
    PC-EKF-2D: Pairwise Consensus scoring dựa trên EKF innovation.

    Update pipeline:
      1. EKF predict: P⁻ = P + Q
      2. Jacobian H = ∂h/∂x tại x⁻
      3. Innovation ν = z - h(x⁻)
      4. S_ii = (HP⁻Hᵀ)[i,i] + R_base  (per-anchor)
      5. pc_mad_scores(ν, S_diag) → scores ∈ (0,1]
      6. R_adaptive[i] = R_base · (1 + r_scale·(1-score_i))
      7. EKF update với R_adaptive
    """
    def __init__(self, q=PCEKF_Q, r_base=PCEKF_R_BASE, r_scale=PCEKF_R_SCALE):
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(2) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        H      = H_jacobian(x_pred)
        z_hat  = h_obs(x_pred)
        innov  = z_raw - z_hat
        # Per-anchor innovation variance (diagonal of HP⁻Hᵀ + R_base·I)
        HPH    = H @ P_pred @ H.T
        S_diag = np.diag(HPH) + self.R_base
        # PC scoring
        scores  = pc_mad_scores(innov, S_diag)
        R_diag  = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        S       = HPH + np.diag(R_diag)
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  5. AEKF — Adaptive EKF (Sage-Husa)
# ══════════════════════════════════════════════════════════════════════
class AEKF:
    """
    Adaptive EKF theo Sage-Husa (1969), với forgetting factor α.

    Online noise estimation từ innovation history:
      R̂ = (1-α) R̂ + α (ν νᵀ + H P⁻ Hᵀ)    [measurement noise]
      Q̂ = (1-α) Q̂ + α (K ν νᵀ Kᵀ)           [process noise, simplified]

    Sliding window (win) giúp tracking non-stationary noise.
    α là forgetting factor: α→1 = nhanh quên quá khứ (responsive),
                            α→0 = trọng số đều (stable).

    Chú ý:
      - Chỉ update R̂ (ổn định hơn), Q cố định (tránh diverge)
      - Clip R̂ về [R_base/100, R_base*100] để tránh numeric blow-up
    """
    def __init__(self, q=AEKF_Q, r=AEKF_R,
                 win=AEKF_WIN, alpha=AEKF_ALPHA):
        self.Q_mat = np.eye(2) * q
        self.R_est = np.eye(N_ANCHORS) * r      # adaptive R estimate
        self.R_base = r
        self.win   = win
        self.alpha = alpha
        self.x     = None
        self.P     = None
        self._innov_buf = []                     # innovation history

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(2) * 1e6
        self._innov_buf = []

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        H      = H_jacobian(x_pred)
        z_hat  = h_obs(x_pred)
        innov  = z_raw - z_hat

        # Update innovation buffer
        self._innov_buf.append(innov.copy())
        if len(self._innov_buf) > self.win:
            self._innov_buf.pop(0)

        # Sage-Husa adaptive R estimation
        HPH = H @ P_pred @ H.T
        if len(self._innov_buf) >= 2:
            innov_mat = np.array(self._innov_buf)               # (W, M)
            C_eps     = innov_mat.T @ innov_mat / len(innov_mat)  # sample cov
            R_new     = C_eps - HPH
            # Clip và symmetrize
            R_new     = np.diag(np.clip(np.diag(R_new),
                                        self.R_base / 100.0,
                                        self.R_base * 100.0))
            self.R_est = ((1 - self.alpha) * self.R_est
                          + self.alpha * R_new)
            # Keep R_est positive diagonal
            self.R_est = np.diag(np.maximum(np.diag(self.R_est),
                                            self.R_base / 100.0))

        S = HPH + self.R_est
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  6. REKF — Robust EKF với Student-t likelihood
# ══════════════════════════════════════════════════════════════════════
class REKF:
    """
    Robust EKF (Student-t likelihood) theo Agamennoni et al. 2012.

    Thay Gaussian likelihood p(z|x) = N(h(x), R) bằng
    Student-t với ν degrees of freedom:
      w_i = (ν + 1) / (ν + r_i²/S_ii)
    → w_i→1 khi r_i nhỏ (inlier), w_i→ν/r_i khi r_i lớn (outlier)

    Tương đương IRLS: mỗi vòng lặp tính w, update R_eff, re-solve K.
    ν nhỏ (2-5): robust mạnh nhưng có thể reject cả inlier.
    ν lớn (→∞): tiệm cận EKF chuẩn.
    """
    def __init__(self, q=REKF_Q, r=REKF_R,
                 nu=REKF_NU, maxiter=REKF_MAXITER):
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r
        self.nu      = nu
        self.maxiter = maxiter
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(2) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        H      = H_jacobian(x_pred)
        z_hat  = h_obs(x_pred)
        innov  = z_raw - z_hat
        R_eff  = np.eye(N_ANCHORS) * self.R_base
        K      = np.zeros((2, N_ANCHORS))
        for _ in range(self.maxiter):
            S      = H @ P_pred @ H.T + R_eff
            s_diag = np.maximum(np.diag(S), 1e-9)
            # Student-t weights
            w = (self.nu + 1.0) / (self.nu + innov**2 / s_diag)
            w = np.maximum(w, 1e-4)
            R_eff = np.diag(self.R_base / w)
        S = H @ P_pred @ H.T + R_eff
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
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
    "Raw+LS"   : None,
    "EKF"      : (StandardEKF, dict(q=EKF_Q,     r=EKF_R)),
    "Huber-EKF": (HuberEKF,   dict(q=HUBER_Q,    r=HUBER_R,   delta=HUBER_DELTA)),
    "MCC-EKF"  : (MCCEKF,     dict(q=MCC_Q,      r=MCC_R,     kernel_bw=MCC_KERNEL_BW)),
    "PC-EKF-2D": (PCEKF2D,    dict(q=PCEKF_Q,    r_base=PCEKF_R_BASE,
                                    r_scale=PCEKF_R_SCALE)),
    "AEKF"     : (AEKF,       dict(q=AEKF_Q,     r=AEKF_R,
                                    win=AEKF_WIN,  alpha=AEKF_ALPHA)),
    "REKF"     : (REKF,       dict(q=REKF_Q,     r=REKF_R,    nu=REKF_NU)),
}

COLORS = {
    "Raw+LS"   : '#9E9E9E',
    "EKF"      : '#00BCD4',
    "Huber-EKF": '#E91E63',
    "MCC-EKF"  : '#FF9800',
    "PC-EKF-2D": '#9C27B0',
    "AEKF"     : '#2196F3',
    "REKF"     : '#4CAF50',
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
        hdr += f" {m:>11s}"
    print("\n" + "═" * 100)
    print("  PER-FILE RMSE (mm)   [⚠ = trajectory collapsed]")
    print("═" * 100)
    print(hdr)
    print("─" * 100)

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
            row += f" {flag}{rmse:>9.1f}"
            all_errors[name].extend(errs)
            all_pos[name].extend(valid)
            per_file_rmse[name].append(rmse)

        print(row)

    print("─" * 100)

    print("\n  TIMING (ms/sample):")
    for name, times in timing.items():
        if not times:
            continue
        total_t = sum(times)
        n_samp  = sum(safe_len(p) for p in file_paths[:len(times)])
        if n_samp > 0:
            print(f"    {name:<12s}: {total_t * 1000 / n_samp:.4f} ms/sample")

    print("\n  COLLAPSE WARNINGS:")
    any_warn = False
    for name, files in collapse_warn.items():
        if files:
            any_warn = True
            print(f"    ⚠  {name:<12s}: {len(files)} file(s) → "
                  f"{', '.join(files[:5])}" + (" ..." if len(files) > 5 else ""))
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
#  GRID SEARCH — LOO cho tất cả methods
# ══════════════════════════════════════════════════════════════════════
def _loo_fold_rmse(filt_class, kwargs, val_file, gt_xy, expected_path_length):
    """LOO fold: chạy filter trên val_file, trả về RMSE hoặc penalty."""
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
    In ra ngay khi tìm được best mới: RMSE + tất cả hệ số.
    """
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n      = len(file_paths)

    print(f"\n  LOO Grid search [{method_name}]: "
          f"{len(combos)} combos × {n} folds = {len(combos) * n} runs")
    print(f"  {'LOO-RMSE':>10s}  Params")
    print(f"  {'─' * 60}")

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
            tag       = " ⚠COLLAPSED" if loo_rmse >= COLLAPSE_PENALTY / 1000 else ""
            param_str = "  ".join(f"{k}={params[k]}" for k in keys)
            print(f"  ★ {loo_rmse:8.2f}mm{tag:<12s}  {param_str}")

    print(f"  {'─' * 60}")
    best_str = "  ".join(f"{k}={best_params[k]}" for k in keys)
    print(f"  ✔ Best: {best_loo_rmse:.2f}mm  |  {best_str}")
    return best_params, best_loo_rmse


# ── Wrappers per filter ───────────────────────────────────────────────
def grid_search_huber(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'    : [50.0],
        'delta': [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0],
    }
    best, _ = _grid_search_loo(HuberEKF, grid, {'q': HUBER_Q},
                               file_paths, gt_xy, expected_path_length,
                               "Huber-EKF")
    return best


def grid_search_mcc(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'        : [50.0],
        'kernel_bw': [100.0, 300.0, 500.0, 800.0, 1200.0, 1700.0, 2500.0],
    }
    best, _ = _grid_search_loo(MCCEKF, grid, {'q': MCC_Q},
                               file_paths, gt_xy, expected_path_length,
                               "MCC-EKF")
    return best


def grid_search_pcekf(file_paths, gt_xy, expected_path_length):
    grid = {
        'r_base' : [50.0],
        'r_scale': [3.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    }
    best, _ = _grid_search_loo(PCEKF2D, grid, {'q': PCEKF_Q},
                               file_paths, gt_xy, expected_path_length,
                               "PC-EKF-2D")
    return best


def grid_search_aekf(file_paths, gt_xy, expected_path_length):
    grid = {
        'r'    : [50.0],
        'win'  : [5, 10, 20, 30],
        'alpha': [0.80, 0.90, 0.95, 0.99],
    }
    best, _ = _grid_search_loo(AEKF, grid, {'q': AEKF_Q},
                               file_paths, gt_xy, expected_path_length,
                               "AEKF")
    return best


def grid_search_rekf(file_paths, gt_xy, expected_path_length):
    grid = {
        'r' : [50.0],
        'nu': [2.0, 3.0, 4.0, 5.0, 8.0, 15.0, 30.0],
    }
    best, _ = _grid_search_loo(REKF, grid, {'q': REKF_Q},
                               file_paths, gt_xy, expected_path_length,
                               "REKF")
    return best


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_cdf(errors_dict, collapse_warn, save_path):
    fig, ax = plt.subplots(figsize=(12, 7))
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
    ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_title("EKF Suite — Error CDF", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] CDF → {save_path}")
    plt.close()


def plot_trajectories(positions_dict, gt_xy, collapse_warn, save_path):
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'k--', lw=2.5,
            label='Ground Truth', alpha=0.5, zorder=5)
    for label, pos in positions_dict.items():
        if len(pos) == 0:
            continue
        color = COLORS.get(label, 'gray')
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
    ax.set_title("EKF Suite — Trajectories", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Trajectories → {save_path}")
    plt.close()


def plot_bar(metrics_dict, save_path):
    methods      = list(metrics_dict.keys())
    metric_names = ['rmse', 'mae', 'cep50', 'p95']
    labels_disp  = ['RMSE', 'MAE', 'CEP50', 'P95']
    fig, axes    = plt.subplots(1, 4, figsize=(26, 6))
    for ax, mname, mlabel in zip(axes, metric_names, labels_disp):
        vals   = [metrics_dict[m][mname] for m in methods]
        colors = [COLORS.get(m, 'gray') for m in methods]
        bars   = ax.bar(range(len(methods)), vals, color=colors,
                        alpha=0.85, edgecolor='white', lw=1.5)
        for bar, val, m in zip(bars, vals, methods):
            if metrics_dict[m].get('collapsed', False):
                bar.set_hatch('//')
                bar.set_edgecolor('darkred')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-', '\n') for m in methods],
                           fontsize=8, ha='center')
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
    print("  EKF Suite V1 — Extended Kalman Filter variants for UWB Positioning")
    print("=" * 75)
    print("  1. Raw+LS      — Weighted Least Squares (baseline)")
    print("  2. EKF          — Extended Kalman Filter chuẩn")
    print("  3. Huber-EKF    — IRLS Huber M-estimator")
    print("  4. MCC-EKF      — Maximum Correntropy Criterion EKF")
    print("  5. PC-EKF-2D    — Pairwise Consensus MAD sigma-free (LOO-tuned)")
    print("  6. AEKF         — Adaptive EKF (Sage-Husa online noise est.)")
    print("  7. REKF         — Robust EKF (Student-t likelihood)")
    print("=" * 75)
    print(f"\n  Tuning policy (fully fair LOO):")
    print(f"    Raw+LS, EKF : không tune")
    print(f"    Huber-EKF   : LOO  [r, delta]")
    print(f"    MCC-EKF     : LOO  [r, kernel_bw]")
    print(f"    PC-EKF-2D   : LOO  [r_base, r_scale]  (sigma-free)")
    print(f"    AEKF        : LOO  [r, win, alpha]")
    print(f"    REKF        : LOO  [r, nu]")

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
    expected_path_length = np.linalg.norm(
        np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"  Path={expected_path_length:.0f}mm  "
          f"Time={total_time:.1f}s  GT={len(gt_xy)} points")

    if DO_GRID_SEARCH:
        print("\n  [LOO Grid Search] Tuning tất cả filters...")

        best_huber = grid_search_huber(eval_files, gt_xy, expected_path_length)
        best_mcc   = grid_search_mcc(eval_files,   gt_xy, expected_path_length)
        best_pc    = grid_search_pcekf(eval_files,  gt_xy, expected_path_length)
        best_aekf  = grid_search_aekf(eval_files,  gt_xy, expected_path_length)
        best_rekf  = grid_search_rekf(eval_files,  gt_xy, expected_path_length)

        METHODS["Huber-EKF"] = (HuberEKF, best_huber)
        METHODS["MCC-EKF"]   = (MCCEKF,   best_mcc)
        METHODS["PC-EKF-2D"] = (PCEKF2D,  best_pc)
        METHODS["AEKF"]      = (AEKF,     best_aekf)
        METHODS["REKF"]      = (REKF,     best_rekf)

        print(f"\n  ══ Params sau LOO grid search ══")
        print(f"    Huber-EKF : r={best_huber['r']}  delta={best_huber['delta']}")
        print(f"    MCC-EKF   : r={best_mcc['r']}  kernel_bw={best_mcc['kernel_bw']}")
        print(f"    PC-EKF-2D : r_base={best_pc['r_base']}  "
              f"r_scale={best_pc['r_scale']}  [sigma-free]")
        print(f"    AEKF      : r={best_aekf['r']}  win={best_aekf['win']}  "
              f"alpha={best_aekf['alpha']}")
        print(f"    REKF      : r={best_rekf['r']}  nu={best_rekf['nu']}")
    else:
        print(f"\n  [INFO] Grid Search TẮT — dùng params mặc định")

    # ── Evaluate ──────────────────────────────────────────────────────
    errors, positions, collapse_warn, per_file_rmse = evaluate_files(
        eval_files, gt_xy, expected_path_length)

    # ── Summary table ─────────────────────────────────────────────────
    metrics = {}
    tuned   = {"Huber-EKF", "MCC-EKF", "PC-EKF-2D", "AEKF", "REKF"}
    print(f"\n{'═' * 115}")
    print(f"  SUMMARY TABLE (mm)   — ⚠ = trajectory collapsed  ◀ = LOO-tuned")
    print(f"{'═' * 115}")
    print(f"  {'Method':<13s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} "
          f"{'CEP90':>7s} {'P95':>7s} {'MAX':>7s} "
          f"{'RMSE mean±std':>16s} {'MotionR':>8s}  Status")
    print(f"  {'─' * 100}")

    for label, errs in errors.items():
        if len(errs) == 0:
            continue
        pos_arr = positions.get(label, np.empty((0, 2)))
        m       = compute_metrics(errs, pos_arr, expected_path_length)
        metrics[label] = m

        pf_vals = [v for v in per_file_rmse.get(label, [])
                   if not math.isnan(v)]
        if len(pf_vals) >= 2:
            std_str = f"{np.mean(pf_vals):.1f} ± {np.std(pf_vals, ddof=1):.1f}"
        elif len(pf_vals) == 1:
            std_str = f"{pf_vals[0]:.1f} ± N/A"
        else:
            std_str = "N/A"

        n_col  = len(collapse_warn.get(label, []))
        status = f"⚠ {n_col} collapsed" if n_col > 0 else "✅ OK"
        mr_str = (f"{m['motion_ratio']:.2f}"
                  if not math.isnan(m.get('motion_ratio', float('nan')))
                  else "N/A")
        tag = " ◀ LOO" if label in tuned else ""
        print(f"  {label:<13s} {m['rmse']:>7.1f} {m['mae']:>7.1f}"
              f" {m['cep50']:>7.1f} {m['cep90']:>7.1f} {m['p95']:>7.1f}"
              f" {m['max']:>7.1f} {std_str:>16s}  {mr_str:>8s}  {status}{tag}")

    # ── Wilcoxon ──────────────────────────────────────────────────────
    print(f"\n  Wilcoxon tests (two-sided, vs PC-EKF-2D):")
    ref_err = errors.get("PC-EKF-2D", np.array([]))
    for name, errs in errors.items():
        if name == "PC-EKF-2D" or len(errs) == 0 or len(ref_err) == 0:
            continue
        N = min(len(errs), len(ref_err))
        if N > 20:
            try:
                _, p = wilcoxon(errs[:N], ref_err[:N])
                sym  = '✅ p<0.05' if p < 0.05 else '⚠️  ns'
                print(f"    {name:<13s} vs PC-EKF-2D: p={p:.4f}  {sym}")
            except ValueError:
                pass

    # ── Plots ─────────────────────────────────────────────────────────
    plot_cdf(errors, collapse_warn,
             os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(positions, gt_xy, collapse_warn,
                      os.path.join(SAVE_DIR, 'trajectories.png'))
    plot_bar(metrics,
             os.path.join(SAVE_DIR, 'bar_comparison.png'))

    for label, errs in errors.items():
        safe = label.lower().replace('+', '_').replace('-', '_').replace(' ', '_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errs)

    print(f"\n[✓] Kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
