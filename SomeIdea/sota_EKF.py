# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — EKF Suite V2  (2-Phase LOO, giống sota.py)
====================================================================
CÁC PHƯƠNG PHÁP:
  1. Raw+LS        — baseline Weighted Least Squares
  2. EKF            — Extended Kalman Filter chuẩn
  3. Huber-EKF      — IRLS Huber M-estimator trên innovation
  4. MCC-EKF        — Maximum Correntropy Criterion EKF
  5. PC-EKF-2D      — Pairwise Consensus MAD-normalized + EKF (sigma-free)
  6. AEKF           — Adaptive EKF (Sage-Husa noise estimator)
  7. REKF           — Robust EKF với Student-t likelihood

WORKFLOW (2 Phase — giống sota.py):
  Raw+LS, EKF   : không tune (không có params)
  Các filter còn lại:
  [Phase 1] LOO Tuning trên cả N files:
            với mỗi combo params: LOO-RMSE = trung bình RMSE qua N folds
            chọn best params có LOO-RMSE thấp nhất
  [Phase 2] Evaluate lại cả N files với best params vừa tìm
            kết quả cuối cùng phản ánh performance thực

CÔNG BẰNG TUNING:
  - Raw+LS, EKF: không có param cần tune → không cần LOO
  - Huber-EKF, MCC-EKF, PC-EKF-2D, AEKF, REKF: đều dùng _grid_search_loo()
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

# ─── Standard EKF (không tune) ───────────────────────────────────────
EKF_Q = 0.001
EKF_R = 50.0

# ─── Huber-EKF ───────────────────────────────────────────────────────
HUBER_Q       = 0.001
HUBER_R       = 50.0
HUBER_DELTA   = 1.5
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

# ─── AEKF ────────────────────────────────────────────────────────────
AEKF_Q        = 0.001
AEKF_R        = 50.0
AEKF_WIN      = 10
AEKF_ALPHA    = 0.95

# ─── REKF ────────────────────────────────────────────────────────────
REKF_Q        = 0.001
REKF_R        = 50.0
REKF_NU       = 4.0
REKF_MAXITER  = 5

DO_GRID_SEARCH = True
DATA_DIR       = "./data"
SAVE_DIR       = "./outputs_sota_EKF"

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
#  EKF CORE — Observation model + Jacobian
# ══════════════════════════════════════════════════════════════════════
def h_obs(x, anchors=ANCHORS):
    px, py = x[0], x[1]
    return np.array([
        math.sqrt((px - ax)**2 + (py - ay)**2 + ANCHOR_HEIGHT**2)
        for ax, ay in anchors
    ])


def H_jacobian(x, anchors=ANCHORS):
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
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        H      = H_jacobian(x_pred)
        innov  = z_raw - h_obs(x_pred)
        S      = H @ P_pred @ H.T + self.R_mat
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  2. HUBER-EKF
# ══════════════════════════════════════════════════════════════════════
class HuberEKF:
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
        innov  = z_raw - h_obs(x_pred)
        R_eff  = np.eye(N_ANCHORS) * self.R_base
        for _ in range(self.maxiter):
            S        = H @ P_pred @ H.T + R_eff
            s_diag   = np.maximum(np.diag(S), 1e-9)
            r_scaled = innov / np.sqrt(s_diag)
            hub_w    = np.where(
                np.abs(r_scaled) <= self.delta,
                1.0,
                self.delta / (np.abs(r_scaled) + 1e-9)
            )
            hub_w = np.maximum(hub_w, 1e-4)
            R_eff = np.diag(self.R_base / hub_w)
        S = H @ P_pred @ H.T + R_eff
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  3. MCC-EKF
# ══════════════════════════════════════════════════════════════════════
class MCCEKF:
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
        innov  = z_raw - h_obs(x_pred)
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
                x_cur = x_new
                break
            x_cur = x_new
        self.x = x_cur
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  4. PC-EKF-2D — MAD Auto-Normalized, Sigma-Free
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


class PCEKF2D:
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
        innov  = z_raw - h_obs(x_pred)
        HPH    = H @ P_pred @ H.T
        S_diag = np.diag(HPH) + self.R_base
        scores = pc_mad_scores(innov, S_diag)
        R_diag = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        S      = HPH + np.diag(R_diag)
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  5. AEKF — Sage-Husa Adaptive EKF
#
#  Sage-Husa noise estimator (Sage & Husa 1969):
#    C_eps = sample covariance của innovation window (trừ mean)
#    R_new = diag( C_eps - H*P_pred*H^T )   (clipped để dương)
#    R_est ← alpha * R_est + (1-alpha) * R_new
#    alpha ∈ (0,1): forgetting factor — lớn = nhớ lâu (ổn định hơn)
#
#  Fix so với phiên bản cũ:
#    1. R_est được reset trong init() — tránh state leak giữa các file
#    2. Sample covariance tính đúng (zero-mean trong window)
#    3. alpha là forgetting factor: alpha*old + (1-alpha)*new
#       (cũ bị ngược: (1-alpha)*old + alpha*new → alpha lớn = không ổn định)
# ══════════════════════════════════════════════════════════════════════
class AEKF:
    def __init__(self, q=AEKF_Q, r=AEKF_R, win=AEKF_WIN, alpha=AEKF_ALPHA):
        self.Q_mat  = np.eye(2) * q
        self.R_base = r
        self.win    = win
        self.alpha  = alpha   # forgetting factor: lớn = nhớ lâu
        self.x = None
        self.P = None
        self.R_est      = None
        self._innov_buf = []

    def init(self, x0):
        self.x          = x0.astype(float).copy()
        self.P          = np.eye(2) * 1e6
        self.R_est      = np.eye(N_ANCHORS) * self.R_base  # reset mỗi file
        self._innov_buf = []

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        H      = H_jacobian(x_pred)
        innov  = z_raw - h_obs(x_pred)

        # Cập nhật buffer innovation
        self._innov_buf.append(innov.copy())
        if len(self._innov_buf) > self.win:
            self._innov_buf.pop(0)

        HPH = H @ P_pred @ H.T

        # Sage-Husa: ước lượng R từ sample covariance (zero-mean) của window
        if len(self._innov_buf) >= 2:
            innov_mat = np.array(self._innov_buf)          # (L, M)
            mean_inn  = innov_mat.mean(axis=0)             # (M,)
            centered  = innov_mat - mean_inn               # (L, M)
            C_eps     = (centered.T @ centered) / (len(innov_mat) - 1)  # (M, M)
            # R_new = C_eps - HPH, clip để giữ dương định
            R_new_diag = np.clip(
                np.diag(C_eps - HPH),
                self.R_base / 100.0,
                self.R_base * 100.0,
            )
            R_new = np.diag(R_new_diag)
            # alpha là forgetting factor: lớn → tin vào R_est cũ hơn
            self.R_est = self.alpha * self.R_est + (1.0 - self.alpha) * R_new
            # Đảm bảo dương
            self.R_est = np.diag(np.maximum(
                np.diag(self.R_est), self.R_base / 100.0))

        S = HPH + self.R_est
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  6. REKF — Robust EKF với Student-t likelihood (Agamennoni et al. 2012)
#
#  IRLS (Iteratively Reweighted Least Squares):
#    weight_i = (nu + 1) / (nu + e_i^2 / r_i)
#    với e_i = innov[i], r_i = R_base (scalar noise variance mỗi anchor)
#    R_eff = diag(R_base / w_i)
#
#  Fix so với phiên bản cũ:
#    1. Normalize innovation bằng R_base (scalar, không phải diag(S))
#       S = HPH^T + R phụ thuộc vào P → không phải đặc trưng của noise
#       Student-t weight phải dùng R_base thuần (Agamennoni 2012 eq.17)
#    2. Track x convergence giữa các vòng IRLS (như MCC-EKF)
#       → dừng sớm khi hội tụ, tránh over-iteration
# ══════════════════════════════════════════════════════════════════════
class REKF:
    def __init__(self, q=REKF_Q, r=REKF_R, nu=REKF_NU, maxiter=REKF_MAXITER):
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
        innov  = z_raw - h_obs(x_pred)

        # IRLS: lặp cập nhật weight và state estimate
        x_cur = x_pred.copy()
        K     = np.zeros((2, N_ANCHORS))
        R_eff = np.eye(N_ANCHORS) * self.R_base
        for _ in range(self.maxiter):
            S = H @ P_pred @ H.T + R_eff
            try:
                K = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                break
            x_new = x_pred + K @ innov

            # Student-t weight: normalize bằng R_base (scalar noise var per anchor)
            # Agamennoni 2012 eq.17: w_i = (nu+1) / (nu + e_i^2 / r_i)
            r_diag = np.maximum(np.diag(R_eff), 1e-9)
            w = (self.nu + 1.0) / (self.nu + innov**2 / r_diag)
            w = np.maximum(w, 1e-4)
            R_eff = np.diag(self.R_base / w)

            # Convergence check
            if np.linalg.norm(x_new - x_cur) < 1e-3:
                x_cur = x_new
                break
            x_cur = x_new

        self.x = x_cur
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
    "Raw+LS"    : None,
    "EKF"       : (StandardEKF, dict(q=EKF_Q,     r=EKF_R)),
    "Huber-EKF" : (HuberEKF,   dict(q=HUBER_Q,    r=HUBER_R,    delta=HUBER_DELTA)),
    "MCC-EKF"   : (MCCEKF,     dict(q=MCC_Q,      r=MCC_R,      kernel_bw=MCC_KERNEL_BW)),
    "PC-EKF-2D" : (PCEKF2D,    dict(q=PCEKF_Q,    r_base=PCEKF_R_BASE,
                                     r_scale=PCEKF_R_SCALE)),
    "AEKF"      : (AEKF,       dict(q=AEKF_Q,     r=AEKF_R,
                                     win=AEKF_WIN,  alpha=AEKF_ALPHA)),
    "REKF"      : (REKF,       dict(q=REKF_Q,     r=REKF_R,     nu=REKF_NU)),
}

COLORS = {
    "Raw+LS"    : '#9E9E9E',
    "EKF"       : '#00BCD4',
    "Huber-EKF" : '#E91E63',
    "MCC-EKF"   : '#FF9800',
    "PC-EKF-2D" : '#9C27B0',
    "AEKF"      : '#2196F3',
    "REKF"      : '#4CAF50',
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
    print("\n" + "═" * 110)
    print("  PER-FILE RMSE (mm)   [⚠ = trajectory collapsed]")
    print("═" * 110)
    print(hdr)
    print("─" * 110)

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

    print("─" * 110)

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
#  LOO PIPELINE
#
#  outer_loo_one_method — nested LOO chuẩn:
#    Outer (N folds):
#      for k = 0..N-1:
#        train = all_files \ {file[k]}         (N-1 file)
#        test  = file[k]                        (1 file)
#        best_params_k = inner_loo(train, grid) → tune KHÔNG nhìn thấy test
#        fold_rmse[k]  = RMSE(test, best_params_k)
#    outer_loo_rmse = mean(fold_rmse)           → unbiased estimate
#
#    Retrain inner LOO trên toàn bộ N files → best_global_params
#    Final eval với best_global_params         → dùng cho plot / summary
# ══════════════════════════════════════════════════════════════════════

# ── Grids cho từng method ─────────────────────────────────────────────
GRIDS = {
    "Huber-EKF": {
        "filt_class"  : HuberEKF,
        "fixed_params": {"q": HUBER_Q},
        "grid"        : {
            "r"    : [50.0],
            "delta": [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0],
        },
    },
    "MCC-EKF": {
        "filt_class"  : MCCEKF,
        "fixed_params": {"q": MCC_Q},
        "grid"        : {
            "r"        : [50.0],
            "kernel_bw": [100.0, 300.0, 500.0, 800.0, 1200.0, 1700.0, 2500.0],
        },
    },
    "PC-EKF-2D": {
        "filt_class"  : PCEKF2D,
        "fixed_params": {"q": PCEKF_Q},
        "grid"        : {
            "r_base" : [50.0],
            "r_scale": [3.0, 5.0, 10.0, 15.0, 20.0, 30.0],
        },
    },
    "AEKF": {
        "filt_class"  : AEKF,
        "fixed_params": {"q": AEKF_Q},
        "grid"        : {
            "r"    : [50.0],
            "win"  : [5, 10, 20, 30],
            "alpha": [0.80, 0.90, 0.95, 0.99],
        },
    },
    "REKF": {
        "filt_class"  : REKF,
        "fixed_params": {"q": REKF_Q},
        "grid"        : {
            "r" : [50.0],
            "nu": [2.0, 3.0, 4.0, 5.0, 8.0, 15.0, 30.0],
        },
    },
}


def _loo_fold_rmse(filt_class, kwargs, val_file, gt_xy, expected_path_length):
    """Chạy filter trên 1 file, trả về RMSE. Collapse → penalty."""
    d = parse_file(val_file)
    if d is None:
        return None
    if filt_class is None:
        p = np.array([LS_position(d[t]) for t in range(len(d))])
    else:
        p = run_filter(filt_class, d, **kwargs)
    valid = p[~np.any(np.isnan(p), axis=1)]
    if is_trajectory_collapsed(valid, expected_path_length):
        return 500.0 + COLLAPSE_PENALTY / 1000.0
    if len(valid) == 0:
        return 500.0
    errs = nearest_gt_error(valid, gt_xy)
    return float(np.sqrt(np.mean(errs**2)))


def _grid_search_loo(filt_class, grid, fixed_params,
                     tune_files, gt_xy, expected_path_length,
                     method_name, verbose=True):
    """
    Inner LOO: duyệt grid, với mỗi combo tính LOO-RMSE trên tune_files.
    verbose=False → im lặng (dùng khi gọi từ outer LOO).
    Trả về (best_params, best_loo_rmse).
    """
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n      = len(tune_files)

    if verbose:
        print(f"\n    Inner LOO [{method_name}]: "
              f"{len(combos)} combos × {n} folds = {len(combos) * n} runs")
        print(f"    {'LOO-RMSE':>10s}  Params")
        print(f"    {'─' * 53}")

    best_loo_rmse = float('inf')
    best_params   = {**fixed_params}

    for combo in combos:
        params = {**fixed_params, **dict(zip(keys, combo))}
        fold_rmses = []
        for val_file in tune_files:
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
            if verbose:
                tag       = " ⚠COLLAPSED" if loo_rmse >= COLLAPSE_PENALTY / 1000 else ""
                param_str = "  ".join(f"{k}={params[k]}" for k in keys)
                print(f"    ★ {loo_rmse:8.2f}mm{tag:<12s}  {param_str}")

    if verbose:
        print(f"    {'─' * 53}")
        best_param_str = "  ".join(f"{k}={best_params[k]}" for k in keys)
        print(f"    ✔ Best: {best_loo_rmse:.2f}mm  |  {best_param_str}")
    return best_params, best_loo_rmse


def outer_loo_one_method(method_name, filt_class_or_none,
                          all_files, gt_xy, expected_path_length):
    """
    Outer LOO chuẩn cho 1 method:

      for k = 0..N-1:
          train = all_files \ {file[k]}           (N-1 file)
          best_params_k = inner_loo(train, grid)   → không nhìn thấy file[k]
          fold_rmse[k]  = RMSE(file[k], best_params_k)

      outer_loo_rmse = mean(fold_rmse)             → unbiased estimate

      Retrain: inner_loo(all N files) → best_global_params

    Returns: (fold_rmses, best_global_params, loo_mean, loo_std)
    """
    N         = len(all_files)
    has_grid  = method_name in GRIDS
    fold_rmses  = []
    fold_params = []

    print(f"\n  ── Outer LOO: {method_name}  (N={N} folds) ──")

    for k in range(N):
        test_file   = all_files[k]
        train_files = [f for i, f in enumerate(all_files) if i != k]
        fname       = os.path.basename(test_file)

        # ── Inner LOO trên N-1 files (im lặng) ────────────────────────
        if has_grid:
            cfg = GRIDS[method_name]
            best_k, _ = _grid_search_loo(
                cfg["filt_class"], cfg["grid"], cfg["fixed_params"],
                train_files, gt_xy, expected_path_length,
                method_name, verbose=False)
        elif method_name == "Raw+LS":
            best_k = {}
        else:
            _, default_kw = METHODS[method_name]
            best_k = default_kw.copy()

        # ── Test trên fold k ───────────────────────────────────────────
        fr = _loo_fold_rmse(filt_class_or_none, best_k, test_file,
                            gt_xy, expected_path_length)
        fr = fr if fr is not None else float('nan')
        fold_rmses.append(fr)
        fold_params.append(best_k.copy())

        col_flag  = " [COLLAPSE]" if fr >= COLLAPSE_PENALTY / 1000 else ""
        if has_grid:
            tune_keys = list(GRIDS[method_name]["grid"].keys())
            param_str = "  params: " + " ".join(
                f"{k2}={best_k[k2]}" for k2 in tune_keys)
        else:
            param_str = ""
        print(f"    Fold {k+1:2d}/{N}  test={fname:<22s}"
              f"  RMSE={fr:7.1f}mm{col_flag}{param_str}")

    # ── Summary ───────────────────────────────────────────────────────
    valid_folds = [r for r in fold_rmses
                   if not math.isnan(r) and r < COLLAPSE_PENALTY / 1000]
    loo_mean = float(np.mean(valid_folds))   if valid_folds else float('nan')
    loo_std  = (float(np.std(valid_folds, ddof=1))
                if len(valid_folds) > 1 else float('nan'))
    n_col    = sum(1 for r in fold_rmses if r >= COLLAPSE_PENALTY / 1000)
    print(f"    → LOO RMSE = {loo_mean:.1f} ± {loo_std:.1f} mm"
          f"  ({n_col}/{N} collapsed)")

    # ── Retrain inner LOO trên toàn bộ N file → best_global_params ────
    if has_grid:
        cfg = GRIDS[method_name]
        print(f"    Retrain inner LOO trên toàn bộ {N} file...")
        best_global, _ = _grid_search_loo(
            cfg["filt_class"], cfg["grid"], cfg["fixed_params"],
            all_files, gt_xy, expected_path_length,
            method_name, verbose=True)
    elif method_name == "Raw+LS":
        best_global = {}
    else:
        _, default_kw = METHODS[method_name]
        best_global = default_kw.copy()

    return fold_rmses, best_global, loo_mean, loo_std


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
    fig, axes    = plt.subplots(1, 4, figsize=(26, 6))
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


def main():
    print("=" * 80)
    print("  EKF Suite — Outer LOO Cross-Validation Chuẩn")
    print("=" * 80)
    print("  1. Raw+LS      — Weighted Least Squares (baseline)")
    print("  2. EKF          — Extended Kalman Filter chuan")
    print("  3. Huber-EKF    — IRLS voi Huber M-estimator")
    print("  4. MCC-EKF      — Maximum Correntropy Criterion EKF")
    print("  5. PC-EKF-2D    — MAD sigma-free consensus (LOO-tuned)")
    print("  6. AEKF         — Adaptive EKF (Sage-Husa, LOO-tuned)")
    print("  7. REKF         — Robust EKF Student-t (LOO-tuned)")
    print("=" * 80)
    print("""
  Outer LOO Pipeline:
    for k = 0..N-1:
        train = all_files \ {file[k]}           (N-1 file)
        test  = file[k]                         (1 file)
        best_params_k = inner_loo(train, grid)  → tune tren N-1 file
        fold_rmse[k]  = RMSE(test, best_params_k)

    outer_loo_rmse = mean(fold_rmse)            → unbiased performance estimate
    best_global    = inner_loo(all_files)       → retrain de production eval
""")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"\n[!] Khong tim thay file .txt trong '{DATA_DIR}/'")
        return
    print(f"  Tim thay {n} file(s) trong '{DATA_DIR}/'")

    np.random.seed(42)
    all_files = [all_files[i] for i in np.random.permutation(n)]

    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    expected_path_length = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"  Path={expected_path_length:.0f}mm  Time={total_time:.1f}s  GT={len(gt_xy)} points")
    print(f"  DO_GRID_SEARCH = {DO_GRID_SEARCH}")

    # ══════════════════════════════════════════════════════════════════
    #  OUTER LOO cho tất cả methods
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  OUTER LOO CROSS-VALIDATION")
    print("═" * 80)

    loo_fold_rmses  = {}   # method → list[float]
    best_global_all = {}   # method → dict
    loo_mean_all    = {}   # method → float
    loo_std_all     = {}   # method → float

    FILT_CLASSES = {
        "Raw+LS"    : None,
        "EKF"       : StandardEKF,
        "Huber-EKF" : HuberEKF,
        "MCC-EKF"   : MCCEKF,
        "PC-EKF-2D" : PCEKF2D,
        "AEKF"      : AEKF,
        "REKF"      : REKF,
    }

    for method_name in METHODS:
        filt_cls = FILT_CLASSES[method_name]
        if DO_GRID_SEARCH or method_name not in GRIDS:
            fold_rmses, best_global, loo_mean, loo_std = outer_loo_one_method(
                method_name, filt_cls, all_files, gt_xy, expected_path_length)
        else:
            # Grid search tắt: dùng default params
            _, default_kw = METHODS[method_name] if METHODS[method_name] else (None, {})
            fold_rmses = []
            for k in range(n):
                fr = _loo_fold_rmse(filt_cls,
                                    default_kw if default_kw else {},
                                    all_files[k], gt_xy, expected_path_length)
                fold_rmses.append(fr if fr is not None else float('nan'))
            valid_f  = [r for r in fold_rmses if not math.isnan(r) and r < COLLAPSE_PENALTY / 1000]
            loo_mean = float(np.mean(valid_f)) if valid_f else float('nan')
            loo_std  = float(np.std(valid_f, ddof=1)) if len(valid_f) > 1 else float('nan')
            best_global = default_kw if default_kw else {}

        loo_fold_rmses[method_name]  = fold_rmses
        best_global_all[method_name] = best_global
        loo_mean_all[method_name]    = loo_mean
        loo_std_all[method_name]     = loo_std

    # ══════════════════════════════════════════════════════════════════
    #  FINAL EVALUATION với best_global_params
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  FINAL EVALUATION (best params retrained tren toan bo N file)")
    print("═" * 80)

    for method_name, best_global in best_global_all.items():
        if method_name == "Raw+LS":
            continue
        filt_cls = FILT_CLASSES[method_name]
        METHODS[method_name] = (filt_cls, best_global)

    errors, positions, collapse_warn, per_file_rmse = evaluate_files(
        all_files, gt_xy, expected_path_length)

    # ── Summary table ─────────────────────────────────────────────────
    TUNED = set(GRIDS.keys())
    print(f"\n{'═' * 135}")
    print("  SUMMARY TABLE (mm)")
    print(f"  LOO RMSE = unbiased estimate tu outer LOO (N folds)")
    print(f"  Full RMSE = evaluation tren toan bo data voi best_global_params")
    print(f"{'═' * 135}")
    print(f"  {'Method':<13s} {'LOO RMSE':>10s} {'±std':>8s} "
          f"{'Full RMSE':>10s} {'MAE':>7s} {'CEP50':>7s} {'CEP90':>7s} "
          f"{'P95':>7s} {'MAX':>7s} {'MotionR':>8s}  Status  BestParams")
    print(f"  {'─' * 125}")

    metrics = {}
    for method_name in METHODS:
        errs    = errors.get(method_name, np.array([]))
        pos_arr = positions.get(method_name, np.empty((0, 2)))
        m       = compute_metrics(errs, pos_arr, expected_path_length)
        metrics[method_name] = m

        loo_m    = loo_mean_all.get(method_name, float('nan'))
        loo_s    = loo_std_all.get(method_name, float('nan'))
        fold_r   = loo_fold_rmses.get(method_name, [])
        n_col_loo= sum(1 for r in fold_r if r >= COLLAPSE_PENALTY / 1000)
        n_col    = len(collapse_warn.get(method_name, []))
        status   = f"⚠ {n_col} col" if n_col > 0 else "✅ OK"
        tag      = " ◀" if method_name in TUNED else "  "
        mr_str   = (f"{m['motion_ratio']:.2f}"
                    if not math.isnan(m.get('motion_ratio', float('nan'))) else "N/A")

        bp = best_global_all.get(method_name, {})
        if method_name in GRIDS:
            tune_keys = list(GRIDS[method_name]["grid"].keys())
            bp_str = " ".join(f"{k}={bp.get(k, '?')}" for k in tune_keys)
        else:
            bp_str = "(fixed)"

        loo_str = f"{loo_m:7.1f}" if not math.isnan(loo_m) else "    N/A"
        std_str = f"{loo_s:6.1f}" if not math.isnan(loo_s) else "   N/A"
        if n_col_loo > 0:
            loo_str += f"[⚠{n_col_loo}]"

        print(f"  {method_name:<13s}{tag} {loo_str:>10s} {std_str:>8s}"
              f" {m['rmse']:>10.1f} {m['mae']:>7.1f}"
              f" {m['cep50']:>7.1f} {m['cep90']:>7.1f}"
              f" {m['p95']:>7.1f} {m['max']:>7.1f}"
              f" {mr_str:>8s}  {status:<10s}  {bp_str}")

    # ── Wilcoxon ──────────────────────────────────────────────────────
    print(f"\n  Wilcoxon tests (two-sided, vs PC-EKF-2D):")
    ref_err = errors.get("PC-EKF-2D", np.array([]))
    for name, errs in errors.items():
        if name == "PC-EKF-2D" or len(errs) == 0 or len(ref_err) == 0:
            continue
        N2 = min(len(errs), len(ref_err))
        if N2 > 20:
            try:
                _, p = wilcoxon(errs[:N2], ref_err[:N2])
                sym  = '✅ p<0.05' if p < 0.05 else '⚠️  ns'
                print(f"    {name:<14s} vs PC-EKF-2D: p={p:.4f}  {sym}")
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
        np.save(os.path.join(SAVE_DIR, f'loo_folds_{safe}.npy'),
                np.array(loo_fold_rmses.get(label, [])))

    print(f"\n[OK] Ket qua luu tai '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
