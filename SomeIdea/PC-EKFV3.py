# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — PC-EKF-2D (V3 — MAD Auto-Normalized, Sigma-Free)
===========================================================================
PC scoring V3:
  - Innovation từ EKF predict step: ν[i] = z[i] - h_i(x̂_pred)
  - S_ii = H_i @ P_pred @ H_i^T + R_base        (per-anchor prediction variance)
  - std_innov[i] = ν[i] / sqrt(S_ii)             → chuẩn hóa ~ N(0,1)
  - MAD normalization: normed[i] = std_innov[i] / (1.4826 * MAD)
    → tự động scale, KHÔNG cần tune sigma nữa
  - Pairwise T-kernel trên normed (sigma=1.0 cố định) → score[i] ∈ [0,1]
  - R_adaptive[i] = R_base * (1 + r_scale*(1 - score[i]))
  - EKF update với R_adaptive (diagonal)

Ưu điểm so với V2:
  - Loại bỏ hoàn toàn tham số sigma
  - MAD robust với outlier (1-2 anchor NLOS không ảnh hưởng scale)
  - Chỉ còn 1 param cần tune: r_scale
  - q và r_base có thể fix (ít nhạy)

PARAMS CẦN TUNE:
  - r_scale : mức phạt outlier  (thử 2 → 30)
  [optional]
  - r_base  : noise baseline mm² (thử 100 → 400)
  - q       : process noise      (thường fix 0.01)

SO SÁNH 3 PHƯƠNG PHÁP:
  1. Raw + LS
  2. EKF-2D         (fixed R)
  3. PC-EKF-v3      (MAD auto-normalized, sigma-free)

Grid search: tự động tìm best (q, r_base, r_scale)

Format file .txt:
  timestamp, d0_slant, d1_slant, d2_slant, d3_slant, v_mm_s, gz_dps
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
    [4000, 8800],
    [0,    8800],
    [0,    0   ],
    [4000, 0   ],
], dtype=float)

ANCHOR_HEIGHT = 1400.0

WAYPOINTS = np.array([
    [800.0,  400.0],
    [3000.0, 400.0],
    [3000.0, 8000.0],
    [800.0,  8000.0],
    [800.0,  400.0],
], dtype=float)

SPEED      = 200.0   # mm/s
GT_SPACING = 5.0     # mm
N_ANCHORS  = 4

# EKF-2D baseline params
EKF2D_Q = 0.01
EKF2D_R = 50.0

# PC-EKF-v3 params  — chỉ cần tune r_scale!
PCEKF_Q       = 0.01
PCEKF_R_BASE  = 50.0
PCEKF_R_SCALE = 30.0   # ← PARAM DUY NHẤT CẦN TUNE (thử 2 → 30)

DO_GRID_SEARCH = True
DATA_DIR = "./data"
SAVE_DIR = "./outputs_PCEKF_v3"


# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS — logic y hệt V3 gốc, chỉ dịch sang @njit
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _pcekf_step_nb(x, P, z, r_base, r_scale, Q_scalar, anchors):
    """
    Một bước PC-EKF V3 hoàn chỉnh: predict + score + update.
    Logic giống hệt class PCEKF_v3 gốc, chỉ dịch sang Numba.
    """
    AH = ANCHOR_HEIGHT

    # ── Predict (giống PCEKF_v3.predict) ──
    P[0, 0] += Q_scalar
    P[1, 1] += Q_scalar

    # ── h_obs + jacobian_H (kết hợp 1 vòng cho cache efficiency) ──
    h  = np.empty(4)
    Hm = np.empty((4, 2))
    for i in range(4):
        dx = x[0] - anchors[i, 0]
        dy = x[1] - anchors[i, 1]
        d  = (dx*dx + dy*dy + AH*AH) ** 0.5
        if d < 1e-6:
            d = 1e-6
        h[i]     = d
        Hm[i, 0] = dx / d
        Hm[i, 1] = dy / d

    # ── Innovation ──
    innov = z - h

    # ── S_diag: HP[i] @ H[i] + r_base  (giống einsum('ij,ij->i', HP, H)) ──
    S_diag = np.empty(4)
    for i in range(4):
        s = 0.0
        for k in range(2):
            row_k = 0.0
            for j in range(2):
                row_k += Hm[i, j] * P[j, k]
            s += row_k * Hm[i, k]
        S_diag[i] = s + r_base

    # ── std_innov[i] = innov[i] / sqrt(S_diag[i]) ──
    std_innov = np.empty(4)
    for i in range(4):
        std_innov[i] = innov[i] / (S_diag[i] ** 0.5 + 1e-9)

    # ── MAD normalize (sort-based median cho N=4, không dùng np.median) ──
    # median của 4 phần tử = trung bình 2 phần tử giữa sau khi sort
    tmp4 = std_innov.copy()
    tmp4.sort()
    med  = (tmp4[1] + tmp4[2]) * 0.5

    abs4 = np.empty(4)
    for i in range(4):
        abs4[i] = abs(std_innov[i] - med)
    abs4.sort()
    mad = (abs4[1] + abs4[2]) * 0.5

    # Giống V3 gốc: normed = std_innov / (1.4826 * mad + 1e-9)
    sigma_hat = 1.4826 * mad + 1e-9
    normed = std_innov / sigma_hat

    # ── Pairwise T-kernel (N=4, unrolled) ──
    # T-kernel: (1 + d^2/(nu*sigma^2))^(-(nu+1)/2), nu=4, sigma=1.0
    # = (1 + d^2/4)^(-2.5)  = (1 + d^2 * 0.25)^(-2.5)
    scores = np.empty(4)
    for i in range(4):
        s = 0.0
        for j in range(4):
            if j != i:
                d = normed[i] - normed[j]
                s += (1.0 + d * d * 0.25) ** (-2.5)
        scores[i] = s / 3.0   # mean over N-1=3 pairs

    # ── R_adaptive (diagonal) ──
    R = np.zeros((4, 4))
    for i in range(4):
        R[i, i] = r_base * (1.0 + r_scale * (1.0 - scores[i]))

    # ── EKF update: S_mat = H P H^T + R ──
    HP    = Hm @ P               # (4, 2)
    S_mat = HP @ Hm.T + R        # (4, 4)

    # Giải S_mat @ X = (H @ P) bằng Gauss-Jordan (njit không có linalg.solve/inv)
    # K^T = S_mat^{-1} @ (H @ P)  →  K = (H @ P)^T @ S_mat^{-T}
    HtP = Hm @ P                 # (4, 2)
    aug = np.empty((4, 6))
    for i in range(4):
        for j in range(4):
            aug[i, j]   = S_mat[i, j]
        for j in range(2):
            aug[i, 4+j] = HtP[i, j]

    for col in range(4):
        # Partial pivoting
        max_v = abs(aug[col, col])
        piv   = col
        for r in range(col + 1, 4):
            if abs(aug[r, col]) > max_v:
                max_v = abs(aug[r, col])
                piv   = r
        if piv != col:
            for j in range(6):
                aug[col, j], aug[piv, j] = aug[piv, j], aug[col, j]
        dv = aug[col, col]
        if abs(dv) < 1e-12:
            dv = 1e-12
        for j in range(6):
            aug[col, j] /= dv
        for r in range(4):
            if r != col:
                f = aug[r, col]
                for j in range(6):
                    aug[r, j] -= f * aug[col, j]

    # Đọc K: K[i, j] = aug[j, 4+i]  →  K shape (2, 4)
    K = np.empty((2, 4))
    for i in range(2):
        for j in range(4):
            K[i, j] = aug[j, 4 + i]

    # ── x update: x = x + K @ innov ──
    Kv = K @ innov
    x[0] += Kv[0]
    x[1] += Kv[1]

    # ── P update: P = (I - K H) P ──
    KH  = K @ Hm
    IKH = np.eye(2) - KH
    P[:] = IKH @ P

    return x, P, scores


# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNEL — EKF-2D BASELINE (R = R_scalar * I)
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _ekf2d_step_nb(x, P, z, Q_scalar, R_scalar, anchors):
    """
    Một bước EKF-2D chuẩn: predict + update với R = R_scalar * I.
    Logic giống hệt EKF2D.step() gốc, dịch sang Numba.
    """
    AH = ANCHOR_HEIGHT

    # ── Predict ──
    P[0, 0] += Q_scalar
    P[1, 1] += Q_scalar

    # ── h_obs + jacobian_H ──
    h  = np.empty(4)
    Hm = np.empty((4, 2))
    for i in range(4):
        dx = x[0] - anchors[i, 0]
        dy = x[1] - anchors[i, 1]
        d  = (dx*dx + dy*dy + AH*AH) ** 0.5
        if d < 1e-6:
            d = 1e-6
        h[i]     = d
        Hm[i, 0] = dx / d
        Hm[i, 1] = dy / d

    # ── Innovation ──
    innov = z - h

    # ── S = H P H^T + R_scalar * I ──
    HP    = Hm @ P
    S_mat = HP @ Hm.T
    for i in range(4):
        S_mat[i, i] += R_scalar

    # ── Gauss-Jordan solve: aug = [S | HP], K = (HP)^T @ S^{-1} ──
    aug = np.empty((4, 6))
    for i in range(4):
        for j in range(4):
            aug[i, j]   = S_mat[i, j]
        for j in range(2):
            aug[i, 4+j] = HP[i, j]

    for col in range(4):
        max_v = abs(aug[col, col])
        piv   = col
        for r in range(col + 1, 4):
            if abs(aug[r, col]) > max_v:
                max_v = abs(aug[r, col])
                piv   = r
        if piv != col:
            for j in range(6):
                aug[col, j], aug[piv, j] = aug[piv, j], aug[col, j]
        dv = aug[col, col]
        if abs(dv) < 1e-12:
            dv = 1e-12
        for j in range(6):
            aug[col, j] /= dv
        for r in range(4):
            if r != col:
                f = aug[r, col]
                for j in range(6):
                    aug[r, j] -= f * aug[col, j]

    K = np.empty((2, 4))
    for i in range(2):
        for j in range(4):
            K[i, j] = aug[j, 4 + i]

    # ── x update ──
    Kv = K @ innov
    x[0] += Kv[0]
    x[1] += Kv[1]

    # ── P update: P = (I - K H) P ──
    KH  = K @ Hm
    IKH = np.eye(2) - KH
    P[:] = IKH @ P

    return x, P


@njit(cache=True)
def _ekf2d_file_nb(raw_dist, x0, Q_scalar, R_scalar, anchors):
    """EKF-2D trên toàn bộ file — full JIT, không cần Python loop."""
    T   = raw_dist.shape[0]
    pos = np.empty((T, 2))
    x   = x0.copy()
    P   = np.eye(2) * 1e6
    for t in range(T):
        x, P = _ekf2d_step_nb(x, P, raw_dist[t], Q_scalar, R_scalar, anchors)
        pos[t, 0] = x[0]
        pos[t, 1] = x[1]
    return pos


# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNEL — LS POSITION (batch)
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _ls_position_nb(distances, anchors):
    """LS linearized trilateration → 2D position, full Numba."""
    x0 = anchors[0, 0]
    y0 = anchors[0, 1]
    d0 = distances[0]
    if d0 < 1.0:
        d0 = 1.0
    N     = anchors.shape[0]
    A     = np.empty((N - 1, 2))
    b_vec = np.empty(N - 1)
    for i in range(1, N):
        xi = anchors[i, 0]
        yi = anchors[i, 1]
        di = distances[i]
        if di < 1.0:
            di = 1.0
        A[i-1, 0] = 2.0 * (xi - x0)
        A[i-1, 1] = 2.0 * (yi - y0)
        b_vec[i-1] = (d0*d0 - di*di) - (x0*x0 - xi*xi) - (y0*y0 - yi*yi)
    # Giải hệ 2×2: (A^T A) pos = A^T b — phân tích trực tiếp
    AtA = A.T @ A      # (2, 2)
    Atb = A.T @ b_vec  # (2,)
    det = AtA[0, 0] * AtA[1, 1] - AtA[0, 1] * AtA[1, 0]
    pos = np.empty(2)
    if abs(det) < 1e-12:
        pos[0] = np.nan
        pos[1] = np.nan
    else:
        pos[0] = (AtA[1, 1] * Atb[0] - AtA[0, 1] * Atb[1]) / det
        pos[1] = (AtA[0, 0] * Atb[1] - AtA[1, 0] * Atb[0]) / det
    return pos


@njit(cache=True)
def _ls_file_nb(dist_mat, anchors):
    """LS position cho toàn bộ timestep — full JIT loop."""
    T   = dist_mat.shape[0]
    pos = np.empty((T, 2))
    for t in range(T):
        p = _ls_position_nb(dist_mat[t], anchors)
        pos[t, 0] = p[0]
        pos[t, 1] = p[1]
    return pos


def _warmup_numba():
    """Trigger JIT compilation khi import — tất cả kernels."""
    _x   = np.array([2000.0, 4400.0])
    _P   = np.eye(2) * 1e6
    _z   = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    _raw = np.ones((2, 4), dtype=np.float64) * 3000.0
    _ekf2d_file_nb(_raw, _x.copy(), 0.01, 50.0, ANCHORS)
    _pcekf_step_nb(_x.copy(), _P.copy(), _z, 50.0, 20.0, 0.01, ANCHORS)
    _ls_file_nb(_raw, ANCHORS)

print("[Numba] Compiling JIT kernels...", end=" ", flush=True)
_warmup_numba()
print("done.")


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
        idx  = np.clip(np.searchsorted(cum_dist, d, 'right') - 1, 0, len(segs)-1)
        frac = (d - cum_dist[idx]) / (seg_len[idx] + 1e-9)
        gt[i] = waypoints[idx] + frac * segs[idx]
    return gt, total_dist / speed


def nearest_gt_error(pos_xy, gt_xy):
    tree = cKDTree(gt_xy)
    errors, _ = tree.query(pos_xy)
    return errors


def ls_position(distances, anchors=ANCHORS):
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b = [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi-x0), 2*(yi-y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    try:
        pos, *_ = np.linalg.lstsq(A.T @ A, A.T @ bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


# ══════════════════════════════════════════════════════════════════════
#  OBSERVATION MODEL (EKF)
# ══════════════════════════════════════════════════════════════════════
def h_obs(state, anchors=ANCHORS):
    x, y = state
    return np.array([
        math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
        for ax, ay in anchors
    ])


def jacobian_H(state, anchors=ANCHORS):
    x, y = state
    H = np.zeros((N_ANCHORS, 2))
    for i, (ax, ay) in enumerate(anchors):
        dist = max(math.sqrt((x-ax)**2 + (y-ay)**2 + ANCHOR_HEIGHT**2), 1e-6)
        H[i, 0] = (x - ax) / dist
        H[i, 1] = (y - ay) / dist
    return H


# ══════════════════════════════════════════════════════════════════════
#  T-DISTRIBUTION KERNEL  (sigma cố định = 1.0, không tune)
# ══════════════════════════════════════════════════════════════════════
def _t_kernel(diff_vec, sigma=1.0):
    nu  = 4.0
    eps = 1e-9
    return (1.0 + diff_vec**2 / (nu * sigma**2 + eps)) ** (-(nu + 1.0) / 2.0)


# ══════════════════════════════════════════════════════════════════════
#  PC SCORING V3 — MAD Auto-Normalized, Sigma-Free
# ══════════════════════════════════════════════════════════════════════
def pc_scores_v3(innovations, S_diag):
    """
    V3: pairwise T-kernel trên MAD-normalized std_innov.

    Bước 1 — Geo-normalize (giống V2):
        std_innov[i] = innovations[i] / sqrt(S_ii)
        S_ii = H_i @ P_pred @ H_i^T + R_base
        → std_innov ≈ N(0,1) khi LOS

    Bước 2 — MAD normalize (mới so với V2):
        mad = median(|std_innov - median(std_innov)|)
        normed[i] = std_innov[i] / (1.4826 * mad + eps)
        → tự động scale, loại bỏ hoàn toàn sigma

    Bước 3 — T-kernel với sigma=1.0 cố định:
        score[i] = mean( T_kernel(normed[i] - normed[j]) ) for j≠i

    Ưu điểm MAD:
      - Robust: 1-2 anchor NLOS không kéo lệch scale
      - Tự adapt theo từng timestep
      - sigma = 1.0 không cần tune
    """
    # Bước 1: geo-normalize
    std_innov = innovations / np.sqrt(np.maximum(S_diag, 1e-9))

    # Bước 2: MAD normalize
    med       = np.median(std_innov)
    mad       = np.median(np.abs(std_innov - med))
    normed    = std_innov / (1.4826 * mad + 1e-9)

    # Bước 3: pairwise T-kernel, sigma=1.0 cố định
    scores = np.zeros(N_ANCHORS)
    for i in range(N_ANCHORS):
        diffs     = np.array([normed[i] - normed[j]
                              for j in range(N_ANCHORS) if j != i])
        scores[i] = np.mean(_t_kernel(diffs, sigma=1.0))
    return scores


# ══════════════════════════════════════════════════════════════════════
#  EKF-2D BASELINE
# ══════════════════════════════════════════════════════════════════════
class EKF2D:
    """EKF với state [x, y], R đồng nhất."""
    def __init__(self, q=EKF2D_Q, r=EKF2D_R):
        self.Q_scalar = q
        self.R_scalar = r
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.copy().astype(float)
        self.P = np.eye(2) * 1e6

    def predict(self):
        self.P = self.P + np.eye(2) * self.Q_scalar

    def update(self, z_raw):
        if self.x is None:
            return np.array([np.nan, np.nan])
        h = h_obs(self.x)
        H = jacobian_H(self.x)
        R = np.eye(N_ANCHORS) * self.R_scalar
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()
        self.x = self.x + K @ (z_raw - h)
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.x.copy()

    def step(self, z_raw):
        self.predict()
        return self.update(z_raw)


# ══════════════════════════════════════════════════════════════════════
#  PC-EKF V3 — MAD Auto-Normalized, Sigma-Free
# ══════════════════════════════════════════════════════════════════════
class PCEKF_v3:
    """
    PC-EKF V3: EKF-2D với PC scoring MAD auto-normalized.
    Không cần tune sigma — chỉ tune r_scale.

    Pipeline mỗi timestep:
      1. EKF predict: P_pred = P + Q*I
      2. h = h_obs(x̂),  H = jacobian_H(x̂)
      3. innovation ν[i] = z_raw[i] - h[i]
      4. S_ii = H_i @ P_pred @ H_i^T + R_base   (scalar per anchor)
      5. std_innov[i] = ν[i] / sqrt(S_ii)
      6. MAD normalize → normed[i]
      7. T-kernel(normed, sigma=1.0) → score[i] ∈ [0,1]
      8. R_adaptive[i] = R_base * (1 + r_scale*(1-score[i]))
      9. EKF update với R_adaptive (diagonal R)
    """
    def __init__(self, q=PCEKF_Q, r_base=PCEKF_R_BASE, r_scale=PCEKF_R_SCALE):
        self.Q_scalar = q
        self.R_base   = r_base
        self.R_scale  = r_scale
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.copy().astype(float)
        self.P = np.eye(2) * 1e6

    def predict(self):
        self.P = self.P + np.eye(2) * self.Q_scalar

    def update(self, z_raw):
        if self.x is None:
            return np.array([np.nan, np.nan]), np.ones(N_ANCHORS)

        h = h_obs(self.x)
        H = jacobian_H(self.x)               # (N_ANCHORS, 2)

        # Innovation
        innov = z_raw - h                    # (N_ANCHORS,)

        # S_ii = H_i @ P @ H_i^T + R_base   (per-anchor scalar)
        HP     = H @ self.P                  # (N_ANCHORS, 2)
        S_diag = np.array([HP[i] @ H[i] + self.R_base
                            for i in range(N_ANCHORS)])

        # PC scoring V3 — MAD auto-normalized, no sigma param
        scores = pc_scores_v3(innov, S_diag)

        # Adaptive R (diagonal)
        R_diag = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        R      = np.diag(R_diag)

        # EKF update
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy(), scores

        self.x = self.x + K @ innov
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.x.copy(), scores

    def step(self, z_raw):
        self.predict()
        return self.update(z_raw)


# ══════════════════════════════════════════════════════════════════════
#  FILTER WRAPPERS
# ══════════════════════════════════════════════════════════════════════
def _init_pos(raw_dist):
    pos = ls_position(raw_dist[0])
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


def ekf2d_filter_file(raw_dist, q=EKF2D_Q, r=EKF2D_R):
    """Dùng Numba kernel — logic giống hệt EKF2D Python, full JIT."""
    x0  = _init_pos(raw_dist).astype(np.float64)
    pos = _ekf2d_file_nb(
        raw_dist.astype(np.float64),
        x0, float(q), float(r), ANCHORS)
    return pos


def pcekf_v3_filter_file(raw_dist, q=PCEKF_Q, r_base=PCEKF_R_BASE,
                          r_scale=PCEKF_R_SCALE):
    """Dùng Numba kernel — kết quả số học giống hệt PCEKF_v3 Python."""
    T    = len(raw_dist)
    x    = _init_pos(raw_dist).astype(np.float64)
    P    = np.eye(2) * 1e6
    pos  = np.full((T, 2), np.nan)
    scrs = np.zeros((T, N_ANCHORS))
    for t in range(T):
        x, P, scrs[t] = _pcekf_step_nb(
            x, P, raw_dist[t].astype(np.float64),
            float(r_base), float(r_scale), float(q), ANCHORS)
        pos[t] = x
    return pos, scrs


# ══════════════════════════════════════════════════════════════════════
#  PARSE FILE
# ══════════════════════════════════════════════════════════════════════
def parse_file(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) < 7: continue
            try:
                d_slant  = [float(parts[i+1]) for i in range(4)]
                d_ground = [slant_to_ground(d) for d in d_slant]
                rows.append(d_ground + [float(parts[5]), float(parts[6])])
            except ValueError:
                continue
    if not rows: return None
    data = np.array(rows, dtype=np.float32)
    return {'dist': data[:, :4], 'v': data[:, 4], 'gz': data[:, 5]}


# ══════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════
def evaluate_files(file_paths, gt_xy,
                   q=PCEKF_Q, r_base=PCEKF_R_BASE, r_scale=PCEKF_R_SCALE):
    raw_pos_all   = []
    ekf_pos_all   = []
    pcekf_pos_all = []
    per_file_rmse = {'Raw + LS': [], 'EKF-2D': [], 'PC-EKF-v3': []}

    W = 95
    print("\n" + "═"*W)
    print(f"  PC-EKF V3 — PER-FILE RESULTS")
    print("═"*W)
    print(f"{'File':<18s} {'Raw+LS':>10s} {'EKF-2D':>10s} "
          f"{'PC-EKF-v3':>12s} | "
          f"{'EKF T_fl':>10s} {'PCEKF T_fl':>11s} {'PCEKF T_smpl':>13s}")
    print("─"*W)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None: continue
        dist_raw = parsed['dist'].astype(float)
        T = len(dist_raw)

        # 1. Raw + LS
        dist_f64 = dist_raw.astype(np.float64)
        raw_pos  = _ls_file_nb(dist_f64, ANCHORS)

        # 2. EKF-2D
        t0 = time.perf_counter()
        ekf_pos = ekf2d_filter_file(dist_raw)
        t1 = time.perf_counter()

        # 3. PC-EKF-v3
        pcekf_pos, _ = pcekf_v3_filter_file(
            dist_raw, q=q, r_base=r_base, r_scale=r_scale)
        t2 = time.perf_counter()

        def filt(p): return p[~np.any(np.isnan(p), axis=1)]
        def rmse(p):
            e = nearest_gt_error(filt(p), gt_xy)
            return float(np.sqrt(np.mean(e**2))) if len(e) > 0 else float('nan')

        r_raw   = rmse(raw_pos)
        r_ekf   = rmse(ekf_pos)
        r_pcekf = rmse(pcekf_pos)

        t_ekf   = (t1 - t0) * 1000
        t_pcekf = (t2 - t1) * 1000
        print(f"{os.path.basename(path):<18s}"
              f"{r_raw:>10.1f}{r_ekf:>10.1f}"
              f"{r_pcekf:>12.1f} | "
              f"{t_ekf:>8.1f}ms {t_pcekf:>9.1f}ms {t_pcekf/T*1000:>11.3f}ms")

        per_file_rmse['Raw + LS'].append(r_raw)
        per_file_rmse['EKF-2D']  .append(r_ekf)
        per_file_rmse['PC-EKF-v3'].append(r_pcekf)

        raw_pos_all  .extend(filt(raw_pos))
        ekf_pos_all  .extend(filt(ekf_pos))
        pcekf_pos_all.extend(filt(pcekf_pos))

    print("─"*W)
    raw_arr   = np.array(raw_pos_all)
    ekf_arr   = np.array(ekf_pos_all)
    pcekf_arr = np.array(pcekf_pos_all)

    errors = {
        'Raw + LS' : nearest_gt_error(raw_arr,   gt_xy),
        'EKF-2D'    : nearest_gt_error(ekf_arr,   gt_xy),
        'PC-EKF-v3' : nearest_gt_error(pcekf_arr, gt_xy),
    }
    positions = {'raw': raw_arr, 'ekf': ekf_arr, 'pcekf': pcekf_arr}
    return errors, positions, per_file_rmse


def compute_metrics(errors, label="", pf_rmse=None):
    m = {k: float(fn(errors)) for k, fn in [
        ('mae',   np.mean),
        ('rmse',  lambda e: np.sqrt(np.mean(e**2))),
        ('cep50', lambda e: np.percentile(e, 50)),
        ('cep90', lambda e: np.percentile(e, 90)),
        ('p95',   lambda e: np.percentile(e, 95)),
        ('max',   np.max),
    ]}
    # mean ± std of per-file RMSE
    if pf_rmse is not None:
        vals = [v for v in pf_rmse if not math.isnan(v)]
        if len(vals) >= 2:
            m['rmse_mean'] = float(np.mean(vals))
            m['rmse_std']  = float(np.std(vals, ddof=1))
        elif len(vals) == 1:
            m['rmse_mean'] = float(vals[0])
            m['rmse_std']  = float('nan')
        else:
            m['rmse_mean'] = float('nan')
            m['rmse_std']  = float('nan')
    else:
        m['rmse_mean'] = float('nan')
        m['rmse_std']  = float('nan')
    if label:
        print(f"\n{'='*50}\n  {label}\n{'='*50}")
        for k, v in m.items(): print(f"  {k.upper():6s}: {v:8.1f} mm")
    return m


# ══════════════════════════════════════════════════════════════════════
#  GRID SEARCH  (chỉ còn 3 params, bỏ sigma)
# ══════════════════════════════════════════════════════════════════════
def grid_search(val_files, gt_xy):
    """
    Grid search cho PC-EKF V3.
    sigma đã bị loại — chỉ còn q, r_base, r_scale.
    """
    grid = {
        'q'       : [0.01],
        'r_base'  : [50.0],
        'r_scale' : [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0],
    }
    keys    = list(grid.keys())
    combos  = list(itertools.product(*[grid[k] for k in keys]))
    n_combo = len(combos)
    print(f"\n  Grid search PC-EKF V3: {n_combo} combinations "
          f"(vs {n_combo * 6} với sigma — giảm 6x)...")

    best_rmse   = float('inf')
    best_params = {}
    results     = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        err, _, pf = evaluate_files(val_files, gt_xy, **params)
        vals   = [v for v in pf['PC-EKF-v3'] if not math.isnan(v)]
        rmse   = float(np.mean(vals)) if vals else float('inf')
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params.copy()
        if (idx + 1) % 20 == 0 or (idx + 1) == n_combo:
            print(f"  [{idx+1}/{n_combo}] best so far: mean RMSE={best_rmse:.1f}mm")

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs (PC-EKF V3):")
    print(f"  {'mean RMSE':>10s} {'Q':>8s} {'R_base':>8s} {'r_scale':>8s}")
    print(f"  {'─'*40}")
    for rmse, p in results[:5]:
        print(f"  {rmse:>9.1f}mm {p['q']:>8.4f} {p['r_base']:>8.1f}"
              f" {p['r_scale']:>8.1f}")
    print(f"\n  Best: mean RMSE={best_rmse:.1f}mm | {best_params}")
    return best_params, best_rmse


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    'Raw + LS' : '#9E9E9E',
    'EKF-2D'    : '#E91E63',
    'PC-EKF-v3' : '#2196F3',
}
LS = {
    'Raw + LS' : ':',
    'EKF-2D'    : '--',
    'PC-EKF-v3' : '-',
}


def plot_cdf(errors_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, errors in errors_dict.items():
        s      = np.sort(errors)
        cdf    = np.arange(1, len(s)+1) / len(s)
        rmse_v = np.sqrt(np.mean(errors**2))
        color  = COLORS.get(label, 'gray')
        lw     = 2.5 if 'PC' in label else 1.8
        ax.plot(s, cdf, lw=lw, color=color, ls=LS.get(label, '-'),
                label=f"{label}  RMSE={rmse_v:.1f}mm")
        ax.axvline(rmse_v, color=color, ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel("Position Error (mm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 2000); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] CDF → {save_path}")
    plt.close()


def plot_trajectories(positions_dict, gt_xy, save_path):
    labels = list(positions_dict.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 7))
    if n == 1: axes = [axes]
    for ax, label in zip(axes, labels):
        pos   = positions_dict[label]
        color = COLORS.get(label, 'gray')
        ax.plot(gt_xy[:,0], gt_xy[:,1], 'k--', lw=2, alpha=0.5, label='GT')
        ax.plot(pos[:,0], pos[:,1], color=color, lw=0.8, alpha=0.7, label=label)
        for nm, pt in zip(["A","B","C","D"], WAYPOINTS[:4]):
            ax.scatter(*pt, s=70, color='black', zorder=10)
            ax.annotate(nm, pt, xytext=(6,4), textcoords="offset points",
                        fontsize=12, fontweight='bold')
        for j, (ax_, ay_) in enumerate(ANCHORS):
            ax.scatter(ax_, ay_, s=90, marker='s', color='red', zorder=10)
            ax.annotate(f"A{j+1}", (ax_,ay_), xytext=(5,5),
                        textcoords="offset points", fontsize=9, color='red')
        rmse = np.sqrt(np.mean(nearest_gt_error(pos, gt_xy)**2))
        ax.set_title(f"{label}\nRMSE={rmse:.1f}mm", fontsize=11, fontweight='bold')
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.legend(fontsize=9); ax.set_aspect('equal')
        ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Trajectories → {save_path}")
    plt.close()


def plot_bar(metrics_dict, save_path):
    methods  = list(metrics_dict.keys())
    mnames   = ['rmse', 'mae', 'cep50', 'p95']
    mlabels  = ['RMSE', 'MAE', 'CEP50', 'P95']
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for ax, mname, mlabel in zip(axes, mnames, mlabels):
        vals   = [metrics_dict[m][mname] for m in methods]
        colors = [COLORS.get(m, 'gray') for m in methods]
        bars   = ax.bar(range(len(methods)), vals, color=colors, alpha=0.85,
                        edgecolor='white', lw=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


def analyze_scores(file_paths, q=PCEKF_Q, r_base=PCEKF_R_BASE,
                   r_scale=PCEKF_R_SCALE, save_path=None):
    """Phân tích scores V3 và R adaptive trên file đầu tiên."""
    parsed = parse_file(file_paths[0])
    if parsed is None: return
    dist_raw = parsed['dist'].astype(float)

    _, scores = pcekf_v3_filter_file(dist_raw, q=q, r_base=r_base, r_scale=r_scale)
    R_adaptive = r_base * (1.0 + r_scale * (1.0 - scores))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']

    for i in range(N_ANCHORS):
        axes[0].plot(scores[:,i], color=colors[i], lw=0.8,
                     label=f'Anchor {i+1}', alpha=0.8)
    axes[0].axhline(0.5, color='gray', ls='--', lw=1, label='threshold 0.5')
    axes[0].set_ylabel('PC Score V3', fontsize=11)
    axes[0].set_title(
        f'PC-EKF V3 Scores (MAD auto-normalized, sigma-free) — {os.path.basename(file_paths[0])}',
        fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.05)

    for i in range(N_ANCHORS):
        axes[1].plot(R_adaptive[:,i], color=colors[i], lw=0.8,
                     label=f'R Anchor {i+1}', alpha=0.8)
    axes[1].set_ylabel('Adaptive R (mm²)', fontsize=11)
    axes[1].set_title(
        f'Adaptive R per Anchor — r_scale={r_scale}  r_base={r_base}',
        fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    for i in range(N_ANCHORS):
        axes[2].plot(dist_raw[:,i], color=colors[i], lw=0.6,
                     label=f'Raw d{i+1}', alpha=0.7)
    axes[2].set_ylabel('Distance (mm)', fontsize=11)
    axes[2].set_xlabel('Timestep', fontsize=11)
    axes[2].set_title('Raw Distances', fontsize=12)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Score analysis → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("PC-EKF V3 — MAD Auto-Normalized, Sigma-Free")
    print("  Không dùng KF 1D phụ trợ")
    print("  Innovation từ EKF predict, normalize S_ii = H_i@P@H_i^T + R_base")
    print("  MAD auto-scale → sigma=1.0 cố định, KHÔNG CẦN tune sigma")
    print(f"  Chỉ tune: r_scale (hiện tại = {PCEKF_R_SCALE})")
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"Không tìm thấy file .txt trong '{DATA_DIR}'"); return
    print(f"Tìm thấy {n} files")

    np.random.seed(42)
    shuffled = [all_files[i] for i in np.random.permutation(n)]

    gt_xy, _ = build_ground_truth()
    total_path = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"Path={total_path:.0f}mm  GT={len(gt_xy)} pts")

    best_params = dict(q=PCEKF_Q, r_base=PCEKF_R_BASE, r_scale=PCEKF_R_SCALE)

    if DO_GRID_SEARCH:
        print(f"\n[Grid Search] Dùng toàn bộ {n} files...")
        best_params, _ = grid_search(shuffled, gt_xy)
    else:
        print("\n[INFO] Grid Search TẮT — dùng params mặc định.")

    eval_files = shuffled

    print(f"\n{'═'*60}\n  Evaluating {len(eval_files)} files\n{'═'*60}")
    print(f"  Params: {best_params}")
    print(f"  NOTE: sigma đã bị loại (MAD auto-normalize)")

    err, positions, per_file_rmse = evaluate_files(eval_files, gt_xy, **best_params)
    metrics = {label: compute_metrics(errors, label, pf_rmse=per_file_rmse.get(label))
               for label, errors in err.items()}

    # Summary
    print(f"\n{'═'*88}")
    print(f"  SUMMARY — PC-EKF V3 (sigma-free)")
    print(f"{'═'*88}")
    print(f"  {'Method':<18s} {'RMSE mean±std':>18s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*70}")
    for label, m in metrics.items():
        tag = " ◀ V3" if 'PC-EKF' in label else ""
        if not math.isnan(m.get('rmse_std', float('nan'))):
            rmse_str = f"{m['rmse_mean']:>6.1f} ± {m['rmse_std']:.1f}"
        else:
            rmse_str = f"{m['rmse_mean']:>6.1f} ± N/A"
        print(f"  {label:<18s} {rmse_str:>18s} {m['mae']:>6.1f} "
              f"{m['cep50']:>6.1f} {m['p95']:>6.1f} {m['max']:>6.1f}{tag}")

    # Wilcoxon
    print(f"\n  Wilcoxon tests (vs PC-EKF-v3):")
    for a in ['Raw + LS', 'EKF-2D']:
        ea, eb = err[a], err['PC-EKF-v3']
        mn = min(len(ea), len(eb))
        if mn > 10:
            try:
                _, p = wilcoxon(ea[:mn], eb[:mn])
                print(f"    {a} vs PC-EKF-v3: p={p:.4f} {'✅' if p<0.05 else '⚠️'}")
            except ValueError:
                pass

    print(f"\n  Best params:")
    print(f"    q={best_params['q']}, r_base={best_params['r_base']}, "
          f"r_scale={best_params['r_scale']}")
    print(f"  [sigma-free: MAD tự động normalize, không cần tune]")

    # Plots
    plot_cdf(err, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(
        {'Raw + LS' : positions['raw'],
         'EKF-2D'    : positions['ekf'],
         'PC-EKF-v3' : positions['pcekf']},
        gt_xy, os.path.join(SAVE_DIR, 'trajectories.png'))
    plot_bar(metrics, os.path.join(SAVE_DIR, 'bar.png'))
    analyze_scores(eval_files, save_path=os.path.join(SAVE_DIR, 'score_analysis.png'),
                   **best_params)

    for label, errors in err.items():
        safe = label.lower().replace(' + ','_').replace(' ','_').replace('-','_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errors)

    print(f"\n[✓] Toàn bộ kết quả → '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
