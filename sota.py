# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — V18: Robust & Adaptive KF
===================================================
THAY ĐỔI SO VỚI V17:
  - PC-KF (distance-space 1D) → PC-UKF-2D (từ V15): PC adaptive R + UKF 2D
    (hệ số đã tuning từ V15, không cần grid search thêm)
  - Sửa lỗi timing: parse_file trả về ndarray, không dùng `or []`
  - Grid search mở rộng cho AEKF và VBAKF (chạy trước evaluate)
  - AEKF: fix Q-adaptive (chỉ adapt R, Q cố định) để tránh diverge
  - VBAKF: rho > 0.95, tau tăng để prior không decay quá nhanh

CÁC PHƯƠNG PHÁP:
  1. Raw + WLS        — baseline
  2. Huber-UKF        — IRLS Huber M-estimator
  3. MCC-UKF          — Maximum Correntropy Criterion UKF
  4. AEKF (IAE)       — Innovation Adaptive Estimation (chỉ adapt R)
  5. VBAKF            — Variational Bayesian Adaptive KF
  6. PC-UKF-2D        — Pairwise Consensus adaptive R + UKF 2D (tuned)

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
    [800.0,  400.0],
    [3000.0, 400.0],
    [3000.0, 8000.0],
    [800.0,  8000.0],
    [800.0,  400.0],
], dtype=float)

SPEED      = 200.0
GT_SPACING = 5.0
N_ANCHORS  = 4

# ─── Huber-UKF ───────────────────────────────────────────────────────
HUBER_Q       = 1.0
HUBER_R       = 5000.0
HUBER_DELTA   = 2.0
HUBER_MAXITER = 10

# ─── MCC-UKF ─────────────────────────────────────────────────────────
MCC_Q          = 1.0
MCC_R          = 5000.0
MCC_KERNEL_BW  = 1000.0
MCC_MAXITER    = 5

# ─── AEKF (chỉ adapt R, Q cố định để tránh diverge) ─────────────────
AEKF_Q0        = 1.0     # Process noise cố định (không adapt)
AEKF_R0        = 500.0   # Initial R — sẽ được grid search
AEKF_WIN       = 50      # Sliding window lớn hơn → estimate ổn định hơn
AEKF_ALPHA_R   = 0.95    # Forgetting factor — sẽ được grid search
AEKF_R_MIN     = 1.0

# ─── VBAKF ────────────────────────────────────────────────────────────
VBAKF_Q0       = 1.0
VBAKF_R0       = 5000.0   # sẽ được grid search
VBAKF_RHO      = 0.98    # Forgetting factor cao hơn → prior decay chậm hơn
VBAKF_MAXITER  = 5

# ─── PC-UKF-2D (đã tuning từ V15, không grid search) ─────────────────
PCUKF_Q        = 0.01
PCUKF_R_BASE   = 200.0
PCUKF_R_SCALE  = 5.0
PCUKF_SIGMA    = 25.0

# ─── UKF common ──────────────────────────────────────────────────────
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

DO_GRID_SEARCH = False
DATA_DIR = "./data"
SAVE_DIR = "./outputs_sota"


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


def wls_position(distances, anchors=ANCHORS, weights=None):
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi - x0), 2*(yi - y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
        w.append(weights[i] if weights is not None else 1.0 / di)
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    W  = np.diag(w)
    try:
        pos, *_ = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


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
    pos = wls_position(dist_raw_0)
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


# ══════════════════════════════════════════════════════════════════════
#  1. HUBER-UKF
# ══════════════════════════════════════════════════════════════════════
class HuberUKF:
    """
    Iteratively Reweighted UKF với Huber M-estimator.
    Outlier (residual lớn) → R_eff tăng → bị dampen tự động.
    """
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

        R_eff  = np.eye(N_ANCHORS) * self.R_base
        x_cur  = x_pred.copy()
        for _ in range(self.maxiter):
            Pzz_eff  = Pzz_no_R + R_eff
            innov    = z_raw - z_hat
            pzz_diag = np.maximum(np.diag(Pzz_eff), 1e-9)
            r_scaled = innov / np.sqrt(pzz_diag)
            hub_w    = np.where(np.abs(r_scaled) <= self.delta,
                                1.0, self.delta / (np.abs(r_scaled) + 1e-9))
            hub_w    = np.maximum(hub_w, 1e-4)
            R_eff    = np.diag(self.R_base / hub_w)
            Pzz_eff  = Pzz_no_R + R_eff
            try:
                K = Pxz @ np.linalg.inv(Pzz_eff)
            except np.linalg.LinAlgError:
                break
            x_new = x_pred + K @ innov
            if np.linalg.norm(x_new - x_cur) < 1e-3:
                x_cur = x_new
                break
            x_cur = x_new

        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_cur
            self.P = make_spd(P_pred)
            return self.x.copy()

        self.x = x_pred + K @ (z_raw - z_hat)
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  2. MCC-UKF
# ══════════════════════════════════════════════════════════════════════
class MCCUKF:
    """
    Maximum Correntropy Criterion UKF.
    Gaussian kernel weight per anchor → outlier có R_eff lớn.
    """
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
            innov  = z_raw - z_hat
            kern_w = self._gaussian_kernel(innov)
            kern_w = np.maximum(kern_w, 1e-4)
            R_eff  = np.diag(self.R_base / kern_w)
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
#  3. AEKF — chỉ adapt R (Q cố định)
# ══════════════════════════════════════════════════════════════════════
class AEKF:
    """
    Adaptive UKF — Innovation-based Adaptive Estimation chỉ cho R.
    Q cố định để tránh diverge (Q-adaptation thường bất ổn với UWB NLOS).

    Sage-Husa IAE:
      C_k = sample cov của innovation trong sliding window
      R_hat = C_k - Pzz_no_R
      R_k = alpha * R_{k-1} + (1-alpha) * R_hat
    """
    def __init__(self, q0=AEKF_Q0, r0=AEKF_R0,
                 win=AEKF_WIN, alpha_r=AEKF_ALPHA_R, r_min=AEKF_R_MIN):
        self.n       = 2
        self.alpha_r = alpha_r
        self.r_min   = r_min
        self.win     = win
        self.Q       = np.eye(2) * q0          # Q cố định
        self.R       = np.eye(N_ANCHORS) * r0  # R khởi tạo
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None
        self._innov_buf = []

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(self.n) * 1e6
        self._innov_buf = []

    def _update_R(self, innov, Pzz_no_R):
        self._innov_buf.append(innov.copy())
        if len(self._innov_buf) > self.win:
            self._innov_buf.pop(0)
        N   = len(self._innov_buf)
        buf = np.array(self._innov_buf)
        # Sample cov: dùng outer product trung bình
        C_innov = (buf.T @ buf) / N
        # R_hat = C_innov - Pzz_no_R (lý thuyết)
        R_hat = C_innov - Pzz_no_R
        # Chỉ giữ diagonal, floor tại r_min
        r_diag = np.maximum(np.diag(R_hat), self.r_min)
        R_hat  = np.diag(r_diag)
        # Exponential forgetting
        self.R = self.alpha_r * self.R + (1 - self.alpha_r) * R_hat
        self.R = make_spd(self.R, eps=self.r_min)

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)

        Pzz_eff = Pzz_no_R + self.R
        innov   = z_raw - z_hat

        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()

        # Adapt R sau khi có K (dùng innovation trước update)
        self._update_R(innov, Pzz_no_R)

        self.x = x_pred + K @ innov
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  4. VBAKF
# ══════════════════════════════════════════════════════════════════════
class VBAKF:
    """
    Variational Bayesian Adaptive Kalman Filter.
    R ~ Inverse-Wishart prior, VB iteration để joint estimate x và R.

    Với rho cao (0.97–0.99): prior decay chậm → ổn định hơn.
    """
    def __init__(self, q0=VBAKF_Q0, r0=VBAKF_R0,
                 rho=VBAKF_RHO, maxiter=VBAKF_MAXITER):
        self.n       = 2
        self.Q       = np.eye(2) * q0
        self.rho     = rho
        self.maxiter = maxiter
        M            = N_ANCHORS
        # IW prior: E[R] = Psi/(nu-M-1) = r0*I
        self.nu      = float(M + 2)          # minimal valid dof
        self.Psi     = np.eye(M) * r0 * 1.0  # Psi s.t. E[R] ~ r0*I
        self.nu_hat  = self.nu
        self.Psi_hat = self.Psi.copy()
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x       = x0.astype(float).copy()
        self.P       = np.eye(self.n) * 1e6
        self.nu_hat  = self.nu
        self.Psi_hat = self.Psi.copy()

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)
        M = N_ANCHORS

        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)

        # VB prior decay (forgetting)
        nu_k  = self.rho * self.nu_hat + (1 - self.rho) * (M + 2.0)
        Psi_k = self.rho * self.Psi_hat

        nu_iter  = nu_k
        Psi_iter = Psi_k.copy()
        innov    = z_raw - z_hat
        x_cur    = x_pred.copy()
        K        = np.zeros((self.n, M))

        for _ in range(self.maxiter):
            dof   = max(nu_iter - M - 1.0, 0.5)
            R_cur = make_spd(Psi_iter / dof, eps=1.0)
            Pzz_eff = Pzz_no_R + R_cur
            try:
                K = Pxz @ np.linalg.inv(Pzz_eff)
            except np.linalg.LinAlgError:
                break
            x_cur    = x_pred + K @ innov
            # Update IW posterior (Huang et al. 2020)
            Psi_iter = Psi_k + np.outer(innov, innov) + Pzz_no_R
            nu_iter  = nu_k + 1.0

        self.Psi_hat = Psi_iter
        self.nu_hat  = max(nu_iter, M + 2.0)
        self.x = x_cur
        self.P = make_spd(P_pred - K @ (Pzz_no_R + make_spd(Psi_iter / max(nu_iter - M - 1.0, 0.5), eps=1.0)) @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  5. PC-UKF-2D (từ V15 — đã tuning)
# ══════════════════════════════════════════════════════════════════════
class _KF1D_for_PC:
    """KF 1D nhẹ, chỉ dùng nội bộ để tính innovation cho PC scoring."""
    def __init__(self):
        self.Q = 0.01; self.R = 200.0
        self.P = 1.0;  self.x = None

    def update(self, z):
        if self.x is None:
            self.x = z
            return 0.0
        P_pred     = self.P + self.Q
        innov      = z - self.x
        K          = P_pred / (P_pred + self.R)
        self.x    += K * innov
        self.P     = (1 - K) * P_pred
        return innov


def _pc_consensus_scores(innovations, sigma=PCUKF_SIGMA):
    innov  = np.abs(innovations)
    scores = np.zeros(N_ANCHORS)
    nu     = 4.0
    for i in range(N_ANCHORS):
        s = 0.0
        for j in range(N_ANCHORS):
            if i == j:
                continue
            diff = innov[i] - innov[j]
            c    = (1.0 + diff**2 / (nu * sigma**2 + 1e-9)) ** (-(nu + 1.0) / 2.0)
            s   += c
        scores[i] = s / (N_ANCHORS - 1)
    return scores


class PCUKF2D:
    """
    PC-UKF-2D: Pairwise Consensus adaptive R + UKF 2D.
    Hệ số đã tuning từ V15 (q=0.01, r_base=200, r_scale=5, sigma=25).
    """
    def __init__(self, q=PCUKF_Q, r_base=PCUKF_R_BASE,
                 r_scale=PCUKF_R_SCALE, sigma=PCUKF_SIGMA):
        self.n       = 2
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.sigma   = sigma
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self._kfs    = [_KF1D_for_PC() for _ in range(N_ANCHORS)]
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(self.n) * 1e6

    def step(self, z_raw):
        if self.x is None:
            return np.full(2, np.nan)

        # PC scores từ KF 1D per-anchor
        innovations = np.array([self._kfs[i].update(z_raw[i]) for i in range(N_ANCHORS)])
        scores      = _pc_consensus_scores(innovations, self.sigma)
        R_diag      = self.R_base * (1.0 + self.R_scale * (1.0 - scores))

        # UKF predict
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat

        # UKF update
        z_hat, Pzz_no_R, Pxz = ukf_measurement_moments(
            x_pred, P_pred, self.Wm, self.Wc, self.c)

        R      = np.diag(R_diag)
        Pzz_eff = Pzz_no_R + R

        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = make_spd(P_pred)
            return self.x.copy()

        innov  = z_raw - z_hat
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
    """Trả về số dòng của file, an toàn với None."""
    data = parse_file(path)
    return len(data) if data is not None else 0


# ══════════════════════════════════════════════════════════════════════
#  METHODS TABLE
# ══════════════════════════════════════════════════════════════════════
# Khởi tạo với params mặc định; sẽ được cập nhật sau grid search
METHODS = {
    "Raw+WLS"   : None,
    "Huber-UKF" : (HuberUKF,  dict(q=HUBER_Q,    r=HUBER_R,    delta=HUBER_DELTA)),
    "MCC-UKF"   : (MCCUKF,    dict(q=MCC_Q,       r=MCC_R,      kernel_bw=MCC_KERNEL_BW)),
    "AEKF-IAE"  : (AEKF,      dict(q0=AEKF_Q0,    r0=AEKF_R0,   win=AEKF_WIN,
                                    alpha_r=AEKF_ALPHA_R)),
    "VBAKF"     : (VBAKF,     dict(q0=VBAKF_Q0,   r0=VBAKF_R0,  rho=VBAKF_RHO)),
    "PC-UKF-2D" : (PCUKF2D,   dict(q=PCUKF_Q,     r_base=PCUKF_R_BASE,
                                    r_scale=PCUKF_R_SCALE, sigma=PCUKF_SIGMA)),
}

COLORS = {
    "Raw+WLS"   : '#9E9E9E',
    "Huber-UKF" : '#E91E63',
    "MCC-UKF"   : '#FF9800',
    "AEKF-IAE"  : '#2196F3',
    "VBAKF"     : '#4CAF50',
    "PC-UKF-2D" : '#9C27B0',
}


# ══════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════
def evaluate_files(file_paths, gt_xy):
    all_errors = {m: [] for m in METHODS}
    all_pos    = {m: [] for m in METHODS}
    timing     = {m: [] for m in METHODS if m != "Raw+WLS"}

    hdr = f"{'File':<18s}"
    for m in METHODS:
        hdr += f" {m:>12s}"
    print("\n" + "═" * 100)
    print("  PER-FILE RMSE (mm)")
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
                pos = np.array([wls_position(dist_raw[t]) for t in range(T)])
            else:
                filt_cls, kwargs = method
                t0  = time.perf_counter()
                pos = run_filter(filt_cls, dist_raw, **kwargs)
                timing[name].append(time.perf_counter() - t0)

            valid = pos[~np.any(np.isnan(pos), axis=1)]
            errs  = nearest_gt_error(valid, gt_xy)
            rmse  = np.sqrt(np.mean(errs**2)) if len(errs) > 0 else float('nan')
            row  += f" {rmse:>12.1f}"
            all_errors[name].extend(errs)
            all_pos[name].extend(valid)

        print(row)

    print("─" * 100)

    # Timing summary — FIX: không dùng `or []` với ndarray
    print("\n  TIMING (ms/sample):")
    for name, times in timing.items():
        if not times:
            continue
        total_t = sum(times)
        n_samp  = sum(safe_len(p) for p in file_paths[:len(times)])
        if n_samp > 0:
            print(f"    {name:<14s}: {total_t * 1000 / n_samp:.4f} ms/sample")

    return (
        {m: np.array(v) for m, v in all_errors.items()},
        {m: np.array(v) for m, v in all_pos.items()},
    )


def compute_metrics(errors):
    if len(errors) == 0:
        return {k: float('nan') for k in ['mae', 'rmse', 'cep50', 'cep90', 'p95', 'max']}
    return {
        'mae'  : float(np.mean(errors)),
        'rmse' : float(np.sqrt(np.mean(errors**2))),
        'cep50': float(np.percentile(errors, 50)),
        'cep90': float(np.percentile(errors, 90)),
        'p95'  : float(np.percentile(errors, 95)),
        'max'  : float(np.max(errors)),
    }


# ══════════════════════════════════════════════════════════════════════
#  GRID SEARCH — AEKF và VBAKF
# ══════════════════════════════════════════════════════════════════════
def _eval_rmse_single(filt_class, kwargs, file_paths, gt_xy):
    """Chạy filter trên tất cả file, trả về RMSE tổng."""
    pos_all = []
    for path in file_paths:
        d = parse_file(path)
        if d is None:
            continue
        p = run_filter(filt_class, d, **kwargs)
        valid = p[~np.any(np.isnan(p), axis=1)]
        pos_all.extend(valid)
    if not pos_all:
        return float('inf')
    errs = nearest_gt_error(np.array(pos_all), gt_xy)
    return float(np.sqrt(np.mean(errs**2)))


def grid_search_aekf(file_paths, gt_xy):
    grid = {
        'r0'     : [100.0, 500.0, 1000.0, 2000.0, 5000.0],
        'win'    : [20, 50, 100],
        'alpha_r': [0.90, 0.95, 0.98, 0.99],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search AEKF: {len(combos)} combinations...")
    best_rmse = float('inf')
    best = dict(q0=AEKF_Q0, r0=AEKF_R0, win=AEKF_WIN, alpha_r=AEKF_ALPHA_R)
    for combo in combos:
        params = dict(zip(keys, combo))
        params['q0'] = AEKF_Q0  # Q cố định
        params['r_min'] = AEKF_R_MIN
        rmse = _eval_rmse_single(AEKF, params, file_paths, gt_xy)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    print(f"  Best AEKF: RMSE={best_rmse:.1f}mm | r0={best['r0']} win={best['win']} alpha_r={best['alpha_r']}")
    return best


def grid_search_vbakf(file_paths, gt_xy):
    grid = {
        'r0' : [100.0, 500.0, 1000.0, 2000.0],
        'rho': [0.95, 0.97, 0.98, 0.99],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search VBAKF: {len(combos)} combinations...")
    best_rmse = float('inf')
    best = dict(q0=VBAKF_Q0, r0=VBAKF_R0, rho=VBAKF_RHO)
    for combo in combos:
        params = dict(zip(keys, combo))
        params['q0'] = VBAKF_Q0
        rmse = _eval_rmse_single(VBAKF, params, file_paths, gt_xy)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    print(f"  Best VBAKF: RMSE={best_rmse:.1f}mm | r0={best['r0']} rho={best['rho']}")
    return best


def grid_search_huber(file_paths, gt_xy):
    grid = {
        'r'    : [500.0, 1000.0, 2000.0, 5000.0],
        'delta': [0.5, 1.0, 1.345, 2.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search Huber-UKF: {len(combos)} combinations...")
    best_rmse = float('inf')
    best = dict(q=HUBER_Q, r=HUBER_R, delta=HUBER_DELTA)
    for combo in combos:
        params = dict(zip(keys, combo))
        params['q'] = HUBER_Q
        rmse = _eval_rmse_single(HuberUKF, params, file_paths, gt_xy)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    print(f"  Best Huber-UKF: RMSE={best_rmse:.1f}mm | r={best['r']} delta={best['delta']}")
    return best


def grid_search_mcc(file_paths, gt_xy):
    grid = {
        'r'        : [500.0, 1000.0, 2000.0, 5000.0],
        'kernel_bw': [100.0, 300.0, 500.0, 1000.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search MCC-UKF: {len(combos)} combinations...")
    best_rmse = float('inf')
    best = dict(q=MCC_Q, r=MCC_R, kernel_bw=MCC_KERNEL_BW)
    for combo in combos:
        params = dict(zip(keys, combo))
        params['q'] = MCC_Q
        rmse = _eval_rmse_single(MCCUKF, params, file_paths, gt_xy)
        if rmse < best_rmse:
            best_rmse = rmse
            best = params.copy()
    print(f"  Best MCC-UKF: RMSE={best_rmse:.1f}mm | r={best['r']} kernel_bw={best['kernel_bw']}")
    return best


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_cdf(errors_dict, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, errors in errors_dict.items():
        if len(errors) == 0:
            continue
        s    = np.sort(errors)
        cdf  = np.arange(1, len(s) + 1) / len(s)
        rmse = np.sqrt(np.mean(errors**2))
        c    = COLORS.get(label, 'gray')
        lw   = 2.5 if label not in ("Raw+WLS",) else 1.5
        ls   = '-'  if label not in ("Raw+WLS",) else '--'
        ax.plot(s, cdf, lw=lw, color=c, ls=ls, label=f"{label}  RMSE={rmse:.1f}mm")
        ax.axvline(rmse, color=c, ls=':', lw=0.8, alpha=0.4)
    ax.set_xlabel("Position Error (mm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_title("V18 — CDF Position Error", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] CDF → {save_path}")
    plt.close()


def plot_trajectories(positions_dict, gt_xy, save_path):
    labels = list(METHODS.keys())
    n      = len(labels)
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes = axes.flatten()
    for ax, label in zip(axes, labels):
        pos   = positions_dict.get(label, np.empty((0, 2)))
        color = COLORS.get(label, 'gray')
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'k--', lw=2, label='GT', alpha=0.4)
        if len(pos) > 0:
            ax.plot(pos[:, 0], pos[:, 1], color=color, lw=0.8, alpha=0.75, label=label)
        for nm, pt in zip(["A", "B", "C", "D"], WAYPOINTS[:4]):
            ax.scatter(*pt, s=70, color='black', zorder=10)
            ax.annotate(nm, pt, textcoords="offset points",
                        xytext=(6, 4), fontsize=12, fontweight='bold')
        for j, (ax_, ay_) in enumerate(ANCHORS):
            ax.scatter(ax_, ay_, s=90, marker='s', color='red', zorder=10)
            ax.annotate(f"A{j+1}", (ax_, ay_), textcoords="offset points",
                        xytext=(5, 5), fontsize=9, color='red')
        if len(pos) > 0:
            errs = nearest_gt_error(pos, gt_xy)
            rmse = np.sqrt(np.mean(errs**2))
            ax.set_title(f"{label}\nRMSE={rmse:.1f}mm", fontsize=11, fontweight='bold')
        else:
            ax.set_title(label, fontsize=11)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, ls='--', alpha=0.3)
    # Ẩn subplot thừa
    for ax in axes[n:]:
        ax.set_visible(False)
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
        vals   = [metrics_dict[m][mname] for m in methods]
        colors = [COLORS.get(m, 'gray') for m in methods]
        bars   = ax.bar(range(len(methods)), vals, color=colors,
                        alpha=0.85, edgecolor='white', lw=1.5)
        for bar, val in zip(bars, vals):
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
    plt.suptitle("V18 — Method Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  V18: Robust & Adaptive UKF — UWB Indoor Positioning")
    print("=" * 70)
    print("  1. Raw+WLS      — Weighted Least Squares (baseline)")
    print("  2. Huber-UKF    — IRLS với Huber M-estimator")
    print("  3. MCC-UKF      — Maximum Correntropy Criterion UKF")
    print("  4. AEKF-IAE     — Sage-Husa adaptive R (Q cố định)")
    print("  5. VBAKF        — Variational Bayesian Adaptive KF")
    print("  6. PC-UKF-2D    — Pairwise Consensus UKF (đã tuning V15)")
    print("=" * 70)

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
    total_path = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"  Path={total_path:.0f}mm  Time={total_time:.1f}s  GT={len(gt_xy)} points")

    # ── Grid search ────────────────────────────────────────────────────
    if DO_GRID_SEARCH:
        print("\n  [Grid Search] Tuning tất cả filter (trừ PC-UKF-2D)...")

        best_huber = grid_search_huber(eval_files, gt_xy)
        best_mcc   = grid_search_mcc(eval_files, gt_xy)
        best_aekf  = grid_search_aekf(eval_files, gt_xy)
        best_vbakf = grid_search_vbakf(eval_files, gt_xy)

        # Cập nhật METHODS với params tốt nhất
        METHODS["Huber-UKF"] = (HuberUKF, best_huber)
        METHODS["MCC-UKF"]   = (MCCUKF,   best_mcc)
        METHODS["AEKF-IAE"]  = (AEKF,     best_aekf)
        METHODS["VBAKF"]     = (VBAKF,    best_vbakf)
        # PC-UKF-2D: KHÔNG thay đổi (đã tuning từ V15)

    # ── Evaluate ──────────────────────────────────────────────────────
    errors, positions = evaluate_files(eval_files, gt_xy)

    # ── Summary table ─────────────────────────────────────────────────
    metrics = {}
    print(f"\n{'═' * 80}")
    print(f"  SUMMARY TABLE (mm)")
    print(f"{'═' * 80}")
    print(f"  {'Method':<14s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'CEP90':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─' * 65}")
    for label, errs in errors.items():
        if len(errs) == 0:
            continue
        m = compute_metrics(errs)
        metrics[label] = m
        print(f"  {label:<14s} {m['rmse']:>7.1f} {m['mae']:>7.1f}"
              f" {m['cep50']:>7.1f} {m['cep90']:>7.1f} {m['p95']:>7.1f} {m['max']:>7.1f}")

    # ── Wilcoxon vs PC-UKF-2D ─────────────────────────────────────────
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
    plot_cdf(errors, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(positions, gt_xy, os.path.join(SAVE_DIR, 'trajectories.png'))
    plot_bar(metrics,  os.path.join(SAVE_DIR, 'bar_comparison.png'))

    # ── Save errors ───────────────────────────────────────────────────
    for label, errs in errors.items():
        safe = label.lower().replace('+', '_').replace('-', '_').replace(' ', '_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errs)

    print(f"\n[✓] Kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()