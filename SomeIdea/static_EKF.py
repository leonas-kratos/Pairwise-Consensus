# -*- coding: utf-8 -*-
"""
sota_MovingLess.py  —  Static (MovingLess) experiment
======================================================
Đọc file dạng:  data{LOS|NLOS}_MovingLess_{x}_{y}.txt
GT point lấy từ tên file: (x, y) mm

Vẽ 4 figure (giống paper Fig 16-19):
  - Fig A : Scatter trajectory  — LOS
  - Fig B : Error theo thời gian — LOS
  - Fig C : Scatter trajectory  — NLOS
  - Fig D : Error theo thời gian — NLOS

Dùng lại toàn bộ filter classes từ sota.py (copy nguyên).
"""

import os
import re
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

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
ANCHORS = np.array([
    [0,    0   ],   # A1
    [0,    4720],   # A2
    [6300, 4720],   # A3
    [6300, 0   ],   # A4
], dtype=float)

ANCHOR_HEIGHT = 1400.0
N_ANCHORS     = 4

# ─── UKF common ──────────────────────────────────────────────────────
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

# ─── Standard UKF ────────────────────────────────────────────────────
UKF_STD_Q = 0.01
UKF_STD_R = 50.0

# ─── Huber-UKF ───────────────────────────────────────────────────────
HUBER_Q       = 0.01
HUBER_R       = 50.0
HUBER_DELTA   = 20.0
HUBER_MAXITER = 5

# ─── MCC-UKF ─────────────────────────────────────────────────────────
MCC_Q         = 0.01
MCC_R         = 50.0
MCC_KERNEL_BW = 1700.0
MCC_MAXITER   = 5

# ─── PC-UKF-v3 ───────────────────────────────────────────────────────
PCUKF_Q       = 0.01
PCUKF_R_BASE  = 50.0
PCUKF_R_SCALE = 2.0

# ─── GUKF ────────────────────────────────────────────────────────────
GUKF_Q      = 0.01
GUKF_R      = 50.0
GUKF_SIGMA  = 5.0
GUKF_N_HALF = 4

DATA_DIR = "./data"
SAVE_DIR = "./outputs_static"

COLORS = {
    "Raw+LS"    : '#9E9E9E',
    "UKF"       : '#00BCD4',
    "Huber-UKF" : '#E91E63',
    "MCC-UKF"   : '#FF9800',
    "PC-UKF-v3" : '#9C27B0',
    "GUKF"      : '#4CAF50',
}


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
    Pzz_diag    : (N,)  — diagonal of Pzz (before adding R)
    N           : int   — number of anchors (compile-time constant)
    r_base      : float — base measurement noise

    Bước 1: std_innov[i] = innovations[i] / sqrt(Pzz_diag[i])
    Bước 2: MAD normalize (sort-based, không dùng np.median)
    Bước 3: T-kernel pairwise (sigma=1 cố định)
    """
    # Bước 1: geo-normalize
    std_innov = np.empty(N)
    for i in range(N):
        denom = Pzz_diag[i]
        if denom < 1e-9: denom = 1e-9
        std_innov[i] = innovations[i] / (denom ** 0.5)

    # Bước 2: MAD normalize (sort-based)
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


def _warmup_numba():
    """Pre-compile tất cả @njit kernels tại import time."""
    _d = np.array([1000.0, 1200.0, 800.0, 900.0])
    _a = np.array([[0.0, 0.0], [0.0, 4720.0], [6300.0, 4720.0], [6300.0, 0.0]])
    _ls_position_nb(_d, _a)
    _ls_file_nb(np.stack([_d]), _a)
    _innov = np.array([10.0, -5.0, 200.0, 3.0])
    _pzz   = np.array([60.0, 60.0, 60.0, 60.0])
    _pc_scores_v3_nb(_innov, _pzz, 4, 50.0)


_warmup_numba()

# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d_slant):
    return math.sqrt(max(d_slant**2 - ANCHOR_HEIGHT**2, 0.0))


def h_obs(state, anchors=ANCHORS):
    x, y = state[0], state[1]
    return np.array([
        math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
        for ax, ay in anchors
    ])


def LS_position(distances, anchors=ANCHORS):
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi - x0), 2*(yi - y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
        w.append(1.0)
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    W  = np.diag(w)
    try:
        pos, *_ = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


# ══════════════════════════════════════════════════════════════════════
#  UKF UTILITIES
# ══════════════════════════════════════════════════════════════════════
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
    return pos if not np.any(np.isnan(pos)) else np.array([3150.0, 2360.0])


# ══════════════════════════════════════════════════════════════════════
#  FILTER CLASSES  (copy từ sota.py)
# ══════════════════════════════════════════════════════════════════════
class StandardUKF:
    def __init__(self, q=UKF_STD_Q, r=UKF_STD_R):
        self.n     = 2
        self.Q_mat = np.eye(2) * q
        self.R_mat = np.eye(N_ANCHORS) * r
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = self.P = None

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
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


class HuberUKF:
    def __init__(self, q=HUBER_Q, r=HUBER_R,
                 delta=HUBER_DELTA, maxiter=HUBER_MAXITER):
        self.n       = 2
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r
        self.delta   = delta
        self.maxiter = maxiter
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = self.P = None

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
            hub_w    = np.where(np.abs(r_scaled) <= self.delta, 1.0,
                                self.delta / (np.abs(r_scaled) + 1e-9))
            hub_w = np.maximum(hub_w, 1e-4)
            R_eff = np.diag(self.R_base / hub_w)
        Pzz_eff = Pzz_no_R + R_eff
        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ (z_raw - z_hat)
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


class MCCUKF:
    def __init__(self, q=MCC_Q, r=MCC_R,
                 kernel_bw=MCC_KERNEL_BW, maxiter=MCC_MAXITER):
        self.n         = 2
        self.Q_mat     = np.eye(2) * q
        self.R_base    = r
        self.kernel_bw = kernel_bw
        self.maxiter   = maxiter
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = self.P = None

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
                x_cur = x_new; break
            x_cur = x_new
        self.x = x_cur
        self.P = make_spd(P_pred - K @ (Pzz_no_R + R_eff) @ K.T)
        return self.x.copy()


def _t_kernel_v3(diff_vec, sigma=1.0):
    nu  = 4.0
    eps = 1e-9
    return (1.0 + diff_vec**2 / (nu * sigma**2 + eps)) ** (-(nu + 1.0) / 2.0)


def pc_scores_v3(innovations, Pzz_diag):
    std_innov = innovations / np.sqrt(np.maximum(Pzz_diag, 1e-9))
    med    = np.median(std_innov)
    mad    = np.median(np.abs(std_innov - med))
    normed = std_innov / (1.4826 * mad + 1e-9)
    scores = np.zeros(N_ANCHORS)
    for i in range(N_ANCHORS):
        diffs     = np.array([normed[i] - normed[j]
                              for j in range(N_ANCHORS) if j != i])
        scores[i] = np.mean(_t_kernel_v3(diffs, sigma=1.0))
    return scores


class PCUKFv3:
    def __init__(self, q=PCUKF_Q, r_base=PCUKF_R_BASE, r_scale=PCUKF_R_SCALE):
        self.n       = 2
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.Wm, self.Wc, self.c = ukf_weights(self.n)
        self.x = self.P = None

    def init(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.eye(self.n) * 1e6

    def _get_z_hat_and_Pzz_diag(self):
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
        x_pred = self.x.copy()
        P_pred = self.P + self.Q_mat
        self.x = x_pred; self.P = P_pred
        z_hat, Pzz_diag = self._get_z_hat_and_Pzz_diag()
        innov   = z_raw - z_hat
        scores  = pc_scores_v3(innov, Pzz_diag)
        R_diag  = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        R_mat   = np.diag(R_diag)
        pts   = sigma_points(x_pred, P_pred, self.c)
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*self.n + 1)])
        z_hat_full = self.Wm @ Z_pts
        Pzz = R_mat.copy()
        Pxz = np.zeros((self.n, N_ANCHORS))
        for i in range(2*self.n + 1):
            dz   = Z_pts[i] - z_hat_full
            dx   = pts[i] - x_pred
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ (z_raw - z_hat_full)
        self.P = make_spd(P_pred - K @ Pzz @ K.T)
        return self.x.copy()


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
        self.x = self.P = None
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
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════
#  METHODS TABLE
# ══════════════════════════════════════════════════════════════════════
METHODS = {
    "Raw+LS"    : None,
    "UKF"       : (StandardUKF, dict(q=UKF_STD_Q, r=UKF_STD_R)),
    "Huber-UKF" : (HuberUKF,   dict(q=HUBER_Q,    r=HUBER_R,   delta=HUBER_DELTA)),
    "MCC-UKF"   : (MCCUKF,     dict(q=MCC_Q,      r=MCC_R,     kernel_bw=MCC_KERNEL_BW)),
    "PC-UKF-v3" : (PCUKFv3,    dict(q=PCUKF_Q,    r_base=PCUKF_R_BASE, r_scale=PCUKF_R_SCALE)),
    "GUKF"      : (GUKF,       dict(q=GUKF_Q,     r=GUKF_R,    sigma=GUKF_SIGMA, n_half=GUKF_N_HALF)),
}


# ══════════════════════════════════════════════════════════════════════
#  PARSE
# ══════════════════════════════════════════════════════════════════════
def parse_file(path):
    """
    Đọc file txt dạng space-separated với header nhiều dòng.
    Format:
      Time(s)   Frame      X(mm)      Y(mm)      [S0 S1 S2 S3]   ← header data
      0.002       1     -382.3      718.1 ...
    Trả về mảng X,Y positions shape (T, 2) — KHÔNG phải distances.
    (Các file này đã là output của filter, không có raw distances.)
    """
    rows = []
    in_data = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Detect header separator line
            if line.startswith('---'):
                in_data = True
                continue
            # Skip header/meta lines before data
            if not in_data:
                continue
            # Skip [EKF] init line
            if line.startswith('['):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[2])
                y = float(parts[3])
                rows.append([x, y])
            except ValueError:
                continue
    return np.array(rows, dtype=np.float64) if rows else None


def gt_from_filename(path):
    """
    Lấy GT point từ tên file — lấy 2 số CUỐI cùng trong tên.
    Ví dụ: Static_EKF_0_800.txt       →  (0, 800)
            Static_PC_EKF_6337_3812.txt → (6337, 3812)
    """
    name = os.path.basename(path)
    # Lấy tất cả số trong tên file, 2 số cuối là GT
    nums = re.findall(r'-?\d+', os.path.splitext(name)[0])
    if len(nums) >= 2:
        return np.array([float(nums[-2]), float(nums[-1])])
    return None


def method_from_filename(path):
    """
    Phân loại method từ tên file (ưu tiên) hoặc tên thư mục cha.
    Thứ tự check: PC_EKF trước EKF để tránh nhầm.
    Static_PC_EKF_*.txt → 'PC-EKF'
    Static_EKF_*.txt    → 'EKF'
    Static_KF_*.txt     → 'KF'
    Static_LS_*.txt     → 'LS'
    """
    name   = os.path.basename(path).upper()
    parent = os.path.basename(os.path.dirname(path)).upper()
    for src in (name, parent):
        if 'PC_EKF' in src or 'PC-EKF' in src:
            return 'PC-EKF'
        if 'EKF' in src:
            return 'EKF'
        if 'KF' in src:
            return 'KF'
        if 'LS' in src:
            return 'LS'
    return os.path.splitext(os.path.basename(path))[0]


# ══════════════════════════════════════════════════════════════════════
#  RUN FILTER ON ONE FILE
# ══════════════════════════════════════════════════════════════════════
def run_one_file(path, method_name):
    """
    Đọc file XY (đã qua filter), trả về pos_array (T,2) và GT point.
    method_name: tên hiển thị (lấy từ tên thư mục cha).
    """
    pos = parse_file(path)
    if pos is None or len(pos) == 0:
        print(f"  [!] Không đọc được data: {path}")
        return None, None

    gt = gt_from_filename(path)
    if gt is None:
        print(f"  [!] Không đọc được GT từ tên file: {path}")
        return None, None

    return pos, gt


# ══════════════════════════════════════════════════════════════════════
#  AGGREGATE ACROSS FILES (cùng nhóm method)
# ══════════════════════════════════════════════════════════════════════
def aggregate_files(file_list, method_name):
    """
    Đọc tất cả file trong một nhóm method, gộp errors lại.
    Trả về:
      all_pos   : np.array (N_total, 2)
      all_errors: np.array (N_total,)
      per_file  : list of (pos, gt, T, fname)
    """
    all_pos    = []
    all_errors = []
    per_file   = []

    for path in file_list:
        pos, gt = run_one_file(path, method_name)
        if pos is None:
            continue
        valid = pos[~np.any(np.isnan(pos), axis=1)]
        errs  = np.linalg.norm(valid - gt, axis=1)
        all_pos.extend(valid.tolist())
        all_errors.extend(errs.tolist())
        per_file.append((pos, gt, len(pos), os.path.basename(path)))

    return (np.array(all_pos)   if all_pos    else np.empty((0, 2))),\
           (np.array(all_errors) if all_errors else np.array([])),\
           per_file


# ══════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════
def print_metrics(errors_by_method, per_file_by_method):
    """
    In RMSE tổng và RMSE từng GT point cho mỗi method.
    """
    print(f"\n{'═'*72}")
    print(f"  STATIC RMSE SUMMARY  (mm)")
    print(f"{'═'*72}")
    for method_name, per_file in per_file_by_method.items():
        errs_all = errors_by_method[method_name]
        if len(errs_all) == 0:
            print(f"\n  [{method_name}]  (không có dữ liệu)")
            continue
        overall_rmse = np.sqrt(np.mean(errs_all**2))
        print(f"\n  [{method_name}]  overall RMSE = {overall_rmse:.1f} mm  (N={len(errs_all)})")
        print(f"  {'File':<35s} {'GT':>18s} {'N':>5s} {'RMSE':>8s} {'MAE':>8s} {'P95':>8s}")
        print(f"  {'─'*65}")
        for pos, gt, T, fname in per_file:
            valid = pos[~np.any(np.isnan(pos), axis=1)]
            errs  = np.linalg.norm(valid - gt, axis=1)
            if len(errs) == 0:
                continue
            rmse = np.sqrt(np.mean(errs**2))
            mae  = np.mean(errs)
            p95  = np.percentile(errs, 95)
            gt_str = f"({gt[0]:.0f}, {gt[1]:.0f})"
            print(f"  {fname:<35s} {gt_str:>18s} {len(errs):>5d} {rmse:>8.1f} {mae:>8.1f} {p95:>8.1f}")


# ══════════════════════════════════════════════════════════════════════
#  PLOT 1 — Scatter trajectory  (giống Fig 16 / Fig 17)
# ══════════════════════════════════════════════════════════════════════
def plot_scatter(all_pos, per_file, env_name, save_path):
    """
    Vẽ scatter tất cả điểm ước lượng từ tất cả file.
    GT points hiển thị dạng ngôi sao đỏ.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, pos in all_pos.items():
        if len(pos) == 0:
            continue
        errs = []
        for results, gt, T, fname in per_file:
            if name not in results:
                continue
            p = results[name]
            valid = p[~np.any(np.isnan(p), axis=1)]
            errs.extend(np.linalg.norm(valid - gt, axis=1))
        rmse  = np.sqrt(np.mean(np.array(errs)**2)) if errs else float('nan')
        color = COLORS.get(name, 'gray')
        ms    = 3 if name != "GUKF" else 4
        alpha = 0.35 if name != "GUKF" else 0.55
        ax.scatter(pos[:, 0], pos[:, 1],
                   s=ms, color=color, alpha=alpha,
                   label=f"{name}  RMSE={rmse:.1f}mm")

    # Vẽ GT points
    gt_pts = np.array([gt for _, gt, _, _ in per_file])
    ax.scatter(gt_pts[:, 0], gt_pts[:, 1],
               s=120, marker='*', color='red', zorder=10,
               label='Ground Truth')

    # Vẽ anchors
    for j, (ax_, ay_) in enumerate(ANCHORS):
        ax.scatter(ax_, ay_, s=80, marker='s', color='black', zorder=10)
        ax.annotate(f"A{j+1}", (ax_, ay_),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=9, color='black')

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(f"Static Positioning Trajectory — {env_name}", fontsize=13)
    ax.legend(fontsize=8, loc='best', markerscale=3)
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Scatter → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 2 — Error theo thời gian  (giống Fig 18 / Fig 19)
# ══════════════════════════════════════════════════════════════════════
def plot_error_time(per_file, env_name, save_path):
    """
    Vẽ sai số Euclidean theo thời gian (sample index).
    Nếu có nhiều file, ghép nối tiếp nhau.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    # Tính error time-series cho từng method
    error_series = {m: [] for m in METHODS}
    for results, gt, T, fname in per_file:
        for name, pos in results.items():
            errs = np.linalg.norm(pos - gt, axis=1)
            errs[np.any(np.isnan(pos), axis=1)] = np.nan
            error_series[name].extend(errs.tolist())

    t_axis = np.arange(len(next(iter(error_series.values()))))

    for name, errs in error_series.items():
        errs = np.array(errs)
        color = COLORS.get(name, 'gray')
        lw    = 1.8 if name == "GUKF" else 1.0
        alpha = 0.9 if name == "GUKF" else 0.6
        ax.plot(t_axis, errs, color=color, lw=lw, alpha=alpha, label=name)

    ax.set_xlabel("Sample index", fontsize=12)
    ax.set_ylabel("Position Error (mm)", fontsize=12)
    ax.set_title(f"Static Positioning Error — {env_name}", fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] Error-time → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  sota_static.py  —  Static UWB Positioning RMSE")
    print("=" * 65)

    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.isdir(DATA_DIR):
        print(f"\n[!] Không tìm thấy thư mục '{DATA_DIR}/'")
        return

    # Glob đệ quy tất cả Static_*.txt (cả subfolder lẫn root)
    all_files = sorted(set(
        glob.glob(os.path.join(DATA_DIR, "**", "Static_*.txt"), recursive=True) +
        glob.glob(os.path.join(DATA_DIR, "Static_*.txt"))
    ))
    if not all_files:
        print(f"\n[!] Không tìm thấy file Static_*.txt trong '{DATA_DIR}/'")
        return

    # Group theo method từ tên file
    from collections import defaultdict as _dd
    files_by_method = _dd(list)
    for f in all_files:
        files_by_method[method_from_filename(f)].append(f)

    method_order = sorted(files_by_method.keys(),
                          key=lambda m: {'LS': 0, 'KF': 1, 'EKF': 2, 'PC-EKF': 3}.get(m, 9))

    print(f"\n  Tìm thấy {len(files_by_method)} nhóm: {method_order}")

    errors_by_method   = {}
    pos_by_method      = {}
    per_file_by_method = {}

    for method_name in method_order:
        files = files_by_method[method_name]
        print(f"\n{'─'*65}")
        print(f"  [{method_name}]  {len(files)} file(s):")
        for f in files:
            gt = gt_from_filename(f)
            print(f"    {os.path.basename(f)}  GT={gt}")

        all_pos, all_errors, per_file = aggregate_files(files, method_name)
        errors_by_method[method_name]   = all_errors
        pos_by_method[method_name]      = all_pos
        per_file_by_method[method_name] = per_file


    if not errors_by_method:
        print("\n[!] Không đọc được dữ liệu nào.")
        return

    print_metrics(errors_by_method, per_file_by_method)

    # ── Collect all unique GT points & methods ─────────────────────────
    cmap = plt.cm.tab10.colors
    method_names  = list(per_file_by_method.keys())
    method_colors = {m: cmap[i % len(cmap)] for i, m in enumerate(method_names)}

    # Build mapping: gt_key → {method: pos_array}
    from collections import defaultdict
    gt_data = defaultdict(dict)   # { (gx,gy): {method_name: pos} }
    for method_name, per_file in per_file_by_method.items():
        for pos, gt, T, fname in per_file:
            key = (int(gt[0]), int(gt[1]))
            gt_data[key][method_name] = pos

    gt_keys = sorted(gt_data.keys())
    n_pts   = len(gt_keys)

    # ── Per-file scatter: grid layout ─────────────────────────────────
    ncols = min(n_pts, 2)
    nrows = math.ceil(n_pts / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle("Static Positioning — Scatter per GT Point", fontsize=14, y=1.01)

    for idx, gt_key in enumerate(gt_keys):
        ax  = axes[idx // ncols][idx % ncols]
        gx, gy = gt_key

        for method_name, pos in gt_data[gt_key].items():
            valid = pos[~np.any(np.isnan(pos), axis=1)]
            errs  = np.linalg.norm(valid - np.array([gx, gy]), axis=1)
            rmse  = np.sqrt(np.mean(errs**2)) if len(errs) > 0 else float('nan')
            color = method_colors[method_name]
            ax.scatter(valid[:, 0], valid[:, 1],
                       s=4, color=color, alpha=0.45,
                       label=f"{method_name}  {rmse:.1f}mm")

        # GT star
        ax.scatter(gx, gy, s=180, marker='*', color='red',
                   zorder=10, label=f"GT ({gx}, {gy})")

        # Anchors
        for j, (ax_, ay_) in enumerate(ANCHORS):
            ax.scatter(ax_, ay_, s=60, marker='s', color='black', zorder=9)
            ax.annotate(f"A{j+1}", (ax_, ay_),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color='black')

        ax.set_title(f"GT = ({gx}, {gy}) mm", fontsize=11)
        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)
        ax.legend(fontsize=8, loc='best', markerscale=3)
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_aspect('equal')

    # Ẩn axes thừa
    for idx in range(n_pts, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    scatter_path = os.path.join(SAVE_DIR, "scatter_per_gt.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"\n[✓] Scatter per GT → {scatter_path}")
    plt.close()

    # ── Error-time: grid layout, mỗi ô = 1 GT point ───────────────────
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle("Static Positioning — Error over Time per GT Point", fontsize=14, y=1.01)

    for idx, gt_key in enumerate(gt_keys):
        ax = axes[idx // ncols][idx % ncols]
        gx, gy = gt_key

        for method_name, pos in gt_data[gt_key].items():
            errs = np.linalg.norm(pos - np.array([gx, gy]), axis=1)
            errs[np.any(np.isnan(pos), axis=1)] = np.nan
            rmse  = np.sqrt(np.nanmean(errs**2))
            color = method_colors[method_name]
            ax.plot(errs, color=color, lw=1.2, alpha=0.85,
                    label=f"{method_name}  {rmse:.1f}mm")

        ax.set_title(f"GT = ({gx}, {gy}) mm", fontsize=11)
        ax.set_xlabel("Sample index", fontsize=10)
        ax.set_ylabel("Error (mm)", fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_ylim(bottom=0)

    for idx in range(n_pts, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    errtime_path = os.path.join(SAVE_DIR, "error_time_per_gt.png")
    plt.savefig(errtime_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Error-time per GT → {errtime_path}")
    plt.close()

    print(f"\n[✓] Tất cả kết quả lưu tại '{SAVE_DIR}/'")



if __name__ == "__main__":
    main()
