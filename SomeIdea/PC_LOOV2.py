# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — V22: thêm Pure Dead Reckoning method
=============================================================================
Thay đổi so với V21:
  - Thêm class PureDR: chỉ dùng v_mm_s + heading_deg, không dùng UWB
  - PureDR khởi tạo cố định tại (800, 400) thay vì dùng LS từ UWB
  - Thêm vào METHODS, COLORS, evaluate_files, loo_tune_and_eval
  - DR-only cho phép đánh giá xem IMU/odometry có đủ tin cậy không

CÁC PHƯƠNG PHÁP:
  0. DR-only      — Pure Dead Reckoning (không UWB), khởi tạo (800, 400)
  1. Raw + LS     — baseline
  2. UKF          — Unscented Kalman Filter + DR motion model
  3. Huber-UKF    — IRLS Huber M-estimator + DR motion model
  4. MCC-UKF      — Maximum Correntropy Criterion UKF + DR motion model
  5. PC-UKF-2D    — Pairwise Consensus V3 + DR motion model
  6. GUKF         — Gaussian-smoothed UKF + DR motion model

Format file .txt:
  timestamp, d0_slant, d1_slant, d2_slant, d3_slant, v_mm_s, heading_deg
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

SPEED      = 200.0    # mm/s — dùng để build GT
GT_SPACING = 5.0      # mm   — dùng để build GT

N_ANCHORS  = 4

# ─── DR-only init position ───────────────────────────────────────────
DR_INIT_POS = np.array([800.0, 400.0])   # ← khởi tạo tại (800, 400)

# ─── Standard UKF ────────────────────────────────────────────────────
UKF_STD_Q = 0.001
UKF_STD_R = 100.0

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

# ─── PC-UKF-2D ───────────────────────────────────────────────────────
PCUKF_Q       = 0.01
PCUKF_R_BASE  = 50.0
PCUKF_R_SCALE = 15.0
PCUKF_SIGMA   = 300.0

# ─── GUKF (Gaussian-smoothed UKF) ────────────────────────────────────
GUKF_Q      = 0.01
GUKF_R      = 50.0
GUKF_SIGMA  = 5.0
GUKF_N_HALF = 4

# ─── UKF common ──────────────────────────────────────────────────────
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

# ─── Grid search flags ───────────────────────────────────────────────
DO_GRID_SEARCH     = False
DO_LOO_GRID_SEARCH = False

DATA_DIR = "./data"
SAVE_DIR = "./outputs_LOO_V2"

# ─── Motion sanity check ─────────────────────────────────────────────
MOTION_RATIO_MIN = 1.00

# ─── Shared fixed params cho grid search — fair comparison ───────────
FIXED_Q = 0.001
FIXED_R = 100.0

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
#  DEAD RECKONING HELPER
# ══════════════════════════════════════════════════════════════════════
def dr_displacement(v_mm_s, heading_deg, dt):
    """
    Tính vector dịch chuyển từ vận tốc và góc heading.
    heading_deg: 0° = +X, 90° = +Y, âm = CCW
    dt: thời gian bước (giây)
    Trả về (dx, dy) tính bằng mm.
    """
    rad = math.radians(-heading_deg)
    dx  = v_mm_s * dt * math.cos(rad)
    dy  = v_mm_s * dt * math.sin(rad)
    return dx, dy


# ══════════════════════════════════════════════════════════════════════
#  0. PURE DEAD RECKONING  (không dùng UWB)
# ══════════════════════════════════════════════════════════════════════
class PureDR:
    """
    Chỉ dùng v_mm_s + heading_deg để tích phân vị trí.
    Không dùng bất kỳ đo lường UWB nào.
    Khởi tạo cố định tại DR_INIT_POS = (800, 400).

    Dùng để kiểm tra: nếu DR-only cho quỹ đạo đúng hình dạng
    thì v/heading từ odometry đáng tin cậy.
    """
    def __init__(self, x0_init=None):
        # x0_init cho phép override nếu cần, mặc định dùng DR_INIT_POS
        self.x0_init = (np.array(x0_init, dtype=float)
                        if x0_init is not None
                        else DR_INIT_POS.copy())
        self.x = None

    def init(self, x0=None):
        # Bỏ qua x0 từ LS — luôn khởi tạo tại x0_init cố định
        self.x = self.x0_init.copy()

    def step(self, row, dt):
        """
        row: [d0, d1, d2, d3, v_mm_s, heading_deg, timestamp_s]
        dt : thời gian bước thực tế (giây)
        Chỉ đọc row[4] (v) và row[5] (heading), bỏ qua UWB.
        """
        if self.x is None:
            return np.full(2, np.nan)
        v       = row[4]
        heading = row[5]
        dx, dy  = dr_displacement(v, heading, dt)
        self.x  = self.x + np.array([dx, dy])
        return self.x.copy()


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


def default_init(row0):
    pos = LS_position(row0[:4])
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


# ══════════════════════════════════════════════════════════════════════
#  1. STANDARD UKF  (+ DR motion model)
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

    def step(self, row, dt):
        if self.x is None:
            return np.full(2, np.nan)
        z_raw   = row[:4]
        v       = row[4]
        heading = row[5]
        dx, dy  = dr_displacement(v, heading, dt)
        x_pred  = self.x + np.array([dx, dy])
        P_pred  = self.P + self.Q_mat
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
#  3. HUBER-UKF  (+ DR motion model)
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

    def step(self, row, dt):
        if self.x is None:
            return np.full(2, np.nan)
        z_raw   = row[:4]
        v       = row[4]
        heading = row[5]
        dx, dy  = dr_displacement(v, heading, dt)
        x_pred  = self.x + np.array([dx, dy])
        P_pred  = self.P + self.Q_mat
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
#  4. MCC-UKF  (+ DR motion model)
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

    def step(self, row, dt):
        if self.x is None:
            return np.full(2, np.nan)
        z_raw   = row[:4]
        v       = row[4]
        heading = row[5]
        dx, dy  = dr_displacement(v, heading, dt)
        x_pred  = self.x + np.array([dx, dy])
        P_pred  = self.P + self.Q_mat
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
#  5. PC-UKF-2D  (V3 — MAD Auto-Normalized + DR motion model)
# ══════════════════════════════════════════════════════════════════════
def _pc_scores_v3(innovations, Pzz_diag):
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

    def _z_hat_and_Pzz_diag(self, x_pred, P_pred):
        pts   = sigma_points(x_pred, P_pred, self.c)
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*self.n + 1)])
        z_hat = self.Wm @ Z_pts
        Pzz_diag = np.full(N_ANCHORS, self.R_base)
        for i in range(2*self.n + 1):
            dz = Z_pts[i] - z_hat
            Pzz_diag += self.Wc[i] * dz**2
        return z_hat, Pzz_diag

    def step(self, row, dt):
        if self.x is None:
            return np.full(2, np.nan)
        z_raw   = row[:4]
        v       = row[4]
        heading = row[5]
        dx, dy  = dr_displacement(v, heading, dt)
        x_pred  = self.x + np.array([dx, dy])
        P_pred  = self.P + self.Q_mat
        self.x  = x_pred
        self.P  = P_pred
        z_hat, Pzz_diag = self._z_hat_and_Pzz_diag(x_pred, P_pred)
        innov  = z_raw - z_hat
        scores = _pc_scores_v3(innov, Pzz_diag)
        R_diag = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        pts   = sigma_points(x_pred, P_pred, self.c)
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*self.n + 1)])
        z_hat = self.Wm @ Z_pts
        R   = np.diag(R_diag)
        Pzz = R.copy()
        Pxz = np.zeros((self.n, N_ANCHORS))
        for i in range(2*self.n + 1):
            dz   = Z_pts[i] - z_hat
            dx_s = pts[i] - x_pred
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx_s, dz)
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
#  6. GUKF — Gaussian Unscented Kalman Filter  (+ DR motion model)
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

    def step(self, row, dt):
        if self.x is None:
            return np.full(2, np.nan)
        z_raw   = row[:4]
        v       = row[4]
        heading = row[5]
        z_smooth = self._smooth(z_raw)
        dx, dy  = dr_displacement(v, heading, dt)
        x_pred  = self.x + np.array([dx, dy])
        P_pred  = self.P + self.Q_mat
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
def run_filter(filt_class, data, **kwargs):
    """
    data: numpy array shape (T, 7):
      [d0, d1, d2, d3, v_mm_s, heading_deg, timestamp_s]

    PureDR: init() bỏ qua x0 từ LS, dùng DR_INIT_POS.
    Các filter khác: init(x0) với x0 từ LS.
    """
    T    = len(data)
    filt = filt_class(**kwargs)
    pos  = np.full((T, 2), np.nan)

    # PureDR dùng x0_init cố định, không cần LS
    x0 = default_init(data[0])
    filt.init(x0)

    timestamps = data[:, 6]
    dts        = np.diff(timestamps)
    dt_median  = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 0.1

    for t in range(T):
        if t == 0:
            dt = dt_median
        else:
            dt = timestamps[t] - timestamps[t - 1]
            if dt <= 0 or dt > 10.0:
                dt = dt_median
        pos[t] = filt.step(data[t], dt)
    return pos


# ══════════════════════════════════════════════════════════════════════
#  METHODS TABLE
# ══════════════════════════════════════════════════════════════════════
METHODS = {
    "DR-only"   : (PureDR,     dict()),                          # ← MỚI
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
    "DR-only"   : '#795548',   # nâu đất                        # ← MỚI
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
    """
    Đọc file .txt → array shape (T, 7):
      [d0_ground, d1_ground, d2_ground, d3_ground, v_mm_s, heading_deg, timestamp_s]

    Format: timestamp, d0_slant, d1_slant, d2_slant, d3_slant, v_mm_s, heading_deg
    Auto-detect đơn vị timestamp (ms vs s).
    """
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
                ts          = float(parts[0])
                d_slant     = [float(parts[i + 1]) for i in range(4)]
                d_ground    = [slant_to_ground(d) for d in d_slant]
                v_mm_s      = float(parts[5])
                heading_deg = float(parts[6])
                rows.append(d_ground + [v_mm_s, heading_deg, ts])
            except ValueError:
                continue
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float64)

    ts_col = arr[:, 6]
    if len(ts_col) >= 2:
        raw_dts   = np.diff(ts_col)
        median_dt = float(np.median(raw_dts[raw_dts > 0])) if np.any(raw_dts > 0) else 0
        if median_dt > 60.0:
            arr[:, 6] = ts_col / 1000.0

    return arr


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
    timing        = {m: [] for m in METHODS if m not in ("Raw+LS", "DR-only")}

    hdr = f"{'File':<18s}"
    for m in METHODS:
        hdr += f" {m:>12s}"
    print("\n" + "═" * 90)
    print("  PER-FILE RMSE (mm)   [⚠ = trajectory collapsed / static]")
    print("═" * 90)
    print(hdr)
    print("─" * 90)

    for path in file_paths:
        data = parse_file(path)
        if data is None:
            continue
        T   = len(data)
        row = f"{os.path.basename(path):<18s}"

        for name, method in METHODS.items():
            if method is None:
                # Raw+LS
                pos = np.array([LS_position(data[t, :4]) for t in range(T)])
            else:
                filt_cls, kwargs = method
                t0  = time.perf_counter()
                pos = run_filter(filt_cls, data, **kwargs)
                elapsed = time.perf_counter() - t0
                if name in timing:
                    timing[name].append(elapsed)

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
    deltas = np.logspace(-1, 2, 10).tolist()
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
    r_scales = np.logspace(0, 2, 10).tolist()
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
    grid = {
        'sigma'  : np.logspace(np.log10(0.3), np.log10(20), 3).tolist(),
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
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, errors in errors_dict.items():
        if len(errors) == 0:
            continue
        s    = np.sort(errors)
        cdf  = np.arange(1, len(s) + 1) / len(s)
        rmse = np.sqrt(np.mean(errors**2))
        c    = COLORS.get(label, 'gray')
        lw   = 2.5 if label not in ("Raw+LS", "DR-only") else 1.5
        ls   = '-'
        if label == "Raw+LS":
            ls = '--'
        elif label == "DR-only":
            ls = '-.'
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

    # Đánh dấu điểm khởi tạo DR-only
    ax.scatter(*DR_INIT_POS, s=200, color='#795548', marker='*',
               zorder=15, label=f'DR init ({DR_INIT_POS[0]:.0f},{DR_INIT_POS[1]:.0f})')

    for label, pos in positions_dict.items():
        color = COLORS.get(label, 'gray')
        if len(pos) == 0:
            continue
        errs  = nearest_gt_error(pos, gt_xy)
        rmse  = np.sqrt(np.mean(errs**2))
        n_col = len(collapse_warn.get(label, []))
        warn  = f" ⚠{n_col}" if n_col > 0 else ""
        lw    = 2.0 if label not in ("Raw+LS", "DR-only") else 1.5
        ls    = '-'
        if label == "Raw+LS":
            ls = '--'
        elif label == "DR-only":
            ls = '-.'
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
    fig, axes    = plt.subplots(1, 4, figsize=(28, 6))
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
#  LEAVE-ONE-OUT CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
def loo_tune_and_eval(all_files, gt_xy, expected_path_length):
    n = len(all_files)
    gs_label = "with grid search" if DO_LOO_GRID_SEARCH else "default params"
    print(f"\n{'═'*70}")
    print(f"  LEAVE-ONE-OUT CV — {n} folds  [{gs_label}]")
    print(f"  DR-only khởi tạo tại ({DR_INIT_POS[0]:.0f}, {DR_INIT_POS[1]:.0f})")
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

        # DR-only KHÔNG grid search — không có tham số cần tune
        methods_fold = {
            "DR-only"   : (PureDR,     dict()),                  # ← MỚI
            "Raw+LS"    : None,
            "UKF"       : (StandardUKF, dict(q=UKF_STD_Q, r=UKF_STD_R)),
            "Huber-UKF" : (HuberUKF,   best_huber),
            "MCC-UKF"   : (MCCUKF,     best_mcc),
            "PC-UKF-2D" : (PCUKF2D,    best_pc),
            "GUKF"      : (GUKF,        best_gukf),
        }

        data = parse_file(test_file)
        if data is None:
            print(f"  [!] Không đọc được {fname}, bỏ qua.")
            continue

        T   = len(data)
        row = f"  {fname:<20s}"

        for name, method in methods_fold.items():
            if method is None:
                pos = np.array([LS_position(data[t, :4]) for t in range(T)])
            else:
                filt_cls, kwargs = method
                pos = run_filter(filt_cls, data, **kwargs)

            valid = pos[~np.any(np.isnan(pos), axis=1)]
            errs  = nearest_gt_error(valid, gt_xy)
            rmse  = float(np.sqrt(np.mean(errs**2))) if len(errs) > 0 else float('nan')

            row += f" {rmse:>12.1f}"
            loo_errors[name].extend(errs)
            loo_pos[name].extend(valid)
            per_file_rmse[name].append(rmse)

        print(row)

    # In thêm bảng DR-only để dễ nhận xét
    print(f"\n{'─'*70}")
    print("  DR-only per-fold RMSE (để kiểm tra drift tích lũy):")
    dr_rmses = per_file_rmse.get("DR-only", [])
    for i, (f, r) in enumerate(zip(all_files, dr_rmses)):
        flag = "⚠ DRIFT?" if not math.isnan(r) and r > 1000 else ""
        print(f"    Fold {i+1}: {os.path.basename(f):<20s}  RMSE={r:>8.1f}mm  {flag}")

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
    print("  V22: UKF + Dead Reckoning — thêm DR-only method")
    print("=" * 70)
    print("  0. DR-only      — Pure DR, không dùng UWB, init (800, 400)")
    print("  1. Raw+LS       — Least Squares baseline")
    print("  2. UKF          — UKF + DR motion model")
    print("  3. Huber-UKF    — Huber M-estimator + DR motion model")
    print("  4. MCC-UKF      — MCC UKF + DR motion model")
    print("  5. PC-UKF-2D    — Pairwise Consensus UKF + DR motion model")
    print("  6. GUKF         — Gaussian-smoothed UKF + DR motion model")
    print(f"  DR init pos        = {DR_INIT_POS}")
    print(f"  dt                 = per-step từ timestamp (auto ms/s detect)")
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
        mode_label = f"LOO-CV ({n} folds)"
        errors, positions, per_file_rmse = loo_tune_and_eval(
            all_files, gt_xy, expected_path_length)

        collapse_warn = {m: [] for m in METHODS}
        for name, pos_arr in positions.items():
            if len(pos_arr) > 0 and is_trajectory_collapsed(
                    pos_arr, expected_path_length):
                collapse_warn[name].append("aggregate")

    else:
        mode_label = "Grid Search (1 file)"
        print(f"\n[!] Chỉ có {n} file — LOO cần ≥ 2 file.")

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

    metrics = print_summary(errors, positions, per_file_rmse,
                            collapse_warn, expected_path_length, mode_label)

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

    # ── Nhận xét nhanh về DR-only ────────────────────────────────────
    dr_errs = errors.get("DR-only", np.array([]))
    if len(dr_errs) > 0:
        dr_rmse = float(np.sqrt(np.mean(dr_errs**2)))
        ls_errs = errors.get("Raw+LS", np.array([]))
        ls_rmse = float(np.sqrt(np.mean(ls_errs**2))) if len(ls_errs) > 0 else float('nan')
        print(f"\n  ── DR-only analysis ──────────────────────────────")
        print(f"     RMSE DR-only : {dr_rmse:.1f} mm")
        print(f"     RMSE Raw+LS  : {ls_rmse:.1f} mm")
        if dr_rmse < ls_rmse * 1.5:
            print("     → DR khá tốt — heading/velocity đáng tin cậy ✅")
        elif dr_rmse < ls_rmse * 3.0:
            print("     → DR drift vừa — có thể fusion tốt với UWB ⚠️")
        else:
            print("     → DR drift nhiều — heading/velocity có lỗi lớn ❌")
        print(f"     (khởi tạo tại {DR_INIT_POS} — đảm bảo đúng vị trí xuất phát!)")


if __name__ == "__main__":
    main()
