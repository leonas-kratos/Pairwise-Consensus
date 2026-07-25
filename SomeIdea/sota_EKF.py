# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — EKF Suite V1  (LOO chuẩn)
====================================================
LOO Cross-Validation chuẩn cho model selection:
  - N file → N fold
  - Fold k: test trên file[k], params được chọn từ LOO trên N-1 file còn lại
  - Outer LOO RMSE = mean của N fold RMSE (unbiased estimate)

CÁC PHƯƠNG PHÁP:
  1. Raw+LS        — baseline Weighted Least Squares
  2. EKF            — Extended Kalman Filter chuẩn
  3. Huber-EKF      — IRLS Huber M-estimator trên innovation
  4. MCC-EKF        — Maximum Correntropy Criterion EKF
  5. PC-EKF-2D      — Pairwise Consensus MAD-normalized + EKF (sigma-free)
  6. AEKF           — Adaptive EKF (Sage-Husa noise estimator)
  7. REKF           — Robust EKF với Student-t likelihood (Agamennoni 2012)

LOO PIPELINE:
  for k in 0..N-1:
      train_files = all_files \ {file[k]}
      test_file   = file[k]
      best_params = inner_loo(train_files, grid)   # LOO trên N-1 file
      fold_rmse   = evaluate(test_file, best_params)
  final_loo_rmse = mean(fold_rmse)                 # unbiased

  Sau khi biết final_loo_rmse:
      best_global_params = inner_loo(all_files, grid)  # retrain toàn bộ
      production_filter  = filter(best_global_params)

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

# ─── Standard EKF (không tune) ───────────────────────────────────────
EKF_Q = 0.001
EKF_R = 50.0

# ─── Huber-EKF ───────────────────────────────────────────────────────
HUBER_Q       = 0.001
HUBER_MAXITER = 5

# ─── MCC-EKF ─────────────────────────────────────────────────────────
MCC_Q         = 0.001
MCC_MAXITER   = 5

# ─── PC-EKF-2D ───────────────────────────────────────────────────────
PCEKF_Q       = 0.001

# ─── AEKF ────────────────────────────────────────────────────────────
AEKF_Q        = 0.001

# ─── REKF ────────────────────────────────────────────────────────────
REKF_Q        = 0.001
REKF_MAXITER  = 5

# ─── LOO config ──────────────────────────────────────────────────────
DO_LOO_TUNING     = True        # False = dùng default params
DATA_DIR          = "./data"
SAVE_DIR          = "./outputs_sota_EKF"

MOTION_RATIO_MIN  = 0.90
COLLAPSE_PENALTY  = 1e6         # mm — penalty khi trajectory collapse


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
    bbox_x = pos_xy[:, 0].max() - pos_xy[:, 0].min()
    bbox_y = pos_xy[:, 1].max() - pos_xy[:, 1].min()
    if math.sqrt(bbox_x**2 + bbox_y**2) < 0.05 * expected_path_length:
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
#  FILTER CLASSES
# ══════════════════════════════════════════════════════════════════════
class StandardEKF:
    def __init__(self, q=EKF_Q, r=EKF_R):
        self.Q_mat = np.eye(2) * q
        self.R_mat = np.eye(N_ANCHORS) * r
        self.x = None; self.P = None

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
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


class HuberEKF:
    def __init__(self, q=HUBER_Q, r=50.0, delta=1.5, maxiter=HUBER_MAXITER):
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r
        self.delta   = delta
        self.maxiter = maxiter
        self.x = None; self.P = None

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
            hub_w    = np.where(np.abs(r_scaled) <= self.delta,
                                1.0, self.delta / (np.abs(r_scaled) + 1e-9))
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


class MCCEKF:
    def __init__(self, q=MCC_Q, r=100.0, kernel_bw=1200.0, maxiter=MCC_MAXITER):
        self.Q_mat     = np.eye(2) * q
        self.R_base    = r
        self.kernel_bw = kernel_bw
        self.maxiter   = maxiter
        self.x = None; self.P = None

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
                x_cur = x_new; break
            x_cur = x_new
        self.x = x_cur
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


def _t_kernel_pairwise(diff_vec, sigma=1.0):
    nu = 4.0
    return (1.0 + diff_vec**2 / (nu * sigma**2 + 1e-9)) ** (-(nu + 1.0) / 2.0)


def pc_mad_scores(innov, S_diag):
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
    def __init__(self, q=PCEKF_Q, r_base=25.0, r_scale=15.0):
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.x = None; self.P = None

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
            self.x = x_pred; self.P = make_spd(P_pred)
            return self.x.copy()
        self.x = x_pred + K @ innov
        self.P = make_spd((np.eye(2) - K @ H) @ P_pred)
        return self.x.copy()


class AEKF:
    def __init__(self, q=AEKF_Q, r=50.0, win=10, alpha=0.95):
        self.Q_mat  = np.eye(2) * q
        self.R_est  = np.eye(N_ANCHORS) * r
        self.R_base = r
        self.win    = win
        self.alpha  = alpha
        self.x = None; self.P = None
        self._innov_buf = []

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
        innov  = z_raw - h_obs(x_pred)
        self._innov_buf.append(innov.copy())
        if len(self._innov_buf) > self.win:
            self._innov_buf.pop(0)
        HPH = H @ P_pred @ H.T
        if len(self._innov_buf) >= 2:
            innov_mat = np.array(self._innov_buf)
            C_eps     = innov_mat.T @ innov_mat / len(innov_mat)
            R_new     = np.diag(np.clip(np.diag(C_eps - HPH),
                                        self.R_base / 100.0,
                                        self.R_base * 100.0))
            self.R_est = ((1 - self.alpha) * self.R_est + self.alpha * R_new)
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


class REKF:
    def __init__(self, q=REKF_Q, r=50.0, nu=4.0, maxiter=REKF_MAXITER):
        self.Q_mat   = np.eye(2) * q
        self.R_base  = r
        self.nu      = nu
        self.maxiter = maxiter
        self.x = None; self.P = None

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
        K      = np.zeros((2, N_ANCHORS))
        for _ in range(self.maxiter):
            S      = H @ P_pred @ H.T + R_eff
            s_diag = np.maximum(np.diag(S), 1e-9)
            w      = (self.nu + 1.0) / (self.nu + innov**2 / s_diag)
            w      = np.maximum(w, 1e-4)
            R_eff  = np.diag(self.R_base / w)
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


def run_ls(dist_raw):
    T   = len(dist_raw)
    pos = np.full((T, 2), np.nan)
    for t in range(T):
        pos[t] = LS_position(dist_raw[t])
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


def file_rmse(filt_class_or_none, path, gt_xy, expected_path_length, **kwargs):
    """
    Chạy filter trên 1 file, trả về RMSE.
    Nếu collapse → trả về COLLAPSE_PENALTY.
    filt_class_or_none = None → Raw+LS.
    """
    dist_raw = parse_file(path)
    if dist_raw is None:
        return None

    if filt_class_or_none is None:
        pos = run_ls(dist_raw)
    else:
        pos = run_filter(filt_class_or_none, dist_raw, **kwargs)

    valid = pos[~np.any(np.isnan(pos), axis=1)]
    if is_trajectory_collapsed(valid, expected_path_length):
        return float(COLLAPSE_PENALTY)
    if len(valid) == 0:
        return float(COLLAPSE_PENALTY)
    errs = nearest_gt_error(valid, gt_xy)
    return float(np.sqrt(np.mean(errs**2)))


# ══════════════════════════════════════════════════════════════════════
#  INNER LOO — chọn best params trên một tập file
# ══════════════════════════════════════════════════════════════════════
def inner_loo(filt_class, grid, fixed_params,
              file_paths, gt_xy, expected_path_length,
              verbose=False, method_name=""):
    """
    LOO model selection trên file_paths (tập train).

    Với mỗi combo params:
        LOO-RMSE = mean_{i} RMSE(file[i], params tuned on file_paths \ {file[i]})
        (nhưng ở đây chúng ta dùng leave-one-out đơn giản: đánh giá từng file
         với params cố định — đây là standard LOO for hyperparameter selection)

    Standard LOO for hyperparam selection:
        score(params) = (1/N) * sum_i RMSE(file[i] | params)
        → chọn params minimize score
        (không nested: params không được tune trên chính file đang test)

    Returns: best_params dict
    """
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    N      = len(file_paths)

    if verbose:
        print(f"\n    Inner LOO [{method_name}]: "
              f"{len(combos)} combos × {N} folds")

    best_score  = float('inf')
    best_params = {**fixed_params}

    for combo in combos:
        params = {**fixed_params, **dict(zip(keys, combo))}
        fold_scores = []
        for val_path in file_paths:
            r = file_rmse(filt_class, val_path, gt_xy, expected_path_length,
                          **params)
            if r is not None:
                fold_scores.append(r)
        if not fold_scores:
            continue
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score  = score
            best_params = params.copy()
            if verbose:
                param_str = "  ".join(f"{k}={params[k]}" for k in keys)
                tag = " [COLLAPSE]" if best_score >= COLLAPSE_PENALTY * 0.9 else ""
                print(f"      ★ {score:8.2f}mm{tag}  {param_str}")

    return best_params, best_score


# ══════════════════════════════════════════════════════════════════════
#  GRIDS
# ══════════════════════════════════════════════════════════════════════
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

# Default params khi không tune
DEFAULT_PARAMS = {
    "EKF"      : {"q": EKF_Q, "r": EKF_R},
    "Huber-EKF": {"q": HUBER_Q, "r": 50.0, "delta": 1.5},
    "MCC-EKF"  : {"q": MCC_Q, "r": 100.0, "kernel_bw": 1200.0},
    "PC-EKF-2D": {"q": PCEKF_Q, "r_base": 25.0, "r_scale": 15.0},
    "AEKF"     : {"q": AEKF_Q, "r": 50.0, "win": 10, "alpha": 0.95},
    "REKF"     : {"q": REKF_Q, "r": 50.0, "nu": 4.0},
}

FILT_CLASSES = {
    "EKF"      : StandardEKF,
    "Huber-EKF": HuberEKF,
    "MCC-EKF"  : MCCEKF,
    "PC-EKF-2D": PCEKF2D,
    "AEKF"     : AEKF,
    "REKF"     : REKF,
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
#  OUTER LOO — đánh giá không bias
# ══════════════════════════════════════════════════════════════════════
def outer_loo(method_name, filt_class_or_none, all_files,
              gt_xy, expected_path_length,
              do_tuning=True):
    """
    Outer LOO chuẩn:
      for k = 0..N-1:
          train_files = all_files \ {all_files[k]}
          test_file   = all_files[k]
          if method cần tune:
              best_params = inner_loo(train_files, grid)   # tune trên N-1 file
          else:
              best_params = DEFAULT_PARAMS[method]
          fold_rmse[k] = RMSE(test_file, best_params)      # test trên 1 file

      loo_rmse = mean(fold_rmse)                            # unbiased estimate
      loo_std  = std(fold_rmse)

    Sau LOO → retrain (inner_loo trên toàn bộ N file) → best_global_params
    để dùng cho production evaluation.

    Returns:
        fold_rmses      : list[float], RMSE mỗi fold
        best_global_params : dict, params tốt nhất retrain trên toàn bộ
    """
    N    = len(all_files)
    need_tune = (method_name in GRIDS) and do_tuning

    print(f"\n  ── Outer LOO: {method_name}  (N={N} folds) ──")

    fold_rmses    = []
    fold_params   = []   # params được chọn ở mỗi fold (for inspection)

    for k in range(N):
        test_file   = all_files[k]
        train_files = [f for i, f in enumerate(all_files) if i != k]

        # ── Inner LOO trên train_files để chọn params ─────────────────
        if need_tune:
            cfg = GRIDS[method_name]
            best_params, inner_rmse = inner_loo(
                cfg["filt_class"], cfg["grid"], cfg["fixed_params"],
                train_files, gt_xy, expected_path_length,
                verbose=False,          # tắt verbose để output gọn
                method_name=method_name,
            )
        elif method_name == "Raw+LS":
            best_params = {}
        else:
            best_params = DEFAULT_PARAMS[method_name].copy()

        # ── Test trên fold k ───────────────────────────────────────────
        r = file_rmse(filt_class_or_none, test_file,
                      gt_xy, expected_path_length, **best_params)
        rmse_val = r if r is not None else float('nan')
        fold_rmses.append(rmse_val)
        fold_params.append(best_params.copy())

        # In kết quả fold ngay lập tức
        fname = os.path.basename(test_file)
        col_flag = " [COLLAPSE]" if rmse_val >= COLLAPSE_PENALTY * 0.9 else ""
        param_str = ""
        if need_tune and best_params:
            tune_keys = list(GRIDS[method_name]["grid"].keys())
            param_str = "  params: " + " ".join(
                f"{k2}={best_params[k2]}" for k2 in tune_keys)
        print(f"    Fold {k+1:2d}/{N}  test={fname:<20s}"
              f"  RMSE={rmse_val:7.1f}mm{col_flag}{param_str}")

    # ── Summary ───────────────────────────────────────────────────────
    valid_rmses = [r for r in fold_rmses if not math.isnan(r)
                   and r < COLLAPSE_PENALTY * 0.9]
    loo_rmse = float(np.mean(fold_rmses)) if fold_rmses else float('nan')
    loo_std  = float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else float('nan')
    n_col    = sum(1 for r in fold_rmses if r >= COLLAPSE_PENALTY * 0.9)

    print(f"    → LOO RMSE = {loo_rmse:.1f} ± {loo_std:.1f} mm"
          f"  ({n_col}/{N} collapsed)")

    # ── Retrain trên toàn bộ N file → best_global_params ─────────────
    if need_tune:
        cfg = GRIDS[method_name]
        print(f"    Retrain inner LOO trên toàn bộ {N} file...")
        best_global_params, _ = inner_loo(
            cfg["filt_class"], cfg["grid"], cfg["fixed_params"],
            all_files, gt_xy, expected_path_length,
            verbose=True, method_name=method_name,
        )
    elif method_name == "Raw+LS":
        best_global_params = {}
    else:
        best_global_params = DEFAULT_PARAMS[method_name].copy()

    return fold_rmses, best_global_params


# ══════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION (dùng best_global_params)
# ══════════════════════════════════════════════════════════════════════
def final_evaluate(method_name, filt_class_or_none,
                   all_files, gt_xy, expected_path_length,
                   best_params):
    """
    Chạy filter với best_params trên toàn bộ file → gom lại errors và positions.
    Đây là evaluation cho visualization / summary table.
    """
    all_errors   = []
    all_positions= []
    per_file_rmse= []
    collapse_warn= []

    for path in all_files:
        dist_raw = parse_file(path)
        if dist_raw is None:
            continue
        if filt_class_or_none is None:
            pos = run_ls(dist_raw)
        else:
            pos = run_filter(filt_class_or_none, dist_raw, **best_params)

        valid    = pos[~np.any(np.isnan(pos), axis=1)]
        collapsed = is_trajectory_collapsed(valid, expected_path_length)
        if collapsed:
            collapse_warn.append(os.path.basename(path))

        errs = nearest_gt_error(valid, gt_xy) if len(valid) > 0 else np.array([])
        rmse = float(np.sqrt(np.mean(errs**2))) if len(errs) > 0 else float('nan')
        per_file_rmse.append(rmse)
        all_errors.extend(errs)
        all_positions.extend(valid)

    return (np.array(all_errors),
            np.array(all_positions) if all_positions else np.empty((0, 2)),
            collapse_warn,
            per_file_rmse)


# ══════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════
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
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_cdf(errors_dict, collapse_warn_dict, loo_rmse_dict, save_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, errors in errors_dict.items():
        if len(errors) == 0:
            continue
        s    = np.sort(errors)
        cdf  = np.arange(1, len(s) + 1) / len(s)
        rmse = np.sqrt(np.mean(errors**2))
        loo  = loo_rmse_dict.get(label, float('nan'))
        c    = COLORS.get(label, 'gray')
        lw   = 2.5 if label != "Raw+LS" else 1.5
        ls   = '-'  if label != "Raw+LS" else '--'
        n_col    = len(collapse_warn_dict.get(label, []))
        warn_tag = f" ⚠{n_col}" if n_col > 0 else ""
        loo_tag  = f"  LOO={loo:.1f}" if not math.isnan(loo) else ""
        ax.plot(s, cdf, lw=lw, color=c, ls=ls,
                label=f"{label}{warn_tag}  RMSE={rmse:.1f}mm{loo_tag}")
        ax.axvline(rmse, color=c, ls=':', lw=0.8, alpha=0.4)
    ax.set_xlabel("Position Error (mm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_title("EKF Suite — Error CDF  (LOO-tuned)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] CDF → {save_path}")
    plt.close()


def plot_trajectories(positions_dict, gt_xy, collapse_warn_dict, save_path):
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'k--', lw=2.5,
            label='Ground Truth', alpha=0.5, zorder=5)
    for label, pos in positions_dict.items():
        if len(pos) == 0:
            continue
        color = COLORS.get(label, 'gray')
        errs  = nearest_gt_error(pos, gt_xy)
        rmse  = np.sqrt(np.mean(errs**2))
        n_col = len(collapse_warn_dict.get(label, []))
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
    ax.set_title("EKF Suite — Trajectories (LOO-tuned params)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Trajectories → {save_path}")
    plt.close()


def plot_bar(metrics_dict, loo_rmse_dict, save_path):
    methods      = list(metrics_dict.keys())
    metric_names = ['rmse', 'mae', 'cep50', 'p95']
    labels_disp  = ['RMSE (full)', 'MAE', 'CEP50', 'P95']
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
        # Thêm LOO RMSE line nếu là cột RMSE
        if mname == 'rmse':
            for i, m in enumerate(methods):
                loo = loo_rmse_dict.get(m, float('nan'))
                if not math.isnan(loo) and loo < COLLAPSE_PENALTY * 0.9:
                    ax.plot([i - 0.4, i + 0.4], [loo, loo],
                            color='black', lw=2, ls='--', alpha=0.6)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-', '\n') for m in methods],
                           fontsize=8, ha='center')
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    axes[0].set_title("RMSE (dashed=LOO estimate)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  EKF Suite V1 — LOO Cross-Validation Chuẩn")
    print("=" * 80)
    print("""
  LOO Pipeline:
    for k = 0..N-1:
        train = all_files \\ {file[k]}          (N-1 file)
        test  = file[k]                         (1 file)
        best_params = inner_loo(train, grid)    → tune trên N-1 file
        fold_rmse[k] = RMSE(test, best_params)  → test trên file chưa thấy

    outer_loo_rmse = mean(fold_rmse)            → unbiased performance estimate
    best_global_params = inner_loo(all_files)   → retrain để production eval
""")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    N = len(all_files)
    if N == 0:
        print(f"\n[!] Không tìm thấy file .txt trong '{DATA_DIR}/'")
        return
    print(f"  Tìm thấy {N} file(s) trong '{DATA_DIR}/'")

    np.random.seed(42)
    all_files = [all_files[i] for i in np.random.permutation(N)]

    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    expected_path_length = float(np.linalg.norm(
        np.diff(WAYPOINTS, axis=0), axis=1).sum())
    print(f"  Path={expected_path_length:.0f}mm  "
          f"Time={total_time:.1f}s  GT={len(gt_xy)} points")
    print(f"  LOO Tuning: {'BẬT' if DO_LOO_TUNING else 'TẮT (dùng default params)'}")

    # ─────────────────────────────────────────────────────────────────
    # Các method cần chạy
    # ─────────────────────────────────────────────────────────────────
    METHODS_TO_RUN = {
        "Raw+LS"   : None,           # không tune, không filter class
        "EKF"      : StandardEKF,    # không tune
        "Huber-EKF": HuberEKF,
        "MCC-EKF"  : MCCEKF,
        "PC-EKF-2D": PCEKF2D,
        "AEKF"     : AEKF,
        "REKF"     : REKF,
    }

    # ─────────────────────────────────────────────────────────────────
    # OUTER LOO cho tất cả methods
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  OUTER LOO CROSS-VALIDATION")
    print("═" * 80)

    loo_fold_rmses  = {}   # method → list[float] (N giá trị)
    best_global_params_all = {}   # method → dict

    for method_name, filt_class in METHODS_TO_RUN.items():
        fold_rmses, best_global = outer_loo(
            method_name, filt_class, all_files,
            gt_xy, expected_path_length,
            do_tuning=DO_LOO_TUNING,
        )
        loo_fold_rmses[method_name]        = fold_rmses
        best_global_params_all[method_name] = best_global

    # ─────────────────────────────────────────────────────────────────
    # FINAL EVALUATION với best_global_params (cho visualization)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  FINAL EVALUATION (best params retrained trên toàn bộ N file)")
    print("═" * 80)

    all_errors_dict    = {}
    all_positions_dict = {}
    collapse_warn_dict = {}
    per_file_rmse_dict = {}

    hdr = f"{'File':<22s}"
    for m in METHODS_TO_RUN:
        hdr += f" {m:>11s}"
    print("\n" + hdr)
    print("─" * (22 + 12 * len(METHODS_TO_RUN)))

    # Thu thập per-file RMSE trong final eval
    pf_matrix = {m: [] for m in METHODS_TO_RUN}
    for path in all_files:
        row = f"{os.path.basename(path):<22s}"
        for method_name, filt_class in METHODS_TO_RUN.items():
            params = best_global_params_all[method_name]
            r = file_rmse(filt_class, path, gt_xy, expected_path_length, **params)
            rmse_val = r if r is not None else float('nan')
            pf_matrix[method_name].append(rmse_val)
            col_flag = "⚠" if rmse_val >= COLLAPSE_PENALTY * 0.9 else " "
            row += f" {col_flag}{rmse_val:>9.1f}"
        print(row)
    print("─" * (22 + 12 * len(METHODS_TO_RUN)))

    # Gom errors/positions
    timing = {}
    for method_name, filt_class in METHODS_TO_RUN.items():
        params = best_global_params_all[method_name]
        t0 = time.perf_counter()
        errs, pos, cwarn, pfr = final_evaluate(
            method_name, filt_class, all_files,
            gt_xy, expected_path_length, params)
        elapsed = time.perf_counter() - t0
        n_samp  = sum(len(parse_file(p) or []) for p in all_files)
        timing[method_name] = elapsed * 1000 / max(n_samp, 1)

        all_errors_dict[method_name]    = errs
        all_positions_dict[method_name] = pos
        collapse_warn_dict[method_name] = cwarn
        per_file_rmse_dict[method_name] = pfr

    # ─────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────
    TUNED_METHODS = set(GRIDS.keys())
    print(f"\n{'═' * 130}")
    print("  SUMMARY TABLE (mm)")
    print(f"  LOO est. = unbiased estimate từ outer LOO")
    print(f"  Full RMSE = evaluation trên toàn bộ data với best_global_params")
    print(f"{'═' * 130}")
    print(f"  {'Method':<13s} {'LOO RMSE':>10s} {'±std':>8s} "
          f"{'Full RMSE':>10s} {'MAE':>7s} {'CEP50':>7s} {'CEP90':>7s} "
          f"{'P95':>7s} {'MAX':>7s} {'MotionR':>8s}  Status  BestParams")
    print(f"  {'─' * 120}")

    metrics_dict  = {}
    loo_rmse_dict = {}   # method → LOO mean RMSE (không collapse)

    for method_name in METHODS_TO_RUN:
        errs   = all_errors_dict[method_name]
        pos_arr= all_positions_dict[method_name]
        m      = compute_metrics(errs, pos_arr, expected_path_length)
        metrics_dict[method_name] = m

        fold_r = loo_fold_rmses[method_name]
        # Chỉ tính LOO mean trên các fold không collapse
        valid_folds = [r for r in fold_r
                       if not math.isnan(r) and r < COLLAPSE_PENALTY * 0.9]
        loo_mean = float(np.mean(valid_folds)) if valid_folds else float('nan')
        loo_std  = (float(np.std(valid_folds, ddof=1))
                    if len(valid_folds) > 1 else float('nan'))
        loo_rmse_dict[method_name] = loo_mean
        n_col    = len(collapse_warn_dict[method_name])
        n_col_loo= sum(1 for r in fold_r if r >= COLLAPSE_PENALTY * 0.9)

        status   = (f"⚠ {n_col} col" if n_col > 0 else "✅ OK")
        tag      = " ◀" if method_name in TUNED_METHODS else "  "
        mr_str   = (f"{m['motion_ratio']:.2f}"
                    if not math.isnan(m.get('motion_ratio', float('nan')))
                    else "N/A")

        # Best params string
        bp = best_global_params_all[method_name]
        if method_name in GRIDS:
            tune_keys = list(GRIDS[method_name]["grid"].keys())
            bp_str = " ".join(f"{k}={bp[k]}" for k in tune_keys)
        else:
            bp_str = "(fixed)"

        loo_str = (f"{loo_mean:7.1f}" if not math.isnan(loo_mean)
                   else "    N/A")
        std_str = (f"{loo_std:6.1f}" if not math.isnan(loo_std)
                   else "   N/A")
        if n_col_loo > 0:
            loo_str += f"[⚠{n_col_loo}]"

        print(f"  {method_name:<13s}{tag} {loo_str:>10s} {std_str:>8s}"
              f" {m['rmse']:>10.1f} {m['mae']:>7.1f}"
              f" {m['cep50']:>7.1f} {m['cep90']:>7.1f}"
              f" {m['p95']:>7.1f} {m['max']:>7.1f}"
              f" {mr_str:>8s}  {status:<10s}  {bp_str}")

    # ─────────────────────────────────────────────────────────────────
    # TIMING
    # ─────────────────────────────────────────────────────────────────
    print(f"\n  TIMING (ms/sample):")
    for name, ms in timing.items():
        if name != "Raw+LS":
            print(f"    {name:<13s}: {ms:.4f} ms/sample")

    # ─────────────────────────────────────────────────────────────────
    # WILCOXON
    # ─────────────────────────────────────────────────────────────────
    print(f"\n  Wilcoxon tests (two-sided, vs PC-EKF-2D):")
    ref_err = all_errors_dict.get("PC-EKF-2D", np.array([]))
    for name, errs in all_errors_dict.items():
        if name == "PC-EKF-2D" or len(errs) == 0 or len(ref_err) == 0:
            continue
        N2 = min(len(errs), len(ref_err))
        if N2 > 20:
            try:
                _, p = wilcoxon(errs[:N2], ref_err[:N2])
                sym  = '✅ p<0.05' if p < 0.05 else '⚠️  ns'
                print(f"    {name:<13s} vs PC-EKF-2D: p={p:.4f}  {sym}")
            except ValueError:
                pass

    # ─────────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────────
    plot_cdf(all_errors_dict, collapse_warn_dict, loo_rmse_dict,
             os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(all_positions_dict, gt_xy, collapse_warn_dict,
                      os.path.join(SAVE_DIR, 'trajectories.png'))
    plot_bar(metrics_dict, loo_rmse_dict,
             os.path.join(SAVE_DIR, 'bar_comparison.png'))

    for method_name, errs in all_errors_dict.items():
        safe = method_name.lower().replace('+', '_').replace('-', '_').replace(' ', '_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errs)
        np.save(os.path.join(SAVE_DIR, f'loo_folds_{safe}.npy'),
                np.array(loo_fold_rmses[method_name]))

    print(f"\n[✓] Kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
