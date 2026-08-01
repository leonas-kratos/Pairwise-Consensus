# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — PC-UKF-2D (V3 — MAD Auto-Normalized, Sigma-Free)
=============================================================================
PC scoring V3:
  - Innovation từ UKF predict step (sigma points): ν[i] = z[i] - z_hat[i]
  - Pzz_diag[i] = sum_k Wc[k]*(Z_pts[k,i]-z_hat[i])^2 + R_base
  - std_innov[i] = ν[i] / sqrt(Pzz_diag[i])             → chuẩn hóa ~ N(0,1)
  - MAD normalization: normed[i] = std_innov[i] / (1.4826 * MAD)
    → tự động scale, KHÔNG cần tune sigma nữa
  - Pairwise T-kernel trên normed (sigma=1.0 cố định) → score[i] ∈ [0,1]
  - R_adaptive[i] = R_base * (1 + r_scale*(1 - score[i]))
  - UKF update với R_adaptive (diagonal)

Ưu điểm so với V2:
  - Loại bỏ hoàn toàn tham số sigma
  - MAD robust với outlier (1-2 anchor NLOS không ảnh hưởng scale)
  - Chỉ còn 1 param cần tune: r_scale

SO SÁNH 3 PHƯƠNG PHÁP:
  1. Raw + LS
  2. UKF-2D           (fixed R)
  3. PC-UKF-v3        (MAD auto-normalized, sigma-free)

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

# UKF-2D baseline params
UKF2D_Q     = 0.01
UKF2D_R     = 50.0
UKF2D_ALPHA = 1e-3
UKF2D_BETA  = 2.0
UKF2D_KAPPA = 0.0

# PC-UKF-v3 params  — chỉ cần tune r_scale!
PCUKF_Q       = 0.01
PCUKF_R_BASE  = 50.0
PCUKF_R_SCALE = 30.0   # ← PARAM DUY NHẤT CẦN TUNE (thử 2 → 30)

DO_GRID_SEARCH = False
DATA_DIR = "./data"
SAVE_DIR = "./outputs_PCUKF_v3"


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


def wls_position(distances, anchors=ANCHORS, weights=None):
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi-x0), 2*(yi-y0)])
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
#  OBSERVATION MODEL
# ══════════════════════════════════════════════════════════════════════
def h_obs(state, anchors=ANCHORS):
    """h_i(x,y) = sqrt((x-ax_i)^2 + (y-ay_i)^2 + H^2)"""
    x, y = state[0], state[1]
    return np.array([
        math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
        for ax, ay in anchors
    ])


# ══════════════════════════════════════════════════════════════════════
#  UNSCENTED TRANSFORM UTILITIES
# ══════════════════════════════════════════════════════════════════════
def ukf_weights(n, alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
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


# ══════════════════════════════════════════════════════════════════════
#  T-DISTRIBUTION KERNEL
# ══════════════════════════════════════════════════════════════════════
def _t_kernel(diff_vec, sigma):
    nu  = 4.0
    eps = 1e-9
    return (1.0 + diff_vec**2 / (nu * sigma**2 + eps)) ** (-(nu + 1.0) / 2.0)


# ══════════════════════════════════════════════════════════════════════
#  PC SCORING V3 — MAD Auto-Normalized, Sigma-Free
# ══════════════════════════════════════════════════════════════════════
def pc_scores_v3(innovations, S_diag):
    """
    V3: pairwise T-kernel trên MAD-normalized std_innov.

    Bước 1 — Geo-normalize:
        std_innov[i] = innovations[i] / sqrt(Pzz_diag[i])
        → std_innov ≈ N(0,1) khi LOS

    Bước 2 — MAD normalize:
        mad = median(|std_innov - median(std_innov)|)
        normed[i] = std_innov[i] / (1.4826 * mad + eps)
        → tự động scale, loại bỏ hoàn toàn sigma

    Bước 3 — T-kernel với sigma=1.0 cố định:
        score[i] = mean( T_kernel(normed[i] - normed[j]) ) for j≠i
    """
    std_innov = innovations / np.sqrt(np.maximum(S_diag, 1e-9))
    med    = np.median(std_innov)
    mad    = np.median(np.abs(std_innov - med))
    normed = std_innov / (1.4826 * mad + 1e-9)
    scores = np.zeros(N_ANCHORS)
    for i in range(N_ANCHORS):
        diffs     = np.array([normed[i] - normed[j]
                              for j in range(N_ANCHORS) if j != i])
        scores[i] = np.mean(_t_kernel(diffs, sigma=1.0))
    return scores


# ══════════════════════════════════════════════════════════════════════
#  UKF 2D BASELINE
# ══════════════════════════════════════════════════════════════════════
class UKF2D:
    """UKF với state [x, y], R đồng nhất."""
    def __init__(self, q=UKF2D_Q, r=UKF2D_R,
                 alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
        self.n = 2
        self.Q_scalar = q
        self.R_scalar = r
        self.Wm, self.Wc, self.c = ukf_weights(self.n, alpha, beta, kappa)
        self.x = None
        self.P = None

    def init(self, x0):
        self.x = x0.copy().astype(float)
        self.P = np.eye(self.n) * 1e6

    def predict(self):
        self.P = self.P + np.eye(self.n) * self.Q_scalar

    def update(self, z_raw, R_diag=None):
        if self.x is None:
            return np.array([np.nan, np.nan])
        n = self.n; Wm = self.Wm; Wc = self.Wc

        pts   = sigma_points(self.x, self.P, self.c)           # (2n+1, n)
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*n+1)]) # (2n+1, N_ANCHORS)
        z_hat = Wm @ Z_pts                                      # (N_ANCHORS,)

        R = np.diag(R_diag) if R_diag is not None else np.eye(N_ANCHORS) * self.R_scalar

        Pzz = R.copy()
        for i in range(2*n+1):
            dz   = Z_pts[i] - z_hat
            Pzz += Wc[i] * np.outer(dz, dz)

        Pxz = np.zeros((n, N_ANCHORS))
        for i in range(2*n+1):
            dx   = pts[i] - self.x
            dz   = Z_pts[i] - z_hat
            Pxz += Wc[i] * np.outer(dx, dz)

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            return self.x.copy()

        innov  = z_raw - z_hat
        self.x = self.x + K @ innov
        self.P = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (self.P + self.P.T) + np.eye(n) * 1e-9
        return self.x.copy()

    def step(self, z_raw, R_diag=None):
        self.predict()
        return self.update(z_raw, R_diag)

    def _get_Pzz_diag(self, R_base):
        """
        Tính diagonal Pzz từ sigma points (không update state).
        Dùng nội bộ cho PC scoring V2.
        """
        n = self.n; Wm = self.Wm; Wc = self.Wc
        pts   = sigma_points(self.x, self.P, self.c)
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*n+1)])
        z_hat = Wm @ Z_pts

        Pzz_diag = np.full(N_ANCHORS, R_base)
        for i in range(2*n+1):
            dz = Z_pts[i] - z_hat
            Pzz_diag += Wc[i] * dz**2
        return z_hat, Pzz_diag


# ══════════════════════════════════════════════════════════════════════
#  PC-UKF V3 — MAD Auto-Normalized, Sigma-Free
# ══════════════════════════════════════════════════════════════════════
class PCUKF2D_v3:
    """
    PC-UKF V3: UKF-2D với PC scoring MAD auto-normalized.
    Không cần tune sigma — chỉ tune r_scale.

    Pipeline mỗi timestep:
      1. UKF predict: P_pred = P + Q*I
      2. Sigma points → propagate qua h_obs → z_hat, Pzz_diag
         Pzz_diag[i] = sum_k Wc[k]*(Z_pts[k,i]-z_hat[i])^2 + R_base
      3. innovation ν[i] = z_raw[i] - z_hat[i]
      4. std_innov[i] = ν[i] / sqrt(Pzz_diag[i])
      5. MAD normalize → normed[i]
      6. T-kernel(normed, sigma=1.0) → score[i] ∈ [0,1]
      7. R_adaptive[i] = R_base * (1 + r_scale*(1-score[i]))
      8. UKF update với R_adaptive
    """
    def __init__(self, q=PCUKF_Q, r_base=PCUKF_R_BASE,
                 r_scale=PCUKF_R_SCALE,
                 alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
        self.R_base  = r_base
        self.R_scale = r_scale
        self._ukf    = UKF2D(q=q, r=r_base, alpha=alpha, beta=beta, kappa=kappa)

    def init(self, x0):
        self._ukf.init(x0)

    def step(self, z_raw):
        """
        Trả về: (pos, scores, R_diag)
        """
        ukf = self._ukf

        # 1. Predict
        ukf.predict()

        if ukf.x is None:
            return np.array([np.nan, np.nan]), np.ones(N_ANCHORS), np.full(N_ANCHORS, self.R_base)

        # 2. Lấy z_hat và Pzz_diag từ UKF (không update state)
        z_hat, Pzz_diag = ukf._get_Pzz_diag(self.R_base)

        # 3. Innovation
        innov = z_raw - z_hat                              # (N_ANCHORS,)

        # 4-5. PC scoring V2
        scores = pc_scores_v3(innov, Pzz_diag)

        # 6. Adaptive R
        R_diag = self.R_base * (1.0 + self.R_scale * (1.0 - scores))

        # 7. UKF update (chỉ update step, predict đã làm ở bước 1)
        pos = ukf.update(z_raw, R_diag=R_diag)
        return pos, scores, R_diag


# ══════════════════════════════════════════════════════════════════════
#  FILTER WRAPPERS
# ══════════════════════════════════════════════════════════════════════
def _init_pos(raw_dist):
    pos = wls_position(raw_dist[0])
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


def ukf2d_filter_file(raw_dist, q=UKF2D_Q, r=UKF2D_R,
                      alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
    T   = len(raw_dist)
    ukf = UKF2D(q=q, r=r, alpha=alpha, beta=beta, kappa=kappa)
    ukf.init(_init_pos(raw_dist))
    pos = np.full((T, 2), np.nan)
    for t in range(T):
        pos[t] = ukf.step(raw_dist[t])
    return pos


def pcukf_v3_filter_file(raw_dist, q=PCUKF_Q, r_base=PCUKF_R_BASE,
                          r_scale=PCUKF_R_SCALE):
    T    = len(raw_dist)
    ukf  = PCUKF2D_v3(q=q, r_base=r_base, r_scale=r_scale)
    ukf.init(_init_pos(raw_dist))
    pos  = np.full((T, 2), np.nan)
    scrs = np.zeros((T, N_ANCHORS))
    for t in range(T):
        pos[t], scrs[t], _ = ukf.step(raw_dist[t])
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
                   q=PCUKF_Q, r_base=PCUKF_R_BASE, r_scale=PCUKF_R_SCALE):
    raw_pos_all   = []
    ukf_pos_all   = []
    pcukf_pos_all = []
    per_file_rmse = {'Raw + LS': [], 'UKF-2D': [], 'PC-UKF-v3': []}

    W = 95
    print("\n" + "═"*W)
    print(f"  PC-UKF V3 — PER-FILE RESULTS")
    print("═"*W)
    print(f"{'File':<18s} {'Raw+LS':>10s} {'UKF-2D':>10s} "
          f"{'PC-UKF-v3':>12s} | "
          f"{'UKF T_fl':>10s} {'PCUKF T_fl':>11s} {'PCUKF T_smpl':>13s}")
    print("─"*W)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None: continue
        dist_raw = parsed['dist'].astype(float)
        T = len(dist_raw)

        # 1. Raw + LS
        raw_pos = np.array([wls_position(dist_raw[t]) for t in range(T)])

        # 2. UKF-2D
        t0 = time.perf_counter()
        ukf_pos = ukf2d_filter_file(dist_raw)
        t1 = time.perf_counter()

        # 3. PC-UKF-v3
        pcukf_pos, _ = pcukf_v3_filter_file(
            dist_raw, q=q, r_base=r_base, r_scale=r_scale)
        t2 = time.perf_counter()

        def filt(p): return p[~np.any(np.isnan(p), axis=1)]
        def rmse(p):
            e = nearest_gt_error(filt(p), gt_xy)
            return float(np.sqrt(np.mean(e**2))) if len(e) > 0 else float('nan')

        r_raw   = rmse(raw_pos)
        r_ukf   = rmse(ukf_pos)
        r_pcukf = rmse(pcukf_pos)

        t_ukf   = (t1 - t0) * 1000
        t_pcukf = (t2 - t1) * 1000
        print(f"{os.path.basename(path):<18s}"
              f"{r_raw:>10.1f}{r_ukf:>10.1f}"
              f"{r_pcukf:>12.1f} | "
              f"{t_ukf:>8.1f}ms {t_pcukf:>9.1f}ms {t_pcukf/T*1000:>11.3f}ms")

        per_file_rmse['Raw + LS'] .append(r_raw)
        per_file_rmse['UKF-2D']   .append(r_ukf)
        per_file_rmse['PC-UKF-v3'].append(r_pcukf)

        raw_pos_all  .extend(filt(raw_pos))
        ukf_pos_all  .extend(filt(ukf_pos))
        pcukf_pos_all.extend(filt(pcukf_pos))

    print("─"*W)
    raw_arr   = np.array(raw_pos_all)
    ukf_arr   = np.array(ukf_pos_all)
    pcukf_arr = np.array(pcukf_pos_all)

    errors = {
        'Raw + LS' : nearest_gt_error(raw_arr,   gt_xy),
        'UKF-2D'    : nearest_gt_error(ukf_arr,   gt_xy),
        'PC-UKF-v3' : nearest_gt_error(pcukf_arr, gt_xy),
    }
    positions = {'raw': raw_arr, 'ukf': ukf_arr, 'pcukf': pcukf_arr}
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
#  GRID SEARCH
# ══════════════════════════════════════════════════════════════════════
def grid_search(val_files, gt_xy):
    """
    Grid search cho PC-UKF V3.
    sigma đã bị loại — chỉ còn q, r_base, r_scale.
    """
    grid = {
        'q'       : [0.01],
        'r_base'  : [50.0],
        'r_scale' : [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n_combo = len(combos)
    print(f"\n  Grid search PC-UKF V3: {n_combo} combinations "
          f"(vs {n_combo * 6} với sigma — giảm 6x)...")

    best_rmse   = float('inf')
    best_params = {}
    results     = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        err, _, pf = evaluate_files(val_files, gt_xy, **params)
        vals   = [v for v in pf['PC-UKF-v3'] if not math.isnan(v)]
        rmse   = float(np.mean(vals)) if vals else float('inf')
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params.copy()
        if (idx + 1) % 20 == 0 or (idx + 1) == n_combo:
            print(f"  [{idx+1}/{n_combo}] best so far: mean RMSE={best_rmse:.1f}mm")

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs (PC-UKF V3):")
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
    'UKF-2D'    : '#E91E63',
    'PC-UKF-v3' : '#2196F3',
}
LS = {
    'Raw + LS' : ':',
    'UKF-2D'    : '--',
    'PC-UKF-v3' : '-',
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
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, mname, mlabel in zip(axes, mnames, mlabels):
        vals   = [metrics_dict[m][mname] for m in methods]
        colors = [COLORS.get(m, 'gray') for m in methods]
        bars   = ax.bar(range(len(methods)), vals, color=colors, alpha=0.85,
                        edgecolor='white', lw=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' + ','\n+ ') for m in methods], fontsize=9)
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


def analyze_scores(file_paths, q=PCUKF_Q, r_base=PCUKF_R_BASE,
                   r_scale=PCUKF_R_SCALE, save_path=None):
    """Phân tích scores V3 và R adaptive của PC-UKF trên file đầu tiên."""
    parsed = parse_file(file_paths[0])
    if parsed is None: return
    dist_raw = parsed['dist'].astype(float)

    _, scores = pcukf_v3_filter_file(dist_raw, q=q, r_base=r_base,
                                      r_scale=r_scale)
    R_adaptive = r_base * (1.0 + r_scale * (1.0 - scores))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']

    for i in range(N_ANCHORS):
        axes[0].plot(scores[:,i], color=colors[i], lw=0.8,
                     label=f'Anchor {i+1}', alpha=0.8)
    axes[0].axhline(0.5, color='gray', ls='--', lw=1, label='threshold 0.5')
    axes[0].set_ylabel('PC Score V3', fontsize=11)
    axes[0].set_title(
        f'PC-UKF V3 Scores (MAD auto-normalized, sigma-free) — {os.path.basename(file_paths[0])}',
        fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.05)

    for i in range(N_ANCHORS):
        axes[1].plot(R_adaptive[:,i], color=colors[i], lw=0.8,
                     label=f'R Anchor {i+1}', alpha=0.8)
    axes[1].set_ylabel('Adaptive R (mm²)', fontsize=11)
    axes[1].set_title(f'Adaptive R per Anchor — r_scale={r_scale}  r_base={r_base}', fontsize=12, fontweight='bold')
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
    print("PC-UKF V3 — MAD Auto-Normalized, Sigma-Free")
    print("  Không dùng KF 1D phụ trợ")
    print("  Innovation từ UKF predict (sigma points), normalize Pzz diagonal")
    print("  MAD auto-scale → sigma=1.0 cố định, KHÔNG CẦN tune sigma")
    print(f"  Chỉ tune: r_scale (hiện tại = {PCUKF_R_SCALE})")
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

    best_params = dict(q=PCUKF_Q, r_base=PCUKF_R_BASE, r_scale=PCUKF_R_SCALE)

    if DO_GRID_SEARCH:
        print(f"\n[Grid Search] Dùng toàn bộ {n} files...")
        best_params, _ = grid_search(shuffled, gt_xy)
    else:
        print("\n[INFO] Grid Search TẮT — dùng params mặc định.")

    eval_files = shuffled

    print(f"\n{'═'*60}\n  Evaluating {len(eval_files)} files\n{'═'*60}")
    print(f"  Params: {best_params}")

    err, positions, per_file_rmse = evaluate_files(eval_files, gt_xy, **best_params)
    metrics = {label: compute_metrics(errors, label, pf_rmse=per_file_rmse.get(label))
               for label, errors in err.items()}

    # Summary
    print(f"\n{'═'*88}")
    print(f"  SUMMARY — PC-UKF V3 (sigma-free)")
    print(f"{'═'*88}")
    print(f"  {'Method':<18s} {'RMSE mean±std':>18s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*70}")
    for label, m in metrics.items():
        tag = " ◀ V3" if 'PC-UKF-v3' in label else ""
        if not math.isnan(m.get('rmse_std', float('nan'))):
            rmse_str = f"{m['rmse_mean']:>6.1f} ± {m['rmse_std']:.1f}"
        else:
            rmse_str = f"{m['rmse_mean']:>6.1f} ± N/A"
        print(f"  {label:<18s} {rmse_str:>18s} {m['mae']:>6.1f} "
              f"{m['cep50']:>6.1f} {m['p95']:>6.1f} {m['max']:>6.1f}{tag}")

    # Wilcoxon
    print(f"\n  Wilcoxon tests (vs PC-UKF-v3):")
    for a in ['Raw + LS', 'UKF-2D']:
        ea, eb = err[a], err['PC-UKF-v3']
        mn = min(len(ea), len(eb))
        if mn > 10:
            try:
                _, p = wilcoxon(ea[:mn], eb[:mn])
                print(f"    {a} vs PC-UKF-v3: p={p:.4f} {'✅' if p<0.05 else '⚠️'}")
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
         'UKF-2D'    : positions['ukf'],
         'PC-UKF-v3' : positions['pcukf']},
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
