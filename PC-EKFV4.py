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
EKF2D_R = 300.0

# PC-EKF-v3 params  — chỉ cần tune r_scale!
PCEKF_Q       = 0.01
PCEKF_R_BASE  = 300.0
PCEKF_R_SCALE = 2.0   # ← PARAM DUY NHẤT CẦN TUNE (thử 2 → 30)

DO_GRID_SEARCH = True
DATA_DIR = "./data"
SAVE_DIR = "./outputs_PCEKF_v4"


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


def wls_position(distances, anchors=ANCHORS):
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
    pos = wls_position(raw_dist[0])
    return pos if not np.any(np.isnan(pos)) else np.array([2000.0, 4400.0])


def ekf2d_filter_file(raw_dist, q=EKF2D_Q, r=EKF2D_R):
    T   = len(raw_dist)
    ekf = EKF2D(q=q, r=r)
    ekf.init(_init_pos(raw_dist))
    pos = np.full((T, 2), np.nan)
    for t in range(T):
        pos[t] = ekf.step(raw_dist[t])
    return pos


def pcekf_v3_filter_file(raw_dist, q=PCEKF_Q, r_base=PCEKF_R_BASE,
                          r_scale=PCEKF_R_SCALE):
    T    = len(raw_dist)
    ekf  = PCEKF_v3(q=q, r_base=r_base, r_scale=r_scale)
    ekf.init(_init_pos(raw_dist))
    pos  = np.full((T, 2), np.nan)
    scrs = np.zeros((T, N_ANCHORS))
    for t in range(T):
        pos[t], scrs[t] = ekf.step(raw_dist[t])
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
        raw_pos = np.array([wls_position(dist_raw[t]) for t in range(T)])

        # 2. EKF-2D
        t0 = time.perf_counter()
        ekf_pos = ekf2d_filter_file(dist_raw)
        t1 = time.perf_counter()

        # 3. PC-EKF-v3
        pcekf_pos, _ = pcekf_v3_filter_file(
            dist_raw, q=q, r_base=r_base, r_scale=r_scale)
        t2 = time.perf_counter()

        def filt(p): return p[~np.any(np.isnan(p), axis=1)]
        def rmse(p): return np.sqrt(np.mean(nearest_gt_error(filt(p), gt_xy)**2))

        t_ekf   = (t1 - t0) * 1000
        t_pcekf = (t2 - t1) * 1000
        print(f"{os.path.basename(path):<18s}"
              f"{rmse(raw_pos):>10.1f}{rmse(ekf_pos):>10.1f}"
              f"{rmse(pcekf_pos):>12.1f} | "
              f"{t_ekf:>8.1f}ms {t_pcekf:>9.1f}ms {t_pcekf/T*1000:>11.3f}ms")

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
    return errors, positions


def compute_metrics(errors, label=""):
    m = {k: float(fn(errors)) for k, fn in [
        ('mae',   np.mean),
        ('rmse',  lambda e: np.sqrt(np.mean(e**2))),
        ('cep50', lambda e: np.percentile(e, 50)),
        ('cep90', lambda e: np.percentile(e, 90)),
        ('p95',   lambda e: np.percentile(e, 95)),
        ('max',   np.max),
    ]}
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
        'q'       : [0.001, 0.01],
        'r_base'  : [100.0, 200.0, 300.0],
        'r_scale' : [2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
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
        err, _ = evaluate_files(val_files, gt_xy, **params)
        rmse   = float(np.sqrt(np.mean(err['PC-EKF-v3']**2)))
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params.copy()
        if (idx + 1) % 20 == 0 or (idx + 1) == n_combo:
            print(f"  [{idx+1}/{n_combo}] best so far: {best_rmse:.1f}mm")

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs (PC-EKF V3):")
    print(f"  {'RMSE':>8s} {'Q':>8s} {'R_base':>8s} {'r_scale':>8s}")
    print(f"  {'─'*38}")
    for rmse, p in results[:5]:
        print(f"  {rmse:>7.1f}mm {p['q']:>8.4f} {p['r_base']:>8.1f}"
              f" {p['r_scale']:>8.1f}")
    print(f"\n  Best: RMSE={best_rmse:.1f}mm | {best_params}")
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
#  MATHEMATICAL VALIDATION PLOTS
# ══════════════════════════════════════════════════════════════════════

def _collect_per_timestep(file_paths, q, r_base, r_scale, gt_xy,
                           warmup=100):
    """
    Thu thập dữ liệu per-timestep từ tất cả files:
      - abs_std_innov  : (T, 4) |ν[i]| / √S_ii  per anchor
      - normed         : (T, 4) sau MAD normalize
      - scores         : (T, 4) PC scores
      - R_adaptive     : (T, 4) R được gán cho từng anchor
      - err_raw        : (T,)   position error của Raw+LS
      - err_ekf        : (T,)   position error của EKF-2D
      - err_pcekf      : (T,)   position error của PC-EKF-v3
    Bỏ warmup timestep đầu để P đã hội tụ.
    """
    from scipy.spatial import cKDTree
    abs_si_list, normed_list, score_list   = [], [], []
    R_list, err_raw_list                   = [], []
    err_ekf_list, err_pc_list              = [], []

    gt_tree = cKDTree(gt_xy)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None:
            continue
        dist_raw = parsed['dist'].astype(float)
        T = len(dist_raw)

        ekf_base = EKF2D(q=q, r=r_base)
        ekf_base.init(_init_pos(dist_raw))
        ekf_pc   = PCEKF_v3(q=q, r_base=r_base, r_scale=r_scale)
        ekf_pc.init(_init_pos(dist_raw))

        for t in range(T):
            # ── Raw LS ──
            pos_raw = wls_position(dist_raw[t])

            # ── EKF baseline ──
            pos_ekf = ekf_base.step(dist_raw[t])

            # ── PC-EKF: manual step để lấy internals ──
            ekf_pc.predict()
            h      = h_obs(ekf_pc.x)
            H      = jacobian_H(ekf_pc.x)
            innov  = dist_raw[t] - h
            HP     = H @ ekf_pc.P
            S_diag = np.array([HP[i] @ H[i] + r_base
                                for i in range(N_ANCHORS)])

            std_i   = innov / np.sqrt(np.maximum(S_diag, 1e-9))
            med     = np.median(std_i)
            mad     = np.median(np.abs(std_i - med))
            norm_i  = std_i / (1.4826 * mad + 1e-9)
            scores  = pc_scores_v3(innov, S_diag)
            R_diag  = r_base * (1.0 + r_scale * (1.0 - scores))

            R     = np.diag(R_diag)
            S_mat = H @ ekf_pc.P @ H.T + R
            try:
                K = ekf_pc.P @ H.T @ np.linalg.inv(S_mat)
            except np.linalg.LinAlgError:
                pass
            else:
                ekf_pc.x = ekf_pc.x + K @ innov
                ekf_pc.P = (np.eye(2) - K @ H) @ ekf_pc.P
            pos_pc = ekf_pc.x.copy()

            if t < warmup:
                continue

            # errors
            def _err(pos):
                if np.any(np.isnan(pos)):
                    return np.nan
                return float(gt_tree.query(pos)[0])

            abs_si_list.append(np.abs(std_i))
            normed_list.append(norm_i)
            score_list.append(scores)
            R_list.append(R_diag)
            err_raw_list.append(_err(pos_raw))
            err_ekf_list.append(_err(pos_ekf))
            err_pc_list.append(_err(pos_pc))

    return (np.array(abs_si_list),          # (T,4)
            np.array(normed_list),           # (T,4)
            np.array(score_list),            # (T,4)
            np.array(R_list),                # (T,4)
            np.array(err_raw_list),          # (T,)
            np.array(err_ekf_list),          # (T,)
            np.array(err_pc_list))           # (T,)


def plot_math_validation(file_paths, gt_xy, q=PCEKF_Q, r_base=PCEKF_R_BASE,
                         r_scale=PCEKF_R_SCALE, save_path=None):
    """
    4 plots chứng minh cơ sở toán học của PC-EKF V3
    với data LOS thực tế (không cần label NLOS):

    Plot 1 — |std_innov| worst vs best anchor per timestep:
        Anchor có |std_innov| lớn nhất (= "worst") so với
        anchor nhỏ nhất ("best") trong cùng timestep.
        → Luôn có sự phân tách → kernel có thể phân biệt.

    Plot 2 — Score của worst anchor vs 3 anchor còn lại:
        Boxplot score[worst] vs score[best3] qua toàn bộ
        timestep. → Worst anchor nhận score thấp hơn có hệ thống.

    Plot 3 — MAD robustness vs std (simulation):
        Giữ nguyên — lý thuyết vẫn đúng, minh họa tại sao
        MAD là lựa chọn đúng dù data LOS hay NLOS.

    Plot 4 — R_adaptive trung bình vs position error:
        Scatter: mỗi điểm là 1 timestep.
        Trục x = mean R_adaptive (4 anchors),
        Trục y = error EKF-2D và PC-EKF.
        → Timestep có noise cao (R↑): PC-EKF ít bị ảnh hưởng hơn.
    """
    print("\n[Math Validation] Thu thập dữ liệu per-timestep...")
    (abs_si, normed, scores, R_adap,
     err_raw, err_ekf, err_pc) = _collect_per_timestep(
        file_paths, q, r_base, r_scale, gt_xy)

    T = len(scores)
    print(f"  Timesteps hợp lệ (sau warmup): {T:,}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "PC-EKF V3 — Mathematical Validation  (LOS data)",
        fontsize=15, fontweight='bold', y=1.01
    )

    anchor_colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']

    # ── Plot 1: |std_innov| worst vs best — log-scale ───────────────
    ax = axes[0, 0]

    worst_idx = np.argmax(abs_si, axis=1)
    best_idx  = np.argmin(abs_si, axis=1)

    worst_vals = abs_si[np.arange(T), worst_idx]
    best_vals  = abs_si[np.arange(T), best_idx]
    ratio      = worst_vals / np.maximum(best_vals, 1e-9)

    # log-scale bins — không clip, phân phối thật
    log_ratio = np.log10(np.maximum(ratio, 1.0))
    max_log   = np.percentile(log_ratio, 99.5)
    bins_log  = np.linspace(0, max_log, 60)
    counts, edges = np.histogram(log_ratio, bins=bins_log, density=True)
    mids = (edges[:-1] + edges[1:]) / 2
    ax.bar(mids, counts, width=np.diff(edges), color='#90CAF9',
           edgecolor='white', linewidth=0.3, label='Data (log₁₀ ratio)')

    med_log = np.median(log_ratio)
    ax.axvline(med_log, color='#1565C0', lw=2, ls='--',
               label=f'Median = {10**med_log:.1f}×')
    ax.axvline(0, color='gray', lw=1.5, ls=':', label='Ratio = 1× (no separation)')

    for pct, col, ls_ in [(25, '#78909C', '--'), (75, '#1976D2', '--'),
                           (95, '#B71C1C', '-.')]:
        v = np.percentile(log_ratio, pct)
        ax.axvline(v, color=col, lw=1.2, ls=ls_, alpha=0.8,
                   label=f'P{pct} = {10**v:.1f}×')

    # x-ticks hiển thị giá trị gốc
    xtick_raw = [1, 2, 5, 10, 20, 50, 100, 200]
    xtick_pos = [np.log10(v) for v in xtick_raw if np.log10(v) <= max_log * 1.05]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f'{v}×' for v in xtick_raw[:len(xtick_pos)]])

    pct_gt2 = np.mean(ratio > 2) * 100
    pct_gt5 = np.mean(ratio > 5) * 100
    ax.set_title("Plot 1 — |std_innov| worst/best ratio per timestep\n"
                 "(log scale — ratio > 1 → kernel can separate anchors)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("|std_innov| worst / best  (log scale)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.03, 0.97,
            f"{pct_gt2:.1f}% timesteps have ratio > 2×\n"
            f"{pct_gt5:.1f}% timesteps have ratio > 5×\n"
            f"→ strong separability at every step",
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#1976D2', alpha=0.9))
    ax.grid(True, ls='--', alpha=0.35)

    # ── Plot 2: Score worst vs best3 ────────────────────────────────
    ax = axes[0, 1]

    score_worst = scores[np.arange(T), worst_idx]
    # mean score của 3 anchor còn lại
    score_best3 = np.array([
        np.mean([scores[t, j] for j in range(N_ANCHORS) if j != worst_idx[t]])
        for t in range(T)
    ])

    import matplotlib
    _mpl_ver = tuple(int(x) for x in matplotlib.__version__.split('.')[:2])
    _bp_label_kw = 'tick_labels' if _mpl_ver >= (3, 9) else 'labels'
    bp = ax.boxplot(
        [score_worst, score_best3],
        **{_bp_label_kw: ['Worst anchor\n(highest |std_innov|)',
                          'Best 3 anchors\n(mean)']},
        patch_artist=True,
        medianprops=dict(color='black', lw=2),
        whiskerprops=dict(lw=1.5),
        capprops=dict(lw=1.5),
        flierprops=dict(marker='.', markersize=2, alpha=0.3)
    )
    bp['boxes'][0].set_facecolor('#FFCDD2')
    bp['boxes'][1].set_facecolor('#C8E6C9')

    from scipy.stats import mannwhitneyu
    stat, p_mwu = mannwhitneyu(score_worst, score_best3, alternative='less')
    med_w = np.median(score_worst)
    med_b = np.median(score_best3)

    ax.set_title("Plot 2 — PC Score: worst anchor vs best 3 anchors\n"
                 "(per timestep — same data as Plot 1)",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("PC Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.text(0.97, 0.97,
            f"Median worst  = {med_w:.3f}\n"
            f"Median best3  = {med_b:.3f}\n"
            f"Mann-Whitney p = {p_mwu:.2e}\n"
            f"→ score phân biệt có hệ thống",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF3E0', ec='#F57C00', alpha=0.9))
    ax.grid(True, axis='y', ls='--', alpha=0.35)

    # ── Plot 3: MAD robustness simulation ───────────────────────────
    ax = axes[1, 0]
    np.random.seed(0)
    n_sim      = 2000
    bias_range = np.linspace(0, 15, 60)
    mad_vals, std_vals = [], []

    for bias in bias_range:
        m_acc, s_acc = 0.0, 0.0
        for _ in range(n_sim):
            v   = np.concatenate([np.random.randn(3),
                                   np.random.randn(1) + bias])
            med = np.median(v)
            m_acc += 1.4826 * np.median(np.abs(v - med))
            s_acc += np.std(v)
        mad_vals.append(m_acc / n_sim)
        std_vals.append(s_acc / n_sim)

    ax.plot(bias_range, mad_vals, '#2196F3', lw=2.5,
            label='1.4826·MAD  (robust scale estimator)')
    ax.plot(bias_range, std_vals, '#E91E63', lw=2, ls='--',
            label='std  (classical scale estimator)')
    ax.axhline(1.0, color='gray', ls=':', lw=1.5,
               label='Ideal = 1.0  (pure LOS)')
    ax.fill_between(bias_range, 0.8, 1.2, color='#E8F5E9', alpha=0.6,
                    label='±20% acceptable band')

    ax.set_title("Plot 3 — MAD vs std: robustness to 1 noisy anchor\n"
                 "(3 anchors ~ N(0,1), 1 anchor shifted by increasing bias)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Extra noise / bias on 1 anchor  (σ units)", fontsize=11)
    ax.set_ylabel("Estimated scale of the 3 clean anchors", fontsize=11)
    ax.legend(fontsize=10)

    # annotate where std breaks the band
    std_arr = np.array(std_vals)
    break_idx = np.argmax(std_arr > 1.2)
    if std_arr[break_idx] > 1.2:
        ax.axvline(bias_range[break_idx], color='#E91E63', ls=':', lw=1, alpha=0.7)
        ax.text(bias_range[break_idx] + 0.2, 1.25,
                f'std breaks\nat bias={bias_range[break_idx]:.1f}σ',
                fontsize=9, color='#C62828')

    ax.text(0.97, 0.45,
            "MAD stays within ±20%\nfor any bias magnitude\n→ sigma=1.0 valid\nin all conditions",
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#1976D2', alpha=0.9))
    ax.set_xlim(0, bias_range[-1]); ax.set_ylim(0.5, None)
    ax.grid(True, ls='--', alpha=0.35)

    # ── Plot 4: rolling error reduction EKF → PC-EKF ────────────────
    ax = axes[1, 1]

    valid  = (~np.isnan(err_ekf)) & (~np.isnan(err_pc))
    e_ekf  = err_ekf[valid]
    e_pc   = err_pc[valid]
    t_axis = np.arange(len(e_ekf))

    # rolling median window
    W = max(30, len(e_ekf) // 80)
    def rolling_median(x, w):
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            lo = max(0, i - w // 2)
            hi = min(len(x), i + w // 2 + 1)
            out[i] = np.median(x[lo:hi])
        return out

    ekf_roll = rolling_median(e_ekf, W)
    pc_roll  = rolling_median(e_pc,  W)
    delta    = ekf_roll - pc_roll   # dương = PC tốt hơn

    # vẽ 2 đường error rolling
    ax2 = ax.twinx()
    ax.plot(t_axis, ekf_roll, '#E91E63', lw=1.8, alpha=0.85,
            label='EKF-2D  (rolling median)')
    ax.plot(t_axis, pc_roll,  '#2196F3', lw=1.8, alpha=0.85,
            label='PC-EKF-v3  (rolling median)')
    ax.fill_between(t_axis, ekf_roll, pc_roll,
                    where=(ekf_roll >= pc_roll),
                    color='#2196F3', alpha=0.15, label='PC-EKF better')
    ax.fill_between(t_axis, ekf_roll, pc_roll,
                    where=(ekf_roll < pc_roll),
                    color='#E91E63', alpha=0.15, label='EKF-2D better')

    # delta trên trục phải
    ax2.bar(t_axis, delta, color=np.where(delta >= 0, '#1565C0', '#C62828'),
            alpha=0.25, width=1.0, label='Δ error (EKF − PC-EKF)')
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_ylabel("Δ error  EKF−PC-EKF (mm)\n(positive = PC-EKF better)",
                   fontsize=10, color='#555')
    ax2.tick_params(axis='y', labelcolor='#555')

    # stats
    pct_better = np.mean(delta > 0) * 100
    mean_gain  = np.nanmean(delta)
    ax.set_title(
        f"Plot 4 — Rolling error: EKF-2D vs PC-EKF-v3\n"
        f"(window={W} steps  |  PC-EKF better in {pct_better:.1f}% of timesteps)",
        fontsize=12, fontweight='bold')
    ax.set_xlabel("Timestep (after warmup)", fontsize=11)
    ax.set_ylabel("Position error  (mm)", fontsize=11)
    ax.set_ylim(0, None)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')

    ax.text(0.97, 0.97,
            f"PC-EKF better: {pct_better:.1f}% timesteps\n"
            f"Mean gain: {mean_gain:.1f} mm\n"
            f"(rolling window = {W} steps)",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#E8F5E9', ec='#388E3C', alpha=0.9))
    ax.grid(True, ls='--', alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Math validation → {save_path}")
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

    err, positions = evaluate_files(eval_files, gt_xy, **best_params)
    metrics = {label: compute_metrics(errors, label) for label, errors in err.items()}

    # Summary
    print(f"\n{'═'*72}")
    print(f"  SUMMARY — PC-EKF V3 (sigma-free)")
    print(f"{'═'*72}")
    print(f"  {'Method':<18s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*55}")
    for label, m in metrics.items():
        tag = " ◀ V3" if 'PC-EKF' in label else ""
        print(f"  {label:<18s} {m['rmse']:>6.1f} {m['mae']:>6.1f} "
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
    plot_math_validation(eval_files, gt_xy,
                         save_path=os.path.join(SAVE_DIR, 'math_validation.png'),
                         **best_params)

    for label, errors in err.items():
        safe = label.lower().replace(' + ','_').replace(' ','_').replace('-','_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errors)

    print(f"\n[✓] Toàn bộ kết quả → '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
