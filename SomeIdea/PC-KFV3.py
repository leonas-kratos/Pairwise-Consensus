# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — PC-KF (V3 — MAD Auto-Normalized, Sigma-Free)
========================================================================
PC scoring V3:
  - Innovation từ KF predict step: ν[i] = z[i] - x̂[i]
  - S_ii = P_pred[i] + R_base                         (per-anchor prediction variance)
  - std_innov[i] = ν[i] / sqrt(S_ii)                  → chuẩn hóa ~ N(0,1)
  - MAD normalization: normed[i] = std_innov[i] / (1.4826 * MAD)
    → tự động scale, KHÔNG cần tune sigma nữa
  - Pairwise T-kernel trên normed (sigma=1.0 cố định) → score[i] ∈ [0,1]
  - R_adaptive[i] = R_base * (1 + r_scale*(1 - score[i]))
  - KF update với R_adaptive, rồi WLS position

Ưu điểm so với V2:
  - Loại bỏ hoàn toàn tham số sigma
  - MAD robust với outlier (1-2 anchor NLOS không ảnh hưởng scale)
  - Chỉ còn 1 param cần tune: r_scale

SO SÁNH 3 PHƯƠNG PHÁP:
  1. Raw + WLS
  2. KF + WLS         (fixed Q, R)
  3. PC-KF-v3 + WLS   (MAD auto-normalized, sigma-free)

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

# KF baseline params
KF_Q = 0.01
KF_R = 200.0

# PC-KF-v3 params  — chỉ cần tune r_scale!
PCKF_Q       = 0.01
PCKF_R_BASE  = 200.0
PCKF_R_SCALE = 10.0   # ← PARAM DUY NHẤT CẦN TUNE (thử 2 → 30)

DO_GRID_SEARCH = True
DATA_DIR = "./data"
SAVE_DIR = "./outputs_PCKF_v3"


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
#  T-DISTRIBUTION KERNEL (M-estimator)
# ══════════════════════════════════════════════════════════════════════
def _t_kernel(diff_vec, sigma):
    """T-distribution kernel — robust với heavy tail (ν=4)."""
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
        std_innov[i] = innovations[i] / sqrt(S_ii)
        S_ii = P_pred[i] + R_base
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
#  KF 1D BASELINE
# ══════════════════════════════════════════════════════════════════════
class KF1D:
    """KF 1D per-anchor, dùng cho baseline KF + WLS."""
    def __init__(self, q=KF_Q, r=KF_R):
        self.Q = q; self.R = r
        self.P = 1.0; self.x = None

    def update(self, z):
        if self.x is None:
            self.x = z; return self.x, 0.0
        P_pred     = self.P + self.Q
        innovation = z - self.x
        K          = P_pred / (P_pred + self.R)
        self.x    += K * innovation
        self.P     = (1 - K) * P_pred
        return self.x, innovation


# ══════════════════════════════════════════════════════════════════════
#  PC-KF V3 — MAD Auto-Normalized, Sigma-Free
# ══════════════════════════════════════════════════════════════════════
class PCKF_v3:
    """
    PC-KF V3: KF 1D per-anchor với PC scoring MAD auto-normalized.
    Không cần tune sigma — chỉ tune r_scale.

    Pipeline mỗi timestep:
      1. Predict: P_pred[i] = P[i] + Q  (per-anchor scalar)
      2. Innovation: ν[i] = z[i] - x̂[i]
      3. S_ii = P_pred[i] + R_base       (per-anchor prediction variance)
      4. std_innov[i] = ν[i] / sqrt(S_ii)  → chuẩn hóa
      5. MAD normalize → normed[i]
      6. T-kernel(normed, sigma=1.0) → score[i] ∈ [0,1]
      7. R_adaptive[i] = R_base * (1 + r_scale * (1 - score[i]))
      8. KF update: K[i] = P_pred[i] / (P_pred[i] + R_adaptive[i])
                    x̂[i] += K[i] * ν[i]
                    P[i]  = (1 - K[i]) * P_pred[i]
      9. WLS position từ filtered distances
    """
    def __init__(self, q=PCKF_Q, r_base=PCKF_R_BASE, r_scale=PCKF_R_SCALE):
        self.Q       = q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.P       = np.ones(N_ANCHORS)
        self.x       = None   # per-anchor distance estimate

    def update(self, d_raw):
        """
        d_raw: (N_ANCHORS,) khoảng cách ground đo được.
        Trả về: (d_filtered, scores)
        """
        if self.x is None:
            self.x = d_raw.copy()
            return self.x.copy(), np.ones(N_ANCHORS)

        # 1. Predict
        P_pred = self.P + self.Q                     # (N_ANCHORS,)

        # 2. Innovation
        innov  = d_raw - self.x                      # (N_ANCHORS,)

        # 3. S_ii = P_pred[i] + R_base
        S_diag = P_pred + self.R_base                # (N_ANCHORS,)

        # 4-5. PC scoring V2 trên std_innov
        scores = pc_scores_v3(innov, S_diag)

        # 6. Adaptive R
        R_adap = self.R_base * (1.0 + self.R_scale * (1.0 - scores))

        # 7. KF update với R_adaptive
        K      = P_pred / (P_pred + R_adap)
        self.x = self.x + K * innov
        self.P = (1 - K) * P_pred

        return self.x.copy(), scores


# ══════════════════════════════════════════════════════════════════════
#  FILTER WRAPPERS
# ══════════════════════════════════════════════════════════════════════
def kf_filter_file(raw_dist, q=KF_Q, r=KF_R):
    T   = len(raw_dist)
    kfs = [KF1D(q=q, r=r) for _ in range(N_ANCHORS)]
    d_kf = np.zeros_like(raw_dist)
    for t in range(T):
        for i in range(N_ANCHORS):
            d_kf[t, i], _ = kfs[i].update(raw_dist[t, i])
    return d_kf.astype(np.float32)


def pckf_v3_filter_file(raw_dist, q=PCKF_Q, r_base=PCKF_R_BASE,
                         r_scale=PCKF_R_SCALE):
    T    = len(raw_dist)
    kf   = PCKF_v3(q=q, r_base=r_base, r_scale=r_scale)
    d_kf = np.zeros_like(raw_dist)
    scrs = np.zeros_like(raw_dist)
    for t in range(T):
        d_kf[t], scrs[t] = kf.update(raw_dist[t])
    return d_kf.astype(np.float32), scrs.astype(np.float32)


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
                   pckf_q=PCKF_Q, pckf_r_base=PCKF_R_BASE,
                   pckf_r_scale=PCKF_R_SCALE):
    raw_pos_all  = []
    kf_pos_all   = []
    pckf_pos_all = []

    W = 95
    print("\n" + "═"*W)
    print(f"  PC-KF V3 — PER-FILE RESULTS")
    print("═"*W)
    print(f"{'File':<18s} {'Raw+WLS':>10s} {'KF+WLS':>10s} {'PC-KF-v3':>12s} | "
          f"{'KF T_file':>11s} {'PCKF T_fl':>11s} {'PCKF T_smpl':>12s}")
    print("─"*W)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None: continue
        dist_raw = parsed['dist']
        T = len(dist_raw)

        # 1. Raw + WLS
        raw_pos = np.array([wls_position(dist_raw[t]) for t in range(T)])

        # 2. KF + WLS
        t0 = time.perf_counter()
        d_kf   = kf_filter_file(dist_raw)
        kf_pos = np.array([wls_position(d_kf[t]) for t in range(T)])
        t1 = time.perf_counter()

        # 3. PC-KF-v3 + WLS
        d_pckf, _ = pckf_v3_filter_file(
            dist_raw, q=pckf_q, r_base=pckf_r_base,
            r_scale=pckf_r_scale)
        pckf_pos = np.array([wls_position(d_pckf[t]) for t in range(T)])
        t2 = time.perf_counter()

        def filt(p): return p[~np.any(np.isnan(p), axis=1)]
        def rmse(p): return np.sqrt(np.mean(nearest_gt_error(filt(p), gt_xy)**2))

        t_kf   = (t1 - t0) * 1000
        t_pckf = (t2 - t1) * 1000
        print(f"{os.path.basename(path):<18s}"
              f"{rmse(raw_pos):>10.1f}{rmse(kf_pos):>10.1f}"
              f"{rmse(pckf_pos):>12.1f} | "
              f"{t_kf:>9.1f}ms {t_pckf:>9.1f}ms {t_pckf/T*1000:>10.3f}ms")

        raw_pos_all .extend(filt(raw_pos))
        kf_pos_all  .extend(filt(kf_pos))
        pckf_pos_all.extend(filt(pckf_pos))

    print("─"*W)
    raw_arr  = np.array(raw_pos_all)
    kf_arr   = np.array(kf_pos_all)
    pckf_arr = np.array(pckf_pos_all)

    errors = {
        'Raw + WLS'      : nearest_gt_error(raw_arr,  gt_xy),
        'KF + WLS'       : nearest_gt_error(kf_arr,   gt_xy),
        'PC-KF-v3 + WLS' : nearest_gt_error(pckf_arr, gt_xy),
    }
    positions = {'raw': raw_arr, 'kf': kf_arr, 'pckf': pckf_arr}
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
#  GRID SEARCH
# ══════════════════════════════════════════════════════════════════════
def grid_search(val_files, gt_xy):
    """
    Grid search cho PC-KF V3.
    sigma đã bị loại — chỉ còn q, r_base, r_scale.
    """
    grid = {
        'q'       : [0.001, 0.01],
        'r_base'  : [50.0, 100.0, 200.0, 300.0, 500.0],
        'r_scale' : [2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n_combo = len(combos)
    print(f"\n  Grid search PC-KF V3: {n_combo} combinations "
          f"(vs {n_combo * 6} với sigma — giảm 6x)...")

    best_rmse   = float('inf')
    best_params = {}
    results     = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        err, _ = evaluate_files(
            val_files, gt_xy,
            pckf_q=params['q'], pckf_r_base=params['r_base'],
            pckf_r_scale=params['r_scale'],
        )
        rmse = float(np.sqrt(np.mean(err['PC-KF-v3 + WLS']**2)))
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params.copy()
        if (idx + 1) % 20 == 0 or (idx + 1) == n_combo:
            print(f"  [{idx+1}/{n_combo}] best so far: {best_rmse:.1f}mm")

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs (PC-KF V3):")
    print(f"  {'RMSE':>8s} {'Q':>8s} {'R_base':>8s} {'r_scale':>8s}")
    print(f"  {'─'*38}")
    for rmse, p in results[:5]:
        print(f"  {rmse:>7.1f}mm {p['q']:>8.3f} {p['r_base']:>8.1f}"
              f" {p['r_scale']:>8.1f}")
    print(f"\n  Best: RMSE={best_rmse:.1f}mm | {best_params}")
    return best_params, best_rmse


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    'Raw + WLS'      : '#9E9E9E',
    'KF + WLS'       : '#E91E63',
    'PC-KF-v3 + WLS' : '#2196F3',
}
LS = {
    'Raw + WLS'      : ':',
    'KF + WLS'       : '--',
    'PC-KF-v3 + WLS' : '-',
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
        ax.set_xticklabels([m.replace(' + ','\n+ ') for m in methods], fontsize=9)
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


def analyze_scores(file_paths, q=PCKF_Q, r_base=PCKF_R_BASE,
                   r_scale=PCKF_R_SCALE, save_path=None):
    """Phân tích scores V3 và R adaptive trên file đầu tiên."""
    parsed = parse_file(file_paths[0])
    if parsed is None: return
    dist_raw = parsed['dist']

    _, scores = pckf_v3_filter_file(dist_raw, q=q, r_base=r_base,
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
        f'PC-KF V3 Scores (MAD auto-normalized, sigma-free) — {os.path.basename(file_paths[0])}',
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
    print("PC-KF V3 — MAD Auto-Normalized, Sigma-Free")
    print("  PC scoring: std_innov[i] = ν[i]/sqrt(S_ii), S_ii = P_pred+R_base")
    print("  MAD auto-scale → sigma=1.0 cố định, KHÔNG CẦN tune sigma")
    print(f"  Chỉ tune: r_scale (hiện tại = {PCKF_R_SCALE})")
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

    best_params = dict(q=PCKF_Q, r_base=PCKF_R_BASE, r_scale=PCKF_R_SCALE)

    if DO_GRID_SEARCH:
        print(f"\n[Grid Search] Dùng toàn bộ {n} files để tìm best params...")
        best_params, _ = grid_search(shuffled, gt_xy)
    else:
        print("\n[INFO] Grid Search TẮT — dùng params mặc định.")

    eval_files = shuffled

    print(f"\n{'═'*60}\n  Evaluating {len(eval_files)} files\n{'═'*60}")
    print(f"  Params: {best_params}")

    err, positions = evaluate_files(
        eval_files, gt_xy,
        pckf_q=best_params['q'], pckf_r_base=best_params['r_base'],
        pckf_r_scale=best_params['r_scale'],
    )
    metrics = {label: compute_metrics(errors, label) for label, errors in err.items()}

    # Summary
    print(f"\n{'═'*72}")
    print(f"  SUMMARY — PC-KF V3 (sigma-free)")
    print(f"{'═'*72}")
    print(f"  {'Method':<22s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*57}")
    for label, m in metrics.items():
        tag = " ◀ V3" if 'PC-KF-v3' in label else ""
        print(f"  {label:<22s} {m['rmse']:>6.1f} {m['mae']:>6.1f} "
              f"{m['cep50']:>6.1f} {m['p95']:>6.1f} {m['max']:>6.1f}{tag}")

    # Wilcoxon
    print(f"\n  Wilcoxon tests (vs PC-KF-v3):")
    for a in ['Raw + WLS', 'KF + WLS']:
        ea, eb = err[a], err['PC-KF-v3 + WLS']
        mn = min(len(ea), len(eb))
        if mn > 10:
            try:
                _, p = wilcoxon(ea[:mn], eb[:mn])
                print(f"    {a} vs PC-KF-v3: p={p:.4f} {'✅' if p<0.05 else '⚠️'}")
            except ValueError:
                pass

    print(f"\n  Best params:")
    print(f"    q={best_params['q']}, r_base={best_params['r_base']}, "
          f"r_scale={best_params['r_scale']}")
    print(f"  [sigma-free: MAD tự động normalize, không cần tune]")

    # Plots
    plot_cdf(err, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(
        {'Raw + WLS'      : positions['raw'],
         'KF + WLS'       : positions['kf'],
         'PC-KF-v3 + WLS' : positions['pckf']},
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
