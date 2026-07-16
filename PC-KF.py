# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — PC-KF (V13a — pure geometric, no learning)
=====================================================================
BẢN CẬP NHẬT: Đồng nhất logic tính RMSE và Ground Truth theo Code 2

SO SÁNH 3 PHƯƠNG PHÁP:
  1. Raw + WLS
  2. KF + WLS          (fixed Q, R)
  3. PC-KF + WLS       (pairwise consensus adaptive R, no learning)

Format file .txt:
  timestamp, d0_slant, d1_slant, d2_slant, d3_slant, v_mm_s, gz_dps
"""

import os
import glob
import math
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
#  CONFIG (Đã đồng nhất hệ đơn vị mm theo Code 2)
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

SPEED          = 200.0  # mm/s — Tốc độ lý thuyết lý tưởng của Code 2
GT_SPACING     = 5.0    # mm   — Khoảng cách bước chuẩn của Code 2
N_ANCHORS      = 4

# KF baseline params
KF_Q = 0.01
KF_R = 300.0

# PC-KF default params
PCKF_Q       = 0.01
PCKF_R_BASE  = 200.0
PCKF_R_SCALE = 20.0    # R_i = R_base * (1 + r_scale * (1 - score_i))
PCKF_SIGMA   = 150.0   # mm — ngưỡng innovation "bình thường"

DO_GRID_SEARCH  = True
DATA_DIR = "./data"
SAVE_DIR = "./outputs_vKFPC"


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY & GROUND TRUTH (Chuẩn Code 2)
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d_slant):
    return math.sqrt(max(d_slant**2 - ANCHOR_HEIGHT**2, 0.0))


def build_ground_truth(waypoints=WAYPOINTS, speed=SPEED, spacing=GT_SPACING):
    """Tạo ground truth hình chữ nhật cố định, khớp hoàn toàn logic Code 2."""
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
    """Tính toán sai số dựa trên KD-Tree tương tự Code 2."""
    tree = cKDTree(gt_xy)
    errors, _ = tree.query(pos_xy)
    return errors


def wls_position(distances, anchors=ANCHORS):
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi-x0), 2*(yi-y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
        # w.append(1.0 / di)
        w.append(1.0 / 1.0)
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    W  = np.diag(w)
    try:
        pos, *_ = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


# ══════════════════════════════════════════════════════════════════════
#  KALMAN FILTERS
# ══════════════════════════════════════════════════════════════════════
class KF1D:
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


class PCKF:
    def __init__(self, q=PCKF_Q, r_base=PCKF_R_BASE, r_scale=PCKF_R_SCALE, sigma=PCKF_SIGMA):
        self.Q       = q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.sigma   = sigma
        self.P       = np.ones(N_ANCHORS)
        self.x       = None

    def _consensus_scores(self, d_raw):
        # Tính residual thay vì innovation
        pos_est = wls_position(d_raw)
        if np.any(np.isnan(pos_est)):
            return np.ones(N_ANCHORS) * 0.5
        
        # Distance từ pos_est đến từng anchor (ground distance)
        residual = np.zeros(N_ANCHORS)
        for i, (ax, ay) in enumerate(ANCHORS):
            d_est = math.sqrt((pos_est[0]-ax)**2 + (pos_est[1]-ay)**2)
            residual[i] = abs(d_raw[i] - d_est)
        
        scores = np.zeros(N_ANCHORS)
        eps = 1e-9
        for i in range(N_ANCHORS):
            s = 0.0; cnt = 0
            for j in range(N_ANCHORS):
                if i == j: continue
                diff = residual[i] - residual[j]
                nu = 4.0
                c = (1.0 + (diff**2) / (nu * self.sigma**2 + eps)) ** (-(nu+1.0)/2.0)
                s += c; cnt += 1
            scores[i] = s / cnt
        return scores

    def update(self, d_raw):
        if self.x is None:
            self.x = d_raw.copy()
            return self.x.copy(), np.ones(N_ANCHORS)

        scores = self._consensus_scores(d_raw)
        R      = self.R_base * (1.0 + self.R_scale * (1.0 - scores))

        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + R)
        self.x = self.x + K * (d_raw - self.x)
        self.P = (1 - K) * P_pred

        return self.x.copy(), scores


def kf_filter_file(raw_dist, q=KF_Q, r=KF_R):
    T    = len(raw_dist)
    kfs  = [KF1D(q=q, r=r) for _ in range(N_ANCHORS)]
    d_kf = np.zeros_like(raw_dist)
    for t in range(T):
        for i in range(N_ANCHORS):
            d_kf[t, i], _ = kfs[i].update(raw_dist[t, i])
    return d_kf.astype(np.float32)


def pckf_filter_file(raw_dist, q=PCKF_Q, r_base=PCKF_R_BASE, r_scale=PCKF_R_SCALE, sigma=PCKF_SIGMA):
    T      = len(raw_dist)
    kf     = PCKF(q=q, r_base=r_base, r_scale=r_scale, sigma=sigma)
    d_kf   = np.zeros_like(raw_dist)
    scores = np.zeros_like(raw_dist)
    for t in range(T):
        d_kf[t], scores[t] = kf.update(raw_dist[t])
    return d_kf.astype(np.float32), scores.astype(np.float32)


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
                vel      = float(parts[5])
                gz       = float(parts[6])
                d_ground = [slant_to_ground(d) for d in d_slant]
                rows.append(d_ground + [vel, gz])
            except ValueError:
                continue
    if not rows: return None
    data = np.array(rows, dtype=np.float32)
    return {'dist': data[:, :4], 'v': data[:, 4], 'gz': data[:, 5]}


# ══════════════════════════════════════════════════════════════════════
#  EVALUATE ALL FILES
# ══════════════════════════════════════════════════════════════════════
import time

def evaluate_files(file_paths, gt_xy, pckf_q=PCKF_Q, pckf_r_base=PCKF_R_BASE,
                   pckf_r_scale=PCKF_R_SCALE, pckf_sigma=PCKF_SIGMA):
    raw_pos_all  = []
    kf_pos_all   = []
    pckf_pos_all = []

    print("\n" + "═"*95)
    print("  PER-FILE RESULTS & EXECUTION TIME")
    print("═"*95)
    print(f"{'File':<18s} {'Raw':>10s} {'KF':>10s} {'PC-KF':>10s} | {'KF T_file':>12s} {'PC T_file':>12s} {'PC T_sample':>12s}")
    print("─"*95)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None: continue

        dist_raw = parsed["dist"]
        T = len(dist_raw)

        # 1. Raw + WLS
        raw_pos = np.array([wls_position(dist_raw[t]) for t in range(T)])

        # 2. KF + WLS (Đo thời gian bao gồm cả bước filter 1D và tính WLS)
        t_start_kf = time.perf_counter()
        d_kf = kf_filter_file(dist_raw)
        kf_pos = np.array([wls_position(d_kf[t]) for t in range(T)])
        t_end_kf = time.perf_counter()
        time_kf_file = t_end_kf - t_start_kf

        # 3. PC-KF + WLS (Đo thời gian bao gồm cả bước PC-filter 1D và tính WLS)
        t_start_pckf = time.perf_counter()
        d_pckf, _ = pckf_filter_file(dist_raw, q=pckf_q, r_base=pckf_r_base, r_scale=pckf_r_scale, sigma=pckf_sigma)
        pckf_pos = np.array([wls_position(d_pckf[t]) for t in range(T)])
        t_end_pckf = time.perf_counter()
        time_pckf_file = t_end_pckf - t_start_pckf
        
        # Thời gian xử lý trung bình 1 mẫu (timestep) của PC-KF + WLS (đơn vị: ms)
        time_pckf_sample_ms = (time_pckf_file / T) * 1000

        def filt(pos):
            return pos[~np.any(np.isnan(pos), axis=1)]

        raw_pos = filt(raw_pos)
        kf_pos = filt(kf_pos)
        pckf_pos = filt(pckf_pos)

        raw_err = nearest_gt_error(raw_pos, gt_xy)
        kf_err = nearest_gt_error(kf_pos, gt_xy)
        pckf_err = nearest_gt_error(pckf_pos, gt_xy)

        raw_rmse = np.sqrt(np.mean(raw_err**2))
        kf_rmse = np.sqrt(np.mean(kf_err**2))
        pckf_rmse = np.sqrt(np.mean(pckf_err**2))

        print(f"{os.path.basename(path):<18s}{raw_rmse:>10.1f}{kf_rmse:>10.1f}{pckf_rmse:>10.1f} | "
              f"{time_kf_file*1000:>10.1f}ms {time_pckf_file*1000:>10.1f}ms {time_pckf_sample_ms:>10.3f}ms")

        raw_pos_all.extend(raw_pos)
        kf_pos_all.extend(kf_pos)
        pckf_pos_all.extend(pckf_pos)

    print("─"*95)
    return {
        "Raw + WLS": nearest_gt_error(np.array(raw_pos_all), gt_xy),
        "KF + WLS": nearest_gt_error(np.array(kf_pos_all), gt_xy),
        "PC-KF + WLS": nearest_gt_error(np.array(pckf_pos_all), gt_xy),
    }, {
        "raw": np.array(raw_pos_all),
        "kf": np.array(kf_pos_all),
        "pckf": np.array(pckf_pos_all),
    }


def compute_metrics(errors, label=""):
    m = {
        'mae'  : float(np.mean(errors)),
        'rmse' : float(np.sqrt(np.mean(errors**2))),
        'cep50': float(np.percentile(errors, 50)),
        'cep90': float(np.percentile(errors, 90)),
        'p95'  : float(np.percentile(errors, 95)),
        'max'  : float(np.max(errors)),
    }
    if label:
        print(f"\n{'='*50}\n  {label}\n{'='*50}")
        for k, v in m.items():
            print(f"  {k.upper():6s}: {v:8.1f} mm")
    return m


# ══════════════════════════════════════════════════════════════════════
#  GRID SEARCH
# ══════════════════════════════════════════════════════════════════════
def grid_search(val_files, gt_xy):
    grid = {
        'q'      : [0.001, 0.01],
        'r_base' : [10.0, 25.0, 50.0, 100.0, 200.0, 300.0],
        'r_scale': [1.0, 5.0, 10.0, 20.0],
        'sigma'  : [1.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search: {len(combos)} combinations...")

    best_rmse   = float('inf')
    best_params = {}
    results     = []

    for combo in combos:
        params = dict(zip(keys, combo))
        err, _ = evaluate_files(
            val_files, gt_xy,
            pckf_q=params['q'], pckf_r_base=params['r_base'],
            pckf_r_scale=params['r_scale'], pckf_sigma=params['sigma'],
        )
        rmse = float(np.sqrt(np.mean(err['PC-KF + WLS']**2)))
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs:")
    print(f"  {'RMSE':>8s} {'Q':>8s} {'R_base':>8s} {'r_scale':>8s} {'sigma':>8s}")
    print(f"  {'─'*48}")
    for rmse, p in results[:5]:
        print(f"  {rmse:>7.1f}mm {p['q']:>8.3f} {p['r_base']:>8.1f} {p['r_scale']:>8.1f} {p['sigma']:>8.1f}")

    print(f"\n  Best: RMSE={best_rmse:.1f}mm | {best_params}")
    return best_params, best_rmse


# ══════════════════════════════════════════════════════════════════════
#  SCORE ANALYSIS & PLOTS
# ══════════════════════════════════════════════════════════════════════
def analyze_scores(file_paths, pckf_q=PCKF_Q, pckf_r_base=PCKF_R_BASE, pckf_r_scale=PCKF_R_SCALE, pckf_sigma=PCKF_SIGMA, save_path=None):
    path = file_paths[0]
    parsed = parse_file(path)
    if parsed is None: return
    dist_raw = parsed['dist']

    _, scores = pckf_filter_file(dist_raw, q=pckf_q, r_base=pckf_r_base, r_scale=pckf_r_scale, sigma=pckf_sigma)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']
    for i in range(N_ANCHORS):
        axes[0].plot(scores[:, i], color=colors[i], lw=0.8, label=f'Anchor {i}', alpha=0.8)
    axes[0].axhline(0.5, color='gray', ls='--', lw=1, label='threshold 0.5')
    axes[0].set_ylabel('Consensus Score', fontsize=11)
    axes[0].set_title(f'PC-KF Consensus Scores — {os.path.basename(path)}', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    for i in range(N_ANCHORS):
        axes[1].plot(dist_raw[:, i], color=colors[i], lw=0.6, label=f'Raw d{i}', alpha=0.7)
    axes[1].set_ylabel('Distance (mm)', fontsize=11)
    axes[1].set_xlabel('Timestep', fontsize=11)
    axes[1].set_title('Raw Distances', fontsize=12)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Score analysis → {save_path}")
    plt.close()


COLORS = {
    'Raw + WLS'  : '#9E9E9E',
    'KF + WLS'   : '#E91E63',
    'PC-KF + WLS': '#2196F3',
}


def plot_cdf(errors_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, errors in errors_dict.items():
        s      = np.sort(errors)
        cdf    = np.arange(1, len(s) + 1) / len(s)
        rmse_v = np.sqrt(np.mean(errors**2))
        color  = COLORS.get(label, 'gray')
        lw     = 2.5 if 'PC-KF' in label else 1.8
        ls     = '-'  if 'PC-KF' in label else '--'
        ax.plot(s, cdf, lw=lw, color=color, ls=ls, label=f"{label}  (RMSE={rmse_v:.1f}mm)")
        ax.axvline(rmse_v, color=color, ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel("Position Error (mm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 2000); ax.set_ylim(0, 1.02) # Giới hạn 2000mm giống Code 2
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] CDF → {save_path}")
    plt.close()


def plot_trajectories(positions_dict, gt_xy, save_path):
    labels = list(positions_dict.keys())
    n      = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 7))
    if n == 1: axes = [axes]
    for ax, label in zip(axes, labels):
        pos   = positions_dict[label]
        color = COLORS.get(label, 'gray')
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'k--', lw=2, label='Ground Truth', alpha=0.5)
        ax.plot(pos[:, 0], pos[:, 1], color=color, lw=0.8, alpha=0.7, label=label)
        for nm, pt in zip(["A","B","C","D"], WAYPOINTS[:4]):
            ax.scatter(*pt, s=70, color='black', zorder=10)
            ax.annotate(nm, pt, textcoords="offset points", xytext=(6,4), fontsize=12, fontweight='bold')
        for j, (ax_, ay_) in enumerate(ANCHORS):
            ax.scatter(ax_, ay_, s=90, marker='s', color='red', zorder=10)
            ax.annotate(f"A{j+1}", (ax_, ay_), textcoords="offset points", xytext=(5,5), fontsize=9, color='red')
        rmse = np.sqrt(np.mean(nearest_gt_error(pos, gt_xy)**2))
        ax.set_title(f"{label}\nRMSE = {rmse:.1f} mm", fontsize=11, fontweight='bold')
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.legend(fontsize=9); ax.set_aspect('equal')
        ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Trajectories → {save_path}")
    plt.close()


def plot_bar(metrics_dict, save_path):
    methods      = list(metrics_dict.keys())
    metric_names = ['rmse', 'mae', 'cep50', 'p95']
    labels_disp  = ['RMSE', 'MAE', 'CEP50', 'P95']
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for ax, mname, mlabel in zip(axes, metric_names, labels_disp):
        vals   = [metrics_dict[m][mname] for m in methods]
        colors = [COLORS.get(m, 'gray') for m in methods]
        bars   = ax.bar(range(len(methods)), vals, color=colors, alpha=0.85, edgecolor='white', lw=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' + ','\n+ ') for m in methods], fontsize=9, ha='center')
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


def debug_pckf(file_path, gt_xy, n_samples=20):
    parsed = parse_file(file_path)
    dist_raw = parsed['dist'].astype(float)
    
    kf = PCKF()
    
    print("\n" + "═"*100)
    print("DEBUG PC-KF — 20 samples đầu")
    print("═"*100)
    print(f"{'t':>4} | {'d_raw':>40} | {'innov':>40} | {'scores':>30} | {'R':>30}")
    print("─"*100)
    
    for t in range(min(n_samples, len(dist_raw))):
        d = dist_raw[t]
        
        if kf.x is None:
            kf.x = d.copy()
            print(f"{t:>4} | INIT")
            continue
        
        innov = np.abs(d - kf.x)
        scores = kf._consensus_scores(d)
        R = kf.R_base * (1.0 + kf.R_scale * (1.0 - scores))
        
        kf.update(d)
        
        print(f"{t:>4} | {str(np.round(d,1)):>40} | {str(np.round(innov,1)):>40} | {str(np.round(scores,3)):>30} | {str(np.round(R,1)):>30}")
    
    print("\n── Phân phối scores toàn bộ file ──")
    kf2 = PCKF()
    all_scores = []
    for t in range(len(dist_raw)):
        _, sc = kf2.update(dist_raw[t])
        all_scores.append(sc)
    all_scores = np.array(all_scores)
    
    print(f"  Score mean per anchor: {np.mean(all_scores, axis=0).round(3)}")
    print(f"  Score std  per anchor: {np.std(all_scores,  axis=0).round(3)}")
    print(f"  Score < 0.3 (nghi ngờ): {(all_scores < 0.3).sum(axis=0)} lần")
    print(f"  Score > 0.7 (tin tưởng): {(all_scores > 0.7).sum(axis=0)} lần")
    
    print("\n── R adaptive distribution ──")
    R_all = kf.R_base * (1.0 + kf.R_scale * (1.0 - all_scores))
    print(f"  R mean: {np.mean(R_all, axis=0).round(1)}")
    print(f"  R min:  {np.min(R_all,  axis=0).round(1)}")
    print(f"  R max:  {np.max(R_all,  axis=0).round(1)}")

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("V13a: PC-KF — Đã cập nhật logic tính toán RMSE chuẩn Code 2")
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"Không tìm thấy file .txt trong '{DATA_DIR}'"); return
    print(f"Tìm thấy {n} files")

    import random; random.seed(42); np.random.seed(42)
    perm        = np.random.permutation(n)
    shuffled    = [all_files[i] for i in perm]
    n_train     = min(0, n)
    n_val       = min(0, max(0, n - n_train))
    train_files = shuffled[:n_train]
    val_files   = shuffled[n_train:n_train + n_val]
    test_files  = shuffled[n_train + n_val:]

    # Khởi tạo GT chuẩn lý thuyết cố định
    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    total_path = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"        Path={total_path:.0f} mm  Time={total_time:.1f}s  GT_points={len(gt_xy)}")

    # Estimate sigma từ data
    innov_list = []
    for path in train_files:
        parsed = parse_file(path)
        if parsed is None: continue
        dist_raw = parsed['dist']
        d_kf     = kf_filter_file(dist_raw)
        innov    = np.abs(dist_raw - d_kf)
        innov_list.append(innov)
    if innov_list:
        all_innov = np.concatenate(innov_list, axis=0)
        sigma_est = float(np.sqrt(np.mean(all_innov**2)))
    else:
        sigma_est = PCKF_SIGMA

    best_params = {
        'q': PCKF_Q, 'r_base': PCKF_R_BASE,
        'r_scale': PCKF_R_SCALE, 'sigma': sigma_est,
    }
    if DO_GRID_SEARCH and val_files:
        best_params, _ = grid_search(val_files, gt_xy)

    eval_files = test_files if test_files else val_files
    if not eval_files:
        eval_files = train_files

    debug_pckf(eval_files[0], gt_xy)

    print(f"\n{'═'*60}\n  Evaluating trên {len(eval_files)} files\n{'═'*60}")
    err, positions = evaluate_files(
        eval_files, gt_xy,
        pckf_q=best_params['q'], pckf_r_base=best_params['r_base'],
        pckf_r_scale=best_params['r_scale'], pckf_sigma=best_params['sigma'],
    )

    metrics = {label: compute_metrics(errors, label) for label, errors in err.items()}

    # Bảng hiển thị
    print(f"\n{'═'*72}")
    print(f"  COMPARISON TABLE — V13a (Static GT)")
    print(f"{'═'*72}")
    print(f"  {'Method':<18s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*52}")
    for label, m in metrics.items():
        print(f"  {label:<18s} {m['rmse']:>6.1f} {m['mae']:>6.1f} {m['cep50']:>6.1f} {m['p95']:>6.1f} {m['max']:>6.1f}")

    # Wilcoxon
    print(f"\n  Wilcoxon tests:")
    for a, b in [('Raw + WLS','PC-KF + WLS'), ('KF + WLS','PC-KF + WLS')]:
        ea, eb = err[a], err[b]
        min_len = min(len(ea), len(eb))
        if min_len > 10:
            try:
                _, p = wilcoxon(ea[:min_len], eb[:min_len])
                print(f"    {a} vs {b}: p={p:.4f} {'✅' if p<0.05 else '⚠️'}")
            except ValueError:
                pass

    # Xuất đồ thị
    plot_cdf(err, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(
        {'Raw + WLS': positions['raw'], 'KF + WLS':  positions['kf'], 'PC-KF + WLS': positions['pckf']},
        gt_xy, os.path.join(SAVE_DIR, 'trajectories.png'),
    )
    plot_bar(metrics, os.path.join(SAVE_DIR, 'bar_comparison.png'))
    analyze_scores(
        eval_files,
        pckf_q=best_params['q'], pckf_r_base=best_params['r_base'],
        pckf_r_scale=best_params['r_scale'], pckf_sigma=best_params['sigma'],
        save_path=os.path.join(SAVE_DIR, 'score_analysis.png'),
    )

    for label, errors in err.items():
        safe = label.lower().replace(' + ','_').replace(' ','_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errors)

    print(f"\n[✓] Toàn bộ dữ liệu đầu ra được lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
