# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — PC-EKF-2D (V14 — 2D state, nonlinear observation)
============================================================================
THAY ĐỔI SO VỚI V13a:
  - EKF và PC-EKF hoạt động trên state 2D [x, y] thay vì 1D per-anchor
  - Observation model phi tuyến: h_i(x,y) = sqrt((x-ax_i)^2 + (y-ay_i)^2 + H^2)
  - Jacobian H tính analytically từ h_i
  - PC scoring vẫn dùng innovation từ bộ lọc 1D KF để tính consensus,
    sau đó truyền R_adaptive vào EKF 2D
  - WLS không còn được dùng để ra vị trí cuối — EKF 2D output thẳng (x,y)

SO SÁNH 3 PHƯƠNG PHÁP:
  1. Raw + WLS
  2. EKF-2D + WLS (khởi tạo bằng WLS, sau đó EKF update 2D)
  3. PC-EKF-2D    (PC adaptive R + EKF 2D, không qua WLS)

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

SPEED      = 200.0   # mm/s
GT_SPACING = 5.0     # mm
N_ANCHORS  = 4

# EKF-2D params
EKF2D_Q     = 0.01    # Process noise (mm^2) — scalar, dùng cho I*Q
EKF2D_R     = 200.0  # Measurement noise (mm^2) per anchor

# PC-EKF-2D params
PCEKF2D_Q       = 0.01
PCEKF2D_R_BASE  = 200.0
PCEKF2D_R_SCALE = 5.0
PCEKF2D_SIGMA   = 25.0

# PC scoring dùng KF 1D để tính innovation (giống V13a)
PC_KF_Q = 0.01
PC_KF_R = 200.0

DO_GRID_SEARCH = False
DATA_DIR = "./data"
SAVE_DIR = "./outputs_vPCEKF"


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


def wls_position(distances, anchors=ANCHORS):
    """WLS dùng để khởi tạo state ban đầu cho EKF."""
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi-x0), 2*(yi-y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
        w.append(1.0 / di)
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    W  = np.diag(w)
    try:
        pos, *_ = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


# ══════════════════════════════════════════════════════════════════════
#  OBSERVATION MODEL PHI TUYẾN
# ══════════════════════════════════════════════════════════════════════
def h_obs(state, anchors=ANCHORS):
    """
    Observation model: h_i(x,y) = sqrt((x-ax_i)^2 + (y-ay_i)^2 + H^2)
    Trả về vector khoảng cách dự đoán từ state (x,y) tới mỗi anchor.
    Đây là hàm phi tuyến — đây chính là điểm EKF khác KF.
    """
    x, y = state
    h = np.zeros(N_ANCHORS)
    for i, (ax, ay) in enumerate(anchors):
        h[i] = math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
    return h


def jacobian_H(state, anchors=ANCHORS):
    """
    Jacobian của h: H = dh/d[x,y], shape (N_ANCHORS, 2)
    dh_i/dx = (x - ax_i) / h_i
    dh_i/dy = (y - ay_i) / h_i
    """
    x, y = state
    H = np.zeros((N_ANCHORS, 2))
    for i, (ax, ay) in enumerate(anchors):
        dist = math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
        dist = max(dist, 1e-6)
        H[i, 0] = (x - ax) / dist
        H[i, 1] = (y - ay) / dist
    return H


# ══════════════════════════════════════════════════════════════════════
#  PC SCORING (dùng KF 1D để tính innovation — giữ nguyên logic V13a)
# ══════════════════════════════════════════════════════════════════════
class _KF1D_for_PC:
    """KF 1D nhẹ, chỉ dùng nội bộ để tính innovation cho PC scoring."""
    def __init__(self, q=PC_KF_Q, r=PC_KF_R):
        self.Q = q; self.R = r
        self.P = 1.0; self.x = None

    def update(self, z):
        if self.x is None:
            self.x = z; return 0.0
        P_pred     = self.P + self.Q
        innovation = z - self.x
        K          = P_pred / (P_pred + self.R)
        self.x    += K * innovation
        self.P     = (1 - K) * P_pred
        return innovation


def pc_consensus_scores(innovations, sigma=PCEKF2D_SIGMA):
    """
    Tính consensus score từ innovation của mỗi anchor.
    Score cao = anchor tin cậy, score thấp = anchor bị nhiễu/lỗi.
    """
    innov = np.abs(innovations)
    scores = np.zeros(N_ANCHORS)
    eps = 1e-9
    for i in range(N_ANCHORS):
        s = 0.0; cnt = 0
        for j in range(N_ANCHORS):
            if i == j: continue
            diff = innov[i] - innov[j]
            nu = 4.0
            c = (1.0 + (diff**2) / (nu * sigma**2 + eps)) ** (-(nu + 1.0) / 2.0)
            s += c; cnt += 1
        scores[i] = s / cnt
    return scores


# ══════════════════════════════════════════════════════════════════════
#  EKF 2D (baseline, R đồng nhất)
# ══════════════════════════════════════════════════════════════════════
class EKF2D:
    """
    Extended Kalman Filter với state [x, y].
    Observation model phi tuyến: h_i(x,y) = sqrt((x-ax)^2+(y-ay)^2+H^2)
    Jacobian H được tính analytically.
    """
    def __init__(self, q=EKF2D_Q, r=EKF2D_R):
        self.Q_scalar = q
        self.R_scalar = r
        self.x = None                     # state [x, y]
        self.P = None                     # covariance 2x2

    def init(self, x0):
        self.x = x0.copy()
        self.P = np.eye(2) * 1e6          # khởi đầu không chắc chắn

    def predict(self):
        # f(x) = x (constant velocity = 0, random walk)
        # F = I
        self.P = self.P + np.eye(2) * self.Q_scalar

    def update(self, z_raw):
        """
        z_raw: vector khoảng cách ground đo được (N_ANCHORS,)
        Trả về state (x,y) sau update.
        """
        if self.x is None:
            return np.array([np.nan, np.nan])

        # Observation model & Jacobian
        h  = h_obs(self.x)
        H  = jacobian_H(self.x)          # (N_ANCHORS, 2)
        R  = np.eye(N_ANCHORS) * self.R_scalar

        # Innovation
        innov = z_raw - h                 # (N_ANCHORS,)

        # Innovation covariance
        S = H @ self.P @ H.T + R          # (N_ANCHORS, N_ANCHORS)

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)   # (2, N_ANCHORS)
        except np.linalg.LinAlgError:
            return self.x.copy()

        # Update
        self.x = self.x + K @ innov
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.x.copy()

    def step(self, z_raw):
        self.predict()
        return self.update(z_raw)


# ══════════════════════════════════════════════════════════════════════
#  PC-EKF 2D (adaptive R từ PC scoring)
# ══════════════════════════════════════════════════════════════════════
class PCEKF2D:
    """
    PC-EKF 2D: PC scoring điều chỉnh R_i cho từng anchor,
    sau đó EKF 2D update state [x, y] với R_diag adaptive.
    """
    def __init__(self, q=PCEKF2D_Q, r_base=PCEKF2D_R_BASE,
                 r_scale=PCEKF2D_R_SCALE, sigma=PCEKF2D_SIGMA):
        self.Q_scalar = q
        self.R_base   = r_base
        self.R_scale  = r_scale
        self.sigma    = sigma
        self.x = None
        self.P = None
        # KF 1D per-anchor chỉ dùng để tính innovation cho PC scoring
        self._kfs = [_KF1D_for_PC() for _ in range(N_ANCHORS)]

    def init(self, x0):
        self.x = x0.copy()
        self.P = np.eye(2) * 1e6

    def predict(self):
        self.P = self.P + np.eye(2) * self.Q_scalar

    def update(self, z_raw):
        if self.x is None:
            return np.array([np.nan, np.nan]), np.ones(N_ANCHORS)

        # Bước 1: tính innovation per-anchor bằng KF 1D phụ trợ
        innovations = np.array([self._kfs[i].update(z_raw[i]) for i in range(N_ANCHORS)])

        # Bước 2: PC consensus score → R adaptive
        scores  = pc_consensus_scores(innovations, self.sigma)
        R_diag  = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        R       = np.diag(R_diag)                # (N_ANCHORS, N_ANCHORS)

        # Bước 3: EKF update 2D với R adaptive
        h = h_obs(self.x)
        H = jacobian_H(self.x)                   # (N_ANCHORS, 2)

        innov = z_raw - h
        S     = H @ self.P @ H.T + R
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
def ekf2d_filter_file(raw_dist, q=EKF2D_Q, r=EKF2D_R):
    T   = len(raw_dist)
    ekf = EKF2D(q=q, r=r)
    pos = np.full((T, 2), np.nan)

    # Khởi tạo bằng WLS từ frame đầu
    init_pos = wls_position(raw_dist[0])
    if not np.any(np.isnan(init_pos)):
        ekf.init(init_pos)
    else:
        ekf.init(np.array([2000.0, 4400.0]))  # center fallback

    for t in range(T):
        pos[t] = ekf.step(raw_dist[t])
    return pos


def pcekf2d_filter_file(raw_dist, q=PCEKF2D_Q, r_base=PCEKF2D_R_BASE,
                         r_scale=PCEKF2D_R_SCALE, sigma=PCEKF2D_SIGMA):
    T    = len(raw_dist)
    ekf  = PCEKF2D(q=q, r_base=r_base, r_scale=r_scale, sigma=sigma)
    pos  = np.full((T, 2), np.nan)
    scrs = np.zeros((T, N_ANCHORS))

    init_pos = wls_position(raw_dist[0])
    if not np.any(np.isnan(init_pos)):
        ekf.init(init_pos)
    else:
        ekf.init(np.array([2000.0, 4400.0]))

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
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════
import time  # Hãy đảm bảo đã import time ở đầu file cùng với các thư viện khác

def evaluate_files(file_paths, gt_xy, q=PCEKF2D_Q, r_base=PCEKF2D_R_BASE,
                   r_scale=PCEKF2D_R_SCALE, sigma=PCEKF2D_SIGMA):
    raw_pos_all   = []
    ekf_pos_all   = []
    pcekf_pos_all = []

    print("\n" + "═"*90)
    print("  PER-FILE RESULTS & EXECUTION TIME")
    print("═"*90)
    # Thêm cột thời gian chạy cho EKF và PC-EKF
    print(f"{'File':<18s} {'Raw+WLS':>10s} {'EKF-2D':>10s} {'PC-EKF-2D':>12s} | {'EKF T_file':>12s} {'PC T_file':>12s} {'PC T_sample':>12s}")
    print("─"*90)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None: continue

        dist_raw = parsed["dist"].astype(float)
        T = len(dist_raw)

        # 1. Đo Raw + WLS (baseline)
        raw_pos = np.array([wls_position(dist_raw[t]) for t in range(T)])

        # 2. Đo EKF-2D
        t_start_ekf = time.perf_counter()
        ekf_pos = ekf2d_filter_file(dist_raw)
        t_end_ekf = time.perf_counter()
        time_ekf_file = t_end_ekf - t_start_ekf  # Tính bằng giây

        # 3. Đo PC-EKF-2D
        t_start_pcekf = time.perf_counter()
        pcekf_pos, _ = pcekf2d_filter_file(dist_raw, q=q, r_base=r_base,
                                             r_scale=r_scale, sigma=sigma)
        t_end_pcekf = time.perf_counter()
        time_pcekf_file = t_end_pcekf - t_start_pcekf  # Tính bằng giây
        
        # Tính thời gian xử lý trung bình cho 1 mẫu (1 timestep) của PC-EKF-2D
        # Đổi từ giây sang mili-giây (ms) hoặc micro-giây (µs) để dễ nhìn
        time_pcekf_sample_ms = (time_pcekf_file / T) * 1000 

        def filt(pos):
            return pos[~np.any(np.isnan(pos), axis=1)]

        raw_pos   = filt(raw_pos)
        ekf_pos   = filt(ekf_pos)
        pcekf_pos = filt(pcekf_pos)

        raw_rmse   = np.sqrt(np.mean(nearest_gt_error(raw_pos,   gt_xy)**2))
        ekf_rmse   = np.sqrt(np.mean(nearest_gt_error(ekf_pos,   gt_xy)**2))
        pcekf_rmse = np.sqrt(np.mean(nearest_gt_error(pcekf_pos, gt_xy)**2))

        # In kết quả kèm thời gian xử lý
        print(f"{os.path.basename(path):<18s}{raw_rmse:>10.1f}{ekf_rmse:>10.1f}{pcekf_rmse:>12.1f} | "
              f"{time_ekf_file*1000:>10.1f}ms {time_pcekf_file*1000:>10.1f}ms {time_pcekf_sample_ms:>10.3f}ms")

        raw_pos_all.extend(raw_pos)
        ekf_pos_all.extend(ekf_pos)
        pcekf_pos_all.extend(pcekf_pos)

    print("─"*90)
    raw_arr   = np.array(raw_pos_all)
    ekf_arr   = np.array(ekf_pos_all)
    pcekf_arr = np.array(pcekf_pos_all)

    return {
        "Raw + WLS"   : nearest_gt_error(raw_arr,   gt_xy),
        "EKF-2D"      : nearest_gt_error(ekf_arr,   gt_xy),
        "PC-EKF-2D"   : nearest_gt_error(pcekf_arr, gt_xy),
    }, {
        "raw"  : raw_arr,
        "ekf"  : ekf_arr,
        "pcekf": pcekf_arr,
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
        'q'      : [0.1, 1.0, 10.0],
        'r_base' : [50.0, 100.0, 200.0, 300.0],
        'r_scale': [5.0, 10.0, 20.0],
        'sigma'  : [50.0, 100.0, 200.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search: {len(combos)} combinations...")

    best_rmse = float('inf'); best_params = {}; results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        err, _ = evaluate_files(val_files, gt_xy, **params)
        rmse = float(np.sqrt(np.mean(err['PC-EKF-2D']**2)))
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse = rmse; best_params = params

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs:")
    print(f"  {'RMSE':>8s} {'Q':>8s} {'R_base':>8s} {'r_scale':>8s} {'sigma':>8s}")
    for rmse, p in results[:5]:
        print(f"  {rmse:>7.1f}mm {p['q']:>8.3f} {p['r_base']:>8.1f} {p['r_scale']:>8.1f} {p['sigma']:>8.1f}")
    print(f"\n  Best: RMSE={best_rmse:.1f}mm | {best_params}")
    return best_params, best_rmse


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    'Raw + WLS' : '#9E9E9E',
    'EKF-2D'    : '#E91E63',
    'PC-EKF-2D' : '#2196F3',
}


def plot_cdf(errors_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, errors in errors_dict.items():
        s      = np.sort(errors)
        cdf    = np.arange(1, len(s) + 1) / len(s)
        rmse_v = np.sqrt(np.mean(errors**2))
        color  = COLORS.get(label, 'gray')
        lw     = 2.5 if 'PC' in label else 1.8
        ls     = '-'  if 'PC' in label else '--'
        ax.plot(s, cdf, lw=lw, color=color, ls=ls, label=f"{label}  (RMSE={rmse_v:.1f}mm)")
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
    n      = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 7))
    if n == 1: axes = [axes]
    for ax, label in zip(axes, labels):
        pos   = positions_dict[label]
        color = COLORS.get(label, 'gray')
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'k--', lw=2, label='Ground Truth', alpha=0.5)
        ax.plot(pos[:, 0],   pos[:, 1],   color=color, lw=0.8, alpha=0.7, label=label)
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
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' + ','\n+ ') for m in methods], fontsize=9, ha='center')
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


def analyze_scores(file_paths, q=PCEKF2D_Q, r_base=PCEKF2D_R_BASE,
                   r_scale=PCEKF2D_R_SCALE, sigma=PCEKF2D_SIGMA, save_path=None):
    path = file_paths[0]
    parsed = parse_file(path)
    if parsed is None: return
    dist_raw = parsed['dist'].astype(float)

    _, scores = pcekf2d_filter_file(dist_raw, q=q, r_base=r_base, r_scale=r_scale, sigma=sigma)
    R_adaptive = r_base * (1.0 + r_scale * (1.0 - scores))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']

    for i in range(N_ANCHORS):
        axes[0].plot(scores[:, i], color=colors[i], lw=0.8, label=f'Anchor {i+1}', alpha=0.8)
    axes[0].axhline(0.5, color='gray', ls='--', lw=1, label='threshold 0.5')
    axes[0].set_ylabel('Consensus Score', fontsize=11)
    axes[0].set_title(f'PC-EKF-2D Consensus Scores — {os.path.basename(path)}', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.05)

    for i in range(N_ANCHORS):
        axes[1].plot(R_adaptive[:, i], color=colors[i], lw=0.8, label=f'R Adaptive Anchor {i+1}', alpha=0.8)
    axes[1].set_ylabel('Adaptive R (mm²)', fontsize=11)
    axes[1].set_title('Adaptive R per Anchor (PC-EKF-2D)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    for i in range(N_ANCHORS):
        axes[2].plot(dist_raw[:, i], color=colors[i], lw=0.6, label=f'Raw d{i}', alpha=0.7)
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
    print("V14: PC-EKF-2D — EKF trên state 2D [x,y] với observation phi tuyến h(x,y)")
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    n = len(all_files)
    if n == 0:
        print(f"Không tìm thấy file .txt trong '{DATA_DIR}'"); return
    print(f"Tìm thấy {n} files")

    import random; random.seed(42); np.random.seed(42)
    perm     = np.random.permutation(n)
    shuffled = [all_files[i] for i in perm]
    n_train  = min(0, n)
    n_val    = min(0, max(0, n - n_train))
    train_files = shuffled[:n_train]
    val_files   = shuffled[n_train:n_train + n_val]
    test_files  = shuffled[n_train + n_val:]

    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    total_path = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"        Path={total_path:.0f} mm  Time={total_time:.1f}s  GT_points={len(gt_xy)}")

    best_params = {
        'q': PCEKF2D_Q, 'r_base': PCEKF2D_R_BASE,
        'r_scale': PCEKF2D_R_SCALE, 'sigma': PCEKF2D_SIGMA,
    }
    if DO_GRID_SEARCH and val_files:
        best_params, _ = grid_search(val_files, gt_xy)
    else:
        print("\n[INFO] Grid Search TẮT — dùng params mặc định.")

    eval_files = test_files if test_files else val_files
    if not eval_files:
        eval_files = train_files if train_files else all_files

    print(f"\n{'═'*60}\n  Evaluating trên {len(eval_files)} files\n{'═'*60}")
    err, positions = evaluate_files(eval_files, gt_xy, **best_params)

    metrics = {label: compute_metrics(errors, label) for label, errors in err.items()}

    print(f"\n{'═'*72}")
    print(f"  COMPARISON TABLE — V14 PC-EKF-2D")
    print(f"{'═'*72}")
    print(f"  {'Method':<18s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*52}")
    for label, m in metrics.items():
        print(f"  {label:<18s} {m['rmse']:>6.1f} {m['mae']:>6.1f} {m['cep50']:>6.1f} {m['p95']:>6.1f} {m['max']:>6.1f}")

    print(f"\n  Wilcoxon tests:")
    for a, b in [('Raw + WLS', 'PC-EKF-2D'), ('EKF-2D', 'PC-EKF-2D')]:
        ea, eb = err[a], err[b]
        min_len = min(len(ea), len(eb))
        if min_len > 10:
            try:
                _, p = wilcoxon(ea[:min_len], eb[:min_len])
                print(f"    {a} vs {b}: p={p:.4f} {'✅' if p<0.05 else '⚠️'}")
            except ValueError:
                pass

    plot_cdf(err, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(
        {'Raw + WLS': positions['raw'], 'EKF-2D': positions['ekf'], 'PC-EKF-2D': positions['pcekf']},
        gt_xy, os.path.join(SAVE_DIR, 'trajectories.png'),
    )
    plot_bar(metrics, os.path.join(SAVE_DIR, 'bar_comparison.png'))
    analyze_scores(eval_files, save_path=os.path.join(SAVE_DIR, 'score_analysis.png'), **best_params)

    for label, errors in err.items():
        safe = label.lower().replace(' + ','_').replace(' ','_').replace('-','_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errors)

    print(f"\n[✓] Toàn bộ kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
