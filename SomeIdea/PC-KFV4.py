# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — PC-KF (V3 — Gaussian Pairwise Consistency, Sigma-Free)
=================================================================================
PC scoring theo cơ chế của sotaV2 (PC-EKF-V4):
  - Innovation từ KF predict step: ν[i] = z[i] - x̂[i]
  - S_ii = P_pred[i] + R_base                         (per-anchor prediction variance)
  - nm[i] = ν[i] / sqrt(S_ii)                         → chuẩn hóa ~ N(0,1) nếu inlier
  - Pairwise Gaussian kernel: score[i] = mean exp(-0.25*(nm[i]-nm[j])²) for j≠i
    (variance=2 vì diff hai N(0,1))
  - R_adaptive[i] = R_base * (1 + r_scale*(1 - score[i]))
  - KF update với R_adaptive, rồi LS position

Khác V2/V3-cũ (MAD + T-kernel):
  - Không có bước MAD normalize thứ hai
  - Dùng Gaussian kernel thay T-kernel
  - Giống hệt _pc_scores_gauss_nb trong PC-EKFV4

SO SÁNH 3 PHƯƠNG PHÁP:
  1. Raw + LS
  2. KF + LS         (fixed Q, R)
  3. PC-KF-v3 + LS   (Gaussian pairwise, sigma-free)

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

# KF baseline params
KF_Q = 0.01
KF_R = 50.0

# PC-KF-v3 params  — chỉ cần tune r_scale!
PCKF_Q       = 0.01
PCKF_R_BASE  = 50.0
PCKF_R_SCALE = 5.0    # ← PARAM DUY NHẤT CẦN TUNE (thử 1 → 30)

DO_GRID_SEARCH = True
DATA_DIR = "./data"
SAVE_DIR = "./outputs_PCKF_v4"


# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _kf_file_nb(raw_dist, q, r):
    """
    KF 1D per-anchor trên toàn bộ file.
    Logic giống hệt kf_filter_file + KF1D.update gốc.
    """
    T = raw_dist.shape[0]
    N = raw_dist.shape[1]
    d_kf = np.zeros((T, N))
    x    = np.zeros(N)
    P    = np.ones(N)
    init = np.zeros(N, dtype=np.bool_)

    for t in range(T):
        for i in range(N):
            z = raw_dist[t, i]
            if not init[i]:
                x[i]    = z
                init[i] = True
                d_kf[t, i] = z
            else:
                P_pred     = P[i] + q
                innovation = z - x[i]
                K          = P_pred / (P_pred + r)
                x[i]      += K * innovation
                P[i]       = (1.0 - K) * P_pred
                d_kf[t, i] = x[i]
    return d_kf


@njit(cache=True)
def _pckf_step_nb(x, P, d_raw, q, r_base, r_scale):
    """
    Một bước PC-KF V3: predict + score (Gaussian pairwise) + update cho 4 anchor.
    Giống hệt cơ chế PC-EKF-V4 (_pc_scores_gauss_nb trong sotaV2).
    """
    N = 4

    # 1. Predict
    P_pred = P + q                          # (N,)

    # 2. Innovation
    innov = d_raw - x                       # (N,)

    # 3. S_diag = P_pred + r_base
    S_diag = P_pred + r_base               # (N,)

    # 4. nm[i] = innov[i] / sqrt(S_diag[i])  ~ N(0,1) nếu inlier
    nm = np.empty(N)
    for i in range(N):
        s = S_diag[i]; s = s if s > 1e-9 else 1e-9
        nm[i] = innov[i] / (s ** 0.5)

    # 5. Pairwise Gaussian kernel (sotaV2/EKFV4):
    #    score[i] = mean exp(-0.25*(nm[i]-nm[j])²) for j≠i
    #    Variance của hiệu = 2 (diff 2 N(0,1)) → -d²/(2*2) = -0.25*d²
    scores = np.empty(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if j != i:
                d = nm[i] - nm[j]
                s += math.exp(-0.25 * d * d)
        scores[i] = s / 3.0   # mean over N-1=3 pairs

    # 6. Adaptive R
    R_adap = np.empty(N)
    for i in range(N):
        R_adap[i] = r_base * (1.0 + r_scale * (1.0 - scores[i]))

    # 7. KF update
    K = P_pred / (P_pred + R_adap)
    x = x + K * innov
    P = (1.0 - K) * P_pred

    return x, P, scores


@njit(cache=True)
def _pckf_file_nb(raw_dist, q, r_base, r_scale):
    """
    PC-KF V3 trên toàn bộ file (Gaussian pairwise, sigma-free).
    """
    T = raw_dist.shape[0]
    N = raw_dist.shape[1]
    d_kf = np.zeros((T, N))
    scrs = np.zeros((T, N))

    # Khởi tạo (timestep đầu tiên)
    x = raw_dist[0].copy()
    P = np.ones(N)
    d_kf[0] = x
    scrs[0] = np.ones(N)

    for t in range(1, T):
        x, P, scrs[t] = _pckf_step_nb(x, P, raw_dist[t], q, r_base, r_scale)
        d_kf[t] = x

    return d_kf, scrs


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
    _raw = np.ones((2, 4), dtype=np.float64) * 3000.0
    _kf_file_nb(_raw, 0.01, 50.0)
    _pckf_file_nb(_raw, 0.01, 50.0, 5.0)
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
#  GAUSSIAN PAIRWISE KERNEL  (theo sotaV2 / PC-EKFV4)
# ══════════════════════════════════════════════════════════════════════
def _gauss_kernel(diff_vec):
    """Gaussian kernel với variance=2 (diff 2 N(0,1)): exp(-d²/4)."""
    return np.exp(-0.25 * diff_vec**2)


# ══════════════════════════════════════════════════════════════════════
#  PC SCORING — Gaussian Pairwise (sotaV2 / PC-EKFV4 mechanism)
# ══════════════════════════════════════════════════════════════════════
def pc_scores_gauss(innovations, S_diag):
    """
    Gaussian pairwise consistency scoring — giống PC-EKFV4 (_pc_scores_gauss_nb).

    Bước 1 — Chuẩn hóa theo S_ii:
        nm[i] = innovations[i] / sqrt(S_ii)
        S_ii = P_pred[i] + R_base
        → nm[i] ~ N(0,1) khi LOS

    Bước 2 — Pairwise Gaussian kernel:
        score[i] = mean exp(-0.25*(nm[i]-nm[j])²)  for j≠i
        (variance=2 vì diff 2 N(0,1) → -d²/(2*2))

    Anchor nào "xa" đám đông → score thấp → R inflate nhiều.
    Không cần MAD normalize, không cần tune sigma.
    """
    nm = innovations / np.sqrt(np.maximum(S_diag, 1e-9))
    scores = np.zeros(N_ANCHORS)
    for i in range(N_ANCHORS):
        diffs     = np.array([nm[i] - nm[j]
                              for j in range(N_ANCHORS) if j != i])
        scores[i] = np.mean(_gauss_kernel(diffs))
    return scores


# ══════════════════════════════════════════════════════════════════════
#  KF 1D BASELINE
# ══════════════════════════════════════════════════════════════════════
class KF1D:
    """KF 1D per-anchor, dùng cho baseline KF + LS."""
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
#  PC-KF V3 — Gaussian Pairwise, Sigma-Free
# ══════════════════════════════════════════════════════════════════════
class PCKF_v3:
    """
    PC-KF V3: KF 1D per-anchor với PC scoring Gaussian pairwise.
    Giống cơ chế PC-EKFV4. Không cần tune sigma — chỉ tune r_scale.

    Pipeline mỗi timestep:
      1. Predict: P_pred[i] = P[i] + Q  (per-anchor scalar)
      2. Innovation: ν[i] = z[i] - x̂[i]
      3. S_ii = P_pred[i] + R_base       (per-anchor prediction variance)
      4. nm[i] = ν[i] / sqrt(S_ii)       → chuẩn hóa ~ N(0,1)
      5. Gaussian pairwise kernel → score[i] ∈ [0,1]
      6. R_adaptive[i] = R_base * (1 + r_scale * (1 - score[i]))
      7. KF update: K[i] = P_pred[i] / (P_pred[i] + R_adaptive[i])
                    x̂[i] += K[i] * ν[i]
                    P[i]  = (1 - K[i]) * P_pred[i]
      8. LS position từ filtered distances
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

        # 4-5. PC scoring Gaussian pairwise (giống PC-EKFV4)
        scores = pc_scores_gauss(innov, S_diag)

        # 6. Adaptive R
        R_adap = self.R_base * (1.0 + self.R_scale * (1.0 - scores))

        # 7. KF update với R_adaptive
        K      = P_pred / (P_pred + R_adap)
        self.x = self.x + K * innov
        self.P = (1 - K) * P_pred

        return self.x.copy(), scores


# ══════════════════════════════════════════════════════════════════════
#  FILTER WRAPPERS — dùng Numba kernel
# ══════════════════════════════════════════════════════════════════════
def kf_filter_file(raw_dist, q=KF_Q, r=KF_R):
    d_kf = _kf_file_nb(raw_dist.astype(np.float64), float(q), float(r))
    return d_kf.astype(np.float32)


def pckf_v3_filter_file(raw_dist, q=PCKF_Q, r_base=PCKF_R_BASE,
                         r_scale=PCKF_R_SCALE):
    d_kf, scrs = _pckf_file_nb(
        raw_dist.astype(np.float64),
        float(q), float(r_base), float(r_scale))
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
                   q=PCKF_Q, r_base=PCKF_R_BASE, r_scale=PCKF_R_SCALE):
    raw_pos_all  = []
    kf_pos_all   = []
    pckf_pos_all = []
    per_file_rmse = {'Raw + LS': [], 'KF + LS': [], 'PC-KF-v3 + LS': []}

    W = 95
    print("\n" + "═"*W)
    print(f"  PC-KF V3 — PER-FILE RESULTS")
    print("═"*W)
    print(f"{'File':<18s} {'Raw+LS':>10s} {'KF+LS':>10s} {'PC-KF-v3':>12s} | "
          f"{'KF T_file':>11s} {'PCKF T_fl':>11s} {'PCKF T_smpl':>12s}")
    print("─"*W)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None: continue
        dist_raw = parsed['dist'].astype(float)
        T = len(dist_raw)

        dist_f64 = dist_raw.astype(np.float64)

        # 1. Raw + LS
        raw_pos = _ls_file_nb(dist_f64, ANCHORS)

        # 2. KF + LS
        t0 = time.perf_counter()
        d_kf   = kf_filter_file(dist_raw)
        kf_pos = _ls_file_nb(d_kf.astype(np.float64), ANCHORS)
        t1 = time.perf_counter()

        # 3. PC-KF-v3 + LS
        d_pckf, _ = pckf_v3_filter_file(
            dist_raw, q=q, r_base=r_base, r_scale=r_scale)
        pckf_pos = _ls_file_nb(d_pckf.astype(np.float64), ANCHORS)
        t2 = time.perf_counter()

        def filt(p): return p[~np.any(np.isnan(p), axis=1)]
        def rmse(p):
            e = nearest_gt_error(filt(p), gt_xy)
            return float(np.sqrt(np.mean(e**2))) if len(e) > 0 else float('nan')

        r_raw  = rmse(raw_pos)
        r_kf   = rmse(kf_pos)
        r_pckf = rmse(pckf_pos)

        t_kf   = (t1 - t0) * 1000
        t_pckf = (t2 - t1) * 1000
        print(f"{os.path.basename(path):<18s}"
              f"{r_raw:>10.1f}{r_kf:>10.1f}"
              f"{r_pckf:>12.1f} | "
              f"{t_kf:>9.1f}ms {t_pckf:>9.1f}ms {t_pckf/T*1000:>10.3f}ms")

        per_file_rmse['Raw + LS']     .append(r_raw)
        per_file_rmse['KF + LS']      .append(r_kf)
        per_file_rmse['PC-KF-v3 + LS'].append(r_pckf)

        raw_pos_all .extend(filt(raw_pos))
        kf_pos_all  .extend(filt(kf_pos))
        pckf_pos_all.extend(filt(pckf_pos))

    print("─"*W)
    raw_arr  = np.array(raw_pos_all)
    kf_arr   = np.array(kf_pos_all)
    pckf_arr = np.array(pckf_pos_all)

    errors = {
        'Raw + LS'      : nearest_gt_error(raw_arr,  gt_xy),
        'KF + LS'       : nearest_gt_error(kf_arr,   gt_xy),
        'PC-KF-v3 + LS' : nearest_gt_error(pckf_arr, gt_xy),
    }
    positions = {'raw': raw_arr, 'kf': kf_arr, 'pckf': pckf_arr}
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
    Grid search cho PC-KF V3.
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
    print(f"\n  Grid search PC-KF V3: {n_combo} combinations "
          f"(vs {n_combo * 6} với sigma — giảm 6x)...")

    best_rmse   = float('inf')
    best_params = {}
    results     = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        err, _, pf = evaluate_files(val_files, gt_xy, **params)
        vals = [v for v in pf['PC-KF-v3 + LS'] if not math.isnan(v)]
        rmse = float(np.mean(vals)) if vals else float('inf')
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params.copy()
        if (idx + 1) % 20 == 0 or (idx + 1) == n_combo:
            print(f"  [{idx+1}/{n_combo}] best so far: mean RMSE={best_rmse:.1f}mm")

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs (PC-KF V3):")
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
    'Raw + LS'      : '#9E9E9E',
    'KF + LS'       : '#E91E63',
    'PC-KF-v3 + LS' : '#2196F3',
}
LS = {
    'Raw + LS'      : ':',
    'KF + LS'       : '--',
    'PC-KF-v3 + LS' : '-',
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


def analyze_scores(file_paths, q=PCKF_Q, r_base=PCKF_R_BASE,
                   r_scale=PCKF_R_SCALE, save_path=None):
    """Phân tích scores V3 và R adaptive trên file đầu tiên."""
    parsed = parse_file(file_paths[0])
    if parsed is None: return
    dist_raw = parsed['dist'].astype(float)

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
        f'PC-KF V3 Scores (Gaussian pairwise, sigma-free) — {os.path.basename(file_paths[0])}',
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
    print("PC-KF V3 — Gaussian Pairwise, Sigma-Free")
    print("  Không dùng MAD normalize")
    print("  Innovation từ KF predict, normalize S_ii = P_pred[i] + R_base")
    print("  Gaussian pairwise kernel → sigma-free, KHÔNG CẦN tune sigma")
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
    print(f"  NOTE: sigma đã bị loại (Gaussian pairwise tự normalize)")

    err, positions, per_file_rmse = evaluate_files(eval_files, gt_xy, **best_params)
    metrics = {label: compute_metrics(errors, label, pf_rmse=per_file_rmse.get(label))
               for label, errors in err.items()}

    # Summary
    print(f"\n{'═'*88}")
    print(f"  SUMMARY — PC-KF V3 (sigma-free)")
    print(f"{'═'*88}")
    print(f"  {'Method':<22s} {'RMSE mean±std':>18s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*72}")
    for label, m in metrics.items():
        tag = " ◀ V3" if 'PC-KF-v3' in label else ""
        if not math.isnan(m.get('rmse_std', float('nan'))):
            rmse_str = f"{m['rmse_mean']:>6.1f} ± {m['rmse_std']:.1f}"
        else:
            rmse_str = f"{m['rmse_mean']:>6.1f} ± N/A"
        print(f"  {label:<22s} {rmse_str:>18s} {m['mae']:>6.1f} "
              f"{m['cep50']:>6.1f} {m['p95']:>6.1f} {m['max']:>6.1f}{tag}")

    # Wilcoxon
    print(f"\n  Wilcoxon tests (vs PC-KF-v3):")
    for a in ['Raw + LS', 'KF + LS']:
        ea, eb = err[a], err['PC-KF-v3 + LS']
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
    print(f"  [sigma-free: Gaussian pairwise tự normalize, không cần tune]")

    # Plots
    plot_cdf(err, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(
        {'Raw + LS'      : positions['raw'],
         'KF + LS'       : positions['kf'],
         'PC-KF-v3 + LS' : positions['pckf']},
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
