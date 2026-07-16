# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — PC-UKF-2D + PC-LS (V15)
==================================================
THAY ĐỔI SO VỚI V14:
  - EKF-2D → UKF-2D: dùng Unscented Transform (sigma points) thay Jacobian
    → không cần tính H analytically, xử lý tốt hơn phi tuyến mạnh
  - PC-EKF-2D → PC-UKF-2D: PC adaptive R + UKF 2D
  - Thêm PC-LS: Weighted Least Squares với trọng số từ PC consensus scores
    (không có filter state, chỉ dùng WLS có trọng số per-anchor)

SO SÁNH 4 PHƯƠNG PHÁP:
  1. Raw + WLS       — baseline không lọc
  2. UKF-2D          — UKF với R đồng nhất
  3. PC-UKF-2D       — PC adaptive R + UKF 2D
  4. PC-LS           — WLS có trọng số PC (không filter)

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

# UKF-2D params
UKF2D_Q     = 0.001    # Process noise (mm^2)
UKF2D_R     = 50.0  # Measurement noise (mm^2) per anchor
UKF2D_ALPHA = 1e-3   # UKF spread parameter
UKF2D_BETA  = 2.0    # UKF distribution parameter (2 optimal for Gaussian)
UKF2D_KAPPA = 0.0    # UKF secondary scaling

# PC-UKF-2D params
PCUKF2D_Q       = 0.001
PCUKF2D_R_BASE  = 50.0
PCUKF2D_R_SCALE = 5.0
PCUKF2D_SIGMA   = 200.0

# PC-LS params (no filter state, just weighted WLS per timestep)
PCLS_R_BASE  = 200.0
PCLS_R_SCALE = 10.0
PCLS_SIGMA   = 200.0

# PC scoring dùng KF 1D để tính innovation
PC_KF_Q = 0.01
PC_KF_R = 200.0

DO_GRID_SEARCH = False
DATA_DIR = "./data"
SAVE_DIR = "./outputs_vPCUKF"


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
    """
    Weighted Least Squares: giải hệ phương trình tuyến tính hoá từ TOA.
    weights: vector trọng số per-anchor (nếu None → WLS đồng nhất = 1/d)
    """
    x0, y0 = anchors[0]
    d0     = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di     = max(distances[i], 1.0)
        rows.append([2*(xi-x0), 2*(yi-y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
        # Trọng số: dùng weights nếu có, ngược lại dùng 1/di
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
#  OBSERVATION MODEL PHI TUYẾN
# ══════════════════════════════════════════════════════════════════════
def h_obs(state, anchors=ANCHORS):
    """h_i(x,y) = sqrt((x-ax_i)^2 + (y-ay_i)^2 + H^2)"""
    x, y = state[0], state[1]
    h = np.zeros(N_ANCHORS)
    for i, (ax, ay) in enumerate(anchors):
        h[i] = math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
    return h


# ══════════════════════════════════════════════════════════════════════
#  UNSCENTED TRANSFORM UTILITIES
# ══════════════════════════════════════════════════════════════════════
def ukf_weights(n, alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
    """
    Tính trọng số UKF cho 2n+1 sigma points.
    lambda_ = alpha^2*(n+kappa) - n
    Wm: trọng số cho mean
    Wc: trọng số cho covariance
    """
    lam = alpha**2 * (n + kappa) - n
    c   = n + lam

    Wm = np.full(2*n + 1, 0.5 / c)
    Wc = np.full(2*n + 1, 0.5 / c)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)

    return Wm, Wc, c   # c = n + lambda (dùng để tạo sigma points)


def sigma_points(x, P, c):
    """
    Tạo 2n+1 sigma points từ mean x và covariance P.
    x: (n,), P: (n,n), c: n+lambda scalar
    Trả về sigma_pts shape (2n+1, n)
    """
    n   = len(x)
    try:
        S = np.linalg.cholesky(c * P)   # Lower triangular, shape (n,n)
    except np.linalg.LinAlgError:
        # Nếu P không SPD, regularize
        S = np.linalg.cholesky(c * (P + np.eye(n) * 1e-6))

    pts = np.zeros((2*n + 1, n))
    pts[0] = x
    for i in range(n):
        pts[i + 1]     = x + S[:, i]
        pts[n + i + 1] = x - S[:, i]
    return pts


# ══════════════════════════════════════════════════════════════════════
#  UKF 2D (thay thế EKF-2D)
# ══════════════════════════════════════════════════════════════════════
class UKF2D:
    """
    Unscented Kalman Filter với state [x, y].
    Observation model phi tuyến: h_i(x,y) = sqrt((x-ax)^2+(y-ay)^2+H^2)
    Không cần Jacobian — dùng sigma points để tính mean/cov xấp xỉ.
    """
    def __init__(self, q=UKF2D_Q, r=UKF2D_R,
                 alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
        self.n         = 2
        self.Q_scalar  = q
        self.R_scalar  = r
        self.Wm, self.Wc, self.c = ukf_weights(self.n, alpha, beta, kappa)
        self.x = None     # state [x, y]
        self.P = None     # covariance (2,2)

    def init(self, x0):
        self.x = x0.copy().astype(float)
        self.P = np.eye(self.n) * 1e6

    def predict(self):
        """
        Predict: f(x) = x (random walk model, không có velocity)
        Sigma points qua f → vẫn là chính nó.
        P_pred = P + Q*I
        """
        self.P = self.P + np.eye(self.n) * self.Q_scalar

    def update(self, z_raw, R_diag=None):
        """
        UKF update:
          1. Tạo sigma points từ x, P
          2. Truyền qua h_obs → sigma points trong không gian đo lường
          3. Tính mean, cov trong không gian đo lường
          4. Cross-covariance → Kalman gain
          5. Update state

        z_raw: (N_ANCHORS,) khoảng cách ground đo được
        R_diag: None → dùng R_scalar * I; hoặc vector (N_ANCHORS,) adaptive
        """
        if self.x is None:
            return np.array([np.nan, np.nan])

        n  = self.n
        Wm = self.Wm
        Wc = self.Wc

        # --- 1. Sigma points ---
        pts = sigma_points(self.x, self.P, self.c)   # (2n+1, n)

        # --- 2. Propagate qua h_obs ---
        Z_pts = np.array([h_obs(pts[i]) for i in range(2*n + 1)])  # (2n+1, N_ANCHORS)

        # --- 3. Mean & cov trong measurement space ---
        z_hat = Wm @ Z_pts                              # (N_ANCHORS,)

        if R_diag is None:
            R = np.eye(N_ANCHORS) * self.R_scalar
        else:
            R = np.diag(R_diag)

        Pzz = R.copy()
        for i in range(2*n + 1):
            dz    = Z_pts[i] - z_hat                   # (N_ANCHORS,)
            Pzz  += Wc[i] * np.outer(dz, dz)

        # --- 4. Cross-covariance Pxz ---
        Pxz = np.zeros((n, N_ANCHORS))
        for i in range(2*n + 1):
            dx    = pts[i] - self.x                    # (n,)
            dz    = Z_pts[i] - z_hat                   # (N_ANCHORS,)
            Pxz  += Wc[i] * np.outer(dx, dz)

        # --- 5. Kalman gain & update ---
        try:
            K = Pxz @ np.linalg.inv(Pzz)              # (n, N_ANCHORS)
        except np.linalg.LinAlgError:
            return self.x.copy()

        innov  = z_raw - z_hat
        self.x = self.x + K @ innov
        self.P = self.P - K @ Pzz @ K.T

        # Đảm bảo P đối xứng, dương
        self.P = 0.5 * (self.P + self.P.T)
        self.P = self.P + np.eye(n) * 1e-9

        return self.x.copy()

    def step(self, z_raw, R_diag=None):
        self.predict()
        return self.update(z_raw, R_diag)


# ══════════════════════════════════════════════════════════════════════
#  PC SCORING (KF 1D phụ trợ — giữ nguyên logic V14)
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


def pc_consensus_scores(innovations, sigma=PCUKF2D_SIGMA, d_raw=None):
    # dùng residual nếu có d_raw, fallback về innovation
    if d_raw is not None:
        pos_est = wls_position(d_raw)
        if not np.any(np.isnan(pos_est)):
            residual = np.zeros(N_ANCHORS)
            for i, (ax, ay) in enumerate(ANCHORS):
                d_est = math.sqrt((pos_est[0]-ax)**2 + (pos_est[1]-ay)**2)
                residual[i] = abs(d_raw[i] - d_est)
        else:
            residual = np.abs(innovations)
    else:
        residual = np.abs(innovations)

    scores = np.zeros(N_ANCHORS)
    eps = 1e-9
    for i in range(N_ANCHORS):
        s = 0.0; cnt = 0
        for j in range(N_ANCHORS):
            if i == j: continue
            diff = residual[i] - residual[j]
            nu = 4.0
            c = (1.0 + (diff**2) / (nu * sigma**2 + eps)) ** (-(nu+1.0)/2.0)
            s += c; cnt += 1
        scores[i] = s / cnt
    return scores


# ══════════════════════════════════════════════════════════════════════
#  PC-UKF 2D (PC adaptive R + UKF 2D)
# ══════════════════════════════════════════════════════════════════════
class PCUKF2D:
    """
    PC-UKF 2D:
      - KF 1D per-anchor → innovation → PC consensus score
      - Score → R_adaptive (anchor tin cậy hơn → R nhỏ hơn)
      - UKF 2D update với R_adaptive
    """
    def __init__(self, q=PCUKF2D_Q, r_base=PCUKF2D_R_BASE,
                 r_scale=PCUKF2D_R_SCALE, sigma=PCUKF2D_SIGMA,
                 alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
        self.R_base   = r_base
        self.R_scale  = r_scale
        self.sigma    = sigma
        self._ukf     = UKF2D(q=q, r=r_base, alpha=alpha, beta=beta, kappa=kappa)
        self._kfs     = [_KF1D_for_PC() for _ in range(N_ANCHORS)]

    def init(self, x0):
        self._ukf.init(x0)

    def step(self, z_raw):
        # Bước 1: innovation per-anchor từ KF 1D phụ trợ
        innovations = np.array([self._kfs[i].update(z_raw[i]) for i in range(N_ANCHORS)])

        # Bước 2: PC consensus score → R adaptive
        scores = pc_consensus_scores(innovations, self.sigma, d_raw=z_raw)
        R_diag = self.R_base * (1.0 + self.R_scale * (1.0 - scores))

        # Bước 3: UKF step với R_adaptive
        pos = self._ukf.step(z_raw, R_diag=R_diag)
        return pos, scores, R_diag


# ══════════════════════════════════════════════════════════════════════
#  PC-LS (Weighted Least Squares với PC scores, không filter state)
# ══════════════════════════════════════════════════════════════════════
class PCLS:
    """
    PC-LS: Weighted Least Squares per timestep.
    Không có filter state (không predict/update theo thời gian).
    Trọng số per-anchor = PC consensus score.

    Ưu điểm: đơn giản, không có lag từ filter.
    Nhược điểm: không có temporal smoothing, nhạy nhiễu hơn UKF.
    """
    def __init__(self, r_base=PCLS_R_BASE, r_scale=PCLS_R_SCALE, sigma=PCLS_SIGMA):
        self.R_base  = r_base
        self.R_scale = r_scale
        self.sigma   = sigma
        self._kfs    = [_KF1D_for_PC() for _ in range(N_ANCHORS)]

    def step(self, z_raw):
        # KF 1D phụ trợ → innovation → PC score
        innovations = np.array([self._kfs[i].update(z_raw[i]) for i in range(N_ANCHORS)])
        scores      = pc_consensus_scores(innovations, self.sigma)

        # Trọng số: score cao → trọng số lớn
        # R_diag tỉ lệ nghịch với độ tin cậy, nên weights = 1/R_diag
        R_diag  = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        weights = 1.0 / (R_diag + 1e-9)    # (N_ANCHORS,)

        # Weighted Least Squares
        pos = wls_position(z_raw, weights=weights)
        return pos, scores


# ══════════════════════════════════════════════════════════════════════
#  FILTER WRAPPERS
# ══════════════════════════════════════════════════════════════════════
def ukf2d_filter_file(raw_dist, q=UKF2D_Q, r=UKF2D_R,
                      alpha=UKF2D_ALPHA, beta=UKF2D_BETA, kappa=UKF2D_KAPPA):
    T   = len(raw_dist)
    ukf = UKF2D(q=q, r=r, alpha=alpha, beta=beta, kappa=kappa)
    pos = np.full((T, 2), np.nan)

    init_pos = wls_position(raw_dist[0])
    ukf.init(init_pos if not np.any(np.isnan(init_pos)) else np.array([2000.0, 4400.0]))

    for t in range(T):
        pos[t] = ukf.step(raw_dist[t])
    return pos


def pcukf2d_filter_file(raw_dist, q=PCUKF2D_Q, r_base=PCUKF2D_R_BASE,
                         r_scale=PCUKF2D_R_SCALE, sigma=PCUKF2D_SIGMA):
    T    = len(raw_dist)
    ekf  = PCUKF2D(q=q, r_base=r_base, r_scale=r_scale, sigma=sigma)
    pos  = np.full((T, 2), np.nan)
    scrs = np.zeros((T, N_ANCHORS))

    init_pos = wls_position(raw_dist[0])
    ekf.init(init_pos if not np.any(np.isnan(init_pos)) else np.array([2000.0, 4400.0]))

    for t in range(T):
        pos[t], scrs[t], _ = ekf.step(raw_dist[t])
    return pos, scrs


def pcls_filter_file(raw_dist, r_base=PCLS_R_BASE, r_scale=PCLS_R_SCALE, sigma=PCLS_SIGMA):
    T    = len(raw_dist)
    pcls = PCLS(r_base=r_base, r_scale=r_scale, sigma=sigma)
    pos  = np.full((T, 2), np.nan)
    scrs = np.zeros((T, N_ANCHORS))

    for t in range(T):
        pos[t], scrs[t] = pcls.step(raw_dist[t])
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
import time

def evaluate_files(file_paths, gt_xy,
                   q=PCUKF2D_Q, r_base=PCUKF2D_R_BASE,
                   r_scale=PCUKF2D_R_SCALE, sigma=PCUKF2D_SIGMA):
    raw_pos_all   = []
    ukf_pos_all   = []
    pcukf_pos_all = []
    pcls_pos_all  = []

    print("\n" + "═"*105)
    print("  PER-FILE RESULTS & EXECUTION TIME")
    print("═"*105)
    print(f"{'File':<18s} {'Raw+WLS':>10s} {'UKF-2D':>10s} {'PC-UKF-2D':>12s} {'PC-LS':>10s} | {'UKF T_file':>11s} {'PCUKF T_fl':>11s} {'PCUKF T_smpl':>12s}")
    print("─"*105)

    for path in file_paths:
        parsed = parse_file(path)
        if parsed is None: continue

        dist_raw = parsed["dist"].astype(float)
        T = len(dist_raw)

        # 1. Raw + WLS
        raw_pos = np.array([wls_position(dist_raw[t]) for t in range(T)])

        # 2. UKF-2D (Đo thời gian chạy bộ lọc UKF không adaptive)
        t_start_ukf = time.perf_counter()
        ukf_pos = ukf2d_filter_file(dist_raw, q=q, r=r_base)
        t_end_ukf = time.perf_counter()
        time_ukf_file = t_end_ukf - t_start_ukf

        # 3. PC-UKF-2D (Đo thời gian chạy PC adaptive R + UKF)
        t_start_pcukf = time.perf_counter()
        pcukf_pos, _ = pcukf2d_filter_file(dist_raw, q=q, r_base=r_base,
                                             r_scale=r_scale, sigma=sigma)
        t_end_pcukf = time.perf_counter()
        time_pcukf_file = t_end_pcukf - t_start_pcukf

        # 4. PC-LS
        pcls_pos, _ = pcls_filter_file(dist_raw, r_base=r_base,
                                        r_scale=r_scale, sigma=sigma)

        # Thời gian xử lý trung bình cho 1 mẫu (timestep) của PC-UKF-2D (đơn vị: ms)
        time_pcukf_sample_ms = (time_pcukf_file / T) * 1000

        def filt(pos):
            return pos[~np.any(np.isnan(pos), axis=1)]

        raw_pos   = filt(raw_pos)
        ukf_pos   = filt(ukf_pos)
        pcukf_pos = filt(pcukf_pos)
        pcls_pos  = filt(pcls_pos)

        raw_rmse   = np.sqrt(np.mean(nearest_gt_error(raw_pos,   gt_xy)**2))
        ukf_rmse   = np.sqrt(np.mean(nearest_gt_error(ukf_pos,   gt_xy)**2))
        pcukf_rmse = np.sqrt(np.mean(nearest_gt_error(pcukf_pos, gt_xy)**2))
        pcls_rmse  = np.sqrt(np.mean(nearest_gt_error(pcls_pos,  gt_xy)**2))

        # In kết quả sai số kèm các cột mốc thời gian (đổi ra ms)
        print(f"{os.path.basename(path):<18s}"
              f"{raw_rmse:>10.1f}{ukf_rmse:>10.1f}"
              f"{pcukf_rmse:>12.1f}{pcls_rmse:>10.1f} | "
              f"{time_ukf_file*1000:>9.1f}ms {time_pcukf_file*1000:>9.1f}ms {time_pcukf_sample_ms:>10.3f}ms")

        raw_pos_all.extend(raw_pos)
        ukf_pos_all.extend(ukf_pos)
        pcukf_pos_all.extend(pcukf_pos)
        pcls_pos_all.extend(pcls_pos)

    print("─"*105)
    raw_arr   = np.array(raw_pos_all)
    ukf_arr   = np.array(ukf_pos_all)
    pcukf_arr = np.array(pcukf_pos_all)
    pcls_arr  = np.array(pcls_pos_all)

    return {
        "Raw + WLS" : nearest_gt_error(raw_arr,   gt_xy),
        "UKF-2D"    : nearest_gt_error(ukf_arr,   gt_xy),
        "PC-UKF-2D" : nearest_gt_error(pcukf_arr, gt_xy),
        "PC-LS"     : nearest_gt_error(pcls_arr,  gt_xy),
    }, {
        "raw"  : raw_arr,
        "ukf"  : ukf_arr,
        "pcukf": pcukf_arr,
        "pcls" : pcls_arr,
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
        'q'      : [0.001, 0.01, 0.1],
        'r_base' : [1.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0],
        'r_scale': [5.0, 10.0, 15.0, 20.0],
        'sigma'  : [10.0, 50.0, 100.0, 150.0, 200.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  Grid search: {len(combos)} combinations...")

    best_rmse = float('inf'); best_params = {}; results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        err, _ = evaluate_files(val_files, gt_xy, **params)
        rmse   = float(np.sqrt(np.mean(err['PC-UKF-2D']**2)))
        results.append((rmse, params))
        if rmse < best_rmse:
            best_rmse = rmse; best_params = params

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 configs:")
    print(f"  {'RMSE':>8s} {'Q':>8s} {'R_base':>8s} {'r_scale':>8s} {'sigma':>8s}")
    for rmse, p in results[:5]:
        print(f"  {rmse:>7.1f}mm {p['q']:>8.3f} {p['r_base']:>8.1f}"
              f" {p['r_scale']:>8.1f} {p['sigma']:>8.1f}")
    print(f"\n  Best: RMSE={best_rmse:.1f}mm | {best_params}")
    return best_params, best_rmse


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    'Raw + WLS' : '#9E9E9E',
    'UKF-2D'    : '#E91E63',
    'PC-UKF-2D' : '#2196F3',
    'PC-LS'     : '#4CAF50',
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
        ax.plot(s, cdf, lw=lw, color=color, ls=ls,
                label=f"{label}  (RMSE={rmse_v:.1f}mm)")
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
            ax.annotate(nm, pt, textcoords="offset points",
                        xytext=(6,4), fontsize=12, fontweight='bold')
        for j, (ax_, ay_) in enumerate(ANCHORS):
            ax.scatter(ax_, ay_, s=90, marker='s', color='red', zorder=10)
            ax.annotate(f"A{j+1}", (ax_, ay_),
                        textcoords="offset points", xytext=(5,5), fontsize=9, color='red')
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
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, mname, mlabel in zip(axes, metric_names, labels_disp):
        vals   = [metrics_dict[m][mname] for m in methods]
        colors = [COLORS.get(m, 'gray') for m in methods]
        bars   = ax.bar(range(len(methods)), vals, color=colors,
                        alpha=0.85, edgecolor='white', lw=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' + ','\n+ ') for m in methods],
                           fontsize=9, ha='center')
        ax.set_ylabel("mm", fontsize=12)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()


def analyze_scores(file_paths, q=PCUKF2D_Q, r_base=PCUKF2D_R_BASE,
                   r_scale=PCUKF2D_R_SCALE, sigma=PCUKF2D_SIGMA, save_path=None):
    path   = file_paths[0]
    parsed = parse_file(path)
    if parsed is None: return
    dist_raw = parsed['dist'].astype(float)

    _, scores = pcukf2d_filter_file(dist_raw, q=q, r_base=r_base,
                                     r_scale=r_scale, sigma=sigma)
    R_adaptive = r_base * (1.0 + r_scale * (1.0 - scores))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']

    for i in range(N_ANCHORS):
        axes[0].plot(scores[:, i], color=colors[i], lw=0.8,
                     label=f'Anchor {i+1}', alpha=0.8)
    axes[0].axhline(0.5, color='gray', ls='--', lw=1, label='threshold 0.5')
    axes[0].set_ylabel('Consensus Score', fontsize=11)
    axes[0].set_title(
        f'PC-UKF-2D Consensus Scores — {os.path.basename(path)}',
        fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.05)

    for i in range(N_ANCHORS):
        axes[1].plot(R_adaptive[:, i], color=colors[i], lw=0.8,
                     label=f'R Adaptive Anchor {i+1}', alpha=0.8)
    axes[1].set_ylabel('Adaptive R (mm²)', fontsize=11)
    axes[1].set_title('Adaptive R per Anchor (PC-UKF-2D)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    for i in range(N_ANCHORS):
        axes[2].plot(dist_raw[:, i], color=colors[i], lw=0.6,
                     label=f'Raw d{i}', alpha=0.7)
    axes[2].set_ylabel('Distance (mm)', fontsize=11)
    axes[2].set_xlabel('Timestep', fontsize=11)
    axes[2].set_title('Raw Distances', fontsize=12)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Score analysis → {save_path}")
    plt.close()

def debug_pcukf(file_path, gt_xy, n_samples=20):
    parsed = parse_file(file_path)
    dist_raw = parsed['dist'].astype(float)
    
    pcukf = PCUKF2D()
    
    # Init
    init_pos = wls_position(dist_raw[0])
    pcukf.init(init_pos if not np.any(np.isnan(init_pos)) else np.array([2000.0, 4400.0]))

    print("\n" + "═"*110)
    print("DEBUG PC-UKF-2D — 20 samples đầu")
    print("═"*110)
    print(f"{'t':>4} | {'d_raw':>40} | {'innov':>40} | {'scores':>30} | {'R':>30}")
    print("─"*110)

    for t in range(min(n_samples, len(dist_raw))):
        d = dist_raw[t]

        # Tính innovation từ KF 1D phụ trợ (trước khi update)
        innov_raw = np.array([pcukf._kfs[i].x for i in range(N_ANCHORS)])
        innov_raw = np.where(innov_raw is None, 0.0, innov_raw)
        innovations = np.array([
            d[i] - pcukf._kfs[i].x if pcukf._kfs[i].x is not None else 0.0
            for i in range(N_ANCHORS)
        ])

        scores = pc_consensus_scores(innovations, pcukf.sigma)
        R_diag = pcukf.R_base * (1.0 + pcukf.R_scale * (1.0 - scores))

        pos, sc, R = pcukf.step(d)

        print(f"{t:>4} | {str(np.round(d,1)):>40} | "
              f"{str(np.round(np.abs(innovations),1)):>40} | "
              f"{str(np.round(scores,3)):>30} | "
              f"{str(np.round(R_diag,1)):>30}")

    print("\n── Phân phối scores toàn bộ file ──")
    pcukf2 = PCUKF2D()
    init_pos = wls_position(dist_raw[0])
    pcukf2.init(init_pos if not np.any(np.isnan(init_pos)) else np.array([2000.0, 4400.0]))

    all_scores = []
    all_R      = []
    for t in range(len(dist_raw)):
        _, sc, R = pcukf2.step(dist_raw[t])
        all_scores.append(sc)
        all_R.append(R)
    all_scores = np.array(all_scores)
    all_R      = np.array(all_R)

    print(f"  Score mean per anchor: {np.mean(all_scores, axis=0).round(3)}")
    print(f"  Score std  per anchor: {np.std(all_scores,  axis=0).round(3)}")
    print(f"  Score < 0.3 (nghi ngờ):  {(all_scores < 0.3).sum(axis=0)} lần")
    print(f"  Score > 0.7 (tin tưởng): {(all_scores > 0.7).sum(axis=0)} lần")

    print("\n── R adaptive distribution ──")
    print(f"  R mean: {np.mean(all_R, axis=0).round(1)}")
    print(f"  R min:  {np.min(all_R,  axis=0).round(1)}")
    print(f"  R max:  {np.max(all_R,  axis=0).round(1)}")

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("V15: PC-UKF-2D + PC-LS")
    print("  UKF-2D  : Unscented Transform, không cần Jacobian")
    print("  PC-UKF-2D: PC adaptive R + UKF 2D")
    print("  PC-LS   : Weighted WLS với trọng số từ PC scores (không filter state)")
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

    gt_xy, total_time = build_ground_truth(WAYPOINTS, SPEED, GT_SPACING)
    total_path = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()
    print(f"  Path={total_path:.0f} mm  Time={total_time:.1f}s  GT_points={len(gt_xy)}")

    best_params = {
        'q': PCUKF2D_Q, 'r_base': PCUKF2D_R_BASE,
        'r_scale': PCUKF2D_R_SCALE, 'sigma': PCUKF2D_SIGMA,
    }
    if DO_GRID_SEARCH and val_files:
        best_params, _ = grid_search(val_files, gt_xy)
    else:
        print("\n[INFO] Grid Search TẮT — dùng params mặc định.")

    eval_files = test_files or val_files or train_files or all_files

    debug_pcukf(eval_files[0], gt_xy)

    print(f"\n{'═'*60}\n  Evaluating trên {len(eval_files)} files\n{'═'*60}")
    err, positions = evaluate_files(eval_files, gt_xy, **best_params)

    metrics = {label: compute_metrics(errors, label) for label, errors in err.items()}

    print(f"\n{'═'*80}")
    print(f"  COMPARISON TABLE — V15 PC-UKF-2D + PC-LS")
    print(f"{'═'*80}")
    print(f"  {'Method':<18s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s}")
    print(f"  {'─'*55}")
    for label, m in metrics.items():
        print(f"  {label:<18s} {m['rmse']:>6.1f} {m['mae']:>6.1f}"
              f" {m['cep50']:>6.1f} {m['p95']:>6.1f} {m['max']:>6.1f}")

    print(f"\n  Wilcoxon tests (vs PC-UKF-2D):")
    for a in ['Raw + WLS', 'UKF-2D', 'PC-LS']:
        ea, eb = err[a], err['PC-UKF-2D']
        min_len = min(len(ea), len(eb))
        if min_len > 10:
            try:
                _, p = wilcoxon(ea[:min_len], eb[:min_len])
                print(f"    {a} vs PC-UKF-2D: p={p:.4f} {'✅' if p<0.05 else '⚠️'}")
            except ValueError:
                pass

    plot_cdf(err, os.path.join(SAVE_DIR, 'cdf.png'))
    plot_trajectories(
        {
            'Raw + WLS' : positions['raw'],
            'UKF-2D'    : positions['ukf'],
            'PC-UKF-2D' : positions['pcukf'],
            'PC-LS'     : positions['pcls'],
        },
        gt_xy, os.path.join(SAVE_DIR, 'trajectories.png'),
    )
    plot_bar(metrics, os.path.join(SAVE_DIR, 'bar_comparison.png'))
    analyze_scores(eval_files, save_path=os.path.join(SAVE_DIR, 'score_analysis.png'),
                   **best_params)

    for label, errors in err.items():
        safe = label.lower().replace(' + ','_').replace(' ','_').replace('-','_')
        np.save(os.path.join(SAVE_DIR, f'errors_{safe}.npy'), errors)

    print(f"\n[✓] Toàn bộ kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()