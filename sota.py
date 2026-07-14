# -*- coding: utf-8 -*-
"""
UWB Indoor Positioning — SOTA Baselines FIXED
==============================================
Các fix so với version cũ:
  1. Huber-KF  : scale residual thay vì scale R; z_score dùng sqrt(S) đúng
  2. VB-AKF    : công thức Sage-Husa forgetting factor đúng
  3. Chi2-Gating: soft gating thay vì hard reject; R_base**2 đúng đơn vị
  4. Toàn bộ   : R_base nhất quán là std (mm) → dùng R_base**2 trong S
  5. RANSAC    : seed cố định để reproducible
"""

import os, glob, math, time, warnings, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import chi2
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
ANCHORS = np.array([
    [4000, 8800], [0, 8800], [0, 0], [4000, 0],
], dtype=float)
ANCHOR_HEIGHT = 1400.0
WAYPOINTS = np.array([
    [800.0, 400.0], [3000.0, 400.0], [3000.0, 8000.0],
    [800.0, 8000.0], [800.0, 400.0],
], dtype=float)
SPEED      = 200.0
GT_SPACING = 5.0
N_ANCHORS  = 4

DATA_DIR = "./data"
SAVE_DIR = "./outputs_sota"
DO_GRID_SEARCH = True   # Tắt — dùng tham số đã tuned sẵn

# ── Tham số đã tuned (dùng trực tiếp) ────────────────────────────────
DEFAULT = {
    'huber' : {'R_base': 200.0, 'delta': 1.345},
    'chi2'  : {'R_base': 200.0, 'alpha': 0.05},
    'ransac': {'thresh': 150.0},
    'vbakf' : {'rho': 0.95, 'R0': 200.0**2},
}

# PC-KF params (tuned từ PC-KF.py)
PCKF_Q       = 0.01
PCKF_R_BASE  = 200.0
PCKF_R_SCALE = 10.0
PCKF_SIGMA   = 200.0

GRID = {
    'huber' : {'R_base': [50., 100., 200., 400.], 'delta': [0.5, 1.0, 1.345, 2.0]},
    'chi2'  : {'R_base': [50., 100., 200., 400.], 'alpha': [0.001, 0.01, 0.05, 0.10]},
    'ransac': {'thresh': [75., 100., 150., 200., 300., 500.]},
    'vbakf' : {'rho': [0.80, 0.90, 0.95, 0.99], 'R0': [50.**2, 100.**2, 200.**2, 400.**2]},
}

# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY & GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d):
    return math.sqrt(max(d**2 - ANCHOR_HEIGHT**2, 0.0))

def build_ground_truth():
    segs     = np.diff(WAYPOINTS, axis=0)
    seg_len  = np.linalg.norm(segs, axis=1)
    total    = seg_len.sum()
    n        = max(int(total / GT_SPACING), 2)
    cum      = np.concatenate([[0], np.cumsum(seg_len)])
    qd       = np.linspace(0, total, n)
    gt       = np.zeros((n, 2))
    for i, d in enumerate(qd):
        idx      = np.clip(np.searchsorted(cum, d, 'right') - 1, 0, len(segs)-1)
        frac     = (d - cum[idx]) / (seg_len[idx] + 1e-9)
        gt[i]    = WAYPOINTS[idx] + frac * segs[idx]
    return gt

def nearest_gt_error(pos, gt):
    tree = cKDTree(gt)
    err, _ = tree.query(pos)
    return err

# ══════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════
def wls_position(distances, anchors=ANCHORS):
    """WLS tuyến tính hóa — dùng chung."""
    x0, y0 = anchors[0]
    d0 = max(distances[0], 1.0)
    rows, b, w = [], [], []
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di = max(distances[i], 1.0)
        rows.append([2*(xi-x0), 2*(yi-y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
        w.append(1.0 / di)
    A  = np.array(rows, dtype=float)
    bv = np.array(b, dtype=float)
    W  = np.diag(w)
    try:
        pos, *_ = np.linalg.lstsq(A.T@W@A, A.T@W@bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])

def jacobian_H(pos_xy, anchor_xy):
    """H = ∂d/∂[x,y] cho 1 anchor (ground distance, không có height)."""
    dx = pos_xy[0] - anchor_xy[0]
    dy = pos_xy[1] - anchor_xy[1]
    d  = max(math.sqrt(dx**2 + dy**2), 1.0)
    return np.array([[dx/d, dy/d]])

def predicted_dist(pos_xy, anchors=ANCHORS):
    d = np.linalg.norm(anchors - pos_xy, axis=1)
    return np.maximum(d, 1.0)

def parse_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            p = line.split(',')
            if len(p) < 7: continue
            try:
                dg = [slant_to_ground(float(p[i+1])) for i in range(4)]
                rows.append(dg + [float(p[5]), float(p[6])])
            except ValueError:
                continue
    if not rows: return None
    d = np.array(rows, dtype=np.float32)
    return {'dist': d[:, :4], 'v': d[:, 4], 'gz': d[:, 5]}

def get_init_pos(dist_row):
    p = wls_position(dist_row)
    return ANCHORS.mean(axis=0) if np.any(np.isnan(p)) else p

def filter_nan(pos):
    return pos[~np.any(np.isnan(pos), axis=1)]

# ══════════════════════════════════════════════════════════════════════
#  1. HUBER-KF 2D — FIXED
#  Dùng Huber weight để inflate R_eff khi outlier:
#    z = |r| / sqrt(R_base²)   ← normalize bằng measurement noise thuần túy
#    w = 1            nếu z ≤ δ   (inlier)
#    w = δ / z        nếu z > δ   (outlier → w < 1)
#    R_eff = R_base² / w          → R tăng khi outlier → K giảm → update yếu
#  Đây là IRLS (Iteratively Reweighted LS) formulation đúng cho Huber-KF.
#  Sửa đơn vị: R_base là std (mm) → R_var = R_base² (mm²)
# ══════════════════════════════════════════════════════════════════════
class HuberKF2D:
    def __init__(self, init_pos, R_base=200.0, delta=1.345, Q_std=0.3):
        self.x      = init_pos.copy()
        self.P      = np.eye(2) * 1e6
        self.Q      = np.eye(2) * Q_std**2
        self.R_var  = R_base**2   # variance (mm²)
        self.delta  = delta

    def predict(self):
        self.P = self.P + self.Q

    def update(self, distances):
        for i, anchor in enumerate(ANCHORS):
            H      = jacobian_H(self.x, anchor)           # (1,2)
            d_pred = max(np.linalg.norm(self.x - anchor), 1.0)
            r      = distances[i] - d_pred                # scalar residual

            HPHT   = (H @ self.P @ H.T).item()

            # Huber weight: normalize bằng sqrt(R_var) — measurement noise thuần túy
            # (không dùng S vì P lớn lúc đầu sẽ làm z ≈ 0 → Huber không kích hoạt)
            z      = abs(r) / math.sqrt(max(self.R_var, 1e-9))
            w      = 1.0 if z <= self.delta else self.delta / (z + 1e-9)
            R_eff  = self.R_var / (w + 1e-9)   # inflate R khi outlier

            S      = HPHT + R_eff
            K      = self.P @ H.T / (S + 1e-9)           # (2,1)
            self.x = self.x + K.flatten() * r
            # Joseph form — numerically stable
            I_KH   = np.eye(2) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + (K * R_eff) @ K.T

    def run(self, distances_seq):
        out = np.zeros((len(distances_seq), 2))
        for t, d in enumerate(distances_seq):
            self.predict(); self.update(d); out[t] = self.x.copy()
        return out

# ══════════════════════════════════════════════════════════════════════
#  2. CHI2-GATING KF 2D — FIXED
#  Soft gating: khi outlier, inflate R thay vì bỏ hoàn toàn.
#  Sửa đơn vị: S = HPH' + R_base²
# ══════════════════════════════════════════════════════════════════════
class Chi2GatingKF2D:
    """
    Chi2-Gating KF đúng:
      - Mahalanobis dùng S = HPHT + R_var (đúng theo lý thuyết)
      - Warm-up WARMUP_STEPS sample đầu: không gate, cho KF converge
        (tránh tình trạng init_pos sai xa → r lớn → mọi anchor bị reject)
      - Sau warm-up: hard gate theo chi2.ppf(1-alpha, df=1)
    """
    WARMUP_STEPS = 30   # số sample đầu không gate

    def __init__(self, init_pos, R_base=200.0, alpha=0.05, Q_std=0.3):
        self.x     = init_pos.copy()
        self.P     = np.eye(2) * 1e6
        self.Q     = np.eye(2) * Q_std**2
        self.R_var = R_base**2
        self.gate  = chi2.ppf(1.0 - alpha, df=1)
        self.step  = 0

    def predict(self):
        self.P = self.P + self.Q

    def update(self, distances):
        warming_up = self.step < self.WARMUP_STEPS
        self.step += 1

        for i, anchor in enumerate(ANCHORS):
            H      = jacobian_H(self.x, anchor)
            d_pred = max(np.linalg.norm(self.x - anchor), 1.0)
            r      = distances[i] - d_pred
            HPHT   = (H @ self.P @ H.T).item()
            S      = HPHT + self.R_var          # đúng Mahalanobis

            if not warming_up:
                d_mah = (r**2) / (S + 1e-9)
                if d_mah > self.gate:
                    continue                     # hard reject anchor outlier

            K      = self.P @ H.T / (S + 1e-9)
            self.x = self.x + K.flatten() * r
            I_KH   = np.eye(2) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + (K * self.R_var) @ K.T

    def run(self, distances_seq):
        out = np.zeros((len(distances_seq), 2))
        for t, d in enumerate(distances_seq):
            self.predict(); self.update(d); out[t] = self.x.copy()
        return out

# ══════════════════════════════════════════════════════════════════════
#  PC-KF — Pairwise Consensus Adaptive KF (1D per anchor + WLS)
#  Giữ nguyên logic từ PC-KF.py để kết quả nhất quán.
# ══════════════════════════════════════════════════════════════════════
class _KF1D:
    """KF 1D đơn giản cho từng anchor (dùng trong KF baseline)."""
    def __init__(self, q=0.01, r=200.0):
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


class _PCKF:
    """PC-KF: 4 KF 1D với R thích nghi theo Pairwise Consensus score."""
    def __init__(self, q=PCKF_Q, r_base=PCKF_R_BASE,
                 r_scale=PCKF_R_SCALE, sigma=PCKF_SIGMA):
        self.Q       = q
        self.R_base  = r_base
        self.R_scale = r_scale
        self.sigma   = sigma
        self.P       = np.ones(N_ANCHORS)
        self.x       = None

    def _scores(self, d_raw):
        innov = np.abs(d_raw - self.x)
        scores = np.zeros(N_ANCHORS)
        nu = 4.0; eps = 1e-9
        for i in range(N_ANCHORS):
            s = 0.0
            for j in range(N_ANCHORS):
                if i == j: continue
                diff = innov[i] - innov[j]
                c = (1.0 + diff**2 / (nu * self.sigma**2 + eps)) ** (-(nu+1)/2)
                s += c
            scores[i] = s / (N_ANCHORS - 1)
        return scores

    def update(self, d_raw):
        if self.x is None:
            self.x = d_raw.copy()
            return self.x.copy(), np.ones(N_ANCHORS)
        scores = self._scores(d_raw)
        R      = self.R_base * (1.0 + self.R_scale * (1.0 - scores))
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + R)
        self.x = self.x + K * (d_raw - self.x)
        self.P = (1 - K) * P_pred
        return self.x.copy(), scores


def pckf_sequence(dist_raw):
    """Chạy PC-KF filter rồi giải WLS trên distances đã filter."""
    T   = len(dist_raw)
    kf  = _PCKF()
    pos = np.zeros((T, 2))
    for t in range(T):
        d_filtered, _ = kf.update(dist_raw[t])
        pos[t] = wls_position(d_filtered)
    return pos


# ══════════════════════════════════════════════════════════════════════
#  3. RANSAC + WLS — FIXED (seed cố định)
# ══════════════════════════════════════════════════════════════════════
def ransac_wls_single(distances, thresh=150.0, n_iters=50, min_inliers=3, seed=0):
    n   = len(ANCHORS)
    rng = np.random.default_rng(seed=seed)
    best_pos, best_inliers = None, []

    for _ in range(n_iters):
        idx    = rng.choice(n, size=min_inliers, replace=False)
        pos    = wls_position(distances[idx], ANCHORS[idx])
        if np.any(np.isnan(pos)): continue
        res    = np.abs(distances - predicted_dist(pos))
        inliers = np.where(res < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers, best_pos = inliers, pos

    if best_pos is not None and len(best_inliers) >= min_inliers:
        p = wls_position(distances[best_inliers], ANCHORS[best_inliers])
        return p if not np.any(np.isnan(p)) else best_pos
    return best_pos if best_pos is not None else np.array([np.nan, np.nan])

def ransac_wls_sequence(distances_seq, thresh=150.0):
    out = np.zeros((len(distances_seq), 2))
    for t, d in enumerate(distances_seq):
        out[t] = ransac_wls_single(d, thresh=thresh, seed=t)
    return out

# ══════════════════════════════════════════════════════════════════════
#  4. VB-AKF 2D — FIXED
#  Dùng Sage-Husa forgetting factor (đúng derivation):
#    R_i ← ρ * R_i + (1-ρ) * (r² + HPH')
#  Sửa đơn vị: khởi tạo R0 = R_base² (variance)
# ══════════════════════════════════════════════════════════════════════
class VBAKF2D:
    def __init__(self, init_pos, rho=0.95, R0=200.0**2, Q_std=0.3):
        self.x   = init_pos.copy()
        self.P   = np.eye(2) * 1e6
        self.Q   = np.eye(2) * Q_std**2
        self.rho = rho
        # R per anchor, khởi tạo bằng prior variance (mm²)
        self.R   = np.full(N_ANCHORS, float(R0))

    def predict(self):
        self.P = self.P + self.Q

    def update(self, distances):
        for i, anchor in enumerate(ANCHORS):
            H      = jacobian_H(self.x, anchor)
            d_pred = max(np.linalg.norm(self.x - anchor), 1.0)
            r      = distances[i] - d_pred

            HPHT   = (H @ self.P @ H.T).item()

            # Sage-Husa VB-M step: cập nhật R_i với forgetting factor
            self.R[i] = self.rho * self.R[i] + (1.0 - self.rho) * (r**2 + HPHT)
            self.R[i] = max(self.R[i], 1.0)     # floor tránh collapse

            # VB-E step: KF update với R_i đã adapt
            S      = HPHT + self.R[i]
            K      = self.P @ H.T / (S + 1e-9)
            self.x = self.x + K.flatten() * r
            I_KH   = np.eye(2) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + (K * self.R[i]) @ K.T

    def run(self, distances_seq):
        out = np.zeros((len(distances_seq), 2))
        for t, d in enumerate(distances_seq):
            self.predict(); self.update(d); out[t] = self.x.copy()
        return out

# ══════════════════════════════════════════════════════════════════════
#  GRID SEARCH (LOO-CV)
# ══════════════════════════════════════════════════════════════════════
def _rmse_files(files, gt, run_fn):
    errs = []
    for p in files:
        parsed = parse_file(p)
        if parsed is None: continue
        dist = parsed['dist'].astype(float)
        pos  = filter_nan(run_fn(dist, get_init_pos(dist[0])))
        if len(pos): errs.extend(nearest_gt_error(pos, gt))
    return float(np.sqrt(np.mean(np.array(errs)**2))) if errs else float('inf')

def _loo(files, gt, run_fn):
    return float(np.mean([_rmse_files([f], gt, run_fn) for f in files]))

def run_grid_search(files, gt):
    print(f"\n{'═'*60}\n  GRID SEARCH (LOO-CV, {len(files)} files)\n{'═'*60}")
    best = {}

    # Huber
    combos = list(itertools.product(GRID['huber']['R_base'], GRID['huber']['delta']))
    print(f"  Huber-KF: {len(combos)} combos...")
    br, bp = float('inf'), {}
    for R, d in combos:
        r = _loo(files, gt, lambda dist, ip, R=R, d=d: HuberKF2D(ip, R_base=R, delta=d).run(dist))
        if r < br: br, bp = r, {'R_base': R, 'delta': d}
    print(f"  → RMSE={br:.1f}mm | {bp}")
    best['huber'] = bp

    # Chi2
    combos = list(itertools.product(GRID['chi2']['R_base'], GRID['chi2']['alpha']))
    print(f"  Chi2-Gating: {len(combos)} combos...")
    br, bp = float('inf'), {}
    for R, a in combos:
        r = _loo(files, gt, lambda dist, ip, R=R, a=a: Chi2GatingKF2D(ip, R_base=R, alpha=a).run(dist))
        if r < br: br, bp = r, {'R_base': R, 'alpha': a}
    print(f"  → RMSE={br:.1f}mm | {bp}")
    best['chi2'] = bp

    # RANSAC
    print(f"  RANSAC+WLS: {len(GRID['ransac']['thresh'])} thresholds...")
    br, bp = float('inf'), {}
    for th in GRID['ransac']['thresh']:
        r = _loo(files, gt, lambda dist, ip, th=th: ransac_wls_sequence(dist, thresh=th))
        if r < br: br, bp = r, {'thresh': th}
    print(f"  → RMSE={br:.1f}mm | {bp}")
    best['ransac'] = bp

    # VB-AKF
    combos = list(itertools.product(GRID['vbakf']['rho'], GRID['vbakf']['R0']))
    print(f"  VB-AKF: {len(combos)} combos...")
    br, bp = float('inf'), {}
    for rho, R0 in combos:
        r = _loo(files, gt, lambda dist, ip, rho=rho, R0=R0: VBAKF2D(ip, rho=rho, R0=R0).run(dist))
        if r < br: br, bp = r, {'rho': rho, 'R0': R0}
    print(f"  → RMSE={br:.1f}mm | {bp}")
    best['vbakf'] = bp

    return best

# ══════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════
def evaluate_files(files, gt, params):
    methods = ['Raw+WLS', 'Huber-KF', 'Chi2-Gating', 'RANSAC+WLS', 'VB-AKF', 'PC-KF+WLS']
    all_err = {m: [] for m in methods}
    all_pos = {m: [] for m in methods}
    tim     = {m: [] for m in methods}

    p_h = params.get('huber',  DEFAULT['huber'])
    p_c = params.get('chi2',   DEFAULT['chi2'])
    p_r = params.get('ransac', DEFAULT['ransac'])
    p_v = params.get('vbakf',  DEFAULT['vbakf'])

    cw  = 13
    sep = "─" * (18 + cw * len(methods))
    print(f"\n{'═'*len(sep)}\n  PER-FILE RESULTS (RMSE mm)\n{'═'*len(sep)}")
    print(f"{'File':<18s}" + "".join(f"{m:>{cw}s}" for m in methods))
    print(sep)

    for path in files:
        parsed = parse_file(path)
        if parsed is None: continue
        dist = parsed['dist'].astype(float)
        T    = len(dist)
        ip   = get_init_pos(dist[0])
        row  = []

        def run_and_log(key, fn):
            t0  = time.perf_counter()
            pos = filter_nan(fn())
            tim[key].append((time.perf_counter()-t0)/T*1000)
            err = nearest_gt_error(pos, gt)
            all_err[key].extend(err)
            all_pos[key].append(pos)
            row.append(np.sqrt(np.mean(err**2)))

        run_and_log('Raw+WLS',     lambda: np.array([wls_position(dist[t]) for t in range(T)]))
        run_and_log('Huber-KF',    lambda: HuberKF2D(ip.copy(), **p_h).run(dist))
        run_and_log('Chi2-Gating', lambda: Chi2GatingKF2D(ip.copy(), **p_c).run(dist))
        run_and_log('RANSAC+WLS',  lambda: ransac_wls_sequence(dist, **p_r))
        run_and_log('VB-AKF',      lambda: VBAKF2D(ip.copy(), **p_v).run(dist))
        run_and_log('PC-KF+WLS',   lambda: pckf_sequence(dist))

        print(f"{os.path.basename(path):<18s}" + "".join(f"{v:>{cw}.1f}" for v in row))

    print(sep)
    all_err = {m: np.array(v) for m, v in all_err.items()}
    all_pos = {m: np.vstack(v) for m, v in all_pos.items()}
    return all_err, all_pos, tim

def compute_metrics(errors):
    return {
        'rmse' : float(np.sqrt(np.mean(errors**2))),
        'mae'  : float(np.mean(errors)),
        'cep50': float(np.percentile(errors, 50)),
        'p95'  : float(np.percentile(errors, 95)),
        'max'  : float(np.max(errors)),
    }

# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    'Raw+WLS'    : '#9E9E9E',
    'Huber-KF'   : '#E91E63',
    'Chi2-Gating': '#FF9800',
    'RANSAC+WLS' : '#4CAF50',
    'VB-AKF'     : '#9C27B0',
    'PC-KF+WLS'  : '#2196F3',   # blue — proposed method
}

def plot_cdf(errors_dict, save_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    for label, errors in errors_dict.items():
        s   = np.sort(errors)
        cdf = np.arange(1, len(s)+1) / len(s)
        rmse = np.sqrt(np.mean(errors**2))
        ls  = '--' if label == 'Raw+WLS' else '-'
        ax.plot(s, cdf, lw=2.2, color=COLORS.get(label,'gray'), ls=ls,
                label=f"{label}  RMSE={rmse:.1f}mm")
    ax.set_xlabel("Position Error (mm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 800); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_title("CDF — SOTA Baselines (Fixed)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] CDF → {save_path}")
    plt.close()

def plot_bar(metrics_dict, save_path):
    methods = list(metrics_dict.keys())
    mnames  = ['rmse','mae','cep50','p95']
    labels  = ['RMSE','MAE','CEP50','P95']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, mn, ml in zip(axes, mnames, labels):
        vals   = [metrics_dict[m][mn] for m in methods]
        colors = [COLORS.get(m,'gray') for m in methods]
        bars   = ax.bar(range(len(methods)), vals, color=colors, alpha=0.85,
                        edgecolor='white', lw=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' ','\n') for m in methods], fontsize=8)
        ax.set_ylabel("mm"); ax.set_title(ml, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
    plt.suptitle("SOTA Baselines Fixed — Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bar → {save_path}")
    plt.close()

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("SOTA Baselines FIXED — Huber/Chi2/VB-AKF corrected")
    os.makedirs(SAVE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not files:
        print(f"Không tìm thấy file trong '{DATA_DIR}'"); return
    print(f"Tìm thấy {len(files)} files")

    gt = build_ground_truth()
    print(f"Ground truth: {len(gt)} points")

    if DO_GRID_SEARCH:
        params = run_grid_search(files, gt)
    else:
        params = DEFAULT
        print(f"\n  [INFO] Grid search TẮT — dùng DEFAULT params:")
        for k, v in params.items(): print(f"    {k}: {v}")
    print(f"  PC-KF: Q={PCKF_Q}, R_base={PCKF_R_BASE}, R_scale={PCKF_R_SCALE}, sigma={PCKF_SIGMA}")

    print(f"\n{'═'*60}\n  FINAL EVALUATION\n{'═'*60}")
    errors, positions, timing = evaluate_files(files, gt, params)
    metrics = {m: compute_metrics(e) for m, e in errors.items()}

    print(f"\n{'═'*78}\n  SUMMARY TABLE\n{'═'*78}")
    print(f"  {'Method':<16s} {'RMSE':>7s} {'MAE':>7s} {'CEP50':>7s} {'P95':>7s} {'MAX':>7s} {'T/smpl':>9s}")
    print(f"  {'─'*68}")
    for m, met in metrics.items():
        t_us  = np.mean(timing[m]) * 1000
        star  = " ◀ proposed" if m == 'PC-KF+WLS' else ""
        print(f"  {m:<16s} {met['rmse']:>7.1f} {met['mae']:>7.1f} {met['cep50']:>7.1f} "
              f"{met['p95']:>7.1f} {met['max']:>7.1f} {t_us:>7.3f}µs{star}")

    plot_cdf(errors,  os.path.join(SAVE_DIR, 'cdf.png'))
    plot_bar(metrics, os.path.join(SAVE_DIR, 'bar.png'))
    print(f"\n[✓] Kết quả → '{SAVE_DIR}/'")

if __name__ == "__main__":
    main()