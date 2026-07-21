# -*- coding: utf-8 -*-
"""
Ablation Study — PC-UKF Hyperparameter Sensitivity
====================================================
So sánh PC-UKF với Standard UKF trên 7 ablation:
  (A) ν        ∈ {1, 2, 4, 8, 16, ∞}
  (B) σ        ∈ {50, 100, 200, 300, 500, 800, 1000}
  (C) R_scale  ∈ {1, 3, 5, 10, 15, 20, 30, 50}
  (D) R_base   ∈ {5, 10, 25, 50, 100, 200, 500}
  (E) q        ∈ {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1}
  (F) Heatmap  R_base × R_scale  (2D)
  (G) Heatmap  ν × σ             (2D)

Hệ số mặc định (giữ cố định khi sweep hệ số khác):
  ν=4,  σ=300,  R_base=25,  R_scale=15,  q=0.001
"""

import os
import glob
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

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

UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

# ── Hệ số mặc định ───────────────────────────────────────────────────
DEFAULT_Q       = 0.001
DEFAULT_R_UKF   = 50.0
DEFAULT_R_BASE  = 25.0
DEFAULT_R_SCALE = 15.0
DEFAULT_SIGMA   = 300.0
DEFAULT_NU      = 4.0

DATA_DIR = "./data"
SAVE_DIR = "./outputs_Hypertune"

# ── Sweep ranges ──────────────────────────────────────────────────────
SWEEP_NU      = [1, 2, 4, 8, 16, 1e9]
SWEEP_SIGMA   = [50, 100, 200, 300, 500, 800, 1000]
SWEEP_RSCALE  = [1, 3, 5, 10, 15, 20, 30, 50]
SWEEP_RBASE   = [5, 10, 25, 50, 100, 200, 500]
SWEEP_Q       = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

# Heatmap grids (nhỏ hơn để chạy nhanh)
HEAT_RBASE    = [5, 10, 25, 50, 100, 200]
HEAT_RSCALE   = [1, 3, 5, 10, 15, 20, 30]
HEAT_NU       = [1, 2, 4, 8, 16, 1e9]
HEAT_SIGMA    = [50, 100, 200, 300, 500, 800]


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY & GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d_slant):
    return math.sqrt(max(d_slant**2 - ANCHOR_HEIGHT**2, 0.0))


def build_ground_truth():
    segs    = np.diff(WAYPOINTS, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    total   = seg_len.sum()
    n_pts   = max(int(total / GT_SPACING), 2)
    cum     = np.concatenate([[0], np.cumsum(seg_len)])
    q_dist  = np.linspace(0, total, n_pts)
    gt      = np.zeros((n_pts, 2))
    for i, d in enumerate(q_dist):
        idx   = np.clip(np.searchsorted(cum, d, 'right') - 1, 0, len(segs)-1)
        frac  = (d - cum[idx]) / (seg_len[idx] + 1e-9)
        gt[i] = WAYPOINTS[idx] + frac * segs[idx]
    return gt


def nearest_gt_error(pos_xy, gt_xy):
    tree = cKDTree(gt_xy)
    errs, _ = tree.query(pos_xy)
    return errs


def LS_position(distances):
    x0, y0 = ANCHORS[0]
    d0 = max(distances[0], 1.0)
    rows, b = [], []
    for i in range(1, N_ANCHORS):
        xi, yi = ANCHORS[i]
        di = max(distances[i], 1.0)
        rows.append([2*(xi - x0), 2*(yi - y0)])
        b.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
    A  = np.array(rows, dtype=float)
    bv = np.array(b, dtype=float)
    try:
        pos, *_ = np.linalg.lstsq(A, bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


# ══════════════════════════════════════════════════════════════════════
#  UKF UTILITIES
# ══════════════════════════════════════════════════════════════════════
def h_obs(state):
    x, y = state[0], state[1]
    return np.array([
        math.sqrt((x - ax)**2 + (y - ay)**2 + ANCHOR_HEIGHT**2)
        for ax, ay in ANCHORS
    ])


def _ukf_weights():
    n   = 2
    lam = UKF_ALPHA**2 * (n + UKF_KAPPA) - n
    c   = n + lam
    Wm  = np.full(2*n + 1, 0.5 / c)
    Wc  = np.full(2*n + 1, 0.5 / c)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - UKF_ALPHA**2 + UKF_BETA)
    return Wm, Wc, c


Wm_G, Wc_G, c_G = _ukf_weights()


def sigma_points(x, P, c):
    n = len(x)
    try:
        S = np.linalg.cholesky(c * P)
    except np.linalg.LinAlgError:
        S = np.linalg.cholesky(c * (P + np.eye(n) * 1e-6))
    pts = np.zeros((2*n + 1, n))
    pts[0] = x
    for i in range(n):
        pts[i+1]   = x + S[:, i]
        pts[n+i+1] = x - S[:, i]
    return pts


def ukf_moments(x_pred, P_pred):
    pts   = sigma_points(x_pred, P_pred, c_G)
    Z_pts = np.array([h_obs(pts[i]) for i in range(len(pts))])
    z_hat = Wm_G @ Z_pts
    n, M  = 2, N_ANCHORS
    Pzz   = np.zeros((M, M))
    Pxz   = np.zeros((n, M))
    for i in range(2*n + 1):
        dz   = Z_pts[i] - z_hat
        dx   = pts[i] - x_pred
        Pzz += Wc_G[i] * np.outer(dz, dz)
        Pxz += Wc_G[i] * np.outer(dx, dz)
    return z_hat, Pzz, Pxz


def make_spd(P):
    P = 0.5 * (P + P.T)
    P += np.eye(len(P)) * 1e-9
    return P


def _default_init(dist_raw_0):
    x0 = LS_position(dist_raw_0)
    return x0 if not np.any(np.isnan(x0)) else np.array([2000.0, 4400.0])


# ══════════════════════════════════════════════════════════════════════
#  STANDARD UKF
# ══════════════════════════════════════════════════════════════════════
def run_standard_ukf(dist_raw, q=DEFAULT_Q, r=DEFAULT_R_UKF):
    Q = np.eye(2) * q
    R = np.eye(N_ANCHORS) * r
    T = len(dist_raw)
    x = _default_init(dist_raw[0]).astype(float)
    P = np.eye(2) * 1e6
    pos = np.full((T, 2), np.nan)
    for t in range(T):
        x_pred = x.copy()
        P_pred = P + Q
        z_hat, Pzz, Pxz = ukf_moments(x_pred, P_pred)
        Pzz_eff = Pzz + R
        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            x, P = x_pred, make_spd(P_pred)
            pos[t] = x
            continue
        x = x_pred + K @ (dist_raw[t] - z_hat)
        P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        pos[t] = x
    return pos


# ══════════════════════════════════════════════════════════════════════
#  PC-UKF
# ══════════════════════════════════════════════════════════════════════
def pc_scores(d_raw, sigma, nu):
    pos_est = LS_position(d_raw)
    if not np.any(np.isnan(pos_est)):
        residual = np.array([
            abs(d_raw[i] - math.sqrt((pos_est[0]-ANCHORS[i,0])**2
                                     + (pos_est[1]-ANCHORS[i,1])**2))
            for i in range(N_ANCHORS)
        ])
    else:
        residual = np.abs(d_raw - np.mean(d_raw))

    scores = np.zeros(N_ANCHORS)
    eps = 1e-9
    for i in range(N_ANCHORS):
        s = 0.0
        for j in range(N_ANCHORS):
            if i == j:
                continue
            diff = residual[i] - residual[j]
            if nu > 1e6:   # ν→∞: tiệm cận Gaussian kernel
                c = math.exp(-0.5 * diff**2 / (sigma**2 + eps))
            else:
                c = (1.0 + diff**2 / (nu * sigma**2 + eps)) ** (-(nu+1.0)/2.0)
            s += c
        scores[i] = s / (N_ANCHORS - 1)
    return scores


def run_pc_ukf(dist_raw, q=DEFAULT_Q, r_base=DEFAULT_R_BASE,
               r_scale=DEFAULT_R_SCALE, sigma=DEFAULT_SIGMA, nu=DEFAULT_NU):
    Q = np.eye(2) * q
    T = len(dist_raw)
    x = _default_init(dist_raw[0]).astype(float)
    P = np.eye(2) * 1e6
    pos = np.full((T, 2), np.nan)
    for t in range(T):
        scores  = pc_scores(dist_raw[t], sigma, nu)
        R_diag  = r_base * (1.0 + r_scale * (1.0 - scores))
        R       = np.diag(R_diag)
        x_pred  = x.copy()
        P_pred  = P + Q
        z_hat, Pzz, Pxz = ukf_moments(x_pred, P_pred)
        Pzz_eff = Pzz + R
        try:
            K = Pxz @ np.linalg.inv(Pzz_eff)
        except np.linalg.LinAlgError:
            x, P = x_pred, make_spd(P_pred)
            pos[t] = x
            continue
        x = x_pred + K @ (dist_raw[t] - z_hat)
        P = make_spd(P_pred - K @ Pzz_eff @ K.T)
        pos[t] = x
    return pos


# ══════════════════════════════════════════════════════════════════════
#  PARSE & EVAL HELPERS
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
                d_slant  = [float(parts[i+1]) for i in range(4)]
                d_ground = [slant_to_ground(d) for d in d_slant]
                rows.append(d_ground)
            except ValueError:
                continue
    return np.array(rows, dtype=np.float64) if rows else None


def eval_rmse(runner_fn, files, gt_xy):
    all_errs = []
    for path in files:
        d = parse_file(path)
        if d is None:
            continue
        pos   = runner_fn(d)
        valid = pos[~np.any(np.isnan(pos), axis=1)]
        if len(valid) > 0:
            all_errs.extend(nearest_gt_error(valid, gt_xy))
    if not all_errs:
        return float('nan')
    return float(np.sqrt(np.mean(np.array(all_errs)**2)))


# ══════════════════════════════════════════════════════════════════════
#  ABLATION 1-D
# ══════════════════════════════════════════════════════════════════════
def ablation_nu(files, gt_xy):
    print("\n── (A) ν — bậc tự do Student-t ────────────────────────────")
    results = {}
    for nu in SWEEP_NU:
        lbl  = "∞" if nu > 1e6 else str(int(nu))
        rmse = eval_rmse(lambda d, nu=nu: run_pc_ukf(d, nu=nu), files, gt_xy)
        results[lbl] = rmse
        print(f"  ν = {lbl:>3s}  → RMSE = {rmse:.1f} mm")
    return results


def ablation_sigma(files, gt_xy):
    print("\n── (B) σ — bandwidth ───────────────────────────────────────")
    results = {}
    for sig in SWEEP_SIGMA:
        rmse = eval_rmse(lambda d, s=sig: run_pc_ukf(d, sigma=s), files, gt_xy)
        results[sig] = rmse
        print(f"  σ = {sig:>5}  → RMSE = {rmse:.1f} mm")
    return results


def ablation_rscale(files, gt_xy):
    print("\n── (C) R_scale — hệ số phạt ────────────────────────────────")
    results = {}
    for rs in SWEEP_RSCALE:
        rmse = eval_rmse(lambda d, rs=rs: run_pc_ukf(d, r_scale=rs), files, gt_xy)
        results[rs] = rmse
        print(f"  R_scale = {rs:>3}  → RMSE = {rmse:.1f} mm")
    return results


def ablation_rbase(files, gt_xy):
    print("\n── (D) R_base — noise baseline ─────────────────────────────")
    results = {}
    for rb in SWEEP_RBASE:
        rmse = eval_rmse(lambda d, rb=rb: run_pc_ukf(d, r_base=rb), files, gt_xy)
        results[rb] = rmse
        print(f"  R_base = {rb:>5}  → RMSE = {rmse:.1f} mm")
    return results


def ablation_q(files, gt_xy):
    print("\n── (E) q — process noise ───────────────────────────────────")
    results = {}
    for q in SWEEP_Q:
        rmse = eval_rmse(lambda d, q=q: run_pc_ukf(d, q=q), files, gt_xy)
        results[q] = rmse
        print(f"  q = {q:.4f}  → RMSE = {rmse:.1f} mm")
    return results


# ══════════════════════════════════════════════════════════════════════
#  ABLATION 2-D HEATMAP
# ══════════════════════════════════════════════════════════════════════
def ablation_heatmap_rbase_rscale(files, gt_xy):
    print("\n── (F) Heatmap R_base × R_scale ────────────────────────────")
    nR, nS = len(HEAT_RBASE), len(HEAT_RSCALE)
    grid   = np.full((nR, nS), np.nan)
    for i, rb in enumerate(HEAT_RBASE):
        for j, rs in enumerate(HEAT_RSCALE):
            rmse = eval_rmse(
                lambda d, rb=rb, rs=rs: run_pc_ukf(d, r_base=rb, r_scale=rs),
                files, gt_xy)
            grid[i, j] = rmse
        print(f"  R_base={rb:>5} done")
    return grid


def ablation_heatmap_nu_sigma(files, gt_xy):
    print("\n── (G) Heatmap ν × σ ───────────────────────────────────────")
    nN, nS = len(HEAT_NU), len(HEAT_SIGMA)
    grid   = np.full((nN, nS), np.nan)
    for i, nu in enumerate(HEAT_NU):
        lbl = "∞" if nu > 1e6 else str(int(nu))
        for j, sig in enumerate(HEAT_SIGMA):
            rmse = eval_rmse(
                lambda d, nu=nu, sig=sig: run_pc_ukf(d, nu=nu, sigma=sig),
                files, gt_xy)
            grid[i, j] = rmse
        print(f"  ν={lbl} done")
    return grid


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════
REF_COLOR = '#E53935'
PC_COLOR  = '#7B1FA2'
OPT_COLOR = '#2E7D32'


def _add_ref(ax, rmse_ukf):
    ax.axhline(rmse_ukf, color=REF_COLOR, ls='--', lw=1.8,
               label=f'Standard UKF ({rmse_ukf:.1f} mm)')


def _mark_best(ax, xs, ys):
    ys = np.array(ys, dtype=float)
    best_i = int(np.nanargmin(ys))
    ax.scatter([xs[best_i]], [ys[best_i]], s=110, color=OPT_COLOR,
               zorder=5, label=f'Best ({ys[best_i]:.1f} mm)')


def plot_1d_ablations(res_nu, res_sigma, res_rscale, res_rbase,
                      res_q, rmse_ukf, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("PC-UKF — Ablation Study (1-D Sensitivity)",
                 fontsize=15, fontweight='bold')

    # helper
    def _draw(ax, xs, ys, xlabel, title, xlog=False, fmt='o-', xtick_labels=None):
        ax.plot(range(len(xs)), ys, fmt, color=PC_COLOR, lw=2, ms=7,
                label='PC-UKF')
        _add_ref(ax, rmse_ukf)
        _mark_best(ax, range(len(xs)), ys)
        ax.set_xticks(range(len(xs)))
        if xtick_labels:
            ax.set_xticklabels(xtick_labels, rotation=20, ha='right',
                               fontsize=9)
        else:
            ax.set_xticklabels([str(x) for x in xs], rotation=20,
                               ha='right', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('RMSE (mm)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)

    # (A) ν
    lbls = list(res_nu.keys())
    _draw(axes[0,0],
          lbls, [res_nu[k] for k in lbls],
          'ν', '(A) Degrees of freedom ν',
          xtick_labels=[f'ν={l}' for l in lbls])

    # (B) σ
    _draw(axes[0,1],
          SWEEP_SIGMA, [res_sigma[k] for k in SWEEP_SIGMA],
          'σ (mm)', '(B) Bandwidth σ',
          xtick_labels=[str(s) for s in SWEEP_SIGMA], fmt='s-')

    # (C) R_scale
    _draw(axes[0,2],
          SWEEP_RSCALE, [res_rscale[k] for k in SWEEP_RSCALE],
          'R_scale', '(C) Penalty scale R_scale', fmt='^-')

    # (D) R_base
    _draw(axes[1,0],
          SWEEP_RBASE, [res_rbase[k] for k in SWEEP_RBASE],
          'R_base', '(D) Noise baseline R_base', fmt='D-')

    # (E) q
    q_labels = [f'{q:.0e}' for q in SWEEP_Q]
    _draw(axes[1,1],
          SWEEP_Q, [res_q[k] for k in SWEEP_Q],
          'q (process noise)', '(E) Process noise q',
          xtick_labels=q_labels, fmt='P-')

    # (F) placeholder — ẩn ô thừa
    axes[1,2].axis('off')
    axes[1,2].text(0.5, 0.5,
                   'Xem file\nablation_heatmaps.png\ncho 2D ablation',
                   ha='center', va='center', fontsize=12,
                   color='gray', transform=axes[1,2].transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] 1-D ablation → {save_path}")
    plt.close()


def plot_heatmaps(grid_rb_rs, grid_nu_sig, rmse_ukf, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("PC-UKF — Ablation Study (2-D Heatmap)",
                 fontsize=14, fontweight='bold')

    def _draw_hmap(ax, grid, row_labels, col_labels, row_name, col_name, title):
        # Đánh dấu ô tốt nhất
        best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)

        im = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r',
                       interpolation='nearest')

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label('RMSE (mm)', fontsize=10)

        # Giá trị trong từng ô
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                txt = f'{val:.0f}' if not np.isnan(val) else ''
                fw  = 'bold' if (i, j) == best_idx else 'normal'
                col = 'white' if (i, j) == best_idx else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8.5, fontweight=fw, color=col)

        # Khoanh ô tốt nhất
        bi, bj = best_idx
        rect = plt.Rectangle((bj-0.5, bi-0.5), 1, 1,
                              linewidth=2.5, edgecolor='#1565C0',
                              facecolor='none')
        ax.add_patch(rect)

        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels([str(c) for c in col_labels],
                           rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels([str(r) for r in row_labels], fontsize=9)
        ax.set_xlabel(col_name, fontsize=11)
        ax.set_ylabel(row_name, fontsize=11)
        ax.set_title(title + f'\n(UKF baseline = {rmse_ukf:.1f} mm)',
                     fontsize=11, fontweight='bold')

    # (F) R_base × R_scale
    _draw_hmap(axes[0], grid_rb_rs,
               HEAT_RBASE, HEAT_RSCALE,
               'R_base', 'R_scale',
               '(F) R_base × R_scale')

    # (G) ν × σ
    nu_labels = ['∞' if n > 1e6 else str(int(n)) for n in HEAT_NU]
    _draw_hmap(axes[1], grid_nu_sig,
               nu_labels, HEAT_SIGMA,
               'ν', 'σ (mm)',
               '(G) ν × σ')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Heatmap → {save_path}")
    plt.close()


def plot_summary_table(results, rmse_ukf, rmse_pc_def, save_path):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis('off')

    header = ['Ablation', 'Tham số',
              'RMSE tốt nhất (mm)', 'Cấu hình tốt nhất',
              'Cải thiện vs UKF']
    rows = []

    def _row(res_dict):
        best_k = min(res_dict, key=res_dict.get)
        best_v = res_dict[best_k]
        imp    = (rmse_ukf - best_v) / rmse_ukf * 100
        return [f'{best_v:.1f}', str(best_k), f'{imp:+.1f}%']

    res_nu, res_sig, res_rs, res_rb, res_q = results

    rows.append(['(A)', 'ν']       + _row(res_nu))
    rows.append(['(B)', 'σ']       + _row(res_sig))
    rows.append(['(C)', 'R_scale'] + _row(res_rs))
    rows.append(['(D)', 'R_base']  + _row(res_rb))
    rows.append(['(E)', 'q']       + _row(res_q))

    # Thêm dòng PC-UKF default
    imp_def = (rmse_ukf - rmse_pc_def) / rmse_ukf * 100
    rows.append(['—', 'PC-UKF (default)',
                 f'{rmse_pc_def:.1f}',
                 f'ν={DEFAULT_NU}, σ={DEFAULT_SIGMA}, '
                 f'Rb={DEFAULT_R_BASE}, Rs={DEFAULT_R_SCALE}, '
                 f'q={DEFAULT_Q}',
                 f'{imp_def:+.1f}%'])

    tbl = ax.table(cellText=rows, colLabels=header,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.15, 1.9)

    n_cols = len(header)
    for j in range(n_cols):
        tbl[0, j].set_facecolor('#4A148C')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    imp_col = n_cols - 1
    for i in range(1, len(rows)+1):
        cell = tbl[i, imp_col]
        val  = rows[i-1][imp_col]
        try:
            v = float(val.replace('%',''))
            cell.set_facecolor('#E8F5E9' if v > 0 else '#FFEBEE')
            cell.set_text_props(
                fontweight='bold',
                color='#1B5E20' if v > 0 else '#B71C1C')
        except Exception:
            pass

    ax.set_title(
        f'Ablation Study Summary  |  Standard UKF baseline = {rmse_ukf:.1f} mm',
        fontsize=12, fontweight='bold', pad=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Bảng tóm tắt → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Ablation Study — PC-UKF Hyperparameter Sensitivity")
    print("  7 ablation: (A)ν  (B)σ  (C)R_scale  (D)R_base  (E)q")
    print("              (F) Heatmap R_base×R_scale")
    print("              (G) Heatmap ν×σ")
    print("=" * 65)

    os.makedirs(SAVE_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not files:
        print(f"[!] Không tìm thấy file .txt trong '{DATA_DIR}/'")
        return
    print(f"  {len(files)} file(s) tìm thấy")

    gt_xy = build_ground_truth()
    print(f"  Ground truth: {len(gt_xy)} points")

    # Baseline
    print("\n── Baseline ─────────────────────────────────────────────────")
    rmse_ukf    = eval_rmse(run_standard_ukf, files, gt_xy)
    rmse_pc_def = eval_rmse(run_pc_ukf, files, gt_xy)
    print(f"  Standard UKF    : {rmse_ukf:.1f} mm")
    print(f"  PC-UKF (default): {rmse_pc_def:.1f} mm  "
          f"[ν={DEFAULT_NU}, σ={DEFAULT_SIGMA}, "
          f"R_base={DEFAULT_R_BASE}, R_scale={DEFAULT_R_SCALE}, "
          f"q={DEFAULT_Q}]")

    # 1-D ablations
    res_nu    = ablation_nu(files, gt_xy)
    res_sigma = ablation_sigma(files, gt_xy)
    res_rs    = ablation_rscale(files, gt_xy)
    res_rb    = ablation_rbase(files, gt_xy)
    res_q     = ablation_q(files, gt_xy)

    # 2-D heatmaps
    grid_rb_rs  = ablation_heatmap_rbase_rscale(files, gt_xy)
    grid_nu_sig = ablation_heatmap_nu_sigma(files, gt_xy)

    # Plots
    plot_1d_ablations(
        res_nu, res_sigma, res_rs, res_rb, res_q, rmse_ukf,
        os.path.join(SAVE_DIR, "ablation_1d.png"))

    plot_heatmaps(
        grid_rb_rs, grid_nu_sig, rmse_ukf,
        os.path.join(SAVE_DIR, "ablation_heatmaps.png"))

    plot_summary_table(
        (res_nu, res_sigma, res_rs, res_rb, res_q),
        rmse_ukf, rmse_pc_def,
        os.path.join(SAVE_DIR, "ablation_summary_table.png"))

    # In kết quả cuối
    print("\n" + "=" * 65)
    print("  KẾT QUẢ TỔNG HỢP")
    print("=" * 65)
    print(f"  Standard UKF            : {rmse_ukf:.1f} mm")
    print(f"  PC-UKF (default)        : {rmse_pc_def:.1f} mm")

    for lbl, res, sweep in [
        ("(A) ν tốt nhất     ", res_nu,    None),
        ("(B) σ tốt nhất     ", res_sigma, None),
        ("(C) R_scale tốt nhất", res_rs,   None),
        ("(D) R_base tốt nhất", res_rb,    None),
        ("(E) q tốt nhất     ", res_q,     None),
    ]:
        bk = min(res, key=res.get)
        bv = res[bk]
        imp = (rmse_ukf - bv) / rmse_ukf * 100
        print(f"  {lbl}: {bk}  → {bv:.1f} mm  ({imp:+.1f}% vs UKF)")

    bri, brj = np.unravel_index(np.nanargmin(grid_rb_rs), grid_rb_rs.shape)
    bni, bnj = np.unravel_index(np.nanargmin(grid_nu_sig), grid_nu_sig.shape)
    nu_lbl   = '∞' if HEAT_NU[bni] > 1e6 else str(int(HEAT_NU[bni]))
    print(f"  (F) R_base×R_scale tốt : R_base={HEAT_RBASE[bri]}, "
          f"R_scale={HEAT_RSCALE[brj]}  → {grid_rb_rs[bri,brj]:.1f} mm")
    print(f"  (G) ν×σ tốt nhất       : ν={nu_lbl}, "
          f"σ={HEAT_SIGMA[bnj]}  → {grid_nu_sig[bni,bnj]:.1f} mm")

    print(f"\n[✓] Tất cả kết quả lưu tại '{SAVE_DIR}/'")
    print(f"    ablation_1d.png")
    print(f"    ablation_heatmaps.png")
    print(f"    ablation_summary_table.png")


if __name__ == "__main__":
    main()