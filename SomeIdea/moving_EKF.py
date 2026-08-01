# -*- coding: utf-8 -*-
"""
Moving_EKF.py  —  Moving experiment analysis
=============================================
Đọc file dạng:  Moving_{METHOD}_{x0}_{y0}_To_{x1}_{y1}.txt
GT là đoạn thẳng từ (x0,y0) → (x1,y1).
GT nội suy theo Time(s) thực tế của từng file (không giả định tốc độ đều theo sample).

Ví dụ tên file:
  Moving_EKF_300_800_To_5830_800.txt
  Moving_KF_300_800_To_5830_800.txt
  Moving_LS_300_800_To_5830_800.txt
  Moving_PC_EKF_300_800_To_5830_800.txt

Vẽ:
  - trajectory_<route>.png  : đường đi ước lượng vs GT (mỗi route 1 subplot)
  - error_time_<route>.png  : sai số theo sample index
"""

import os
import re
import glob
import math
import warnings
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
ANCHORS = np.array([
    [0,    0   ],   # A1
    [0,    4720],   # A2
    [6300, 4720],   # A3
    [6300, 0   ],   # A4
], dtype=float)

ANCHOR_HEIGHT = 1400.0

DATA_DIR = "./data"
SAVE_DIR = "./outputs_moving"

# Màu cho từng method
METHOD_COLORS = {
    "LS"     : '#9E9E9E',
    "KF"     : '#00BCD4',
    "EKF"    : '#E91E63',
    "PC-EKF" : '#9C27B0',
}

METHOD_ORDER = ["LS", "KF", "EKF", "PC-EKF"]


# ══════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS
# ══════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _ls_position_nb(distances, anchors):
    """Giải LS position bằng nghiệm giải tích 2×2 (không dùng np.linalg.inv)."""
    x0 = anchors[0, 0]; y0 = anchors[0, 1]
    d0 = distances[0]
    if d0 < 1.0: d0 = 1.0
    N = anchors.shape[0]
    A = np.empty((N - 1, 2))
    b_vec = np.empty(N - 1)
    for i in range(1, N):
        xi = anchors[i, 0]; yi = anchors[i, 1]
        di = distances[i]
        if di < 1.0: di = 1.0
        A[i-1, 0] = 2.0 * (xi - x0)
        A[i-1, 1] = 2.0 * (yi - y0)
        b_vec[i-1] = (d0*d0 - di*di) - (x0*x0 - xi*xi) - (y0*y0 - yi*yi)
    AtA = A.T @ A
    Atb = A.T @ b_vec
    det = AtA[0, 0]*AtA[1, 1] - AtA[0, 1]*AtA[1, 0]
    pos = np.empty(2)
    if abs(det) < 1e-12:
        pos[0] = np.nan; pos[1] = np.nan
    else:
        pos[0] = (AtA[1, 1]*Atb[0] - AtA[0, 1]*Atb[1]) / det
        pos[1] = (AtA[0, 0]*Atb[1] - AtA[1, 0]*Atb[0]) / det
    return pos


@njit(cache=True)
def _ls_file_nb(dist_mat, anchors):
    """Batch LS cho toàn bộ file: dist_mat (T, N_anchors) → pos (T, 2)."""
    T = dist_mat.shape[0]
    pos = np.empty((T, 2))
    for t in range(T):
        p = _ls_position_nb(dist_mat[t], anchors)
        pos[t, 0] = p[0]; pos[t, 1] = p[1]
    return pos


def _warmup_numba():
    """Compile tất cả @njit kernels khi import."""
    _d = np.array([1000.0, 1200.0, 800.0, 900.0])
    _a = np.array([[0.0, 0.0], [0.0, 4720.0], [6300.0, 4720.0], [6300.0, 0.0]])
    _ls_position_nb(_d, _a)
    _ls_file_nb(np.stack([_d]), _a)


_warmup_numba()


# ══════════════════════════════════════════════════════════════════════
#  PARSE FILE
# ══════════════════════════════════════════════════════════════════════
def parse_file(path):
    """
    Đọc file txt output của filter.
    Format sau dấu ---:
      Time(s)  Frame   X(mm)   Y(mm)   [d0 d1 d2 d3] [s0 s1 s2 s3]
    Trả về:
      times : np.array (T,)   — timestamp tính từ đầu file (s)
      pos   : np.array (T, 2) — cột X, Y (mm)
    """
    times = []
    rows  = []
    in_data = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('---'):
                in_data = True
                continue
            if not in_data:
                continue
            if line.startswith('['):       # [EKF] init ...
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                times.append(t)
                rows.append([x, y])
            except ValueError:
                continue
    if not rows:
        return None, None
    times = np.array(times, dtype=np.float64)
    # Chuẩn hoá về t=0 tại sample đầu tiên
    times -= times[0]
    return times, np.array(rows, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════
#  THÔNG TIN TỪ TÊN FILE
# ══════════════════════════════════════════════════════════════════════
def parse_moving_filename(path):
    """
    Lấy method, start, end từ tên file.
    Pattern: Moving_{METHOD}_{x0}_{y0}_To_{x1}_{y1}.txt
             Moving_PC_EKF_{x0}_{y0}_To_{x1}_{y1}.txt

    Trả về (method_str, (x0,y0), (x1,y1)) hoặc None nếu không parse được.
    """
    name = os.path.splitext(os.path.basename(path))[0]   # bỏ .txt

    # Tách phần "To" (không phân biệt hoa thường)
    m = re.match(
        r'Moving_(.+?)_(-?\d+)_(-?\d+)_To_(-?\d+)_(-?\d+)$',
        name, re.IGNORECASE
    )
    if not m:
        return None, None, None

    method_raw = m.group(1).upper()          # VD: "PC_EKF" hoặc "EKF"
    x0, y0     = float(m.group(2)), float(m.group(3))
    x1, y1     = float(m.group(4)), float(m.group(5))

    # Chuẩn hoá tên method
    if 'PC_EKF' in method_raw or 'PC-EKF' in method_raw:
        method = 'PC-EKF'
    elif 'EKF' in method_raw:
        method = 'EKF'
    elif 'KF' in method_raw:
        method = 'KF'
    elif 'LS' in method_raw:
        method = 'LS'
    else:
        method = method_raw

    return method, (x0, y0), (x1, y1)


def route_key(start, end):
    """Key duy nhất cho mỗi route: '300_800_To_5830_800'."""
    return f"{int(start[0])}_{int(start[1])}_To_{int(end[0])}_{int(end[1])}"


# ══════════════════════════════════════════════════════════════════════
#  KHOẢNG CÁCH VUÔNG GÓC ĐẾN ĐOẠN THẲNG GT
# ══════════════════════════════════════════════════════════════════════
def dist_point_to_segment(points, start, end):
    """
    Tính khoảng cách ngắn nhất từ mỗi điểm trong `points` đến
    đoạn thẳng start→end (không phải đường thẳng vô hạn).

    Công thức: chiếu điểm lên đoạn thẳng, clamp t∈[0,1],
    rồi lấy khoảng cách Euclidean đến điểm chiếu.

    Không cần timestamp, không cần đồng bộ sample — đúng với mọi
    frame rate và mọi số dòng khác nhau giữa các method.
    """
    s = np.array(start, dtype=float)
    e = np.array(end,   dtype=float)
    seg      = e - s
    seg_len2 = seg @ seg

    if seg_len2 < 1e-12:          # start == end (điểm tĩnh)
        diff = points - s
        return np.linalg.norm(diff, axis=1), np.tile(s, (len(points), 1))

    vecs = points - s
    t    = np.clip(vecs @ seg / seg_len2, 0.0, 1.0)
    proj = s + np.outer(t, seg)   # (T, 2) điểm gần nhất trên đoạn
    errs = np.linalg.norm(points - proj, axis=1)
    return errs, proj


# ══════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(pos, start, end):
    """
    RMSE/MAE/P95 = khoảng cách vuông góc ngắn nhất từng điểm đến
    đoạn thẳng GT start→end.

    Không phụ thuộc timestamp hay số sample — công bằng với mọi method.
    """
    valid_mask = ~np.any(np.isnan(pos), axis=1)
    errs_full  = np.full(len(pos), np.nan)

    if valid_mask.any():
        errs_valid, _ = dist_point_to_segment(pos[valid_mask], start, end)
        errs_full[valid_mask] = errs_valid

    e = errs_full[~np.isnan(errs_full)]
    if len(e) == 0:
        return dict(rmse=np.nan, mae=np.nan, p95=np.nan, errors=errs_full)
    return dict(
        rmse   = np.sqrt(np.mean(e**2)),
        mae    = np.mean(e),
        p95    = np.percentile(e, 95),
        errors = errs_full,
    )


def print_metrics(route_results):
    """
    route_results: { route_key: { method: {pos, gt, metrics, fname} } }
    """
    print(f"\n{'═'*75}")
    print(f"  MOVING RMSE SUMMARY  (mm)")
    print(f"{'═'*75}")

    for rkey in sorted(route_results.keys()):
        methods = route_results[rkey]
        # lấy route info từ method đầu tiên
        first = next(iter(methods.values()))
        start = first['start']
        end   = first['end']
        print(f"\n  Route: ({int(start[0])},{int(start[1])}) → ({int(end[0])},{int(end[1])})")
        print(f"  {'Method':<10s} {'File':<40s} {'N':>5s} {'RMSE':>8s} {'MAE':>8s} {'P95':>8s}")
        print(f"  {'─'*75}")

        for method in METHOD_ORDER:
            if method not in methods:
                continue
            d = methods[method]
            m = d['metrics']
            n = np.sum(~np.isnan(m['errors']))
            print(f"  {method:<10s} {d['fname']:<40s} {n:>5d} "
                  f"{m['rmse']:>8.1f} {m['mae']:>8.1f} {m['p95']:>8.1f}")


# ══════════════════════════════════════════════════════════════════════
#  PLOT 1 — Trajectory (đường đi)
# ══════════════════════════════════════════════════════════════════════
def plot_trajectory(route_results, save_dir):
    """
    Mỗi route 1 figure.
    Vẽ GT (đường đứt nét đỏ) + estimated trajectory của từng method.
    """
    for rkey, methods in route_results.items():
        first  = next(iter(methods.values()))
        start  = first['start']
        end    = first['end']
        title  = f"Moving Trajectory  ({int(start[0])},{int(start[1])}) → ({int(end[0])},{int(end[1])})"

        fig, ax = plt.subplots(figsize=(10, 7))

        # GT đường thẳng
        gt_line = np.array([start, end])
        ax.plot(gt_line[:, 0], gt_line[:, 1],
                color='red', lw=2.5, ls='--', zorder=8, label='Ground Truth')
        ax.scatter([start[0], end[0]], [start[1], end[1]],
                   s=120, color='red', zorder=9)
        ax.annotate('Start', start, textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color='red')
        ax.annotate('End', end, textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color='red')

        for method in METHOD_ORDER:
            if method not in methods:
                continue
            d     = methods[method]
            pos   = d['pos']
            m     = d['metrics']
            color = METHOD_COLORS.get(method, 'gray')
            valid = pos[~np.any(np.isnan(pos), axis=1)]
            ax.plot(valid[:, 0], valid[:, 1],
                    color=color, lw=1.2, alpha=0.75,
                    label=f"{method}  RMSE={m['rmse']:.1f}mm")

        # Anchors
        for j, (ax_, ay_) in enumerate(ANCHORS):
            ax.scatter(ax_, ay_, s=80, marker='s', color='black', zorder=10)
            ax.annotate(f"A{j+1}", (ax_, ay_),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=9, color='black')

        ax.set_xlabel("X (mm)", fontsize=12)
        ax.set_ylabel("Y (mm)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"trajectory_{rkey}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Trajectory → {save_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 2 — Error theo thời gian
# ══════════════════════════════════════════════════════════════════════
def plot_error_time(route_results, save_dir):
    """
    Mỗi route 1 figure.
    Trục X: sample index, trục Y: error (mm).
    """
    for rkey, methods in route_results.items():
        first = next(iter(methods.values()))
        start = first['start']
        end   = first['end']
        title = (f"Moving Error over Time  "
                 f"({int(start[0])},{int(start[1])}) → ({int(end[0])},{int(end[1])})")

        fig, ax = plt.subplots(figsize=(11, 5))

        for method in METHOD_ORDER:
            if method not in methods:
                continue
            d     = methods[method]
            errs  = d['metrics']['errors']
            m     = d['metrics']
            color = METHOD_COLORS.get(method, 'gray')
            lw    = 1.8 if method == 'PC-EKF' else 1.1
            alpha = 0.9 if method == 'PC-EKF' else 0.65
            t_ax  = d['times']          # trục X là Time(s) thực tế
            ax.plot(t_ax, errs, color=color, lw=lw, alpha=alpha,
                    label=f"{method}  RMSE={m['rmse']:.1f}mm  MAE={m['mae']:.1f}mm")

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Perpendicular Error to GT (mm)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"error_time_{rkey}.png")
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Error-time → {save_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 3 — RMSE bar chart so sánh tất cả route + method
# ══════════════════════════════════════════════════════════════════════
def plot_rmse_bar(route_results, save_dir):
    """
    Bar chart: trục X = route, mỗi group = các method.
    """
    routes  = sorted(route_results.keys())
    methods = [m for m in METHOD_ORDER
               if any(m in route_results[r] for r in routes)]

    n_routes  = len(routes)
    n_methods = len(methods)
    if n_routes == 0 or n_methods == 0:
        return

    x       = np.arange(n_routes)
    width   = 0.8 / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_routes), 5))

    for i, method in enumerate(methods):
        rmse_vals = []
        for rkey in routes:
            if method in route_results[rkey]:
                rmse_vals.append(route_results[rkey][method]['metrics']['rmse'])
            else:
                rmse_vals.append(np.nan)
        color = METHOD_COLORS.get(method, 'gray')
        bars  = ax.bar(x + offsets[i], rmse_vals, width * 0.9,
                       label=method, color=color, alpha=0.85)
        for bar, val in zip(bars, rmse_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f"{val:.0f}", ha='center', va='bottom', fontsize=8)

    # Nhãn trục X: route ngắn gọn
    xlabels = []
    for rkey in routes:
        # "300_800_To_5830_800" → "300,800→5830,800"
        xlabels.append(rkey.replace('_To_', '→').replace('_', ','))

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("RMSE (mm)", fontsize=12)
    ax.set_title("Moving Experiment — RMSE Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', ls='--', alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "rmse_bar_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] RMSE bar → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Moving_EKF.py  —  Moving UWB Positioning Analysis")
    print("=" * 65)

    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.isdir(DATA_DIR):
        print(f"\n[!] Không tìm thấy thư mục '{DATA_DIR}/'")
        return

    # Glob đệ quy tất cả Moving_*.txt
    all_files = sorted(set(
        glob.glob(os.path.join(DATA_DIR, "**", "Moving_*.txt"), recursive=True) +
        glob.glob(os.path.join(DATA_DIR, "Moving_*.txt"))
    ))

    if not all_files:
        print(f"\n[!] Không tìm thấy file Moving_*.txt trong '{DATA_DIR}/'")
        return

    print(f"\n  Tìm thấy {len(all_files)} file(s):")
    for f in all_files:
        print(f"    {os.path.basename(f)}")

    # ── Đọc và phân nhóm theo route ────────────────────────────────────
    # route_results[rkey][method] = { pos, gt, start, end, metrics, fname }
    route_results = {}

    for path in all_files:
        method, start, end = parse_moving_filename(path)
        if start is None:
            print(f"  [!] Không parse được tên: {os.path.basename(path)}")
            continue

        times, pos = parse_file(path)
        if pos is None or len(pos) == 0:
            print(f"  [!] Không đọc được data: {os.path.basename(path)}")
            continue

        # RMSE = khoảng cách vuông góc ngắn nhất đến đoạn thẳng GT
        # → không cần timestamp, không phụ thuộc số sample
        met = compute_metrics(pos, start, end)

        rkey = route_key(start, end)
        if rkey not in route_results:
            route_results[rkey] = {}

        route_results[rkey][method] = dict(
            pos     = pos,
            times   = times,
            start   = start,
            end     = end,
            metrics = met,
            fname   = os.path.basename(path),
        )

        print(f"  [OK] {os.path.basename(path):45s} "
              f"N={len(pos):4d}  RMSE={met['rmse']:7.1f}mm")

    if not route_results:
        print("\n[!] Không đọc được dữ liệu nào.")
        return

    # ── In metrics ─────────────────────────────────────────────────────
    print_metrics(route_results)

    # ── Vẽ ─────────────────────────────────────────────────────────────
    print(f"\n  Đang vẽ...")
    plot_trajectory(route_results, SAVE_DIR)
    plot_error_time(route_results, SAVE_DIR)
    plot_rmse_bar(route_results, SAVE_DIR)

    print(f"\n[✓] Tất cả kết quả lưu tại '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
