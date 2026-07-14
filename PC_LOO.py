# -*- coding: utf-8 -*-
"""
LOO Grid Search cho PC-KF, PC-EKF, PC-UKF
==========================================
Công tắc RUN_KF / RUN_EKF / RUN_UKF để chạy từng con độc lập.

Chạy: python loo_pc_search.py
"""

import os, sys, glob, itertools
import numpy as np

DATA_DIR = "./data"

# ══════════════════════════════════════════════════════════════════════
#  CÔNG TẮC
# ══════════════════════════════════════════════════════════════════════
RUN_KF  = False
RUN_EKF = False
RUN_UKF = True

# ══════════════════════════════════════════════════════════════════════
#  PARAMS HIỆN TẠI
# ══════════════════════════════════════════════════════════════════════
CURRENT = {
    'PC-KF' : dict(pckf_q=0.01,  pckf_r_base=200.0, pckf_r_scale=10.0,  pckf_sigma=200.0),
    'PC-EKF': dict(q=0.1,         r_base=300.0,       r_scale=20.0,        sigma=50.0),
    'PC-UKF': dict(q=0.1,         r_base=300.0,       r_scale=20.0,        sigma=50.0),
}

# ══════════════════════════════════════════════════════════════════════
#  GRID
# ══════════════════════════════════════════════════════════════════════
GRID_KF = {
    'pckf_q'      : [0.001, 0.01, 0.1],
    'pckf_r_base' : [25.0, 50.0, 100.0, 200.0, 300.0, 500.0],
    'pckf_r_scale': [5.0, 10.0, 20.0, 30.0],
    'pckf_sigma'  : [25.0, 50.0, 100.0, 150.0, 200.0, 300.0],
}

GRID_EKF = {
    'q'      : [0.01, 0.1, 1.0, 10.0],
    'r_base' : [50.0, 100.0, 200.0, 300.0, 500.0],
    'r_scale': [5.0, 10.0, 20.0, 30.0],
    'sigma'  : [25.0, 50.0, 100.0, 200.0, 300.0],
}

# UKF: best nằm ở biên cũ (q=0.1 min, r_base=300 max, sigma=50 min)
# → mở rộng ra ngoài biên đó
# q nhỏ hơn: thêm 0.01, 0.001
# r_base lớn hơn: thêm 500, 800
# r_scale: giữ quanh 20-30, thêm 40
# sigma nhỏ hơn: thêm 10, 25
# 4×4×3×4 = 192 combos — vẫn nhanh hơn run cũ (400×8=3200)
GRID_UKF = {
    'q'      : [0.001, 0.01, 0.1, 1.0],
    'r_base' : [300.0, 500.0, 800.0, 1000.0],
    'r_scale': [20.0, 30.0, 40.0],
    'sigma'  : [10.0, 25.0, 50.0, 100.0],
}


# ══════════════════════════════════════════════════════════════════════
#  IMPORT MODULES
# ══════════════════════════════════════════════════════════════════════
import importlib.util, io, contextlib

def silent_import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Importing modules...", end=" ", flush=True)
mod_kf  = silent_import("PC_KF",  "PC-KF.py")  if RUN_KF  else None
mod_ekf = silent_import("PC_EKF", "PC-EKF.py") if RUN_EKF else None
mod_ukf = silent_import("PC_UKF", "PC-UKF.py") if RUN_UKF else None
_first  = next(m for m in [mod_kf, mod_ekf, mod_ukf] if m is not None)
print("OK")


# ══════════════════════════════════════════════════════════════════════
#  LOO HELPERS
# ══════════════════════════════════════════════════════════════════════
def loo_rmse(all_files, gt_xy, eval_fn, err_key, params):
    scores = []
    for i in range(len(all_files)):
        val = [all_files[i]]
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                err_dict, *_ = eval_fn(val, gt_xy, **params)
                e = err_dict[err_key]
                scores.append(float(np.sqrt(np.mean(e**2))))
            except Exception:
                scores.append(float('inf'))
    return float(np.mean(scores))


def grid_loo(name, all_files, gt_xy, eval_fn, err_key, grid, current_params,
             top_n=5, early_stop=80):
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total  = len(combos)

    print(f"\n{'═'*62}")
    print(f"  LOO Grid Search — {name}")
    print(f"  {total} combinations × {len(all_files)} LOO folds = {total*len(all_files)} runs")
    print(f"  Grid: { {k: grid[k] for k in keys} }")
    print(f"{'═'*62}")

    rmse_current = loo_rmse(all_files, gt_xy, eval_fn, err_key, current_params)
    print(f"  Baseline (params hiện tại): RMSE = {rmse_current:.2f} mm")
    print(f"  Searching...", flush=True)

    results = []
    best_so_far = float('inf')
    no_improve  = 0
    stopped_early = False

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        rmse   = loo_rmse(all_files, gt_xy, eval_fn, err_key, params)
        results.append((rmse, params))

        if rmse < best_so_far:
            best_so_far = rmse
            no_improve  = 0
            print(f"    [{idx+1}/{total}] ★ New best: {rmse:.2f} mm  {params}",
                  flush=True)
        else:
            no_improve += 1

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{total} done, best so far: {best_so_far:.2f} mm "
                  f"(no improve: {no_improve}/{early_stop})", flush=True)

        if no_improve >= early_stop:
            print(f"\n  ⏹  Early stop tại combo {idx+1}/{total} "
                  f"(no improve {early_stop} combo liên tiếp)", flush=True)
            stopped_early = True
            break

    results.sort(key=lambda x: x[0])
    best_rmse, best_params = results[0]

    # Cảnh báo nếu best vẫn nằm ở biên grid
    _warn = []
    for k, v in best_params.items():
        vals = grid[k]
        if v == min(vals): _warn.append(f"{k}={v} (MIN biên)")
        if v == max(vals): _warn.append(f"{k}={v} (MAX biên)")
    if _warn:
        print(f"\n  ⚠️  Best vẫn nằm ở biên grid: {', '.join(_warn)}")
        print(f"     → Cân nhắc mở rộng grid thêm!")

    print(f"\n  Top {top_n} configs:")
    print(f"  {'RMSE':>8s}  params")
    print(f"  {'─'*55}")
    for rmse, p in results[:top_n]:
        tag = " ← BEST" if rmse == best_rmse else ""
        print(f"  {rmse:>7.2f}mm  {p}{tag}")

    improvement = rmse_current - best_rmse
    pct = improvement / rmse_current * 100
    print(f"\n  {'─'*55}")
    print(f"  Baseline : {rmse_current:.2f} mm  {current_params}")
    print(f"  LOO best : {best_rmse:.2f} mm  {best_params}")
    if improvement > 0:
        print(f"  → LOO cải thiện {improvement:.2f} mm ({pct:.1f}%) ✅")
    else:
        print(f"  → Params hiện tại đã tốt hơn ({abs(improvement):.2f} mm) 🏆")

    return best_params, best_rmse, rmse_current


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not all_files:
        print(f"Không tìm thấy .txt trong '{DATA_DIR}'"); return
    print(f"Found {len(all_files)} files")

    gt_xy, *_ = _first.build_ground_truth()
    print(f"Ground truth: {len(gt_xy)} points")
    print(f"Running: KF={RUN_KF} | EKF={RUN_EKF} | UKF={RUN_UKF}\n")

    summary = {}

    if RUN_KF:
        best, rmse, baseline = grid_loo(
            name="PC-KF", all_files=all_files, gt_xy=gt_xy,
            eval_fn=mod_kf.evaluate_files, err_key="PC-KF + WLS",
            grid=GRID_KF, current_params=CURRENT['PC-KF'],
        )
        summary['PC-KF'] = dict(baseline=baseline, loo=rmse, params=best)

    if RUN_EKF:
        best, rmse, baseline = grid_loo(
            name="PC-EKF", all_files=all_files, gt_xy=gt_xy,
            eval_fn=mod_ekf.evaluate_files, err_key="PC-EKF-2D",
            grid=GRID_EKF, current_params=CURRENT['PC-EKF'],
        )
        summary['PC-EKF'] = dict(baseline=baseline, loo=rmse, params=best)

    if RUN_UKF:
        best, rmse, baseline = grid_loo(
            name="PC-UKF", all_files=all_files, gt_xy=gt_xy,
            eval_fn=mod_ukf.evaluate_files, err_key="PC-UKF-2D",
            grid=GRID_UKF, current_params=CURRENT['PC-UKF'],
        )
        summary['PC-UKF'] = dict(baseline=baseline, loo=rmse, params=best)

    if not summary:
        print("Không có method nào được bật."); return

    print(f"\n\n{'═'*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Method':<10s} {'Baseline':>12s} {'LOO Best':>12s} {'Improve':>10s}  Best params")
    print(f"  {'─'*66}")
    for name, s in summary.items():
        imp = s['baseline'] - s['loo']
        pct = imp / s['baseline'] * 100
        tag = f"+{pct:.1f}%" if imp > 0 else f"{pct:.1f}%"
        print(f"  {name:<10s} {s['baseline']:>11.2f}mm {s['loo']:>11.2f}mm "
              f"{tag:>10s}  {s['params']}")

    print(f"\n  Copy params vào file gốc:")
    for name, s in summary.items():
        print(f"\n  # {name}")
        for k, v in s['params'].items():
            print(f"  {k.upper():<20s} = {v}")


if __name__ == "__main__":
    main()