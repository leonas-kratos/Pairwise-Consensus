# -*- coding: utf-8 -*-
"""
LS — UWB Realtime
==================
Least Squares thuần (WLS), không filter.
Mỗi frame cho ra 1 ước lượng vị trí độc lập.
"""

import sys, math, time, csv, argparse
import numpy as np
import serial
import re
from _robot_mixin import start_robot, stop_robot

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
SERIAL_PORT = '/dev/serial/by-id/usb-SEGGER_J-Link_000760192143-if00'
BAUD_RATE   = 921600

ANCHORS = np.array([
    [0,    0   ],
    [0,    4720],
    [6300, 4720],
    [6300, 0   ],
], dtype=float)

ANCHOR_IDS    = [0x1001, 0x1002, 0x1003, 0x1004]
ANCHOR_HEIGHT = 1400.0
N_ANCHORS     = 4

LOG_FILE = "ls_log.csv"

# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════════════
def slant_to_ground(d):
    return math.sqrt(max(d * d - ANCHOR_HEIGHT * ANCHOR_HEIGHT, 0.0))


def ls_position(z_ground):
    """
    WLS dùng tất cả C(N,2) cặp anchor.
    Linearize: d_i² - d_j² = 2(x_j-x_i)x + 2(y_j-y_i)y + (x_i²-x_j²+y_i²-y_j²)
    Trả về [x, y] mm.
    """
    rows, b = [], []
    for i in range(N_ANCHORS):
        for j in range(i + 1, N_ANCHORS):
            xi, yi = ANCHORS[i]; di = max(z_ground[i], 1.0)
            xj, yj = ANCHORS[j]; dj = max(z_ground[j], 1.0)
            rows.append([2*(xi - xj), 2*(yi - yj)])
            b.append((dj**2 - di**2) + (xi**2 - xj**2) + (yi**2 - yj**2))
    A  = np.array(rows, dtype=float)
    bv = np.array(b,    dtype=float)
    try:
        pos, *_ = np.linalg.lstsq(A, bv, rcond=None)
        return pos
    except Exception:
        return np.array([np.nan, np.nan])


# ══════════════════════════════════════════════════════════════════════
#  SERIAL PARSER
# ══════════════════════════════════════════════════════════════════════
class SerialParser:
    def __init__(self):
        self.buf = {}

    def feed(self, line):
        if isinstance(line, bytes):
            line = line.decode("ascii", errors="ignore")
        line = line.strip()
        if not line:
            return None
        m = re.search(r'(0x[0-9A-Fa-f]+)\s*,\s*([0-9]+(?:\.[0-9]+)?)', line)
        if m is None:
            return None
        try:
            aid  = int(m.group(1), 16)
            dist = float(m.group(2))
        except Exception:
            return None
        if aid not in ANCHOR_IDS:
            return None
        self.buf[ANCHOR_IDS.index(aid)] = dist
        if len(self.buf) == N_ANCHORS:
            frame = np.array([self.buf[i] for i in range(N_ANCHORS)], dtype=float)
            self.buf.clear()
            return frame
        return None


# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run(port, baud, log_file):
    parser  = SerialParser()
    frame_n = 0
    t0      = None

    print(f"Đang kết nối {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    print("OK. Đang chờ dữ liệu... (Ctrl+C để dừng)\n")
    start_robot()
    print(f"{'Time(s)':>8}  {'Frame':>6}  {'X(mm)':>9}  {'Y(mm)':>9}")
    print("-" * 40)

    with open(log_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'frame', 'x_mm', 'y_mm',
                    'd0', 'd1', 'd2', 'd3'])
        try:
            while True:
                raw   = ser.readline()
                if not raw:
                    continue
                frame = parser.feed(raw)
                if frame is None:
                    continue

                if t0 is None:
                    t0 = time.perf_counter()

                z_g = np.array([slant_to_ground(d) for d in frame])
                pos = ls_position(z_g)
                frame_n += 1
                t = time.perf_counter() - t0

                print(f"{t:8.3f}  {frame_n:6d}  {pos[0]:9.1f}  {pos[1]:9.1f}")
                w.writerow([f"{t:.4f}", frame_n,
                             f"{pos[0]:.2f}", f"{pos[1]:.2f}",
                             *[f"{d:.1f}" for d in frame]])
                f.flush()

        except KeyboardInterrupt:
            pass
        finally:
            stop_robot()
            ser.close()
            print(f"\n[✓] Dừng. {frame_n} frames. Log → {log_file}")


# ══════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════
def demo():
    rng       = np.random.default_rng(42)
    WAYPOINTS = np.array([[800,400],[3000,400],[3000,8000],[800,8000],[800,400]], dtype=float)
    segs      = np.diff(WAYPOINTS, axis=0)
    seg_len   = np.linalg.norm(segs, axis=1)
    cum       = np.concatenate([[0], np.cumsum(seg_len)])
    total     = seg_len.sum()

    gt_pts = []
    for d in np.linspace(0, total, 500):
        idx  = np.clip(np.searchsorted(cum, d, 'right') - 1, 0, len(segs)-1)
        frac = (d - cum[idx]) / (seg_len[idx] + 1e-9)
        gt_pts.append(WAYPOINTS[idx] + frac * segs[idx])

    start_robot()
    print(f"{'Step':>5}  {'X(mm)':>9}  {'Y(mm)':>9}  {'Err(mm)':>8}")
    print("-" * 40)

    errors = []
    for i, gt in enumerate(gt_pts):
        nlos = rng.normal(300, 80) if 150 < i < 300 else 0.0
        z = np.array([
            math.sqrt((gt[0]-ax)**2 + (gt[1]-ay)**2 + ANCHOR_HEIGHT**2)
            + rng.normal(0, 50) + (nlos if j == 2 else 0.0)
            for j, (ax, ay) in enumerate(ANCHORS)
        ])
        z_g = np.array([slant_to_ground(d) for d in z])
        pos = ls_position(z_g)
        err = np.linalg.norm(pos - gt)
        errors.append(err)
        print(f"{i:5d}  {pos[0]:9.1f}  {pos[1]:9.1f}  {err:8.1f}")
        time.sleep(0.005)

    errors = np.array(errors)
    print(f"\nRMSE={np.sqrt(np.mean(errors**2)):.1f}mm  "
          f"MAE={np.mean(errors):.1f}mm  "
          f"P95={np.percentile(errors, 95):.1f}mm")
    stop_robot()


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="LS — UWB Realtime")
    ap.add_argument("--port",  default=SERIAL_PORT)
    ap.add_argument("--baud",  type=int, default=BAUD_RATE)
    ap.add_argument("--log",   default=LOG_FILE)
    ap.add_argument("--demo",  action="store_true")
    args = ap.parse_args()

    print("LS — UWB Realtime")
    print(f"  Anchors : {[hex(i) for i in ANCHOR_IDS]}")
    print(f"  Height  : {ANCHOR_HEIGHT} mm\n")

    if args.demo:
        demo()
    else:
        run(args.port, args.baud, args.log)

if __name__ == "__main__":
    main()
