"""Convert the 718 Blickfeld CSV frames into a single compressed NumPy cache.

CSV parsing dominates runtime otherwise (~4 MB of ASCII per frame).
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "00_lidar_data")
OUT = os.path.join(ROOT, "cache")

COLS = ["X", "Y", "Z", "DISTANCE", "INTENSITY", "POINT_ID", "RETURN_ID", "AMBIENT", "TIMESTAMP"]


def frame_files():
    files = []
    for part in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, part)
        if os.path.isdir(d):
            files += glob.glob(os.path.join(d, "*.csv"))
    return sorted(files, key=lambda p: int(re.search(r"frame-(\d+)\.csv$", os.path.basename(p)).group(1)))


def main():
    os.makedirs(OUT, exist_ok=True)
    files = frame_files()
    print(f"{len(files)} frames", flush=True)

    xyz, dia, counts, frame_ids, t_start = [], [], [], [], []
    for i, f in enumerate(files):
        df = pd.read_csv(f, sep=";", usecols=COLS, dtype=np.float64)
        xyz.append(df[["X", "Y", "Z"]].to_numpy(np.float32))
        dia.append(df[["DISTANCE", "INTENSITY", "AMBIENT"]].to_numpy(np.float32))
        counts.append(len(df))
        frame_ids.append(int(re.search(r"frame-(\d+)\.csv$", os.path.basename(f)).group(1)))
        t_start.append(df["TIMESTAMP"].min())
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    np.savez(
        os.path.join(OUT, "frames.npz"),
        xyz=np.concatenate(xyz),
        dia=np.concatenate(dia),
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        frame_ids=np.array(frame_ids, np.int32),
        t_start=np.array(t_start, np.float64),
    )
    print("wrote", os.path.join(OUT, "frames.npz"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
