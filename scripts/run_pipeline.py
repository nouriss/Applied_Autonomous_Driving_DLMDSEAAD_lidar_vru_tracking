"""Run detection, classification and tracking over the whole recording."""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from lidar_track.classify import RuleClassifier  # noqa: E402
from lidar_track.detect import Detector  # noqa: E402
from lidar_track.io import Sequence  # noqa: E402
from lidar_track.preprocess import PlaneSmoother  # noqa: E402
from lidar_track.track import Tracker  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--tag", default="full")
    # 0.30 is the adopted value from scripts/tune_plane_smoothing.py
    # (outputs/tune_plane_smoothing.csv): cuts the fixed post at (-11.9, 31.7)
    # ("post_1")'s pedestrian-classification flicker from 10.1% to 1.9% of the
    # frames it is present in, at zero cost to detection recall,
    # classification rate, MOTA, IDF1 or false-alarm count - confirmed by a
    # position-matched diff of every per-frame label against the unsmoothed
    # baseline: all 148 changes land on known static structure (post_1,
    # post_2, and one previously undocumented site near (-10, 35.5)), none on
    # any of the four real tracked objects. 0.15 (matching the value already
    # used for /tf display) suppresses the flicker further (0.3%) but costs
    # one extra frame on the cyclist track past its ground-truth window,
    # dropping MOTA 0.817->0.800 - 0.30 is the value that is actually free.
    # Pass 0 to disable and reproduce the pre-adoption, per-frame-only fit.
    ap.add_argument("--plane-alpha", type=float, default=0.30,
                    help="temporally smooth the ground plane used for "
                         "segmentation and classification (PlaneSmoother in "
                         "preprocess.py) with this EMA weight; 0 disables it, "
                         "reproducing the raw per-frame RANSAC fit. See "
                         "scripts/tune_plane_smoothing.py.")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    seq = Sequence()
    stop = args.stop if args.stop is not None else len(seq)
    plane_smoother = PlaneSmoother(alpha=args.plane_alpha) if args.plane_alpha > 0 else None
    det = Detector(plane_smoother=plane_smoother)
    clf = RuleClassifier()
    # kinematic_speed_mid/scale/weight are the best point of the sweep in
    # scripts/tune_kinematic_classifier.py (outputs/tune_kinematic_classifier.csv):
    # 86/94 -> 87/94 on this recording's raw per-frame classification rate,
    # at the cost of one new error at the car/cyclist crossing point (see
    # development_log.md for the full frame-by-frame trade-off table before
    # adopting this). classifier=clf is what turns Tracker._kinematic_relabel()
    # on at all - passing None (the default) reproduces the pre-adoption
    # behaviour exactly.
    trk = Tracker(dt=seq.dt, classifier=clf, kinematic_speed_mid=1.6,
                 kinematic_speed_scale=0.8, kinematic_weight=0.3)

    det_rows, plane_rows = [], []
    t0 = time.time()
    for i in range(args.start, stop):
        f = seq[i]
        r = det(f)
        for d in r.detections:
            d.label, d.score = clf(d.features)
        # trk.update() mutates d.label/d.score in place for any detection
        # whose pedestrian/cyclist ambiguity it arbitrates with kinematics
        # (Tracker._kinematic_relabel) - det_rows below must log what came
        # out of that, not the pre-tracker shape-only guess, or the adopted
        # gain would never show up in outputs/detections_full.csv or
        # evaluate.py's classification metric.
        trk.update(r.detections, r.frame_id, r.t)
        for d in r.detections:
            fr = d.features
            det_rows.append(dict(frame=r.frame_id, t=r.t, label=d.label, score=d.score,
                                 **{k: getattr(fr, k) for k in
                                    ["n_points", "cx", "cy", "cz", "rng", "length", "width",
                                     "height", "h_min", "h_max", "yaw", "density", "occupancy",
                                     "linearity", "planarity", "sphericity",
                                     "intensity_mean", "intensity_std", "fill_ratio", "aspect"]}))
        plane_rows.append(dict(frame=r.frame_id, t=r.t, height=r.plane.height,
                               pitch=r.plane.pitch_deg, roll=r.plane.roll_deg,
                               inliers=r.plane.n_inliers, n_valid=r.n_valid,
                               n_ground=r.n_ground, n_obstacle=r.n_obstacle,
                               n_det=len(r.detections)))
        if (i - args.start) % 50 == 0:
            print(f"  {i - args.start}/{stop - args.start}  {len(trk.tracks)} live tracks", flush=True)

    dt = time.time() - t0
    print(f"\n{stop - args.start} frames in {dt:.1f}s ({dt / (stop - args.start):.3f}s/frame)")

    pd.DataFrame(det_rows).to_csv(os.path.join(OUT, f"detections_{args.tag}.csv"), index=False)
    pd.DataFrame(plane_rows).to_csv(os.path.join(OUT, f"planes_{args.tag}.csv"), index=False)

    rows = []
    for tr in trk.all_tracks():
        for (fid, t, x, y, vx, vy, lab) in tr.history:
            rows.append(dict(tid=tr.tid, frame=fid, t=t, x=x, y=y, vx=vx, vy=vy,
                             label=lab, final_label=tr.label, hits=tr.hits,
                             confirmed=tr.confirmed))
    tdf = pd.DataFrame(rows)
    tdf.to_csv(os.path.join(OUT, f"tracks_{args.tag}.csv"), index=False)

    summary = [dict(tid=t.tid, label=t.label, votes=t.label_votes, hits=t.hits,
                    confirmed=t.confirmed, born=t.born_frame, disp=t.displacement,
                    path=t.path_length, straightness=t.straightness,
                    median_speed=t.median_speed, hit_ratio=t.hit_ratio,
                    dim_variation=t.dim_variation, dynamic=trk.is_dynamic(t),
                    history=t.history)
                   for t in trk.all_tracks()]
    with open(os.path.join(OUT, f"tracker_{args.tag}.pkl"), "wb") as fh:
        pickle.dump(summary, fh)

    s = pd.DataFrame([{k: v for k, v in d.items() if k not in ("history", "votes")}
                      for d in summary])
    s["f0"] = [d["history"][0][0] for d in summary]
    s["f1"] = [d["history"][-1][0] for d in summary]
    s["n"] = [len(d["history"]) for d in summary]
    s.to_csv(os.path.join(OUT, f"track_summary_{args.tag}.csv"), index=False)

    real = s[s.dynamic & s.confirmed].sort_values("disp", ascending=False)
    print(f"\n{len(s)} tracks total, {int(s.confirmed.sum())} confirmed, "
          f"{len(real)} confirmed and dynamic")
    print(real[["tid", "label", "f0", "f1", "n", "hits", "hit_ratio", "disp",
                "straightness", "median_speed", "dim_variation"]].head(25).round(2).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
