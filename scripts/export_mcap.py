"""Write an MCAP recording of the pipeline for replay in Lichtblick Suite.

    python scripts/export_mcap.py --start 2190 --stop 2240 --tag car_cyclist
    python scripts/export_mcap.py --tag full          # whole recording

Open the resulting .mcap in Lichtblick and add a 3D panel; every topic is
published in the `ground` frame.

Topics
------
/tf              sensor -> ground transform (smoothed for display; see below)
/lidar/points    the raw cloud, unprocessed: no gating, no ground referencing,
                 published in the sensor frame with its native intensity
                 channel. Always included, by standing request - a reference
                 view of exactly what the sensor delivered, kept even when
                 every other topic is trimmed for a simpler viewing session.
/lidar/ground    ground-plane points only
/objects/moving        point cloud of every point belonging to a confirmed,
                       moving road user this frame (a track reported by
                       Tracker.is_dynamic in track.py), coloured by class
/objects/moving_boxes  one oriented CubePrimitive + text label per moving
                       detection, same set as /objects/moving above, so a box
                       and its points always describe the same object. A
                       track that misses a detection this frame (occlusion,
                       a thin/distant cluster) still draws - at reduced
                       opacity, from its last known size and the Kalman
                       filter's predicted position - rather than vanishing;
                       see moving_scene() in mcap_export.py.
/objects/static        point cloud of every other obstacle point - parked or
                       misclassified vehicles, buildings, vegetation,
                       stationary pedestrians, unclustered noise - coloured by
                       class where a cluster was classified, dim grey otherwise
/tracks                confirmed dynamic tracks: id, label, speed, trajectory
                       trail (SceneUpdate: LinePrimitive + TextPrimitive).
                       Always included, by standing request, same as
                       /lidar/points.

"""
import argparse
import os
import sys

import numpy as np
from mcap_protobuf.writer import Writer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from lidar_track.classify import RuleClassifier  # noqa: E402
from lidar_track.detect import Detector  # noqa: E402
from lidar_track.io import Sequence  # noqa: E402
from lidar_track.mcap_export import (  # noqa: E402
    class_colored_points, ground_transform, moving_scene,
    point_cloud_xyzi, point_cloud_xyzrgb, track_scene,
)
from lidar_track.preprocess import PlaneSmoother, segment_ground, valid_mask  # noqa: E402
from lidar_track.track import Tracker  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

MATCH_DIST = 2.0  # m, same association radius used throughout evaluation
GROUND_RGB = np.array([60, 60, 68], np.uint8)
# Comfortably above the tallest point observed in this recording (~14.7 m
# above ground); see the comment where this is used, below.
VIS_MAX_HEIGHT = 50.0


def shaded(base_rgb, z, z_lo=0.0, z_hi=4.0):
    """Base colour brightened with height, purely so the ground has depth."""
    k = np.clip((z - z_lo) / max(z_hi - z_lo, 1e-6), 0.0, 1.0)
    out = base_rgb[None, :].astype(np.float32) * (0.55 + 0.45 * k[:, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=None, help="first frame id")
    ap.add_argument("--stop", type=int, default=None, help="last frame id")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    seq = Sequence()
    lo = 0 if args.start is None else int(np.where(seq.frame_ids == args.start)[0][0])
    hi = len(seq) if args.stop is None else int(np.where(seq.frame_ids == args.stop)[0][0]) + 1

 
    det = Detector(plane_smoother=PlaneSmoother(alpha=0.30))
    clf = RuleClassifier()
    trk = Tracker(dt=seq.dt, classifier=clf, kinematic_speed_mid=1.6,
                 kinematic_speed_scale=0.8, kinematic_weight=0.3)
    path = os.path.join(OUT, f"lidar_{args.tag}.mcap")

    with open(path, "wb") as fh, Writer(fh) as w:
        for i in range(lo, hi):
            f = seq[i]
            r = det(f)
            for d in r.detections:
                d.label, d.score = clf(d.features)
            trk.update(r.detections, r.frame_id, r.t)
         
            displayed = [tr for tr in trk.tracks if tr.confirmed and tr.ever_dynamic]
            live = [tr for tr in displayed if tr.misses == 0]  # matched this frame
            coasting = [tr for tr in displayed if tr.misses > 0]  # on prediction alone
            t = f.t
            log_t = int(t * 1e9)

           
            w.write_message("/tf", ground_transform(r.plane, t), log_time=log_t, publish_time=log_t)
            w.write_message("/lidar/points", point_cloud_xyzi(f.xyz, f.intensity, t, "sensor"),
                            log_time=log_t, publish_time=log_t)

            keep = valid_mask(f)
            xyz = f.xyz[keep].astype(np.float64)
          
            ground, obstacle, h = segment_ground(xyz, r.plane, max_obstacle_height=VIS_MAX_HEIGHT)
            R = r.plane.rotation()
            xyz_g = xyz @ R.T
            xyz_g[:, 2] = h

            w.write_message("/lidar/ground",
                            point_cloud_xyzrgb(xyz_g[ground], shaded(GROUND_RGB, xyz_g[ground, 2]),
                                               t, "ground"),
                            log_time=log_t, publish_time=log_t)

          
            class_rgb = class_colored_points(len(xyz_g), r.detections)
            moving_mask = np.zeros(len(xyz_g), bool)
            moving_dets = []
            for d in r.detections:
                if d.label == "clutter":
                    continue
               
                nearest, best_d = None, MATCH_DIST
                for lt in live:
                    dd = np.hypot(lt.xy[0] - d.features.cx, lt.xy[1] - d.features.cy)
                    if dd <= best_d:
                        nearest, best_d = lt, dd
                if nearest is not None:
                    moving_mask[d.point_idx] = True
                    moving_dets.append((d, nearest.label))
            static_mask = obstacle & ~moving_mask

            w.write_message("/objects/moving",
                            point_cloud_xyzrgb(xyz_g[moving_mask], class_rgb[moving_mask], t, "ground"),
                            log_time=log_t, publish_time=log_t)
            w.write_message("/objects/moving_boxes", moving_scene(moving_dets, coasting, t, "ground"),
                            log_time=log_t, publish_time=log_t)
            w.write_message("/objects/static",
                            point_cloud_xyzrgb(xyz_g[static_mask], class_rgb[static_mask], t, "ground"),
                            log_time=log_t, publish_time=log_t)
            w.write_message("/tracks", track_scene(live + coasting, t, "ground"),
                            log_time=log_t, publish_time=log_t)

            if (i - lo) % 25 == 0:
                print(f"  {i - lo}/{hi - lo}  frame {r.frame_id}  {len(live)} live tracks  "
                      f"{len(coasting)} coasting  {len(moving_dets)} moving objects  "
                      f"{int(moving_mask.sum())} moving pts  "
                      f"{int(static_mask.sum())} static pts", flush=True)

    print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print("Open in Lichtblick -> 3D panel -> /lidar/points, /lidar/ground, "
          "/objects/moving, /objects/moving_boxes, /objects/static, /tracks")


if __name__ == "__main__":
    sys.exit(main())
