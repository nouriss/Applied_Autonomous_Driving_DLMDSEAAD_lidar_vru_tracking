# LiDAR detection and tracking for vehicles and pedestrians

Case study for ** Localisation, Motion Planning and Sensor Fusion**,

Detection, classification and tracking of vehicles, cyclists and pedestrians in
point clouds from a Blickfeld Cube 1 solid-state LiDAR. The algorithm is built
for verification and validation, not for real-time deployment.

## Result

Over the full 718-frame (295 s) recording the pipeline reports exactly the four
moving road users present, with correct classes and no false alarms.

| Measure | Result |
| --- | --- |
| Reported dynamic tracks | 4 (cyclist, car, 2 × pedestrian) |
| False-alarm tracks | 0 |
| MOTA / MOTP | 0.817 / 0.080 m |
| IDF1 | 0.910 |
| Identity switches / fragmentations | 0 / 0 |
| Mostly tracked | 4 of 4 |
| Detection recall (object-frames) | 0.817 |
| Classification rate, per frame / per track | 0.926 / 4 of 4 |



## Approach

The sensor is stationary in this recording, so background subtraction would
isolate moving objects almost trivially. It is used **only** to build ground
truth, never in the delivered detector, because it cannot run on a moving
vehicle. The detector is ego-motion agnostic throughout:

```
frame -> range/FOV gate (datasheet limits)
      -> RANSAC ground plane, re-estimated every frame
      -> rotate into a ground-referenced frame (z = height above road)
      -> split ground from obstacles by height
      -> range-adaptive DBSCAN (radius scales with return spacing)
      -> cluster features (extent, ground contact, vertical continuity, shape)
      -> hard-gated class envelopes -> {pedestrian, cyclist, car, clutter}
      -> Kalman filter + Hungarian association over a Mahalanobis gate
      -> kinematic pedestrian/cyclist tie-break (track speed, ties only)
      -> dynamic-track test -> reported road users
```

The scene contains a post with pedestrian dimensions and a wall with truck
dimensions, both present in over 95 % of frames. Shape alone cannot reject them,
so rejection happens at track level: a track is reported only if its motion was
observed rather than extrapolated, is straight rather than jittering, and the
object kept its shape while moving.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Place the recording under `00_lidar_data/` (four `*_part_N` folders of CSV
frames), then:

```bash
python scripts/build_cache.py        # CSV -> cache/frames.npz  (~2 min)
python scripts/run_pipeline.py       # detection + tracking     (~3.5 min)
python scripts/export_mcap.py        # replayable recording

```

## Viewing the result

`scripts/export_mcap.py` writes an MCAP recording for
[Lichtblick Suite](https://github.com/lichtblick-suite). Open
`outputs/lidar_full.mcap`, add a 3D panel, and enable:

| Topic | Contents |
| --- | --- |
| `/tf` | sensor-to-ground transform (the same temporally-smoothed plane the detector itself uses) |
| `/lidar/points` | raw cloud, unprocessed, native intensity channel |
| `/lidar/ground` | ground-plane points only |
| `/objects/moving` | point cloud of every confirmed, moving road user, by class |
| `/objects/moving_boxes` | one oriented box + label per moving detection, same set as above |
| `/objects/static` | point cloud of everything else obstacle-side, by class where classified |
| `/tracks` | confirmed tracks: id, label, speed, trajectory trail |

`outputs/lidar_car_cyclist.mcap` is a 51-frame extract of the car/cyclist
crossing, which is the most informative 20 s of the recording.

## Layout

```
src/lidar_track/     reusable package (io, preprocess, cluster, classify,
                     track, background, mcap_export)
scripts/             reproduction entry points, in the order listed above
outputs/             results, figures and exported recordings
cache/               frames.npz (generated, not committed)
00_lidar_data/       input recording (not committed)
```

## Note on the data

The recording is 718 contiguous frames at 2.435 Hz, measured, against the 1–30 Hz
the datasheet permits. At that rate a pedestrian advances ~0.6 m and a car at
50 km/h ~5.7 m between frames, which is longer than the car; this drives the
association design. The sensor sits 3.19 m above the road pitched 6.3° down, and
the 30° vertical field of view leaves everything below 1.35 m invisible at 5 m
range. One of the two pedestrians is lost to exactly this blind zone.
