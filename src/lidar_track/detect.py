"""Per-frame detection: gate -> ground segmentation -> clustering -> features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cluster import ClusterFeatures, cluster_points, extract_features
from .io import Frame
from .preprocess import GroundPlane, fit_ground, segment_ground, valid_mask


@dataclass
class Detection:
    frame_id: int
    t: float
    features: ClusterFeatures
    label: str = "unknown"
    score: float = 0.0
    point_idx: np.ndarray = field(default=None, repr=False)

    @property
    def position(self) -> np.ndarray:
        """Ground-plane position (x, y) in the ground-referenced frame."""
        return np.array([self.features.cx, self.features.cy])


@dataclass
class FrameResult:
    frame_id: int
    t: float
    plane: GroundPlane
    detections: list
    n_valid: int
    n_ground: int
    n_obstacle: int


class Detector:
    def __init__(self, ground_tol: float = 0.20, max_obstacle_height: float = 4.0,
                 min_samples: int = 6, alpha: float = 4.0, min_points: int = 8,
                 plane_smoother=None):
        self.ground_tol = ground_tol
        self.max_obstacle_height = max_obstacle_height
        self.min_samples = min_samples
        self.alpha = alpha
        self.min_points = min_points
        # Off by default: every point's height above ground - the
        # ground/obstacle split, and h_min/height for every classification
        # gate - is computed from one shared per-frame plane fit, whose few
        # centimetres of RANSAC sampling noise this smooths out before it is
        # used, when a caller opts in. See PlaneSmoother in preprocess.py for
        # the full reasoning and scripts/tune_plane_smoothing.py for the
        # sweep an adopted alpha would be chosen from.
        self.plane_smoother = plane_smoother

    def __call__(self, frame: Frame) -> FrameResult:
        keep = valid_mask(frame)
        xyz = frame.xyz[keep].astype(np.float64)
        rng = frame.distance[keep].astype(np.float64)
        inten = frame.intensity[keep].astype(np.float64)

        plane = fit_ground(xyz, rng)
        if self.plane_smoother is not None:
            plane = self.plane_smoother.update(plane)
        ground, obstacle, h = segment_ground(xyz, plane, self.ground_tol, self.max_obstacle_height)

        # Work in a ground-referenced frame: z becomes height above ground.
        R = plane.rotation()
        xyz_g = xyz @ R.T
        xyz_g[:, 2] = h

        oi = np.where(obstacle)[0]
        labels = cluster_points(xyz_g[oi], rng[oi], self.min_samples, self.alpha)

        dets = []
        for c in range(labels.max() + 1 if len(labels) else 0):
            sel = oi[labels == c]
            if len(sel) < self.min_points:
                continue
            f = extract_features(xyz_g[sel], inten[sel], float(rng[sel].mean()))
            dets.append(Detection(frame_id=frame.frame_id, t=frame.t, features=f, point_idx=sel))

        return FrameResult(
            frame_id=frame.frame_id, t=frame.t, plane=plane, detections=dets,
            n_valid=int(keep.sum()), n_ground=int(ground.sum()), n_obstacle=int(obstacle.sum()),
        )
