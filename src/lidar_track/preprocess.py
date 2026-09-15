"""Gating, ground-plane estimation and ground-referenced transformation.

The ground plane is re-estimated per frame rather than calibrated once. On this
recording the sensor never moves, so a fixed plane would do; estimating it per
frame is what the same code needs on the moving vehicle, and the frame-to-frame
spread of the estimate is itself a useful stability metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

# Blickfeld Cube 1 datasheet limits, used as the validity gate.
RANGE_MIN, RANGE_MAX = 5.0, 250.0
AZ_LIMIT, EL_LIMIT = 35.0, 15.0


@dataclass
class GroundPlane:
    normal: np.ndarray  # unit normal, +Z half-space
    height: float  # sensor height above the plane, m
    n_inliers: int
    n_points: int

    @property
    def pitch_deg(self) -> float:
        return float(np.degrees(np.arcsin(-self.normal[1])))

    @property
    def roll_deg(self) -> float:
        return float(np.degrees(np.arcsin(self.normal[0])))

    def height_above(self, xyz: np.ndarray) -> np.ndarray:
        """Signed height of each point above the plane."""
        return xyz @ self.normal + self.height

    def rotation(self) -> np.ndarray:
        """Rotation taking the plane normal onto +Z."""
        n, z = self.normal, np.array([0.0, 0.0, 1.0])
        v = np.cross(n, z)
        s, c = np.linalg.norm(v), float(n @ z)
        if s < 1e-9:
            return np.eye(3)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def valid_mask(frame) -> np.ndarray:
    """Points inside the datasheet range and field-of-view limits."""
    return (
        (frame.distance >= RANGE_MIN)
        & (frame.distance <= RANGE_MAX)
        & (np.abs(frame.azimuth) <= AZ_LIMIT)
        & (np.abs(frame.elevation) <= EL_LIMIT)
    )


def fit_ground(xyz: np.ndarray, distance: np.ndarray, max_range: float = 40.0,
               thresh: float = 0.12, iters: int = 1000, seed: int = 0) -> GroundPlane:
    """RANSAC plane fit, restricted to the lower part of the near field.

    Open3D's segment_plane has no seed argument of its own; it draws its
    random 3-point samples from a process-global generator, so two identical
    calls on the exact same input can return different planes purely from
    RANSAC sampling noise (measured: up to 6 cm in height, 0.09 deg in pitch
    on a single, unchanged frame). Reseeding that global generator before
    every call makes the fit a deterministic function of its input, which a
    verification and validation tool needs to be.
    """
    o3d.utility.random.seed(seed)
    band = (distance < max_range) & (xyz[:, 2] < np.percentile(xyz[:, 2], 60))
    if band.sum() < 100:
        band = distance < max_range
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[band].astype(np.float64)))
    model, inliers = pc.segment_plane(distance_threshold=thresh, ransac_n=3,
                                      num_iterations=iters, probability=0.9999)
    n = np.array(model[:3], float)
    d = float(model[3])
    k = np.linalg.norm(n)
    n, d = n / k, d / k
    if n[2] < 0:  # keep the normal pointing up
        n, d = -n, -d
    return GroundPlane(normal=n, height=d, n_inliers=len(inliers), n_points=int(band.sum()))


def segment_ground(xyz: np.ndarray, plane: GroundPlane, tol: float = 0.20,
                   max_obstacle_height: float = 4.0):
    """Split points into (ground, obstacle) by height above the fitted plane.

    The upper bound removes tree canopy and building facades above any traffic
    participant, which is where most of this scene's clutter lives.
    """
    h = plane.height_above(xyz)
    ground = h < tol
    obstacle = (h >= tol) & (h <= max_obstacle_height)
    return ground, obstacle, h


class PlaneSmoother:
    """Exponential moving average on the ground-plane estimate.
    """

    def __init__(self, alpha: float = 0.15):
        """alpha: weight on the new estimate. Lower = smoother but slower to
        follow a genuine change, e.g. the plane tilting as a real vehicle
        pitches under braking. 0.15 averages over roughly a 2 s window at
        this recording's 2.4 Hz frame rate, comfortably longer than the
        millimetre-scale range noise but short enough to track real motion.
        """
        self.alpha = alpha
        self._normal = np.zeros(3)
        self._height = 0.0
        self._seeded = False

    def update(self, plane: GroundPlane) -> GroundPlane:
        if not self._seeded:
            self._normal = plane.normal.copy()
            self._height = plane.height
            self._seeded = True
        else:
            a = self.alpha
            self._normal = a * plane.normal + (1 - a) * self._normal
            self._normal /= np.linalg.norm(self._normal)
            self._height = a * plane.height + (1 - a) * self._height
        return GroundPlane(normal=self._normal.copy(), height=self._height,
                           n_inliers=plane.n_inliers, n_points=plane.n_points)
