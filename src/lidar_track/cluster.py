"""Range-adaptive Euclidean clustering and per-cluster feature extraction.

A fixed DBSCAN radius cannot serve the whole working range: with the 0.4 deg
horizontal scan pattern the spacing between neighbouring returns grows from
about 3 cm at 5 m to 28 cm at 40 m, so a radius tight enough to separate a
pedestrian from a parked car nearby will shatter the same pedestrian at 30 m.
Points are therefore clustered in overlapping range bands, each with its own
radius, and clusters sharing points in the overlap are merged.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.cluster import DBSCAN

ANG_RES_RAD = np.radians(0.4)  # datasheet horizontal scan pattern


def adaptive_eps(r: np.ndarray | float, alpha: float = 4.0,
                 eps_min: float = 0.25, eps_max: float = 1.10) -> np.ndarray | float:
    """Neighbourhood radius as a multiple of the local inter-return spacing."""
    return np.clip(alpha * np.asarray(r) * ANG_RES_RAD, eps_min, eps_max)


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_points(xyz: np.ndarray, rng: np.ndarray, min_samples: int = 6,
                   alpha: float = 4.0, bands=(5, 15, 25, 40, 60, 260),
                   overlap: float = 3.0) -> np.ndarray:
    """DBSCAN per range band with union-find merging across band overlaps.

    Returns a label per point, -1 for noise.
    """
    labels = np.full(len(xyz), -1, np.int64)
    if len(xyz) == 0:
        return labels

    next_label = 0
    band_labels = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        sel = np.where((rng >= lo - overlap) & (rng < hi + overlap))[0]
        if len(sel) < min_samples:
            continue
        eps = float(adaptive_eps(0.5 * (lo + hi), alpha))
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz[sel])
        out = np.full(len(sel), -1, np.int64)
        valid = lab >= 0
        out[valid] = lab[valid] + next_label
        next_label += lab.max() + 1 if valid.any() else 0
        band_labels.append((sel, out))

    if not band_labels:
        return labels

    # A point seen by two bands ties their labels together.
    uf = _UnionFind(next_label)
    seen: dict[int, int] = {}
    for sel, out in band_labels:
        for idx, lb in zip(sel, out):
            if lb < 0:
                continue
            if idx in seen:
                uf.union(seen[idx], lb)
            else:
                seen[idx] = lb

    remap: dict[int, int] = {}
    for idx, lb in seen.items():
        root = uf.find(lb)
        if root not in remap:
            remap[root] = len(remap)
        labels[idx] = remap[root]
    return labels


@dataclass
class ClusterFeatures:
    n_points: int
    cx: float
    cy: float
    cz: float
    rng: float
    length: float  # oriented extent, long axis in ground plane
    width: float  # oriented extent, short axis
    height: float  # extent above ground
    h_min: float  # height of lowest point above ground
    h_max: float
    yaw: float  # long-axis heading, rad
    density: float  # points per m^3 of oriented box
    occupancy: float  # observed points / points expected at this range
    linearity: float
    planarity: float
    sphericity: float
    intensity_mean: float
    intensity_std: float
    fill_ratio: float  # fraction of 0.2 m height slices that contain points
    aspect: float  # height / max(length, width)


def extract_features(xyz_g: np.ndarray, intensity: np.ndarray, rng: float) -> ClusterFeatures:
    """Features in the ground-referenced frame (z = height above ground)."""
    c = xyz_g.mean(0)
    xy = xyz_g[:, :2] - c[:2]

    # Oriented extents from the XY principal axis.
    if len(xy) >= 3:
        cov = np.cov(xy.T)
        w, v = np.linalg.eigh(cov)
        axis = v[:, np.argmax(w)]
    else:
        axis = np.array([1.0, 0.0])
    yaw = float(np.arctan2(axis[1], axis[0]))
    R = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    proj = xy @ R.T
    length = float(np.ptp(proj[:, 0]))
    width = float(np.ptp(proj[:, 1]))

    h_min, h_max = float(xyz_g[:, 2].min()), float(xyz_g[:, 2].max())
    height = h_max - h_min

    vol = max(length * width * height, 1e-3)
    # Angular footprint a solid object of this size subtends at this range.
    expected = max((length * height) / (rng * ANG_RES_RAD) ** 2, 1.0)

    if len(xyz_g) >= 3:
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyz_g.T)))[::-1]
        ev = np.maximum(ev, 1e-12)
        s = ev.sum()
        lin, pla, sph = float((ev[0] - ev[1]) / s), float((ev[1] - ev[2]) / s), float(ev[2] / s)
    else:
        lin = pla = sph = 0.0

    # Vertical continuity: a standing person or a vehicle body fills the height
    # slices it spans, whereas a facade corner or a branch leaves gaps.
    if height > 0.2:
        edges = np.arange(h_min, h_max + 0.2, 0.2)
        occupied = np.histogram(xyz_g[:, 2], bins=edges)[0] > 0
        fill = float(occupied.mean()) if len(occupied) else 1.0
    else:
        fill = 1.0

    return ClusterFeatures(
        n_points=len(xyz_g), cx=float(c[0]), cy=float(c[1]), cz=float(c[2]), rng=float(rng),
        length=length, width=width, height=height, h_min=h_min, h_max=h_max, yaw=yaw,
        density=len(xyz_g) / vol, occupancy=len(xyz_g) / expected,
        linearity=lin, planarity=pla, sphericity=sph,
        intensity_mean=float(intensity.mean()), intensity_std=float(intensity.std()),
        fill_ratio=fill, aspect=height / max(length, width, 1e-3),
    )


def features_to_dict(f: ClusterFeatures) -> dict:
    return asdict(f)
