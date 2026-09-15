"""Frame access for the Blickfeld Cube 1 recording.

Sensor frame convention measured from the data: +Y forward (boresight),
+X right, +Z up, sensor at the origin.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "frames.npz")


@dataclass
class Frame:
    index: int
    frame_id: int
    t: float  # seconds, relative to the first frame
    xyz: np.ndarray  # (N, 3) float32
    distance: np.ndarray  # (N,) float32, metres
    intensity: np.ndarray  # (N,) float32
    ambient: np.ndarray  # (N,) float32

    @property
    def azimuth(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.xyz[:, 0], self.xyz[:, 1]))

    @property
    def elevation(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.xyz[:, 2], np.hypot(self.xyz[:, 0], self.xyz[:, 1])))


class Sequence:
    """Random access to the cached recording."""

    def __init__(self, path: str = CACHE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run scripts/build_cache.py first")
        z = np.load(path)
        self._xyz = z["xyz"]
        self._dia = z["dia"]
        self._off = z["offsets"]
        self.frame_ids = z["frame_ids"]
        self._t = z["t_start"]
        self.t = (self._t - self._t[0]) / 1e9

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, i: int) -> Frame:
        a, b = self._off[i], self._off[i + 1]
        dia = self._dia[a:b]
        return Frame(
            index=i,
            frame_id=int(self.frame_ids[i]),
            t=float(self.t[i]),
            xyz=self._xyz[a:b],
            distance=dia[:, 0],
            intensity=dia[:, 1],
            ambient=dia[:, 2],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def dt(self) -> float:
        """Nominal frame period in seconds."""
        return float(np.median(np.diff(self.t)))
