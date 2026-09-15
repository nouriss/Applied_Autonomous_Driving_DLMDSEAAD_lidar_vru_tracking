"""Temporal range-image background model for a stationary sensor.

Used as a ground-truth labelling aid and as a reference baseline, NOT as the
primary detector: it assumes the sensor never moves, which does not hold on the
target vehicle.

A moving object always shortens the measured range in its angular cell, so the
free-space range per cell is recovered as a high percentile over time.
"""
from __future__ import annotations

import numpy as np

from .io import Frame, Sequence

AZ_MIN, AZ_MAX = -40.0, 40.0
EL_MIN, EL_MAX = -16.0, 16.0


class BackgroundMap:
    def __init__(self, bin_deg: float = 0.25, percentile: float = 85.0):
        self.bin_deg = bin_deg
        self.percentile = percentile
        self.n_az = int(round((AZ_MAX - AZ_MIN) / bin_deg))
        self.n_el = int(round((EL_MAX - EL_MIN) / bin_deg))
        self.bg: np.ndarray | None = None

    def _cells(self, frame: Frame):
        ia = ((frame.azimuth - AZ_MIN) / self.bin_deg).astype(np.int32)
        ie = ((frame.elevation - EL_MIN) / self.bin_deg).astype(np.int32)
        ok = (ia >= 0) & (ia < self.n_az) & (ie >= 0) & (ie < self.n_el)
        return ia, ie, ok

    def fit(self, seq: Sequence, stride: int = 5) -> "BackgroundMap":
        idx = range(0, len(seq), stride)
        stack = np.full((len(list(idx)), self.n_az, self.n_el), np.nan, np.float32)
        for k, i in enumerate(range(0, len(seq), stride)):
            f = seq[i]
            ia, ie, ok = self._cells(f)
            # Nearest return per cell: a cell straddling an object edge also
            # sees the ground behind it, and the farthest return would make the
            # object itself look like foreground against its own background.
            g = np.full((self.n_az, self.n_el), np.inf, np.float32)
            np.minimum.at(g, (ia[ok], ie[ok]), f.distance[ok])
            g[np.isposinf(g)] = np.nan
            stack[k] = g
        with np.errstate(all="ignore"):
            self.bg = np.nanpercentile(stack, self.percentile, axis=0).astype(np.float32)
        self.observed = np.isfinite(stack).sum(axis=0)
        return self

    def foreground(self, frame: Frame, margin: float = 0.8, min_obs: int = 5) -> np.ndarray:
        """Boolean mask of points closer than the learned free-space range."""
        if self.bg is None:
            raise RuntimeError("call fit() first")
        ia, ie, ok = self._cells(frame)
        mask = np.zeros(len(frame.distance), bool)
        b = self.bg[ia[ok], ie[ok]]
        n = self.observed[ia[ok], ie[ok]]
        mask[ok] = np.isfinite(b) & (n >= min_obs) & (frame.distance[ok] < b - margin)
        return mask
