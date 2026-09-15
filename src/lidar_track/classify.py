"""Cluster classification into traffic-participant classes.

Two interchangeable classifiers share one interface:

``RuleClassifier``   transparent dimension boxes, deterministic and auditable,
                     which is what a verification and validation tool wants.
``ForestClassifier`` random forest over the same feature vector, trained on the
                     labelled clusters, used to bound what the features support.

Class dimension boxes follow the passenger-car and vulnerable-road-user
envelopes used in the automotive literature and in the Euro NCAP VRU test
specifications, widened to absorb the fact that a LiDAR sees only the faces of
an object that point towards it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster import ClusterFeatures

CLASSES = ["pedestrian", "cyclist", "car", "large_vehicle", "clutter"]

FEATURE_NAMES = [
    "n_points", "rng", "length", "width", "height", "h_min", "h_max",
    "density", "occupancy", "linearity", "planarity", "sphericity",
    "intensity_mean", "intensity_std", "fill_ratio", "aspect",
]


def feature_vector(f: ClusterFeatures) -> np.ndarray:
    return np.array([getattr(f, k) for k in FEATURE_NAMES], float)


@dataclass
class DimBox:
    name: str
    length: tuple
    width: tuple
    height: tuple
    h_min_max: float  # lowest point must be at least this close to the ground
    fill_min: float  # minimum vertical continuity
    aspect: tuple = (0.0, 1e9)  # height / footprint


# Envelopes are hard gates, not soft preferences: a cluster either falls inside
# the physical envelope of the class or it is clutter. See scripts/tune_rules.py
# for the sweep behind each bound.
#
# Two bounds are set by measurement geometry rather than by object size. The
# cyclist length floor is 1.05 m, not the ~1.8 m of an actual bicycle, because a
# bicycle viewed end-on projects to roughly its own width; holding the floor at
# 1.30 m cost 8 percentage points of recall for no reduction in false alarms.
# The car length floor is 2.60 m, just under the shortest production passenger
# car, rather than the 2.00 m the sweep nominally preferred: going below the
# physical minimum buys 1 percentage point of recall on this recording while
# admitting clusters that cannot be cars, and the recording is too short for its
# false-alarm count to argue otherwise.
DIM_BOXES = [
    DimBox("pedestrian", length=(0.20, 1.30), width=(0.20, 1.05), height=(1.30, 2.10),
           h_min_max=0.45, fill_min=0.70, aspect=(1.15, 9.0)),
    DimBox("cyclist", length=(1.05, 2.40), width=(0.25, 1.25), height=(1.30, 2.10),
           h_min_max=0.45, fill_min=0.60, aspect=(0.70, 1.60)),
    DimBox("car", length=(2.60, 5.80), width=(1.30, 2.40), height=(1.20, 2.20),
           h_min_max=0.60, fill_min=0.55, aspect=(0.25, 0.85)),
]

# Retired from the delivered classifier, and deliberately kept here rather than
# deleted, because the reason is a finding worth preserving.


LARGE_VEHICLE_BOX = DimBox("large_vehicle", length=(5.00, 12.0), width=(2.00, 3.10),
                           height=(2.20, 4.20), h_min_max=0.60, fill_min=0.55,
                           aspect=(0.25, 0.85))

# Beyond NEAR_RANGE, a pedestrian or cyclist's true height is increasingly
# under-measured: its angular height (~true_height / range) shrinks enough
# that the scan does not reliably catch its topmost point every frame.

HEIGHT_NEAR_RANGE = 25.0  # m, unchanged below this - recall is already good here
HEIGHT_FAR_RANGE = 45.0  # m, floor reaches its minimum by here
HEIGHT_FLOOR_MIN = 0.90  # m, the lowest the floor ever goes, inside the window below
HEIGHT_MAX_RANGE = 45.0


def range_relaxed_height_floor(base_floor: float, rng: float) -> float:
    if rng <= HEIGHT_NEAR_RANGE or rng >= HEIGHT_MAX_RANGE:
        return base_floor
    k = (base_floor - HEIGHT_FLOOR_MIN) / (HEIGHT_FAR_RANGE - HEIGHT_NEAR_RANGE)
    return max(HEIGHT_FLOOR_MIN, base_floor - k * (rng - HEIGHT_NEAR_RANGE))


_RANGE_ADAPTIVE_HEIGHT = {"pedestrian", "cyclist"}

# The same "measured extent under-reads true extent at range" effect, but for a
# car's *length*, and driven by aspect rather than by angular resolution alone.
# A car approaching the sensor head-on presents its front face and only a sliver
# of flank, so its measured length is the foreshortened projection of the true
# box, not the box. The flank opens up as it closes and the bearing across it
# widens, which is why the measurement grows monotonically with proximity.
#

CAR_LENGTH_NEAR_RANGE = 25.0  # m, unchanged below this - the flank is visible here
CAR_LENGTH_FAR_RANGE = 32.0  # m, floor reaches its minimum by here
CAR_LENGTH_FLOOR_MIN = 2.30  # m, the lowest the floor ever goes, inside the window
CAR_LENGTH_MAX_RANGE = 45.0  # m, past this the floor snaps back; see above


def range_relaxed_car_length_floor(base_floor: float, rng: float) -> float:
    if rng <= CAR_LENGTH_NEAR_RANGE or rng >= CAR_LENGTH_MAX_RANGE:
        return base_floor
    k = (base_floor - CAR_LENGTH_FLOOR_MIN) / (CAR_LENGTH_FAR_RANGE - CAR_LENGTH_NEAR_RANGE)
    return max(CAR_LENGTH_FLOOR_MIN, base_floor - k * (rng - CAR_LENGTH_NEAR_RANGE))


_RANGE_ADAPTIVE_LENGTH = {"car"}


class RuleClassifier:
    """Dimension-box classifier with ground-contact and continuity gates.

    Ground contact (``h_min``) removes tree canopy and upper-facade returns;
    vertical continuity (``fill_ratio``) removes facade corners and branch
    fragments that happen to span a plausible height; the aspect gate separates
    an upright person from a car-shaped footprint of similar volume.
    """

    def __init__(self, boxes=None, min_points: int = 8, min_occupancy: float = 0.10):
        self.boxes = boxes or DIM_BOXES
        self.min_points = min_points
        self.min_occupancy = min_occupancy

    def _height_floor(self, box: DimBox, f: ClusterFeatures) -> float:
        if box.name in _RANGE_ADAPTIVE_HEIGHT:
            return range_relaxed_height_floor(box.height[0], f.rng)
        return box.height[0]

    def _length_floor(self, box: DimBox, f: ClusterFeatures) -> float:
        if box.name in _RANGE_ADAPTIVE_LENGTH:
            return range_relaxed_car_length_floor(box.length[0], f.rng)
        return box.length[0]

    def _passes(self, box: DimBox, f: ClusterFeatures) -> bool:
        return (
            self._length_floor(box, f) <= f.length <= box.length[1]
            and box.width[0] <= f.width <= box.width[1]
            and self._height_floor(box, f) <= f.height <= box.height[1]
            and f.h_min <= box.h_min_max
            and f.fill_ratio >= box.fill_min
            and box.aspect[0] <= f.aspect <= box.aspect[1]
        )

    def _margin(self, box: DimBox, f: ClusterFeatures) -> float:
        """How centrally the cluster sits inside the envelope, in [0, 1]."""
        h_lo = self._height_floor(box, f)
        l_lo = self._length_floor(box, f)
        m = 1.0
        for val, (lo, hi) in [(f.length, (l_lo, box.length[1])), (f.width, box.width),
                              (f.height, (h_lo, box.height[1]))]:
            span = max(hi - lo, 1e-6)
            m = min(m, 2.0 * min(val - lo, hi - val) / span)
        return float(np.clip(m, 0.0, 1.0))

    def fits(self, f: ClusterFeatures) -> dict:
        """Every class whose hard gates this cluster independently passes,
        each mapped to its margin (how centrally it sits in that envelope).

        ``__call__`` collapses this to a single winner, which is right for a
        detector with no other evidence to break a tie. A caller that *does*
        have other evidence - Tracker, using a track's own speed history to
        arbitrate a genuine pedestrian/cyclist tie - needs to see every class
        that passed, not just the best one, to tell a real tie (two classes
        independently admissible) from a clear verdict (one admissible class,
        arbitrating it would be overriding the shape evidence rather than
        resolving an ambiguity in it).
        """
        return {b.name: self._margin(b, f) for b in self.boxes if self._passes(b, f)}

    def __call__(self, f: ClusterFeatures) -> tuple:
        if f.n_points < self.min_points or f.occupancy < self.min_occupancy:
            return "clutter", 0.0
        fits = self.fits(f)
        if not fits:
            return "clutter", 1.0
        best = max(fits, key=fits.get)
        return best, 0.5 + 0.5 * fits[best]


class ForestClassifier:
    """Random forest over the same feature vector."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 8, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed,
            class_weight="balanced_subsample", n_jobs=-1,
        )
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForestClassifier":
        self.clf.fit(X, y)
        self.fitted = True
        return self

    def __call__(self, f: ClusterFeatures) -> tuple:
        if not self.fitted:
            raise RuntimeError("call fit() first")
        x = feature_vector(f)[None, :]
        p = self.clf.predict_proba(x)[0]
        i = int(np.argmax(p))
        return str(self.clf.classes_[i]), float(p[i])
