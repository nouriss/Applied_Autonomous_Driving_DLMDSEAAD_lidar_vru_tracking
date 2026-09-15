"""Multi-object tracking: constant-velocity Kalman filters with global nearest
neighbour association.

Design notes
------------
The recording runs at 2.435 Hz. In one frame period a pedestrian moves ~0.6 m
and a car at 50 km/h moves ~5.7 m, which is larger than a car. Association
therefore cannot rely on position overlap and uses a Mahalanobis gate driven by
the filter's own predicted covariance, so a fast, uncertain track gets a wide
gate while a slow, well-observed one stays tight.

Static scene structure (posts, hedges, facade corners) survives the shape
classifier because it genuinely has pedestrian-like dimensions. It is rejected
here instead: a track must accumulate real displacement before it is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

CHI2_GATE_2DOF = 13.82  # p = 0.999


def make_filter(x0: np.ndarray, dt: float, sigma_a: float, sigma_z: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    kf.R = np.eye(2) * sigma_z**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_a**2, block_size=2, order_by_dim=False)
    kf.Q = q
    kf.x = np.array([x0[0], x0[1], 0.0, 0.0])
    # Initial speed uncertainty sets the size of the first association gate.
    # 8 m/s made the gate wide enough to swallow neighbouring static clutter;
    # 3 m/s covers a VRU immediately and a vehicle within two updates.
    kf.P = np.diag([sigma_z**2, sigma_z**2, 3.0**2, 3.0**2])
    return kf


@dataclass
class Track:
    tid: int
    kf: KalmanFilter
    label: str
    label_votes: dict = field(default_factory=dict)
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    born_frame: int = 0
    born_t: float = 0.0
    history: list = field(default_factory=list)  # (frame_id, t, x, y, vx, vy, label)
    start_xy: np.ndarray = None
    dims: list = field(default_factory=list)  # (length, width, height) per associated detection
    # Latched once is_dynamic() first holds. is_dynamic() answers "has this
    # track proven itself a moving road user", which is a judgement about the
    # track's whole life, but one of its terms - hit_ratio - is
    # hits/len(history), and history grows on a coasted frame while hits does not. 
    ever_dynamic: bool = False

    @property
    def xy(self) -> np.ndarray:
        return self.kf.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.kf.x[2:]))

    @property
    def displacement(self) -> float:
        if self.start_xy is None:
            return 0.0
        return float(np.linalg.norm(self.kf.x[:2] - self.start_xy))

    @property
    def path_length(self) -> float:
        if len(self.history) < 2:
            return 0.0
        p = np.array([[h[2], h[3]] for h in self.history])
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    @property
    def straightness(self) -> float:
        """Net displacement over distance travelled.

        A real road user advances along a path; a jittering static cluster
        accumulates path length without going anywhere, so it scores near zero
        even when a mis-association gives it a large apparent displacement.
        """
        p = self.path_length
        return float(self.displacement / p) if p > 1e-6 else 0.0

    @property
    def median_speed(self) -> float:
        if len(self.history) < 3:
            return 0.0
        v = np.array([np.hypot(h[4], h[5]) for h in self.history])
        return float(np.median(v))

    @property
    def hit_ratio(self) -> float:
        return self.hits / max(len(self.history), 1)

    @property
    def dim_variation(self) -> float:
        """Largest coefficient of variation among the track's box dimensions.

        A rigid road user keeps its extent within measurement noise as it moves,
        while a wind-disturbed foliage cluster changes shape every frame. This
        separates the two without relying on how far the centroid appears to
        travel, which mis-association can fake.
        """
        if len(self.dims) < 3:
            return np.inf
        d = np.array(self.dims)
        return float(np.max(d.std(0) / np.maximum(d.mean(0), 1e-3)))

    def vote(self, label: str, score: float):
        self.label_votes[label] = self.label_votes.get(label, 0.0) + score
        self.label = max(self.label_votes, key=self.label_votes.get)


class Tracker:
    def __init__(self, dt: float = 0.4107, sigma_a: float = 2.5, sigma_z: float = 0.40,
                 max_gate_m: float = 5.0, confirm_hits: int = 3, max_misses: int = 3,
                 min_displacement: float = 0.5, min_speed: float = 0.5,
                 min_straightness: float = 0.55, min_hits: int = 3,
                 min_hit_ratio: float = 0.60, max_dim_variation: float = 0.35,
                 classifier=None, kinematic_speed_mid: float = 2.0,
                 kinematic_speed_scale: float = 0.5, kinematic_weight: float = 0.5,
                 kinematic_min_history: int = 3):
        self.dt = dt
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        self.max_gate_m = max_gate_m
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.min_displacement = min_displacement
        self.min_speed = min_speed
        self.min_straightness = min_straightness
        self.min_hits = min_hits
        self.min_hit_ratio = min_hit_ratio
        self.max_dim_variation = max_dim_variation
        # See _kinematic_relabel() for what these do and why they are
        # deliberately inert (classifier=None) unless a caller opts in.
        self.classifier = classifier
        self.kinematic_speed_mid = kinematic_speed_mid
        self.kinematic_speed_scale = kinematic_speed_scale
        self.kinematic_weight = kinematic_weight
        self.kinematic_min_history = kinematic_min_history
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 0

    def _kinematic_relabel(self, tr: Track, d) -> None:
        """Resolve a pedestrian/cyclist tie using the track's own prior speed,
        for the one detection just matched to this track this frame.

        The shape classifier is deliberately weak to avoid false positives
        """
        if self.classifier is None or d.label not in ("pedestrian", "cyclist"):
            return
        if len(tr.history) < self.kinematic_min_history:
            return
        fits = self.classifier.fits(d.features)
        if "pedestrian" not in fits or "cyclist" not in fits:
            return  # not a genuine tie - only one class independently passed
        v = tr.median_speed
        bias = self.kinematic_weight * float(np.clip(
            (v - self.kinematic_speed_mid) / self.kinematic_speed_scale, -1.0, 1.0))
        # bias > 0 (track has been moving fast) favours cyclist; < 0 favours
        # pedestrian. Added to each class's own shape margin, so a strong
        # shape margin still wins against a weak kinematic push either way.
        cyc_score, ped_score = fits["cyclist"] + bias, fits["pedestrian"] - bias
        if cyc_score > ped_score:
            d.label, d.score = "cyclist", 0.5 + 0.5 * np.clip(fits["cyclist"], 0.0, 1.0)
        else:
            d.label, d.score = "pedestrian", 0.5 + 0.5 * np.clip(fits["pedestrian"], 0.0, 1.0)

    def _gate_cost(self, track: Track, z: np.ndarray) -> float:
        S = track.kf.H @ track.kf.P @ track.kf.H.T + track.kf.R
        y = z - track.kf.H @ track.kf.x
        if np.linalg.norm(y) > self.max_gate_m:
            return np.inf
        d2 = float(y @ np.linalg.solve(S, y))
        return d2 if d2 <= CHI2_GATE_2DOF else np.inf

    def update(self, detections, frame_id: int, t: float):
        for tr in self.tracks:
            tr.kf.predict()
            tr.age += 1

        obs = [d for d in detections if d.label != "clutter"]
        cost = np.full((len(self.tracks), len(obs)), np.inf)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(obs):
                cost[i, j] = self._gate_cost(tr, d.position)

        matched_t, matched_o = set(), set()
        if cost.size and np.isfinite(cost).any():
            big = 1e6
            c = np.where(np.isfinite(cost), cost, big)
            ri, ci = linear_sum_assignment(c)
            for r, cc in zip(ri, ci):
                if np.isfinite(cost[r, cc]):
                    tr, d = self.tracks[r], obs[cc]
                    tr.kf.update(d.position)
                    tr.hits += 1
                    tr.misses = 0
                    self._kinematic_relabel(tr, d)
                    tr.vote(d.label, max(d.score, 1e-3))
                    tr.dims.append((d.features.length, d.features.width, d.features.height))
                    matched_t.add(r)
                    matched_o.add(cc)

        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.misses += 1
            if tr.hits >= self.confirm_hits:
                tr.confirmed = True
            tr.history.append((frame_id, t, *tr.kf.x[:2], *tr.kf.x[2:], tr.label))
            # Latch on the frame the evidence first supports it; see
            # Track.ever_dynamic for why this must not be re-evaluated away.
            if not tr.ever_dynamic and self.is_dynamic(tr):
                tr.ever_dynamic = True

        for j, d in enumerate(obs):
            if j in matched_o:
                continue
            tr = Track(tid=self._next_id,
                       kf=make_filter(d.position, self.dt, self.sigma_a, self.sigma_z),
                       label=d.label, born_frame=frame_id, born_t=t,
                       start_xy=d.position.copy())
            tr.vote(d.label, max(d.score, 1e-3))
            tr.dims.append((d.features.length, d.features.width, d.features.height))
            tr.history.append((frame_id, t, *tr.kf.x[:2], 0.0, 0.0, tr.label))
            self.tracks.append(tr)
            self._next_id += 1

        alive = []
        for tr in self.tracks:
            if tr.misses > self.max_misses:
                self.finished.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        return self.reported()

    def is_dynamic(self, t: Track) -> bool:
        """Evidence that a track is a moving road user rather than scene clutter.

        Static structure with road-user geometry (posts, wall corners) passes
        the shape classifier, so it is rejected here. Displacement alone is not
        enough, because a single mis-association hands a static cluster a large
        apparent jump

        """
        return (t.hits >= self.min_hits
                and t.hit_ratio >= self.min_hit_ratio
                and t.displacement >= self.min_displacement
                and t.straightness >= self.min_straightness
                and t.median_speed >= self.min_speed
                and t.dim_variation <= self.max_dim_variation)

    def reported(self) -> list:
        """Confirmed tracks currently observed and judged dynamic."""
        return [t for t in self.tracks
                if t.confirmed and t.misses == 0 and self.is_dynamic(t)]

    def all_tracks(self) -> list:
        return self.finished + self.tracks
