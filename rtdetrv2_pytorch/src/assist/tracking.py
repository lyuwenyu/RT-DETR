"""Lightweight detection tracking across frames.

Greedy IoU association with a range gate -- no Kalman filter, which would add
tuning surface without buying anything the guidance layer actually consumes.

Tracking earns its place here for three reasons, none of them cosmetic:

  * Flicker suppression. A detector that drops an object for one frame would
    otherwise produce a spurious "gone" and then re-announce it.
  * Closing speed. Range differenced over time gives time-to-contact, which is
    what separates "a parked car" from "a car coming at you". Proximity alone
    cannot make that distinction, and it is the distinction that matters.
  * Announcement memory. A track remembers that it has been spoken about, so a
    bench the user is walking past is mentioned once rather than every frame.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Track:
    id: int
    name: str
    box: tuple
    score: float
    range_m: float
    bearing: float
    hits: int = 1
    misses: int = 0
    first_time: float = 0.0
    last_time: float = 0.0
    range_rate: float = 0.0          # metres/second, negative = approaching
    announced_zone: Optional[str] = None
    last_announced: float = -1e9

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    @property
    def closing_speed(self) -> float:
        """Positive when the object is getting nearer."""
        return max(0.0, -self.range_rate)

    def time_to_contact(self) -> float:
        """Seconds until contact at the current closing speed."""
        speed = self.closing_speed
        if speed < 0.05:
            return float('inf')
        return self.range_m / speed


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self._next_id = 1

    def update(self, observations, now):
        """observations: iterable of (Detection, range_m, bearing)."""
        dt_default = 1.0 / 15.0
        matched_tracks, matched_obs = self._associate(observations)

        for ti, oi in zip(matched_tracks, matched_obs):
            det, rng, bearing = observations[oi]
            trk = self.tracks[ti]
            dt = max(now - trk.last_time, 1e-3) if trk.last_time else dt_default

            new_range = self._ema(trk.range_m, rng)
            rate = (new_range - trk.range_m) / dt
            trk.range_rate = self._ema(trk.range_rate, rate)

            trk.box, trk.score, trk.name = det.box, det.score, det.name
            trk.range_m, trk.bearing = new_range, bearing
            trk.hits += 1
            trk.misses = 0
            trk.last_time = now

        for i, trk in enumerate(self.tracks):
            if i not in matched_tracks:
                trk.misses += 1

        for oi, (det, rng, bearing) in enumerate(observations):
            if oi not in matched_obs:
                self.tracks.append(Track(id=self._next_id, name=det.name,
                                         box=det.box, score=det.score,
                                         range_m=rng, bearing=bearing,
                                         first_time=now, last_time=now))
                self._next_id += 1

        self.tracks = [t for t in self.tracks
                       if t.misses <= self.cfg.track_max_misses]
        return self.tracks

    def _ema(self, old, new):
        a = self.cfg.range_ema_alpha
        if not np.isfinite(old):
            return new
        return (1.0 - a) * old + a * new

    def _associate(self, observations):
        matched_t, matched_o = [], []
        if not self.tracks or not observations:
            return matched_t, matched_o

        pairs = []
        for ti, trk in enumerate(self.tracks):
            for oi, (det, rng, _) in enumerate(observations):
                if det.name != trk.name:
                    continue
                iou = _iou(trk.box, det.box)
                if iou < self.cfg.track_iou_threshold:
                    continue
                # Guard against two same-class objects at very different
                # depths whose boxes happen to overlap.
                if np.isfinite(trk.range_m) and np.isfinite(rng) \
                        and abs(trk.range_m - rng) > max(1.5, 0.5 * trk.range_m):
                    continue
                pairs.append((iou, ti, oi))

        for _, ti, oi in sorted(pairs, reverse=True):
            if ti in matched_t or oi in matched_o:
                continue
            matched_t.append(ti)
            matched_o.append(oi)
        return matched_t, matched_o


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
