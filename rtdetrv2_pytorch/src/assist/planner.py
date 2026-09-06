"""Free-space corridor search -- the part that answers "where can I walk?".

Rays are cast from the user outward across the forward arc. Each returns the
distance to the first blocked cell of the dilated occupancy grid, which is the
free walking distance along that bearing. The chosen heading maximises free
distance while preferring to keep going straight.

The hysteresis in `Corridor.choose` is not a refinement, it is what makes the
output usable. Two openings of near-equal merit -- a doorway either side of a
pillar, a corridor branching -- will trade places frame to frame under any
pure argmax, and a guidance system that alternates "bear left"/"bear right"
several times a second is worse than no guidance at all. A challenger must
therefore beat the incumbent heading by a margin, and hold that advantage for
several consecutive frames, before the system commits to it.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Corridor:
    bearings: np.ndarray        # radians, negative = left
    free: np.ndarray            # metres of clear walking per bearing
    blocked: np.ndarray         # bool: a real obstacle stopped this ray
    blocked_by_dropoff: np.ndarray   # bool per bearing, this frame only
    dropoff_confirmed: np.ndarray    # ...and seen for enough consecutive frames
    heading_index: int
    heading_bearing: float
    heading_free: float
    is_clear: bool


class CorridorPlanner:
    def __init__(self, cfg, hfov_deg=None):
        self.cfg = cfg
        # Never plan into bearings the camera cannot see. Beyond the field of
        # view nothing is ever rasterised, so those rays would read as
        # invitingly empty and could steer the user into unobserved space.
        span_deg = cfg.ray_span_deg
        if hfov_deg is not None:
            span_deg = min(span_deg, hfov_deg * 0.92)
        self.span_deg = span_deg
        span = math.radians(span_deg)
        self.bearings = np.linspace(-span / 2.0, span / 2.0, cfg.n_rays)
        self._heading_index = cfg.n_rays // 2
        self._challenger = None
        self._challenger_frames = 0
        self._drop_streak = np.zeros(cfg.n_rays, dtype=np.int32)

    @property
    def straight_index(self):
        return self.cfg.n_rays // 2

    def plan(self, grid):
        free, by_drop, blocked = self._cast(grid)

        # Debounce the drop-off layer. Planning stays cautious about a
        # single-frame reading -- the ray is shortened either way -- but
        # *announcing* one demands several consecutive frames of evidence.
        # "Stop" is the most alarming thing this system can say, and it was
        # the only warning with no temporal confirmation behind it: one noisy
        # depth frame could shout at the user to halt on open ground. Crying
        # wolf is precisely how someone learns to ignore a warning that will
        # one day be real.
        self._drop_streak = np.where(by_drop, self._drop_streak + 1, 0)
        confirmed = self._drop_streak >= self.cfg.dropoff_confirm_frames

        idx = self._choose(free)
        return Corridor(
            bearings=self.bearings,
            free=free,
            blocked=blocked,
            blocked_by_dropoff=by_drop,
            dropoff_confirmed=confirmed,
            heading_index=idx,
            heading_bearing=float(self.bearings[idx]),
            heading_free=float(free[idx]),
            is_clear=bool(free[self.straight_index] >= self.cfg.clear_distance),
        )

    def _cast(self, grid):
        """March every ray at once until the dilated grid blocks it.

        Vectorised over rays and steps together. The scalar version of this
        cost 25-80 ms per frame -- more than the detector and the depth network
        combined -- because it performed a thousand individual grid lookups in
        Python. Same result, roughly two orders of magnitude cheaper.
        """
        cfg = self.cfg
        step = cfg.grid_cell * 0.5
        n_steps = int(cfg.grid_depth / step)

        r = (np.arange(1, n_steps + 1) * step)[None, :]           # (1, S)
        sin_t = np.sin(self.bearings)[:, None]                    # (R, 1)
        cos_t = np.cos(self.bearings)[:, None]
        x, z = r * sin_t, r * cos_t                               # (R, S)

        inside = (z < cfg.grid_depth) & (np.abs(x) <= cfg.grid_width / 2.0)
        ix, iz = grid.to_cell(x, z)
        in_grid = inside & (ix >= 0) & (ix < grid.n_x)             & (iz >= 0) & (iz < grid.n_z)

        obs = np.zeros_like(in_grid)
        drop = np.zeros_like(in_grid)
        gi, gj = iz[in_grid], ix[in_grid]
        obs[in_grid] = grid.free_obstacle[gi, gj].astype(bool)
        drop[in_grid] = grid.free_dropoff[gi, gj].astype(bool)

        hit = obs | drop
        radii = np.broadcast_to(r, x.shape)

        # First blocked step, and first step that leaves the mapped area. The
        # earlier of the two ends the ray; only the former is a real hazard.
        any_hit = hit.any(axis=1)
        first_hit = np.argmax(hit, axis=1)
        any_out = (~inside).any(axis=1)
        first_out = np.argmax(~inside, axis=1)

        idx_hit = np.where(any_hit, first_hit, n_steps - 1)
        idx_out = np.where(any_out, first_out, n_steps - 1)

        blocked = any_hit & (~any_out | (first_hit <= first_out))
        end_idx = np.where(blocked, idx_hit, idx_out)

        rows = np.arange(cfg.n_rays)
        free = np.where(any_hit | any_out,
                        radii[rows, end_idx],
                        cfg.grid_depth).astype(np.float64)
        by_drop = blocked & drop[rows, idx_hit]

        return free, by_drop, blocked

    def _choose(self, free):
        """Best bearing by free distance, biased straight ahead, with hysteresis."""
        cfg = self.cfg
        deviation = np.abs(self.bearings)
        # Credit for going straight, tapering to zero at the edge of the arc.
        bonus = cfg.heading_straight_bonus * (1.0 - deviation / max(deviation.max(), 1e-6))
        score = free + bonus

        best = int(np.argmax(score))
        if best == self._heading_index:
            self._challenger = None
            self._challenger_frames = 0
            return self._heading_index

        if score[best] < score[self._heading_index] + cfg.heading_switch_margin:
            self._challenger = None
            self._challenger_frames = 0
            return self._heading_index

        # A clearly better bearing must stay better for several frames.
        if best == self._challenger:
            self._challenger_frames += 1
        else:
            self._challenger = best
            self._challenger_frames = 1

        if self._challenger_frames >= cfg.heading_switch_frames:
            self._heading_index = best
            self._challenger = None
            self._challenger_frames = 0

        return self._heading_index
