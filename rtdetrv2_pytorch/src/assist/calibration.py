"""Recover metric scale for a monocular depth map, using the detector itself.

Monocular depth networks emit *relative inverse* depth: a value d that grows as
things get closer, related to true depth Z by an unknown affine transform

    1 / Z = a * d + b

so the raw map cannot answer "how many metres away". But the detector is
already finding people, cars and chairs, and those have well-known real-world
heights. Pinhole ranging on such a box gives an independent Z_geo, and pairing
it with the depth map's reading at the same box yields one (d, 1/Z) sample.
Enough samples determine (a, b) by least squares.

That is the trick this module implements: the object detector calibrates the
depth network, so a single ordinary camera yields metric depth with no depth
sensor, no stereo rig and no user calibration step.

RANSAC rather than plain least squares, because individual samples are noisy in
ways that are not Gaussian -- a partially occluded person, a child, a box whose
bottom is cut off by the frame edge -- and a single such outlier would drag an
ordinary fit badly.
"""

from collections import deque

import numpy as np

from .geometry import pinhole_range, box_interior


class ScaleCalibrator:
    """Maintains a rolling affine fit from relative inverse depth to 1/metres."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._samples = deque(maxlen=cfg.calib_window)
        self.a = cfg.calib_fallback_a
        self.b = cfg.calib_fallback_b
        self.calibrated = False
        self.inlier_count = 0
        self.last_fit_time = None

    # -- sampling -----------------------------------------------------------

    def observe(self, detections, depth_rel, intr, now):
        """Harvest (d, 1/Z_geo) pairs from known-height detections."""
        heights = self.cfg.known_heights
        added = 0

        for det in detections:
            real_h = heights.get(det.name)
            if real_h is None or det.score < self.cfg.calib_score_threshold:
                continue

            x1, y1, x2, y2 = det.box
            # A box clipped by the top or bottom edge has a truncated pixel
            # height, which would under-estimate range. Skip those.
            if y1 <= 1.0 or y2 >= intr.height - 1.0:
                continue

            z_geo = pinhole_range(y2 - y1, real_h, intr.fy)
            if not np.isfinite(z_geo) or not (0.3 < z_geo < 30.0):
                continue

            sl = box_interior(det.box, depth_rel.shape)
            if sl is None:
                continue
            patch = depth_rel[sl]
            if patch.size < 4:
                continue

            d_obj = float(np.median(patch))
            if not np.isfinite(d_obj):
                continue

            self._samples.append((d_obj, 1.0 / z_geo))
            added += 1

        if added:
            self._fit(now)
        return added

    # -- fitting ------------------------------------------------------------

    def _fit(self, now):
        if len(self._samples) < self.cfg.calib_min_samples:
            return

        pts = np.asarray(self._samples, dtype=np.float64)
        d, inv_z = pts[:, 0], pts[:, 1]

        spread = float(d.max() - d.min())
        if spread < 1e-3:
            return

        # A two-parameter affine fit needs the samples to actually span a range
        # of depths. When everything visible sits at roughly the same distance
        # -- two people and a dog walking together, say -- slope and intercept
        # trade off almost freely, and the best-fitting line through that
        # cluster can be wildly wrong once extrapolated to the rest of the
        # scene. Observed in practice: a fit of b = -1.6 that placed the ground
        # 40 cm below the camera and killed the ground-plane search entirely.
        #
        # So when the samples are clustered, drop to the one-parameter model
        # 1/Z = a*d, which is the physically motivated one anyway: a depth
        # network reads ~0 at infinity, and that is exactly what b = 0 says.
        if spread < self.cfg.calib_min_spread * max(abs(float(d.max())), 1e-6):
            self._fit_proportional(d, inv_z, now)
            return

        rng = np.random.default_rng(0)
        n = len(d)
        best_inliers = None
        best_count = 0

        for _ in range(self.cfg.calib_ransac_iters):
            i, j = rng.choice(n, size=2, replace=False)
            dd = d[j] - d[i]
            if abs(dd) < 1e-6:
                continue
            a = (inv_z[j] - inv_z[i]) / dd
            b = inv_z[i] - a * d[i]
            if a <= 0:                       # closer must mean nearer
                continue
            # b < 0 puts a sign change inside the valid range: depth would run
            # to infinity and then turn negative. There is no such camera.
            if b < 0:
                continue

            resid = np.abs(a * d + b - inv_z)
            inliers = resid < self.cfg.calib_ransac_tol
            count = int(inliers.sum())
            if count > best_count:
                best_count, best_inliers = count, inliers

        if best_inliers is None or best_count < self.cfg.calib_min_samples:
            return

        # Refit on the consensus set for a less arbitrary answer than the
        # two-point hypothesis that produced it.
        a, b = np.polyfit(d[best_inliers], inv_z[best_inliers], 1)
        if a <= 0:
            return
        if b < 0:
            # The consensus set supports the slope but not a negative offset;
            # keep the slope and pin the intercept at its physical floor.
            self._fit_proportional(d[best_inliers], inv_z[best_inliers], now)
            return

        self.a, self.b = float(a), float(b)
        self.inlier_count = best_count
        self.calibrated = True
        self.last_fit_time = now

    def _fit_proportional(self, d, inv_z, now):
        """Fit 1/Z = a*d with b pinned to zero, robustly via the median ratio."""
        ok = d > 1e-6
        if int(ok.sum()) < self.cfg.calib_min_samples:
            return
        a = float(np.median(inv_z[ok] / d[ok]))
        if not np.isfinite(a) or a <= 0:
            return

        self.a, self.b = a, 0.0
        self.inlier_count = int(ok.sum())
        self.calibrated = True
        self.last_fit_time = now

    # -- use ----------------------------------------------------------------

    def is_fresh(self, now):
        """Whether the current fit is recent enough to state distances aloud."""
        if not self.calibrated or self.last_fit_time is None:
            return False
        return (now - self.last_fit_time) <= self.cfg.calib_max_age

    def to_metric(self, depth_rel, max_range=30.0):
        """Convert a relative inverse-depth map to metres."""
        inv_z = self.a * depth_rel + self.b
        # Guard the pole at inv_z -> 0: anything that flat is effectively
        # infinitely far, which for guidance purposes is max_range.
        inv_z = np.maximum(inv_z, 1.0 / max_range)
        return (1.0 / inv_z).astype(np.float32)
