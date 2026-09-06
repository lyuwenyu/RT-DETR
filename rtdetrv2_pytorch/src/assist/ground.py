"""Ground-plane estimation and height-above-floor classification.

Given a metric point cloud, RANSAC-fit the floor the user is standing on, then
classify every point by its signed height above that plane:

    |h| small   ->  floor      (walkable)
    h positive  ->  obstacle   (something sticking up)
    h negative  ->  DROP-OFF   (the floor is lower than it should be)

That last case is the reason this module exists and is not optional. A curb, a
descending stair, a kerb-side gutter or an unfenced platform edge has no COCO
class and no bounding box -- no amount of object detection will ever find one.
It is visible only as geometry: ground that sits below the plane the user is
currently walking on. For a blind walker it is also the highest-stakes hazard
in the scene, so it is derived here rather than inferred from detections.

Fitting the plane per frame, instead of assuming a fixed camera height and zero
pitch, means the system self-calibrates to how the user actually carries the
camera and stays correct as they lean, climb or look down.
"""

from dataclasses import dataclass

import numpy as np

# Camera-frame "up" is -Y, because Y points down.
_UP = np.array([0.0, -1.0, 0.0])


@dataclass
class Plane:
    normal: np.ndarray      # unit vector pointing up, away from the floor
    d: float                # plane is n . p + d = 0; equals camera height
    tilt_deg: float         # deviation of the normal from straight up
    inlier_frac: float

    def height_of(self, points: np.ndarray) -> np.ndarray:
        """Signed height above the plane; positive is up."""
        return points @ self.normal + self.d

    @property
    def camera_height(self) -> float:
        return self.d


class GroundEstimator:
    """Per-frame RANSAC floor fit, holding the last good plane when a fit fails."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.plane = None
        self.stale_frames = 0

    def update(self, points: np.ndarray, vs: np.ndarray, image_height: int):
        """Fit the floor using points from the lower part of the image.

        The bottom third is where the ground in front of the user's feet
        projects, so it is the region most likely to be floor and least likely
        to be wall, sky or torso.
        """
        cfg = self.cfg
        seed_mask = (vs > image_height * 0.55) & np.isfinite(points[:, 2]) \
            & (points[:, 2] > 0.2) & (points[:, 2] < cfg.grid_depth)
        seed = points[seed_mask]

        fit = self._ransac(seed) if len(seed) >= 30 else None

        if fit is None:
            # Hold the previous plane rather than guessing. Guidance degrades to
            # obstacle-only, which the caller signals via `stale_frames`.
            self.stale_frames += 1
            return self.plane

        self.plane = fit
        self.stale_frames = 0
        return fit

    def _ransac(self, pts: np.ndarray):
        cfg = self.cfg
        rng = np.random.default_rng(0)
        n = len(pts)
        best = None
        best_count = 0

        # Score hypotheses against a bounded random subset. Counting inliers
        # over every seed point on every iteration dominated the whole guidance
        # budget, and a few thousand samples rank candidate planes just as well.
        if n > cfg.ground_score_samples:
            scoring = pts[rng.choice(n, size=cfg.ground_score_samples,
                                     replace=False)]
        else:
            scoring = pts

        for _ in range(cfg.ground_ransac_iters):
            idx = rng.choice(n, size=3, replace=False)
            p0, p1, p2 = pts[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            if normal[1] > 0:            # orient "up" (-Y in camera frame)
                normal = -normal
            d = -float(normal @ p0)

            # A floor is roughly horizontal and roughly a body-height below the
            # camera. Anything else is a wall, a table top or a bad sample.
            tilt = np.degrees(np.arccos(np.clip(normal @ _UP, -1.0, 1.0)))
            if tilt > cfg.ground_max_tilt_deg:
                continue
            if not (cfg.ground_min_cam_height <= d <= cfg.ground_max_cam_height):
                continue

            count = int((np.abs(scoring @ normal + d) < cfg.ground_inlier_tol).sum())
            if count > best_count:
                best_count, best = count, (normal, d, tilt)

        if best is None:
            return None

        frac = best_count / float(len(scoring))
        if frac < cfg.ground_min_inlier_frac:
            return None

        normal, d, tilt = best

        # Refine on the consensus set: the least-squares plane through all
        # inliers is the eigenvector of least variance.
        inliers = pts[np.abs(pts @ normal + d) < cfg.ground_inlier_tol]
        centroid = inliers.mean(axis=0)
        _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
        refined = vt[-1]
        if refined[1] > 0:
            refined = -refined
        refined_d = -float(refined @ centroid)

        tilt = float(np.degrees(np.arccos(np.clip(refined @ _UP, -1.0, 1.0))))
        if tilt > cfg.ground_max_tilt_deg or \
           not (cfg.ground_min_cam_height <= refined_d <= cfg.ground_max_cam_height):
            return None

        return Plane(normal=refined, d=refined_d, tilt_deg=tilt, inlier_frac=frac)

    def classify(self, points: np.ndarray, plane: Plane):
        """Split points into (floor, obstacle, dropoff) boolean masks."""
        cfg = self.cfg
        h = plane.height_of(points)
        floor = np.abs(h) < cfg.floor_band
        obstacle = h > cfg.obstacle_height
        dropoff = h < -cfg.dropoff_depth
        return floor, obstacle, dropoff, h
