"""Camera geometry: intrinsics, back-projection, bearings and pinhole ranging.

Coordinate conventions used throughout the assist package:

    image   u = column (right positive), v = row (down positive)
    camera  X = right, Y = down, Z = forward (metres)
    bearing radians, 0 = straight ahead, positive = to the user's right

Keeping every conversion in this one module means the rest of the pipeline can
speak in metres and radians and never re-derive a projection.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> 'Intrinsics':
        """Approximate intrinsics from image size and horizontal field of view.

        Assumes square pixels and a centred principal point. This is the
        laptop-webcam fallback; on Android, Camera2 reports true intrinsics and
        this approximation disappears.
        """
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    def bearing_of_u(self, u):
        """Horizontal bearing (radians) of image column(s) u."""
        return np.arctan2(np.asarray(u, dtype=np.float64) - self.cx, self.fx)

    def backproject(self, depth: np.ndarray, stride: int = 1):
        """Lift a metric depth map to a camera-frame point cloud.

        Returns (points, us, vs) where points is (N, 3) as X (right), Y (down),
        Z (forward). `stride` subsamples the pixel grid -- the guidance loop
        does not need every pixel, and striding is the cheapest way to keep the
        per-frame cost flat as camera resolution grows.
        """
        h, w = depth.shape
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[0:h:stride, 0:w:stride]

        us = us.ravel().astype(np.float64)
        vs = vs.ravel().astype(np.float64)
        z = z.ravel().astype(np.float64)

        x = (us - self.cx) * z / self.fx
        y = (vs - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=1), us, vs


def pinhole_range(box_height_px: float, real_height_m: float, fy: float) -> float:
    """Distance to an object of known real height from its pixel height.

    Z = fy * H / h_px. Returns inf for a degenerate (zero-height) box rather
    than raising, so callers can filter with a finite check.
    """
    if box_height_px <= 0.0:
        return float('inf')
    return float(fy) * float(real_height_m) / float(box_height_px)


def box_centre_bearing(box, intr: Intrinsics) -> float:
    """Bearing (radians) of a box's horizontal centre."""
    x1, _, x2, _ = box
    return float(np.arctan2((x1 + x2) / 2.0 - intr.cx, intr.fx))


def box_interior(box, shape, shrink: float = 0.25):
    """Slices for the central part of a box, for robust depth sampling.

    Sampling the whole box bleeds background depth in around the object's
    silhouette; shrinking toward the centre keeps the median on the object.
    Returns None when the shrunken box is empty.
    """
    h, w = shape
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    x1 += bw * shrink
    x2 -= bw * shrink
    y1 += bh * shrink
    y2 -= bh * shrink

    c0, c1 = int(max(0, round(x1))), int(min(w, round(x2)))
    r0, r1 = int(max(0, round(y1))), int(min(h, round(y2)))
    if c1 <= c0 or r1 <= r0:
        return None
    return slice(r0, r1), slice(c0, c1)
