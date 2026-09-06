import math

import numpy as np
import pytest

from src.assist.geometry import (Intrinsics, pinhole_range, box_centre_bearing,
                                 box_interior)


def test_hfov_maps_to_image_edges():
    intr = Intrinsics.from_hfov(640, 480, 65.0)
    left, centre, right = np.degrees(intr.bearing_of_u([0, 320, 640]))
    assert centre == pytest.approx(0.0, abs=1e-9)
    assert left == pytest.approx(-32.5, abs=0.1)
    assert right == pytest.approx(32.5, abs=0.1)


def test_pinhole_range_is_inverse_in_pixel_height():
    intr = Intrinsics.from_hfov(640, 480, 65.0)
    near = pinhole_range(400, 1.7, intr.fy)
    far = pinhole_range(200, 1.7, intr.fy)
    assert far == pytest.approx(2 * near)


def test_pinhole_range_degenerate_box_is_infinite():
    assert math.isinf(pinhole_range(0, 1.7, 500.0))


def test_backprojection_recovers_known_point():
    intr = Intrinsics.from_hfov(640, 480, 65.0)
    depth = np.full((480, 640), 5.0, dtype=np.float32)
    pts, us, vs = intr.backproject(depth, stride=40)

    # the principal-point pixel must land straight ahead at exactly Z
    i = np.argmin((us - intr.cx) ** 2 + (vs - intr.cy) ** 2)
    assert pts[i, 2] == pytest.approx(5.0)
    # and X should equal (u - cx) * Z / fx everywhere
    expected_x = (us - intr.cx) * 5.0 / intr.fx
    assert np.allclose(pts[:, 0], expected_x)


def test_box_interior_shrinks_and_clips():
    sl = box_interior((100, 100, 200, 200), (480, 640), shrink=0.25)
    rows, cols = sl
    assert (rows.start, rows.stop) == (125, 175)
    assert (cols.start, cols.stop) == (125, 175)


def test_box_interior_rejects_degenerate_box():
    assert box_interior((10, 10, 10, 10), (480, 640)) is None


def test_box_centre_bearing_sign():
    intr = Intrinsics.from_hfov(640, 480, 65.0)
    assert box_centre_bearing((0, 0, 100, 100), intr) < 0      # left is negative
    assert box_centre_bearing((540, 0, 640, 100), intr) > 0    # right is positive
