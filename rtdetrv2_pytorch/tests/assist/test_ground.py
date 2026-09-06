import numpy as np
import pytest

from src.assist.config import AssistConfig
from src.assist.ground import GroundEstimator

CAM_H = 1.40       # camera carried 1.4 m above the floor


def _floor(n=4000, z_max=6.0, seed=0, height=CAM_H):
    """Flat floor in camera frame: Y is down, so floor points sit at Y = +cam_h."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.uniform(0.4, z_max, n)
    y = np.full(n, height)
    return np.stack([x, y, z], axis=1)


def _vs(points, image_height=480):
    """Row indices that all pass the 'lower part of the image' seed filter."""
    return np.full(len(points), image_height * 0.9)


def test_fits_flat_floor_and_recovers_camera_height():
    est = GroundEstimator(AssistConfig())
    pts = _floor()

    plane = est.update(pts, _vs(pts), image_height=480)

    assert plane is not None
    assert plane.camera_height == pytest.approx(CAM_H, abs=0.02)
    assert plane.tilt_deg == pytest.approx(0.0, abs=1.0)
    assert plane.inlier_frac > 0.9


def test_floor_points_classify_as_floor():
    cfg = AssistConfig()
    est = GroundEstimator(cfg)
    pts = _floor()
    plane = est.update(pts, _vs(pts), 480)

    floor, obstacle, dropoff, h = est.classify(pts, plane)
    assert floor.all()
    assert not obstacle.any()
    assert not dropoff.any()
    assert np.abs(h).max() < cfg.floor_band


def test_upright_object_classifies_as_obstacle():
    est = GroundEstimator(AssistConfig())
    floor = _floor()
    plane = est.update(floor, _vs(floor), 480)

    # a 1 m tall box standing on the floor 2 m ahead
    n = 300
    rng = np.random.default_rng(1)
    obj = np.stack([rng.uniform(-0.3, 0.3, n),
                    rng.uniform(CAM_H - 1.0, CAM_H - 0.2, n),   # above the floor
                    rng.uniform(1.9, 2.1, n)], axis=1)

    _, obstacle, dropoff, h = est.classify(obj, plane)
    assert obstacle.all()
    assert not dropoff.any()
    assert h.min() > 0.15


def test_step_down_classifies_as_dropoff():
    """The hazard with no COCO class: ground lower than the plane underfoot."""
    est = GroundEstimator(AssistConfig())
    floor = _floor()
    plane = est.update(floor, _vs(floor), 480)

    n = 300
    rng = np.random.default_rng(2)
    lower = np.stack([rng.uniform(-1.0, 1.0, n),
                      np.full(n, CAM_H + 0.30),      # 30 cm BELOW the floor
                      rng.uniform(3.0, 4.0, n)], axis=1)

    floor_m, obstacle, dropoff, h = est.classify(lower, plane)
    assert dropoff.all()
    assert not obstacle.any()
    assert not floor_m.any()
    assert h.max() < -0.15


def test_rejects_wall_shaped_plane():
    """A vertical surface must never be accepted as the floor."""
    est = GroundEstimator(AssistConfig())
    rng = np.random.default_rng(3)
    n = 4000
    wall = np.stack([rng.uniform(-2, 2, n),
                     rng.uniform(-1, 1, n),
                     np.full(n, 3.0)], axis=1)     # constant Z: a wall

    assert est.update(wall, _vs(wall), 480) is None


def test_rejects_implausible_camera_height():
    """A table top 30 cm below the camera is not the floor."""
    est = GroundEstimator(AssistConfig())
    pts = _floor(height=0.3)

    assert est.update(pts, _vs(pts), 480) is None


def test_holds_last_good_plane_when_fit_fails():
    est = GroundEstimator(AssistConfig())
    good = _floor()
    plane = est.update(good, _vs(good), 480)

    junk = np.zeros((5, 3))
    held = est.update(junk, _vs(junk), 480)

    assert held is plane
    assert est.stale_frames == 1


def test_tilted_camera_still_fits():
    """Looking 15 deg down must not break the fit -- users do not hold cameras level."""
    est = GroundEstimator(AssistConfig())
    pts = _floor()
    t = np.radians(15.0)
    rot = np.array([[1, 0, 0],
                    [0, np.cos(t), -np.sin(t)],
                    [0, np.sin(t), np.cos(t)]])
    tilted = pts @ rot.T

    plane = est.update(tilted, _vs(tilted), 480)
    assert plane is not None
    assert plane.tilt_deg == pytest.approx(15.0, abs=1.5)
    assert plane.camera_height == pytest.approx(CAM_H, abs=0.05)
