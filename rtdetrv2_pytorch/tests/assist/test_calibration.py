import numpy as np
import pytest

from src.assist.config import AssistConfig
from src.assist.calibration import ScaleCalibrator
from src.assist.detections import Detection
from src.assist.geometry import Intrinsics, box_interior

W, H = 640, 480
TRUE_A, TRUE_B = 0.42, 0.03


def _scene(ranges, intr, a=TRUE_A, b=TRUE_B, noise=0.0, name='person',
           real_h=1.70, seed=0, x0=60):
    """Build detections plus a matching relative-inverse-depth map.

    A person at range Z projects to a box of height fy*H/Z, and the depth net
    would read d such that 1/Z = a*d + b. We paint that d into the box.
    """
    rng = np.random.default_rng(seed)
    depth_rel = np.zeros((H, W), dtype=np.float32)
    dets = []

    for k, z in enumerate(ranges):
        h_px = intr.fy * real_h / z
        cx = x0 + k * 70
        y1 = intr.cy - h_px / 2.0
        box = (cx - h_px / 6.0, y1, cx + h_px / 6.0, y1 + h_px)

        d = ((1.0 / z) - b) / a
        if noise:
            d += rng.normal(0.0, noise)

        sl = box_interior(box, depth_rel.shape)
        depth_rel[sl] = d
        dets.append(Detection(name=name, label=0, score=0.9, box=box))

    return dets, depth_rel


def _intr():
    return Intrinsics.from_hfov(W, H, 65.0)


def test_uncalibrated_until_enough_samples():
    cfg = AssistConfig()
    cal = ScaleCalibrator(cfg)
    intr = _intr()
    dets, depth = _scene([3.0, 4.0], intr)

    cal.observe(dets, depth, intr, now=0.0)
    assert not cal.calibrated
    assert cal.a == cfg.calib_fallback_a


def test_recovers_known_affine_scale():
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    dets, depth = _scene([1.5, 2.0, 3.0, 4.0, 5.5, 7.0], intr)

    cal.observe(dets, depth, intr, now=1.0)

    assert cal.calibrated
    assert cal.a == pytest.approx(TRUE_A, rel=0.05)
    assert cal.b == pytest.approx(TRUE_B, abs=0.02)


def test_metric_conversion_round_trips_to_real_distances():
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    ranges = [1.5, 2.0, 3.0, 4.0, 5.5, 7.0]
    dets, depth = _scene(ranges, intr)
    cal.observe(dets, depth, intr, now=1.0)

    for det, z in zip(dets, ranges):
        sl = box_interior(det.box, depth.shape)
        got = float(np.median(cal.to_metric(depth)[sl]))
        assert got == pytest.approx(z, rel=0.08)


def test_survives_noise():
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    dets, depth = _scene([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0],
                         intr, noise=0.03, seed=3)
    cal.observe(dets, depth, intr, now=1.0)

    assert cal.calibrated
    assert cal.a == pytest.approx(TRUE_A, rel=0.20)


def test_ransac_rejects_wrong_height_outliers():
    """A child, or a partly occluded adult, gives a badly wrong Z_geo."""
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    good, depth = _scene([1.5, 2.0, 3.0, 4.0, 5.0, 6.0], intr)

    # three "people" that are really 0.9 m tall -> pinhole ranging halves their
    # distance, so their (d, 1/Z) pairs sit far off the true line.
    bad, bad_depth = _scene([2.5, 3.5, 4.5], intr, real_h=0.90, seed=9, x0=430)
    for det in bad:
        sl = box_interior(det.box, depth.shape)
        depth[sl] = bad_depth[sl]

    n_used = cal.observe(good + bad, depth, intr, now=1.0)

    assert cal.calibrated
    assert cal.a == pytest.approx(TRUE_A, rel=0.15)
    # every outlier must be excluded from the consensus set: the 1.5 m person
    # is dropped by the edge-clip guard (569 px tall in a 480 px frame), so
    # 5 good + 3 bad samples are harvested and exactly the 5 good ones fit.
    assert n_used == 8
    assert cal.inlier_count == 5


def test_edge_clipped_boxes_are_skipped():
    """A box touching the frame edge has a truncated height -> bad range."""
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    clipped = [Detection('person', 0, 0.9, (100.0, 0.0, 150.0, 300.0))]
    depth = np.full((H, W), 1.0, dtype=np.float32)

    assert cal.observe(clipped, depth, intr, now=0.0) == 0


def test_low_confidence_detections_are_not_used():
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    dets, depth = _scene([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], intr)
    for d in dets:
        d.score = 0.50           # above detection bar, below calibration bar

    assert cal.observe(dets, depth, intr, now=0.0) == 0
    assert not cal.calibrated


def test_unknown_classes_contribute_nothing():
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    dets, depth = _scene([2.0, 3.0, 4.0, 5.0, 6.0], intr, name='tv')

    assert cal.observe(dets, depth, intr, now=0.0) == 0


def test_freshness_expires():
    cfg = AssistConfig()
    cal = ScaleCalibrator(cfg)
    intr = _intr()
    dets, depth = _scene([1.5, 2.0, 3.0, 4.0, 5.5, 7.0], intr)
    cal.observe(dets, depth, intr, now=10.0)

    assert cal.is_fresh(10.0 + cfg.calib_max_age - 0.1)
    assert not cal.is_fresh(10.0 + cfg.calib_max_age + 0.1)


def test_to_metric_clamps_far_field():
    cal = ScaleCalibrator(AssistConfig())
    flat = np.zeros((4, 4), dtype=np.float32)      # d = 0 -> 1/Z = b -> huge Z
    assert np.all(cal.to_metric(flat, max_range=30.0) <= 30.0 + 1e-3)


def test_clustered_samples_fall_back_to_a_proportional_fit():
    """Objects all at one distance must not produce a wild extrapolation.

    Regression test for a real failure: two people and a dog walking together
    gave four samples in a narrow depth band, the affine fit returned an
    intercept of -1.6, and every distance in the scene collapsed to under a
    metre -- which in turn put the 'floor' 40 cm below the camera and disabled
    ground-plane detection completely.
    """
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    dets, depth = _scene([3.0, 3.1, 3.2, 3.15, 3.05, 2.95], intr)

    cal.observe(dets, depth, intr, now=1.0)

    assert cal.calibrated
    assert cal.b == 0.0
    metric = cal.to_metric(depth)
    for det, z in zip(dets, [3.0, 3.1, 3.2, 3.15, 3.05, 2.95]):
        sl = box_interior(det.box, depth.shape)
        assert float(np.median(metric[sl])) == pytest.approx(z, rel=0.15)


def test_intercept_is_never_negative():
    """b < 0 means depth runs to infinity and then flips sign. No such camera."""
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    dets, depth = _scene([2.0, 2.5, 3.0, 4.0, 5.0, 6.0], intr, b=-0.4)

    cal.observe(dets, depth, intr, now=1.0)
    assert cal.b >= 0.0


def test_far_field_stays_far_after_a_clustered_fit():
    """The failure mode was near-field collapse; guard the whole range."""
    cal = ScaleCalibrator(AssistConfig())
    intr = _intr()
    dets, depth = _scene([3.0, 3.1, 3.2, 3.15, 3.05, 2.95], intr)
    cal.observe(dets, depth, intr, now=1.0)

    # a much smaller relative-depth reading must map to a much larger distance
    faint = np.full((4, 4), float(np.median(depth[depth > 0])) / 4.0, np.float32)
    assert float(np.median(cal.to_metric(faint))) > 8.0
