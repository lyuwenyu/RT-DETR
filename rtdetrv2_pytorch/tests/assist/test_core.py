"""End-to-end tests of the pure guidance core on synthetic scenes.

Each scene is rendered the way a real camera would see it: a flat floor
1.4 m below a level camera, objects standing on that floor, and a relative
inverse-depth map consistent with the geometry. Everything the user would
hear is then asserted from the core's output alone.
"""

import math

import numpy as np
import pytest

from src.assist.config import AssistConfig
from src.assist.core import GuidanceCore
from src.assist.detections import Detection
from src.assist.geometry import Intrinsics

W, H = 640, 480
CAM_H = 1.40
FAR = 30.0


def _intr():
    return Intrinsics.from_hfov(W, H, 65.0)


def _floor_depth(intr, cam_h=CAM_H, step_at=None, step_drop=0.0):
    """Metric depth of a flat floor seen by a level camera.

    A floor point at row v satisfies Y = (v - cy) * Z / fy = cam_h, so
    Z = cam_h * fy / (v - cy). Rows at or above the horizon see only far
    background. Note this means a level chest-height camera sees no ground
    nearer than about 3 m -- which is itself a real property of the setup.

    With `step_at`, the ground beyond that distance sits `step_drop` metres
    lower, so it projects further away for the same image row: exactly how a
    kerb or a descending stair looks to a camera.
    """
    vs = np.arange(H, dtype=np.float64)
    denom = vs - intr.cy
    visible = denom > 1.0

    upper = np.full(H, FAR)
    upper[visible] = cam_h * intr.fy / denom[visible]

    if step_at is None:
        row_z = upper
    else:
        lower = np.full(H, FAR)
        lower[visible] = (cam_h + step_drop) * intr.fy / denom[visible]
        # rows whose upper-floor depth is past the edge instead see the
        # lowered ground behind it
        row_z = np.where(upper > step_at, lower, upper)

    z = np.repeat(row_z[:, None], W, axis=1)
    return np.clip(z, 0.3, FAR)


def _stand(z_map, intr, name, range_m, real_h, bearing_deg=0.0):
    """Paint an upright object onto the depth map and return its Detection."""
    h_px = intr.fy * real_h / range_m
    w_px = h_px * 0.35
    u_c = intr.cx + intr.fx * math.tan(math.radians(bearing_deg))
    v_feet = intr.cy + CAM_H * intr.fy / range_m
    v_head = v_feet - h_px

    x1, x2 = u_c - w_px / 2, u_c + w_px / 2
    y1, y2 = v_head, v_feet

    c0, c1 = int(max(0, x1)), int(min(W, x2))
    r0, r1 = int(max(0, y1)), int(min(H, y2))
    z_map[r0:r1, c0:c1] = range_m

    return Detection(name=name, label=0, score=0.9,
                     box=(float(x1), float(y1), float(x2), float(y2)))


def _rel(z_map, a=1.0, b=0.0):
    """Metric depth -> relative inverse depth, as a depth network emits it."""
    return ((1.0 / z_map - b) / a).astype(np.float32)


class Run:
    """Result of driving the core for several frames.

    The arbiter deliberately speaks once and then falls silent, so a test that
    only looked at the final frame would see None for a phrase that was in fact
    delivered. Everything spoken across the run is collected here.
    """

    def __init__(self, last, spoken):
        self.last = last
        self.spoken = spoken

    def __getattr__(self, name):
        return getattr(self.last, name)

    def said(self, word):
        return any(word in s for s in self.spoken)


def _run(core, dets, z_map, frames=4, t0=100.0, dt=0.1, a=1.0, b=0.0):
    depth = _rel(z_map, a, b)
    spoken = []
    for i in range(frames):
        out = core.update(dets, depth, now=t0 + i * dt)
        if out.speech:
            spoken.append(out.speech)
    return Run(out, spoken)


# -- the empty case ----------------------------------------------------------

def test_clear_floor_is_silent_and_walkable():
    """Silence is the correct output for an open path."""
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    out = _run(core, [], _floor_depth(intr))

    assert out.plane is not None
    assert out.plane.camera_height == pytest.approx(CAM_H, abs=0.1)
    assert out.corridor is not None
    assert out.corridor.is_clear
    assert out.speech is None
    assert not out.beep.active


def test_a_whole_walk_down_a_clear_corridor_says_nothing():
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    depth = _rel(_floor_depth(intr))

    said = [core.update([], depth, now=100.0 + i * 0.1).speech for i in range(100)]
    assert all(s is None for s in said)


# -- objects -----------------------------------------------------------------

def test_person_ahead_is_detected_ranged_and_announced():
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    z = _floor_depth(intr)
    person = _stand(z, intr, 'person', 2.5, 1.70)

    out = _run(core, [person], z)

    assert out.hazards, 'a person 2.5 m ahead must register as a hazard'
    top = out.hazards[0]
    assert top.name == 'person'
    assert top.range_m == pytest.approx(2.5, rel=0.15)
    assert out.said('person')


def test_object_bearing_is_reported_on_the_correct_side():
    intr = _intr()
    z = _floor_depth(intr)
    person = _stand(z, intr, 'person', 3.0, 1.70, bearing_deg=-25.0)

    core = GuidanceCore(AssistConfig(), intr)
    out = _run(core, [person], z)

    assert out.hazards[0].bearing < 0
    assert out.said('left')


def test_obstacle_ahead_shortens_the_corridor():
    intr = _intr()
    z = _floor_depth(intr)
    person = _stand(z, intr, 'person', 2.0, 1.70)

    core = GuidanceCore(AssistConfig(), intr)
    out = _run(core, [person], z)

    straight = out.corridor.free[out.corridor.bearings.size // 2]
    assert straight < 2.2
    assert not out.corridor.is_clear


def test_beep_activates_for_a_close_object():
    intr = _intr()
    z = _floor_depth(intr)
    _stand(z, intr, 'person', 1.2, 1.70)
    person = _stand(z, intr, 'person', 1.2, 1.70)

    core = GuidanceCore(AssistConfig(), intr)
    out = _run(core, [person], z)

    assert out.beep.active
    assert out.beep.interval < 1.0


# -- drop-offs ---------------------------------------------------------------

def test_step_down_is_found_from_geometry_alone():
    """No detection, no class, no box -- only the ground plane reveals it."""
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    # a level chest-height camera only sees the ground from ~3 m out, so
    # the step has to sit within the band of floor it can actually see
    z = _floor_depth(intr, step_at=4.5, step_drop=0.30)

    out = _run(core, [], z)

    assert out.corridor is not None
    assert out.corridor.blocked_by_dropoff.any()
    assert any(h.is_dropoff for h in out.hazards)
    assert 'stop, step down' in out.spoken


def test_flat_floor_produces_no_false_dropoff():
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    out = _run(core, [], _floor_depth(intr))

    assert not out.corridor.blocked_by_dropoff.any()
    assert not any(h.is_dropoff for h in out.hazards)


# -- scale calibration through the full pipeline -----------------------------

def test_scale_is_recovered_end_to_end_from_known_heights():
    """The keystone claim: the detector calibrates the depth net's scale."""
    intr = _intr()
    cfg = AssistConfig()
    core = GuidanceCore(cfg, intr)

    a, b = 0.55, 0.02          # the depth net's unknown affine transform
    z = _floor_depth(intr)
    people = [_stand(z, intr, 'person', r, 1.70, bearing_deg=d)
              for r, d in ((2.0, -22.0), (3.5, -8.0), (5.0, 8.0), (7.0, 22.0))]

    out = _run(core, people, z, frames=3, a=a, b=b)

    assert core.calibrator.calibrated
    assert core.calibrator.a == pytest.approx(a, rel=0.15)
    assert out.metric_ok
    # and the recovered ranges must be right, not merely self-consistent
    by_bearing = sorted(out.hazards, key=lambda h: h.bearing)
    assert by_bearing[0].range_m == pytest.approx(2.0, rel=0.25)


def test_distances_are_not_spoken_before_calibration():
    """Better vague than confidently wrong."""
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    z = _floor_depth(intr)
    tv = _stand(z, intr, 'tv', 2.5, 0.60)      # not a known-height class

    out = _run(core, [tv], z)

    assert not core.calibrator.calibrated
    assert not out.metric_ok
    if out.speech:
        assert 'metre' not in out.speech


# -- robustness --------------------------------------------------------------

def test_degenerate_depth_map_does_not_crash():
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    out = core.update([], np.zeros((H, W), dtype=np.float32), now=1.0)

    assert out.speech is None
    assert out.corridor is None or not out.corridor.blocked.any()


def test_core_is_pure_across_repeated_identical_frames():
    """Same input, advancing time: no runaway state, no growing chatter."""
    intr = _intr()
    core = GuidanceCore(AssistConfig(), intr)
    z = _floor_depth(intr)
    person = _stand(z, intr, 'person', 2.5, 1.70)
    depth = _rel(z)

    said = [core.update([person], depth, now=100.0 + i * 0.1).speech
            for i in range(60)]
    spoken = [s for s in said if s]

    # 6 seconds of an unchanging scene stays quiet: the person is named once,
    # and the rest is at most an occasional steer around them.
    assert sum('person' in s for s in spoken) == 1
    assert len(spoken) <= 4
