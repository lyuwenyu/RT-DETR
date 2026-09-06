import math

import numpy as np
import pytest

from src.assist.config import AssistConfig
from src.assist.hazards import (assess, zone_for, IMMEDIATE, NEAR, CONTEXT,
                                IGNORE)
from src.assist.planner import Corridor
from src.assist.tracking import Track


def _corridor(n=9, heading=0.0, dropoff_at=None, free=6.0):
    bearings = np.linspace(math.radians(-30), math.radians(30), n)
    blocked = np.zeros(n, dtype=bool)
    by_drop = np.zeros(n, dtype=bool)
    frees = np.full(n, float(free))
    if dropoff_at is not None:
        idx, dist = dropoff_at
        blocked[idx] = True
        by_drop[idx] = True
        frees[idx] = dist
    return Corridor(bearings=bearings, free=frees, blocked=blocked,
                    blocked_by_dropoff=by_drop, dropoff_confirmed=by_drop.copy(),
                    heading_index=int(np.argmin(np.abs(bearings - heading))),
                    heading_bearing=heading, heading_free=free,
                    is_clear=not blocked.any())


def _track(name='person', rng=2.0, bearing=0.0, tid=1, hits=3, rate=0.0):
    return Track(id=tid, name=name, box=(0, 0, 10, 10), score=0.9,
                 range_m=rng, bearing=bearing, hits=hits, range_rate=rate)


def test_zone_boundaries():
    cfg = AssistConfig()
    inf = float('inf')
    assert zone_for(1.0, inf, cfg) == IMMEDIATE
    assert zone_for(2.5, inf, cfg) == NEAR
    assert zone_for(5.0, inf, cfg) == CONTEXT
    assert zone_for(20.0, inf, cfg) == IGNORE


def test_fast_approach_is_immediate_even_when_far():
    """A car 6 m away closing at 5 m/s is not a 'context' item."""
    cfg = AssistConfig()
    assert zone_for(6.0, 1.2, cfg) == IMMEDIATE


def test_unconfirmed_tracks_are_not_reported():
    cfg = AssistConfig()
    hz = assess([_track(hits=1)], _corridor(), cfg)
    assert hz == []


def test_non_navigation_classes_are_ignored():
    cfg = AssistConfig()
    assert assess([_track(name='toothbrush')], _corridor(), cfg) == []


def test_closer_object_outranks_a_farther_one():
    cfg = AssistConfig()
    hz = assess([_track(rng=5.0, tid=1), _track(rng=1.2, tid=2)],
                _corridor(), cfg)
    assert hz[0].track_id == 2


def test_approaching_object_outranks_a_static_one_at_equal_range():
    cfg = AssistConfig()
    static = _track(rng=4.0, tid=1, rate=0.0)
    closing = _track(rng=4.0, tid=2, rate=-2.5)     # coming at the user

    hz = assess([static, closing], _corridor(), cfg)
    assert hz[0].track_id == 2
    assert hz[0].ttc < 3.0


def test_object_in_the_chosen_path_outranks_one_off_to_the_side():
    cfg = AssistConfig()
    ahead = _track(rng=3.0, bearing=0.0, tid=1)
    aside = _track(rng=3.0, bearing=math.radians(28), tid=2)

    hz = assess([ahead, aside], _corridor(heading=0.0), cfg)
    assert hz[0].track_id == 1
    assert hz[0].in_path
    assert not hz[1].in_path


def test_dropoff_outranks_every_object():
    cfg = AssistConfig()
    hz = assess([_track(rng=0.8, tid=1)],
                _corridor(dropoff_at=(4, 2.0)), cfg)

    assert hz[0].is_dropoff
    assert hz[0].zone == IMMEDIATE
    assert hz[0].name == 'step down'


def test_dropoff_is_immediate_regardless_of_distance():
    cfg = AssistConfig()
    hz = assess([], _corridor(dropoff_at=(4, 6.0)), cfg)
    assert hz[0].zone == IMMEDIATE


def test_blocked_but_not_dropoff_produces_no_dropoff_hazard():
    cfg = AssistConfig()
    cor = _corridor()
    cor.blocked[4] = True                # an ordinary obstacle
    hz = assess([], cor, cfg)
    assert not any(h.is_dropoff for h in hz)


def test_non_finite_range_is_dropped():
    cfg = AssistConfig()
    assert assess([_track(rng=float('nan'))], _corridor(), cfg) == []


def test_dropoff_ahead_is_immediate_and_in_path():
    cfg = AssistConfig()
    hz = assess([], _corridor(heading=0.0, dropoff_at=(4, 2.0)), cfg)
    assert hz[0].is_dropoff and hz[0].in_path and hz[0].zone == IMMEDIATE


def test_dropoff_to_the_side_is_advisory_not_immediate():
    """The path edge is worth knowing about; it is not a reason to stop."""
    cfg = AssistConfig()
    hz = assess([], _corridor(heading=0.0, dropoff_at=(0, 2.0)), cfg)
    assert hz[0].is_dropoff
    assert not hz[0].in_path
    assert hz[0].zone == NEAR


def test_distant_side_dropoff_is_ignored_entirely():
    cfg = AssistConfig()
    hz = assess([], _corridor(heading=0.0, dropoff_at=(0, 5.5)), cfg)
    assert not any(h.is_dropoff for h in hz)


def test_unconfirmed_dropoff_produces_no_hazard():
    cfg = AssistConfig()
    cor = _corridor(dropoff_at=(4, 2.0))
    cor.dropoff_confirmed[:] = False
    assert not any(h.is_dropoff for h in assess([], cor, cfg))
