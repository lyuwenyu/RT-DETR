import numpy as np
import pytest

from src.assist.config import AssistConfig
from src.assist.detections import Detection
from src.assist.tracking import Tracker


def _obs(name, box, rng, bearing=0.0, score=0.9):
    return (Detection(name=name, label=0, score=score, box=box), rng, bearing)


def test_same_object_keeps_one_id_across_frames():
    trk = Tracker(AssistConfig())
    trk.update([_obs('person', (100, 100, 200, 400), 5.0)], now=0.0)
    trk.update([_obs('person', (105, 100, 205, 400), 4.8)], now=0.1)
    tracks = trk.update([_obs('person', (110, 100, 210, 400), 4.6)], now=0.2)

    assert len(tracks) == 1
    assert tracks[0].id == 1
    assert tracks[0].hits == 3


def test_new_id_for_a_disjoint_object():
    trk = Tracker(AssistConfig())
    trk.update([_obs('person', (100, 100, 200, 400), 5.0)], now=0.0)
    tracks = trk.update([_obs('person', (100, 100, 200, 400), 5.0),
                         _obs('person', (500, 100, 600, 400), 5.0)], now=0.1)

    assert {t.id for t in tracks} == {1, 2}


def test_different_classes_never_associate():
    trk = Tracker(AssistConfig())
    trk.update([_obs('person', (100, 100, 200, 400), 5.0)], now=0.0)
    tracks = trk.update([_obs('chair', (100, 100, 200, 400), 5.0)], now=0.1)

    assert len(tracks) == 2


def test_track_survives_a_dropped_frame():
    """A one-frame detector miss must not delete the object."""
    cfg = AssistConfig()
    trk = Tracker(cfg)
    trk.update([_obs('person', (100, 100, 200, 400), 5.0)], now=0.0)
    tracks = trk.update([], now=0.1)

    assert len(tracks) == 1
    assert tracks[0].misses == 1


def test_track_expires_after_sustained_absence():
    cfg = AssistConfig()
    trk = Tracker(cfg)
    trk.update([_obs('person', (100, 100, 200, 400), 5.0)], now=0.0)
    for i in range(cfg.track_max_misses + 1):
        tracks = trk.update([], now=0.1 * (i + 1))

    assert tracks == []


def test_approaching_object_has_finite_time_to_contact():
    """The distinction that proximity alone cannot make."""
    trk = Tracker(AssistConfig())
    for i, rng in enumerate([8.0, 7.0, 6.0, 5.0, 4.0, 3.0]):
        tracks = trk.update([_obs('person', (100, 100, 200, 400), rng)],
                            now=i * 0.5)

    t = tracks[0]
    assert t.closing_speed > 0.5
    assert np.isfinite(t.time_to_contact())
    assert t.time_to_contact() < 10.0


def test_static_object_has_infinite_time_to_contact():
    trk = Tracker(AssistConfig())
    for i in range(6):
        tracks = trk.update([_obs('person', (100, 100, 200, 400), 4.0)],
                            now=i * 0.5)

    assert tracks[0].closing_speed == pytest.approx(0.0, abs=0.05)
    assert not np.isfinite(tracks[0].time_to_contact())


def test_receding_object_is_not_treated_as_closing():
    trk = Tracker(AssistConfig())
    for i, rng in enumerate([2.0, 3.0, 4.0, 5.0, 6.0]):
        tracks = trk.update([_obs('person', (100, 100, 200, 400), rng)],
                            now=i * 0.5)

    assert tracks[0].closing_speed == 0.0
    assert not np.isfinite(tracks[0].time_to_contact())


def test_range_is_smoothed_against_a_single_bad_reading():
    trk = Tracker(AssistConfig())
    for i in range(4):
        trk.update([_obs('person', (100, 100, 200, 400), 4.0)], now=i * 0.1)
    tracks = trk.update([_obs('person', (100, 100, 200, 400), 12.0)], now=0.5)

    # a lone outlier must not teleport the track
    assert tracks[0].range_m < 9.0


def test_unconfirmed_track_is_flagged_until_a_second_hit():
    trk = Tracker(AssistConfig())
    tracks = trk.update([_obs('person', (100, 100, 200, 400), 5.0)], now=0.0)
    assert not tracks[0].confirmed

    tracks = trk.update([_obs('person', (100, 100, 200, 400), 5.0)], now=0.1)
    assert tracks[0].confirmed
