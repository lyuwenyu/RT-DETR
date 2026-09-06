"""Tests for the audio output layer -- the parts that are pure logic."""

import math

import numpy as np
import pytest

from src.assist.arbiter import BeepState
from src.assist.io.sonifier import Sonifier, equal_power_pan, SAMPLE_RATE
from src.assist.io.speech import NullSpeaker, Speaker


# -- stereo panning ----------------------------------------------------------

def test_pan_extremes_and_centre():
    l, r = equal_power_pan(-1.0)
    assert l == pytest.approx(1.0) and r == pytest.approx(0.0, abs=1e-9)

    l, r = equal_power_pan(1.0)
    assert l == pytest.approx(0.0, abs=1e-9) and r == pytest.approx(1.0)

    l, r = equal_power_pan(0.0)
    assert l == pytest.approx(r)


def test_pan_preserves_power_across_the_sweep():
    """Equal power, so a hazard crossing the front does not dip in loudness."""
    for pan in np.linspace(-1, 1, 21):
        l, r = equal_power_pan(pan)
        assert l ** 2 + r ** 2 == pytest.approx(1.0)


# -- tone synthesis ----------------------------------------------------------

def test_inactive_beep_renders_silence():
    son = Sonifier()
    son.apply(BeepState(active=False))
    assert np.abs(son.render(2048)).max() == 0.0


def test_active_beep_renders_audio():
    son = Sonifier()
    son.apply(BeepState(active=True, interval=0.05, frequency=660.0))
    block = son.render(SAMPLE_RATE // 4)
    assert np.abs(block).max() > 0.01


def test_faster_interval_produces_more_beeps_per_second():
    """Beep rate is the distance cue, so it has to actually change."""
    def onsets(interval):
        son = Sonifier()
        son.apply(BeepState(active=True, interval=interval, frequency=660.0))
        env = np.abs(son.render(SAMPLE_RATE))          # one second
        loud = env.max(axis=1) > 0.01
        return int(np.sum(loud[1:] & ~loud[:-1]))

    assert onsets(0.15) > onsets(0.6)


def test_beep_pans_to_the_named_side():
    son = Sonifier()
    son.apply(BeepState(active=True, interval=0.05, frequency=660.0, pan=-1.0))
    block = son.render(SAMPLE_RATE // 4)
    assert np.abs(block[:, 0]).max() > np.abs(block[:, 1]).max()

    son.apply(BeepState(active=True, interval=0.05, frequency=660.0, pan=1.0))
    block = son.render(SAMPLE_RATE // 4)
    assert np.abs(block[:, 1]).max() > np.abs(block[:, 0]).max()


def test_output_stays_within_range():
    son = Sonifier(volume=1.0)
    son.apply(BeepState(active=True, interval=0.05, frequency=880.0))
    assert np.abs(son.render(SAMPLE_RATE // 2)).max() <= 1.0


# -- speech mailbox ----------------------------------------------------------

def test_more_urgent_phrase_replaces_a_waiting_one():
    """Single slot, not a queue: the newer, more urgent warning wins."""
    sp = Speaker(enabled=False)
    sp.say('bench, right, 4 metres', priority=1)
    sp.say('stop, step down', priority=3)

    assert sp._pending[1] == 'stop, step down'


def test_less_urgent_phrase_does_not_displace_a_waiting_one():
    sp = Speaker(enabled=False)
    sp.say('stop, step down', priority=3)
    assert not sp.say('bench, right, 4 metres', priority=1)
    assert sp._pending[1] == 'stop, step down'


def test_equal_priority_phrase_replaces_with_the_newer_state():
    """Between two equally urgent warnings, the fresher one is the true one."""
    sp = Speaker(enabled=False)
    sp.say('person, left', priority=2)
    sp.say('person, ahead', priority=2)
    assert sp._pending[1] == 'person, ahead'


def test_empty_phrase_is_ignored():
    sp = Speaker(enabled=False)
    assert not sp.say('')
    assert sp._pending is None


def test_null_speaker_records_without_audio():
    sp = NullSpeaker().start()
    sp.say('person, ahead')
    sp.say('bear left')
    sp.stop()
    assert sp.spoken == ['person, ahead', 'bear left']
