import math

import numpy as np
import pytest

from src.assist.arbiter import Arbiter
from src.assist.config import AssistConfig
from src.assist.hazards import Hazard, IMMEDIATE, NEAR, CONTEXT
from src.assist.phrasing import direction_word, distance_words, hazard_phrase
from src.assist.planner import Corridor


def _corridor(free=6.0, heading=0.0, clear=True, n=9):
    bearings = np.linspace(math.radians(-30), math.radians(30), n)
    return Corridor(bearings=bearings,
                    free=np.full(n, free),
                    blocked=np.zeros(n, dtype=bool),
                    blocked_by_dropoff=np.zeros(n, dtype=bool),
                    dropoff_confirmed=np.zeros(n, dtype=bool),
                    heading_index=int(np.argmin(np.abs(bearings - heading))),
                    heading_bearing=heading,
                    heading_free=free,
                    is_clear=clear)


def _hazard(name='person', zone=NEAR, rng=2.5, bearing=0.0, tid=1, urgency=5.0,
            kind='object', in_path=False):
    return Hazard(kind=kind, name=name, zone=zone, urgency=urgency,
                  range_m=rng, bearing=bearing, track_id=tid, in_path=in_path)


# -- the default state is silence -------------------------------------------

def test_clear_path_says_nothing():
    arb = Arbiter(AssistConfig())
    out = arb.update([], _corridor(clear=True), now=100.0)

    assert out.speech is None
    assert not out.beep.active


def test_distant_context_object_is_not_announced():
    """Terse mode: things 6 m away are not worth interrupting a walk for."""
    arb = Arbiter(AssistConfig())
    out = arb.update([_hazard(zone=CONTEXT, rng=6.0)], _corridor(), now=100.0)

    assert out.speech is None


# -- throttling and de-duplication ------------------------------------------

def test_same_object_is_announced_once_not_every_frame():
    arb = Arbiter(AssistConfig())
    hz = [_hazard(zone=NEAR, rng=2.5)]

    said = [arb.update(hz, _corridor(), now=100.0 + i * 0.1).speech
            for i in range(30)]

    assert sum(s is not None for s in said) == 1


def test_object_is_re_announced_when_it_gets_closer():
    """Escalation is news; repetition is not."""
    cfg = AssistConfig()
    arb = Arbiter(cfg)
    arb.update([_hazard(zone=NEAR, rng=2.5)], _corridor(), now=100.0)

    out = arb.update([_hazard(zone=IMMEDIATE, rng=1.2)], _corridor(), now=101.0)
    assert out.speech is not None


def test_repeat_only_after_the_cooldown():
    cfg = AssistConfig()
    arb = Arbiter(cfg)
    hz = [_hazard(zone=NEAR, rng=2.5)]
    arb.update(hz, _corridor(), now=100.0)

    assert arb.update(hz, _corridor(), now=100.0 + cfg.speech_repeat_cooldown - 1).speech is None
    assert arb.update(hz, _corridor(), now=100.0 + cfg.speech_repeat_cooldown + 1).speech is not None


def test_minimum_gap_between_utterances_is_respected():
    cfg = AssistConfig()
    arb = Arbiter(cfg)
    arb.update([_hazard(tid=1, zone=NEAR)], _corridor(), now=100.0)

    # a different object, but too soon to say anything
    out = arb.update([_hazard(tid=2, zone=NEAR)], _corridor(), now=100.3)
    assert out.speech is None

    out = arb.update([_hazard(tid=2, zone=NEAR)], _corridor(),
                     now=100.0 + cfg.speech_min_interval + 0.1)
    assert out.speech is not None


def test_immediate_hazard_may_cut_the_gap_short_but_not_to_zero():
    arb = Arbiter(AssistConfig())
    arb.update([_hazard(tid=1, zone=NEAR)], _corridor(), now=100.0)

    assert arb.update([_hazard(tid=2, zone=IMMEDIATE, rng=1.0)],
                      _corridor(), now=100.2).speech is None
    assert arb.update([_hazard(tid=2, zone=IMMEDIATE, rng=1.0)],
                      _corridor(), now=100.9).speech is not None


# -- drop-offs ---------------------------------------------------------------

def test_dropoff_outranks_an_ordinary_obstacle():
    arb = Arbiter(AssistConfig())
    drop = _hazard(kind='dropoff', name='step down', zone=IMMEDIATE,
                   rng=1.8, urgency=100.0, in_path=True)
    obj = _hazard(zone=IMMEDIATE, rng=1.0, urgency=9.0)

    out = arb.update([drop, obj], _corridor(), now=100.0)
    assert out.speech == 'stop, step down'


def test_dropoff_beside_the_path_informs_without_alarming():
    """Shouting 'stop' about ground the user is not walking toward is how a
    warning system teaches people to ignore it."""
    arb = Arbiter(AssistConfig())
    drop = _hazard(kind='dropoff', name='step down', zone=NEAR, rng=1.8,
                   bearing=math.radians(-30), urgency=20.0, in_path=False)

    out = arb.update([drop], _corridor(), now=100.0)
    assert out.speech == 'step down, left'
    assert 'stop' not in out.speech


def test_dropoff_gets_its_own_low_tone():
    cfg = AssistConfig()
    arb = Arbiter(cfg)
    drop = _hazard(kind='dropoff', zone=IMMEDIATE, rng=1.8, urgency=100.0)

    beep = arb.update([drop], _corridor(), now=100.0).beep
    assert beep.dropoff
    assert beep.frequency == cfg.beep_freq_dropoff
    assert beep.frequency < cfg.beep_freq_obstacle


# -- steering ----------------------------------------------------------------

def test_blocked_path_produces_a_steer():
    arb = Arbiter(AssistConfig())
    out = arb.update([], _corridor(free=1.5, heading=math.radians(25),
                                   clear=False), now=100.0)
    assert out.speech == 'bear right'


def test_steer_is_not_repeated_every_frame():
    cfg = AssistConfig()
    arb = Arbiter(cfg)
    cor = _corridor(free=1.5, heading=math.radians(-25), clear=False)

    said = [arb.update([], cor, now=100.0 + i * 0.5).speech for i in range(10)]
    assert said.count('bear left') <= 2


def test_no_steer_when_the_heading_is_already_straight():
    arb = Arbiter(AssistConfig())
    out = arb.update([], _corridor(free=1.5, heading=0.0, clear=False),
                     now=100.0)
    assert out.speech is None


# -- beeps -------------------------------------------------------------------

def test_beep_rate_rises_as_the_object_gets_closer():
    arb = Arbiter(AssistConfig())
    far = arb.update([_hazard(rng=3.5)], _corridor(), now=100.0).beep
    near = arb.update([_hazard(rng=0.8)], _corridor(), now=101.0).beep

    assert near.interval < far.interval


def test_beep_is_silent_beyond_the_alert_range():
    cfg = AssistConfig()
    arb = Arbiter(cfg)
    beep = arb.update([_hazard(rng=cfg.beep_range_far + 1.0)],
                      _corridor(), now=100.0).beep
    assert not beep.active


def test_beep_pans_toward_the_hazard():
    arb = Arbiter(AssistConfig())
    left = arb.update([_hazard(bearing=math.radians(-30))], _corridor(), now=100.0).beep
    right = arb.update([_hazard(bearing=math.radians(30))], _corridor(), now=101.0).beep

    assert left.pan < -0.3
    assert right.pan > 0.3


def test_beep_is_never_blocked_by_speech_throttling():
    """The whole reason beeps are a separate channel."""
    arb = Arbiter(AssistConfig())
    hz = [_hazard(zone=IMMEDIATE, rng=0.9)]
    arb.update(hz, _corridor(), now=100.0)

    out = arb.update(hz, _corridor(), now=100.05)
    assert out.speech is None          # speech correctly throttled
    assert out.beep.active             # but the warning still gets through


# -- phrasing ----------------------------------------------------------------

def test_direction_words_span_the_arc():
    assert direction_word(0.0) == 'ahead'
    assert direction_word(math.radians(-15)) == 'slightly left'
    assert direction_word(math.radians(15)) == 'slightly right'
    assert direction_word(math.radians(-35)) == 'left'
    assert direction_word(math.radians(35)) == 'right'


def test_distance_words_round_sensibly():
    assert distance_words(0.6) == 'very close'
    assert distance_words(1.0) == '1 metre'
    assert distance_words(2.0) == '2 metres'
    assert distance_words(2.4) == '2.5 metres'
    assert distance_words(5.2) == '5 metres'


def test_immediate_phrase_omits_the_distance():
    phrase = hazard_phrase(_hazard(zone=IMMEDIATE, rng=1.1, bearing=0.0))
    assert phrase == 'person, ahead'


def test_uncalibrated_scale_suppresses_spoken_distances():
    """Better vague than confidently wrong -- the user acts on what they hear."""
    phrase = hazard_phrase(_hazard(zone=NEAR, rng=2.5), metric_ok=False)
    assert 'metre' not in phrase
    assert phrase == 'person, ahead'


def test_calibrated_scale_includes_the_distance():
    phrase = hazard_phrase(_hazard(zone=NEAR, rng=2.5), metric_ok=True)
    assert phrase == 'person, ahead, 2.5 metres'


def test_uncalibrated_scale_does_not_unlock_emergency_preemption():
    """With no trustworthy range, everything can look IMMEDIATE at once.

    Regression test: on footage where calibration could not converge, every
    detection landed in the immediate zone and the 0.8 s preemption floor
    produced continuous chatter at a user who was in no danger.
    """
    cfg = AssistConfig()
    arb = Arbiter(cfg)
    arb.update([_hazard(tid=1, zone=IMMEDIATE, rng=1.0)], _corridor(),
               now=100.0, metric_ok=False)

    out = arb.update([_hazard(tid=2, zone=IMMEDIATE, rng=1.0)], _corridor(),
                     now=100.9, metric_ok=False)
    assert out.speech is None

    out = arb.update([_hazard(tid=2, zone=IMMEDIATE, rng=1.0)], _corridor(),
                     now=100.0 + cfg.speech_min_interval + 0.1, metric_ok=False)
    assert out.speech is not None


def test_dropoff_still_preempts_without_calibration():
    """Ground geometry does not depend on a ranged estimate being trusted."""
    arb = Arbiter(AssistConfig())
    arb.update([_hazard(tid=1, zone=NEAR)], _corridor(), now=100.0,
               metric_ok=False)

    drop = _hazard(kind='dropoff', name='step down', zone=IMMEDIATE, rng=1.5,
                   urgency=100.0, in_path=True)
    out = arb.update([drop], _corridor(), now=100.9, metric_ok=False)
    assert out.speech == 'stop, step down'
