"""Decide what the user hears right now, across two very different channels.

This is where a technically correct system becomes a usable one. The pipeline
upstream can produce a ranked hazard list every frame; saying all of it, or even
just the top of it every frame, would be unbearable within a minute.

Two channels, deliberately independent:

  * Beeps carry continuous proximity. They are never queued and never blocked,
    because a spoken warning about something 80 cm away arrives after the user
    has already walked into it. Rate encodes range, stereo pan encodes bearing,
    and drop-offs get a distinctly low tone so they are never mistaken for an
    ordinary obstacle.

  * Speech carries identity, and is rationed hard. One utterance at a time, a
    minimum gap between them, and a single-slot mailbox rather than a queue --
    if something more urgent arrives it replaces what was waiting instead of
    lining up behind it. A backlog of stale warnings is worse than silence.

Silence is the default and correct state. When the path is clear the system
says nothing at all; the beep channel already conveys that nothing is near.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from .hazards import IMMEDIATE, NEAR, rank_of
from .phrasing import hazard_phrase, steering_phrase


@dataclass
class BeepState:
    active: bool = False
    interval: float = 1.0      # seconds between beeps
    frequency: float = 660.0
    pan: float = 0.0           # -1 fully left, +1 fully right
    dropoff: bool = False


@dataclass
class Announcement:
    speech: Optional[str] = None
    priority: int = 0
    beep: BeepState = field(default_factory=BeepState)


class Arbiter:
    def __init__(self, cfg):
        self.cfg = cfg
        self._last_speech_time = -1e9
        self._last_steer_time = -1e9
        self._last_steer_phrase = None
        self._spoken = {}          # track_id (or 'dropoff') -> (zone, time)

    def update(self, hazards, corridor, now, metric_ok=True):
        beep = self._beep_for(hazards[0] if hazards else None)
        speech, priority = self._speech_for(hazards, corridor, now, metric_ok)
        if speech is not None:
            self._last_speech_time = now
        return Announcement(speech=speech, priority=priority, beep=beep)

    # -- beeps --------------------------------------------------------------

    def _beep_for(self, hazard):
        cfg = self.cfg
        if hazard is None or hazard.range_m > cfg.beep_range_far:
            return BeepState(active=False)

        # Linear in range between the near and far anchors, so the acceleration
        # of the beeping is itself the distance cue.
        span = max(cfg.beep_range_far - cfg.beep_range_near, 1e-6)
        t = (hazard.range_m - cfg.beep_range_near) / span
        t = min(max(t, 0.0), 1.0)
        interval = cfg.beep_min_interval + t * (cfg.beep_max_interval - cfg.beep_min_interval)

        max_bearing = math.radians(45.0)
        pan = max(-1.0, min(1.0, math.sin(hazard.bearing) / math.sin(max_bearing)))

        return BeepState(
            active=True,
            interval=interval,
            frequency=cfg.beep_freq_dropoff if hazard.is_dropoff else cfg.beep_freq_obstacle,
            pan=pan,
            dropoff=hazard.is_dropoff,
        )

    # -- speech -------------------------------------------------------------

    def _speech_for(self, hazards, corridor, now, metric_ok):
        cfg = self.cfg
        since = now - self._last_speech_time

        for hazard in hazards:
            if not self._worth_saying(hazard, now):
                continue
            # An immediate hazard may cut the normal gap short, but not to zero:
            # rapid-fire warnings stop being parseable.
            #
            # That shortcut is only earned when the distances behind it are
            # trustworthy. With no scale calibration -- nothing of known height
            # in view -- ranges are arbitrary, so everything can look
            # IMMEDIATE at once and the system chatters continuously at a user
            # who is in no danger. When scale is unknown, fall back to the
            # normal spacing; a drop-off, which comes from ground geometry
            # rather than from a range estimate, still gets through first by
            # urgency.
            urgent = hazard.zone == IMMEDIATE and (metric_ok or hazard.is_dropoff)
            floor = 0.8 if urgent else cfg.speech_min_interval
            if since < floor:
                continue
            self._mark_spoken(hazard, now)
            return hazard_phrase(hazard, metric_ok=metric_ok), rank_of(hazard.zone)

        if since < cfg.speech_min_interval:
            return None, 0

        return self._steer(corridor, now)

    def _worth_saying(self, hazard, now):
        cfg = self.cfg
        if hazard.is_dropoff:
            # Not tracked, so throttle by time alone.
            last = self._spoken.get('dropoff')
            return last is None or (now - last[1]) >= cfg.speech_min_interval

        if hazard.zone not in (IMMEDIATE, NEAR):
            return False

        prev = self._spoken.get(hazard.track_id)
        if prev is None:
            return True
        zone, when = prev
        if rank_of(hazard.zone) > rank_of(zone):      # got closer: say it again
            return True
        return (now - when) >= cfg.speech_repeat_cooldown

    def _mark_spoken(self, hazard, now):
        key = 'dropoff' if hazard.is_dropoff else hazard.track_id
        self._spoken[key] = (hazard.zone, now)

    def _steer(self, corridor, now):
        cfg = self.cfg
        if corridor is None or corridor.is_clear:
            # Path is open: say nothing at all.
            self._last_steer_phrase = None
            return None, 0

        phrase = steering_phrase(corridor)
        if phrase is None:
            return None, 0
        if phrase == self._last_steer_phrase and \
                (now - self._last_steer_time) < cfg.steer_cooldown:
            return None, 0

        self._last_steer_time = now
        self._last_steer_phrase = phrase
        return phrase, 1

    def forget(self, track_ids):
        """Drop announcement memory for tracks that no longer exist."""
        alive = set(track_ids) | {'dropoff'}
        self._spoken = {k: v for k, v in self._spoken.items() if k in alive}
