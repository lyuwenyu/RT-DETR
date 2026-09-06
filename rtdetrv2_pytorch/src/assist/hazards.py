"""Turn tracks and the corridor into ranked hazards.

Two sources feed in. Tracks give named objects with a range, a bearing and a
closing speed. The corridor gives geometry-only hazards -- principally
drop-offs, which have no detection to attach to.

Ranking exists because a walker can absorb roughly one piece of information at
a time. The job here is not to describe the scene, it is to decide what the
single most important thing is right now.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

IMMEDIATE = 'immediate'
NEAR = 'near'
CONTEXT = 'context'
IGNORE = 'ignore'

_ZONE_RANK = {IMMEDIATE: 3, NEAR: 2, CONTEXT: 1, IGNORE: 0}


@dataclass
class Hazard:
    kind: str                  # 'object' or 'dropoff'
    name: str
    zone: str
    urgency: float
    range_m: float
    bearing: float
    ttc: float = float('inf')
    track_id: Optional[int] = None
    in_path: bool = False

    @property
    def is_dropoff(self):
        return self.kind == 'dropoff'


def zone_for(range_m, ttc, cfg):
    if range_m <= cfg.zone_immediate or ttc <= cfg.ttc_immediate:
        return IMMEDIATE
    if range_m <= cfg.zone_near:
        return NEAR
    if range_m <= cfg.zone_context:
        return CONTEXT
    return IGNORE


def assess(tracks, corridor, cfg):
    """Rank everything worth reacting to, most urgent first."""
    hazards = []

    for trk in tracks:
        if not trk.confirmed or trk.name not in cfg.nav_classes:
            continue
        if not np.isfinite(trk.range_m):
            continue

        ttc = trk.time_to_contact()
        zone = zone_for(trk.range_m, ttc, cfg)
        if zone == IGNORE:
            continue

        in_path = _in_path(trk.bearing, trk.range_m, corridor)
        hazards.append(Hazard(
            kind='object', name=trk.name, zone=zone,
            urgency=_urgency(trk.range_m, ttc, in_path,
                             cfg.nav_classes[trk.name], cfg),
            range_m=trk.range_m, bearing=trk.bearing, ttc=ttc,
            track_id=trk.id, in_path=in_path,
        ))

    hazards.extend(_dropoff_hazards(corridor, cfg))
    hazards.sort(key=lambda h: h.urgency, reverse=True)
    return hazards


def _dropoff_hazards(corridor, cfg):
    """Drop-offs, with severity set by whether the user is walking into one.

    A step down straight ahead earns an unconditional "stop". A step down off
    to one side is worth knowing about -- it is usually the edge of the path --
    but shouting "stop" about ground the user is not walking toward is a false
    alarm in the only sense that matters, and a system that does it stops being
    believed.
    """
    if corridor is None:
        return []

    mask = corridor.blocked & corridor.dropoff_confirmed
    if not mask.any():
        return []

    idx = int(np.argmin(np.where(mask, corridor.free, np.inf)))
    rng = float(corridor.free[idx])
    bearing = float(corridor.bearings[idx])

    if len(corridor.bearings) > 1:
        half_step = float(np.mean(np.diff(corridor.bearings)))
    else:
        half_step = 0.2
    aligned = abs(bearing - corridor.heading_bearing) <= max(half_step, 0.15)

    if aligned:
        return [Hazard(kind='dropoff', name='step down', zone=IMMEDIATE,
                       urgency=100.0 + (cfg.zone_context - rng),
                       range_m=rng, bearing=bearing, in_path=True)]

    # Off to the side: advisory only, and only while it is close enough to
    # matter. It still outranks ordinary objects at the same distance.
    if rng > cfg.zone_near:
        return []
    return [Hazard(kind='dropoff', name='step down', zone=NEAR,
                   urgency=20.0 + (cfg.zone_near - rng),
                   range_m=rng, bearing=bearing, in_path=False)]


def _urgency(range_m, ttc, in_path, class_weight, cfg):
    # Proximity dominates, time-to-contact escalates anything closing fast, and
    # being in the chosen corridor matters more than being off to one side.
    proximity = cfg.zone_context / max(range_m, 0.3)
    closing = 0.0 if not np.isfinite(ttc) else 3.0 * (cfg.ttc_immediate / max(ttc, 0.2))
    path_factor = 1.6 if in_path else 1.0
    return float((proximity + closing) * class_weight * path_factor)


def _in_path(bearing, range_m, corridor):
    """Does this object sit inside the corridor the user is being steered into?"""
    if corridor is None:
        return True
    half_step = float(np.mean(np.diff(corridor.bearings))) if len(corridor.bearings) > 1 else 0.2
    return abs(bearing - corridor.heading_bearing) <= max(half_step, 0.15)


def rank_of(zone):
    return _ZONE_RANK.get(zone, 0)
