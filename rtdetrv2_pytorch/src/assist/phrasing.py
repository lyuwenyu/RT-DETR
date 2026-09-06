"""Turn hazards and headings into the short phrases the user actually hears.

Everything here optimises for time-to-understand, not grammar. A spoken
sentence costs well over a second, during which the user keeps walking, so
phrases are built from a small fixed vocabulary in a fixed order:

    <what> <where> <how far>        "person, left, 2 metres"

Fixed word order matters more than it looks: it lets a regular user parse the
first word and stop listening. Distances are only spoken when the depth scale
is actually calibrated -- a confidently wrong "3 metres" is worse than the
vaguer "close", because the user will trust it and step accordingly.
"""

import math

from .hazards import IMMEDIATE


def direction_word(bearing_rad):
    deg = math.degrees(bearing_rad)
    if abs(deg) < 8.0:
        return 'ahead'
    side = 'right' if deg > 0 else 'left'
    if abs(deg) < 22.0:
        return 'slightly ' + side
    return side


def distance_words(range_m):
    """Spoken distance, rounded to a precision the estimate can support."""
    if range_m < 1.0:
        return 'very close'
    if range_m < 3.0:
        half = round(range_m * 2) / 2.0
        if half == int(half):
            n = int(half)
            return '1 metre' if n == 1 else '{} metres'.format(n)
        return '{:.1f} metres'.format(half)
    return '{} metres'.format(int(round(range_m)))


def hazard_phrase(hazard, metric_ok=True):
    """The phrase for a single hazard, terse in proportion to its urgency."""
    if hazard.is_dropoff:
        if hazard.in_path:
            # Directly ahead: no name, no distance, no hedging -- it has to land fast.
            return 'stop, step down'
        # Beside the path: inform, do not alarm.
        return 'step down, {}'.format(direction_word(hazard.bearing))

    where = direction_word(hazard.bearing)

    if hazard.zone == IMMEDIATE:
        # At this range the distance number is stale before it is spoken.
        return '{}, {}'.format(hazard.name, where)

    if not metric_ok:
        return '{}, {}'.format(hazard.name, where)

    return '{}, {}, {}'.format(hazard.name, where, distance_words(hazard.range_m))


def steering_phrase(corridor):
    """A steer only when there is somewhere to steer to."""
    deg = math.degrees(corridor.heading_bearing)
    if abs(deg) < 8.0:
        return None
    return 'bear right' if deg > 0 else 'bear left'
