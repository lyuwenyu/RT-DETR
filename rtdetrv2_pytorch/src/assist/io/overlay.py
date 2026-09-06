"""Debug visualisation of what the guidance core is thinking.

The user of this system never sees a screen. This view exists entirely for
tuning: without being able to watch the corridor rays lie down on the real
floor, and the bird's-eye grid fill in beside them, every threshold in
config.py would have to be guessed at from how a walk felt.

Two panels. The camera image carries detections, the fitted ground plane's
horizon, and the corridor rays drawn where they actually fall on the floor. The
inset carries the top-down occupancy grid the planner truly reasons over.
"""

import math

import cv2
import numpy as np

_GREEN = (80, 230, 120)
_RED = (60, 60, 240)
_AMBER = (40, 190, 250)
_BLUE = (240, 180, 60)
_GREY = (150, 150, 150)
_WHITE = (255, 255, 255)

def draw(frame, out, intr, cfg, fps=None, timings=None, last_speech=None):
    """Annotate a copy of `frame` with everything the core decided."""
    vis = frame.copy()

    if out.plane is not None and out.corridor is not None:
        _draw_corridor(vis, out.corridor, out.plane, intr, cfg)

    _draw_tracks(vis, out)
    _draw_grid_inset(vis, out, cfg)
    _draw_status(vis, out, fps, timings, last_speech)
    return vis


def _draw_tracks(vis, out):
    hazard_by_id = {h.track_id: h for h in out.hazards if h.track_id is not None}

    for trk in out.tracks:
        if not trk.confirmed:
            continue
        x1, y1, x2, y2 = [int(v) for v in trk.box]
        hazard = hazard_by_id.get(trk.id)

        if hazard is None:
            color = _GREY
        elif hazard.zone == 'immediate':
            color = _RED
        elif hazard.zone == 'near':
            color = _AMBER
        else:
            color = _GREEN

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = '{} {:.1f}m'.format(trk.name, trk.range_m)
        if trk.closing_speed > 0.3:
            label += ' <{:.1f}m/s'.format(trk.closing_speed)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def _ground_basis(plane):
    """Forward and right unit vectors lying in the ground plane."""
    n = plane.normal
    fwd = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ n) * n
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return None, None, None
    fwd = fwd / norm
    right = np.cross(fwd, n)
    foot = -plane.d * n            # the camera's position projected onto the floor
    return foot, fwd, right


def _project(pt, intr, margin=4000):
    """Project a camera-frame point, or None if it is behind or far off-frame.

    Ground points close to the user project far below the image; keeping them
    would hand cv2 wild coordinates and draw lines across the whole frame.
    """
    if pt[2] <= 0.05:
        return None
    u = intr.fx * pt[0] / pt[2] + intr.cx
    v = intr.fy * pt[1] / pt[2] + intr.cy
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    if not (-margin < u < intr.width + margin
            and -margin < v < intr.height + margin):
        return None
    return int(round(u)), int(round(v))


def _draw_corridor(vis, corridor, plane, intr, cfg):
    """Lay the planner's rays down on the floor where they actually fall."""
    foot, fwd, right = _ground_basis(plane)
    if foot is None:
        return

    for i, theta in enumerate(corridor.bearings):
        reach = corridor.free[i]
        chosen = (i == corridor.heading_index)

        if corridor.blocked_by_dropoff[i]:
            color = (255, 80, 255)          # magenta: the drop-off case
        elif corridor.blocked[i]:
            color = _RED
        else:
            color = _GREEN

        pts = []
        for r in np.linspace(0.3, max(reach, 0.31), 14):
            p = foot + r * (math.cos(theta) * fwd + math.sin(theta) * right)
            uv = _project(p, intr)
            if uv is not None:
                pts.append(uv)

        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False,
                          color, 4 if chosen else 1, cv2.LINE_AA)
            if chosen:
                cv2.circle(vis, pts[-1], 7, _WHITE, 2, cv2.LINE_AA)


def _draw_grid_inset(vis, out, cfg, size=180):
    """Top-down view of the occupancy grid the planner reasons over."""
    h, w = vis.shape[:2]
    panel = np.full((size, size, 3), 28, dtype=np.uint8)

    if out.corridor is not None:
        n_z = int(round(cfg.grid_depth / cfg.grid_cell))
        n_x = int(round(cfg.grid_width / cfg.grid_cell))

        def to_px(x_m, z_m):
            px = int((x_m + cfg.grid_width / 2) / cfg.grid_width * size)
            py = int(size - (z_m / cfg.grid_depth) * size)
            return px, py

        # metre rings, so distances are readable at a glance
        for r in range(1, int(cfg.grid_depth) + 1):
            _, py = to_px(0, r)
            cv2.line(panel, (0, py), (size, py), (48, 48, 48), 1)

        for i, theta in enumerate(out.corridor.bearings):
            reach = out.corridor.free[i]
            chosen = (i == out.corridor.heading_index)
            if out.corridor.blocked_by_dropoff[i]:
                color = (255, 80, 255)
            elif out.corridor.blocked[i]:
                color = _RED
            else:
                color = _GREEN
            end = to_px(reach * math.sin(theta), reach * math.cos(theta))
            cv2.line(panel, to_px(0, 0), end, color, 3 if chosen else 1,
                     cv2.LINE_AA)

    cv2.circle(panel, (size // 2, size - 2), 4, _WHITE, -1)
    cv2.rectangle(panel, (0, 0), (size - 1, size - 1), (90, 90, 90), 1)
    cv2.putText(panel, 'top-down', (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                _GREY, 1, cv2.LINE_AA)

    y0, x0 = 8, w - size - 8
    if y0 + size <= h and x0 >= 0:
        vis[y0:y0 + size, x0:x0 + size] = panel


def _draw_status(vis, out, fps, timings, last_speech):
    lines = []
    if fps is not None:
        lines.append('{:.1f} fps'.format(fps))
    if timings:
        lines.append('det {det:.0f} / depth {depth:.0f} / guide {guide:.0f} ms'
                     .format(**timings))

    if out.plane is not None:
        lines.append('floor {:.2f} m, tilt {:.0f} deg'.format(
            out.plane.camera_height, out.plane.tilt_deg))
    else:
        lines.append('floor: NOT FOUND')

    lines.append('scale: {}'.format('calibrated' if out.metric_ok else 'estimating'))

    if out.corridor is not None:
        lines.append('heading {:+.0f} deg, {:.1f} m clear'.format(
            math.degrees(out.corridor.heading_bearing), out.corridor.heading_free))

    y = 22
    for text in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _WHITE, 1, cv2.LINE_AA)
        y += 20

    if last_speech:
        h = vis.shape[0]
        cv2.putText(vis, '" {} "'.format(last_speech), (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, '" {} "'.format(last_speech), (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BLUE, 2, cv2.LINE_AA)
