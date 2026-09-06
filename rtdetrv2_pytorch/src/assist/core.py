"""The guidance core: detections and a depth map in, guidance out.

`GuidanceCore.update()` is pure in the sense that matters: it opens no camera,
touches no audio device, writes no file and blocks on nothing. It is a function
from (what the detector saw, what the depth net saw, what time it is) to a
description of what the user should hear.

That boundary is deliberate and is the main architectural decision in this
package. It makes the interesting behaviour -- ranging, drop-off detection,
corridor choice, speech rationing -- testable without hardware, and it means the
eventual Android port is a transcription of one module rather than an
archaeology expedition through code with a microphone wired into it.

Both inputs are expressed in the frame's own pixel coordinates: boxes come from
the postprocessor already scaled to the original frame, and `depth_rel` is
expected at frame resolution too, so the caller is responsible for resizing the
depth network's output. Keeping one coordinate system here avoids a whole class
of off-by-a-resize bugs.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .arbiter import Arbiter, Announcement
from .calibration import ScaleCalibrator
from .geometry import Intrinsics, box_interior, box_centre_bearing
from .ground import GroundEstimator
from .hazards import assess
from .occupancy import OccupancyGrid
from .planner import CorridorPlanner
from .tracking import Tracker


@dataclass
class GuidanceOutput:
    announcement: Announcement
    hazards: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    corridor: Optional[object] = None
    plane: Optional[object] = None
    depth_m: Optional[np.ndarray] = None
    metric_ok: bool = False

    @property
    def speech(self):
        return self.announcement.speech

    @property
    def beep(self):
        return self.announcement.beep


class GuidanceCore:
    def __init__(self, cfg, intrinsics: Intrinsics):
        self.cfg = cfg
        self.intr = intrinsics
        self.calibrator = ScaleCalibrator(cfg)
        self.ground = GroundEstimator(cfg)
        self.grid = OccupancyGrid(cfg)
        self.planner = CorridorPlanner(cfg, hfov_deg=cfg.hfov_deg)
        self.tracker = Tracker(cfg)
        self.arbiter = Arbiter(cfg)

    def update(self, detections, depth_rel, now) -> GuidanceOutput:
        cfg = self.cfg
        intr = self.intr

        # 1. Let the detector calibrate the depth net's unknown scale, then
        #    read the map in metres.
        self.calibrator.observe(detections, depth_rel, intr, now)
        depth_m = self.calibrator.to_metric(depth_rel)
        metric_ok = self.calibrator.is_fresh(now)

        # 2. Lift to 3D and find the floor the user is standing on.
        points, _, vs = intr.backproject(depth_m, stride=cfg.backproject_stride)
        plane = self.ground.update(points, vs, intr.height)

        # 3. Where can they walk? Only answerable once the floor is known.
        corridor = None
        if plane is not None:
            _, obstacle, dropoff, _ = self.ground.classify(points, plane)
            self.grid.build(points, obstacle, dropoff)
            corridor = self.planner.plan(self.grid)

        # 4. Range and bearing every navigation-relevant detection.
        tracks = self.tracker.update(self._observe(detections, depth_m), now)
        self.arbiter.forget([t.id for t in tracks])

        # 5. Rank, then decide what is actually worth saying.
        hazards = assess(tracks, corridor, cfg)
        announcement = self.arbiter.update(hazards, corridor, now,
                                           metric_ok=metric_ok)

        return GuidanceOutput(announcement=announcement, hazards=hazards,
                              tracks=tracks, corridor=corridor, plane=plane,
                              depth_m=depth_m, metric_ok=metric_ok)

    def _observe(self, detections, depth_m):
        """(Detection, range, bearing) for everything worth tracking."""
        out = []
        for det in detections:
            if det.name not in self.cfg.nav_classes:
                continue
            rng = self._range_of(det, depth_m)
            if not np.isfinite(rng):
                continue
            out.append((det, rng, box_centre_bearing(det.box, self.intr)))
        return out

    def _range_of(self, det, depth_m):
        """Distance to a detection, from the metric depth map.

        The median over the box's central region rather than its full extent:
        depth bleeds in from the background around an object's silhouette, and
        a mean would be dragged outward by it.
        """
        sl = box_interior(det.box, depth_m.shape)
        if sl is None:
            return float('inf')
        patch = depth_m[sl]
        if patch.size == 0:
            return float('inf')
        return float(np.median(patch))
