"""Tunable parameters for the blind-assistance guidance core.

Every magic number the guidance pipeline depends on lives here, so that field
tuning is a matter of editing one dataclass rather than hunting through modules.
Distances are metres, angles are degrees at the config boundary (radians
internally), and times are seconds.
"""

from dataclasses import dataclass, field


# Real-world heights (metres) of COCO classes tall enough, and rigid enough, to
# serve as pinhole ranging references. Used to calibrate the depth net's scale.
# Deliberately conservative: only classes whose height varies little in practice.
KNOWN_HEIGHTS = {
    'person':     1.70,
    'bicycle':    1.10,
    'car':        1.50,
    'motorcycle': 1.20,
    'bus':        3.00,
    'truck':      3.20,
    'chair':      0.85,
    'dog':        0.50,
    'stop sign':  2.10,
    'parking meter': 1.20,
    'fire hydrant':  0.75,
}

# Classes worth reporting to a walker, with a priority weight that biases the
# urgency score. Anything not listed is detected but never announced -- a
# 'toothbrush' is not navigation-relevant even at 1 metre.
NAV_CLASSES = {
    'person':        1.30,
    'bicycle':       1.20,
    'car':           1.50,
    'motorcycle':    1.40,
    'bus':           1.50,
    'truck':         1.50,
    'train':         1.50,
    'traffic light': 0.60,
    'stop sign':     0.60,
    'fire hydrant':  1.00,
    'parking meter': 1.00,
    'bench':         1.00,
    'chair':         1.00,
    'couch':         1.00,
    'bed':           1.00,
    'dining table':  1.00,
    'toilet':        0.80,
    'tv':            0.70,
    'potted plant':  0.90,
    'refrigerator':  1.00,
    'suitcase':      1.00,
    'backpack':      0.90,
    'dog':           1.20,
    'cat':           1.00,
    'horse':         1.30,
}


@dataclass
class AssistConfig:
    # --- detection ---------------------------------------------------------
    score_threshold: float = 0.45     # below this a detection is discarded
    calib_score_threshold: float = 0.60   # stricter bar for calibration refs

    # --- camera ------------------------------------------------------------
    hfov_deg: float = 65.0            # horizontal field of view of the camera
    backproject_stride: int = 4       # subsample the depth map when lifting to 3D

    # --- depth scale calibration ------------------------------------------
    calib_window: int = 150           # rolling (d_inv, 1/Z) sample budget
    calib_min_samples: int = 5        # below this, fall back to defaults
    calib_ransac_iters: int = 80
    calib_ransac_tol: float = 0.06    # inlier tolerance in inverse-depth units
    calib_min_spread: float = 0.25    # d-range needed before fitting an intercept
    calib_fallback_a: float = 1.0     # 1/Z = a*d + b before calibration exists
    calib_fallback_b: float = 0.0
    calib_max_age: float = 5.0        # seconds a fit stays trusted without refresh

    # --- ground plane ------------------------------------------------------
    ground_seed_frac: float = 0.25    # bottom fraction of rows that seeds the fit
    ground_ransac_iters: int = 60
    ground_score_samples: int = 2000  # cap on points used to rank RANSAC planes
    ground_inlier_tol: float = 0.05   # metres from plane to count as floor
    ground_min_inlier_frac: float = 0.25
    ground_max_tilt_deg: float = 30.0 # reject planes not roughly horizontal
    ground_min_cam_height: float = 0.8
    ground_max_cam_height: float = 1.8
    floor_band: float = 0.12          # |h| below this is walkable floor
    obstacle_height: float = 0.15     # h above this sticks up -> obstacle
    dropoff_depth: float = 0.15       # h below this -> drop-off / step down
    dropoff_confirm_frames: int = 3   # consecutive frames before announcing one

    # --- occupancy grid ----------------------------------------------------
    grid_width: float = 4.0           # metres, lateral extent (centred on user)
    grid_depth: float = 6.0           # metres, forward extent
    grid_cell: float = 0.10           # metres per cell
    body_radius: float = 0.35         # half a shoulder width; obstacle dilation

    # --- corridor planner --------------------------------------------------
    n_rays: int = 9
    ray_span_deg: float = 60.0        # total spread, clamped to camera FOV
    heading_straight_bonus: float = 0.8   # metres of credit for going straight
    heading_switch_margin: float = 0.5    # challenger must beat by this
    heading_switch_frames: int = 3        # ...for this many consecutive frames
    clear_distance: float = 4.0       # free distance treated as "path is clear"

    # --- tracking ----------------------------------------------------------
    track_iou_threshold: float = 0.3
    track_max_misses: int = 5
    track_min_hits: int = 2           # frames before a track may be announced
    range_ema_alpha: float = 0.5

    # --- hazard zones ------------------------------------------------------
    zone_immediate: float = 1.5       # metres
    zone_near: float = 3.0
    zone_context: float = 8.0
    ttc_immediate: float = 1.5        # seconds to contact -> IMMEDIATE

    # --- speech arbitration ------------------------------------------------
    speech_min_interval: float = 2.0  # max one utterance per this many seconds
    speech_repeat_cooldown: float = 8.0   # re-announce same track after this
    steer_cooldown: float = 3.0       # min gap between steering cues

    # --- beeps -------------------------------------------------------------
    beep_max_interval: float = 1.0    # seconds between beeps at far range
    beep_min_interval: float = 0.10   # ...and at touching distance
    beep_range_far: float = 4.0
    beep_range_near: float = 0.5
    beep_freq_obstacle: float = 660.0
    beep_freq_dropoff: float = 220.0  # distinctly low: never confusable

    known_heights: dict = field(default_factory=lambda: dict(KNOWN_HEIGHTS))
    nav_classes: dict = field(default_factory=lambda: dict(NAV_CLASSES))
