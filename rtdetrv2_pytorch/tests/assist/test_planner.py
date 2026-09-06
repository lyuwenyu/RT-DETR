import numpy as np
import pytest

from src.assist.config import AssistConfig
from src.assist.occupancy import OccupancyGrid
from src.assist.planner import CorridorPlanner


def _pillar(x_centre, z, half_w=0.25, n=400, seed=0):
    """A column of obstacle points standing at (x_centre, z)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(x_centre - half_w, x_centre + half_w, n),
                     rng.uniform(-0.5, 0.5, n),
                     rng.uniform(z - 0.15, z + 0.15, n)], axis=1)


def _grid_with(cfg, obstacles=None, dropoffs=None):
    obstacles = obstacles if obstacles is not None else np.zeros((0, 3))
    dropoffs = dropoffs if dropoffs is not None else np.zeros((0, 3))
    pts = np.vstack([obstacles, dropoffs])
    obs_mask = np.zeros(len(pts), dtype=bool)
    drop_mask = np.zeros(len(pts), dtype=bool)
    obs_mask[:len(obstacles)] = True
    drop_mask[len(obstacles):] = True
    return OccupancyGrid(cfg).build(pts, obs_mask, drop_mask)


def test_empty_scene_is_clear_and_straight():
    cfg = AssistConfig()
    corridor = CorridorPlanner(cfg).plan(_grid_with(cfg))

    assert corridor.is_clear
    assert corridor.heading_bearing == pytest.approx(0.0)
    # nothing may be reported as blocked in an empty scene; outer rays still
    # terminate at the edge of the mapped area, which is not a hazard
    assert not corridor.blocked.any()
    assert corridor.free[cfg.n_rays // 2] >= cfg.clear_distance


def test_obstacle_dead_ahead_shortens_the_straight_ray():
    cfg = AssistConfig()
    corridor = CorridorPlanner(cfg).plan(_grid_with(cfg, _pillar(0.0, 2.0)))

    straight = corridor.free[cfg.n_rays // 2]
    assert straight < 2.1
    assert not corridor.is_clear


def test_heading_steers_around_a_central_obstacle():
    """The whole point of the corridor: pick a bearing that is actually open."""
    cfg = AssistConfig()
    planner = CorridorPlanner(cfg)
    grid = _grid_with(cfg, _pillar(0.0, 1.8))

    for _ in range(cfg.heading_switch_frames + 1):
        corridor = planner.plan(grid)

    assert abs(corridor.heading_bearing) > 0.1        # not straight any more
    assert corridor.heading_free > 3.0                # and genuinely open


def test_heading_prefers_the_open_side():
    cfg = AssistConfig()
    planner = CorridorPlanner(cfg)
    # wall of obstacles across the entire left half
    left_wall = np.vstack([_pillar(x, 2.0, seed=i)
                           for i, x in enumerate(np.arange(-1.9, 0.0, 0.2))])
    grid = _grid_with(cfg, left_wall)

    for _ in range(cfg.heading_switch_frames + 1):
        corridor = planner.plan(grid)

    assert corridor.heading_bearing > 0               # must go right


def test_dropoff_blocks_the_path_and_is_reported_as_such():
    cfg = AssistConfig()
    hole = _pillar(0.0, 2.5, half_w=0.6)
    corridor = CorridorPlanner(cfg).plan(_grid_with(cfg, dropoffs=hole))

    straight = cfg.n_rays // 2
    assert corridor.free[straight] < 2.6
    assert corridor.blocked[straight]
    assert corridor.blocked_by_dropoff[straight]      # distinct from an obstacle


def test_body_width_dilation_rejects_a_gap_too_narrow_to_walk_through():
    """Two pillars 0.4 m apart: a point ray fits, a person does not."""
    cfg = AssistConfig()
    gap = np.vstack([_pillar(-0.32, 2.0, half_w=0.2, seed=1),
                     _pillar(0.32, 2.0, half_w=0.2, seed=2)])
    corridor = CorridorPlanner(cfg).plan(_grid_with(cfg, gap))

    assert corridor.free[cfg.n_rays // 2] < 2.1


def test_wide_gap_stays_walkable():
    cfg = AssistConfig()
    gap = np.vstack([_pillar(-1.1, 2.0, half_w=0.2, seed=1),
                     _pillar(1.1, 2.0, half_w=0.2, seed=2)])
    corridor = CorridorPlanner(cfg).plan(_grid_with(cfg, gap))

    assert corridor.free[cfg.n_rays // 2] >= cfg.clear_distance


def test_heading_does_not_oscillate_between_equal_openings():
    """Two symmetric doorways must not make the system chatter left/right.

    Without hysteresis an argmax flips whenever noise tips the balance, and the
    guidance becomes unusable. Here the two sides swap advantage every frame.
    """
    cfg = AssistConfig()
    planner = CorridorPlanner(cfg)

    left = _grid_with(cfg, _pillar(0.55, 2.0, half_w=0.5, seed=1))
    right = _grid_with(cfg, _pillar(-0.55, 2.0, half_w=0.5, seed=2))

    headings = []
    for i in range(12):
        headings.append(planner.plan(left if i % 2 else right).heading_index)

    # at most one committed change across the whole alternating sequence
    switches = sum(a != b for a, b in zip(headings, headings[1:]))
    assert switches <= 1


def test_heading_commits_only_after_sustained_advantage():
    cfg = AssistConfig()
    planner = CorridorPlanner(cfg)
    # offset obstacle: straight ahead is blocked, the right side stays open
    scene = _grid_with(cfg, _pillar(-0.3, 1.8, half_w=0.5))

    first = planner.plan(scene).heading_index
    assert first == cfg.n_rays // 2               # holds straight initially

    for _ in range(cfg.heading_switch_frames - 1):
        planner.plan(scene)
    committed = planner.plan(scene).heading_index

    assert committed != cfg.n_rays // 2           # commits once sustained
    assert planner.bearings[committed] > 0        # toward the open side


def test_grid_cell_mapping_round_trips():
    cfg = AssistConfig()
    grid = OccupancyGrid(cfg)
    ix, iz = grid.to_cell(0.0, 0.0)
    assert (int(ix), int(iz)) == (grid.n_x // 2, 0)

    ix, iz = grid.to_cell(-cfg.grid_width / 2.0, cfg.grid_depth - 0.01)
    assert int(ix) == 0 and int(iz) == grid.n_z - 1


def test_out_of_bounds_is_not_blocked():
    cfg = AssistConfig()
    grid = _grid_with(cfg)
    assert grid.blocked_at(99.0, 99.0) == (False, False)


def test_dropoff_must_persist_before_it_is_confirmed():
    """A single noisy depth frame must not be able to shout 'stop'.

    The ray is still shortened immediately -- planning stays cautious -- but
    the announcement waits for evidence.
    """
    cfg = AssistConfig()
    planner = CorridorPlanner(cfg)
    hole = _grid_with(cfg, dropoffs=_pillar(0.0, 2.5, half_w=0.6))

    first = planner.plan(hole)
    assert first.blocked_by_dropoff.any()          # seen
    assert not first.dropoff_confirmed.any()       # but not yet believed
    assert first.free[cfg.n_rays // 2] < 2.6       # and already avoided

    for _ in range(cfg.dropoff_confirm_frames - 1):
        out = planner.plan(hole)
    assert out.dropoff_confirmed.any()


def test_transient_dropoff_never_gets_confirmed():
    """Alternating noise must never accumulate into a confirmation."""
    cfg = AssistConfig()
    planner = CorridorPlanner(cfg)
    hole = _grid_with(cfg, dropoffs=_pillar(0.0, 2.5, half_w=0.6))
    clear = _grid_with(cfg)

    for i in range(20):
        out = planner.plan(hole if i % 2 else clear)
        assert not out.dropoff_confirmed.any()


def test_confirmation_resets_once_the_dropoff_goes_away():
    cfg = AssistConfig()
    planner = CorridorPlanner(cfg)
    hole = _grid_with(cfg, dropoffs=_pillar(0.0, 2.5, half_w=0.6))
    for _ in range(cfg.dropoff_confirm_frames + 2):
        planner.plan(hole)

    out = planner.plan(_grid_with(cfg))
    assert not out.dropoff_confirmed.any()
