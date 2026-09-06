"""Bird's-eye occupancy grid built from a classified point cloud.

Obstacle and drop-off points are dropped onto a top-down grid of the floor in
front of the user. Occupied cells are then dilated by half a shoulder width, so
that asking "can I walk along this ray" reduces to marching a single-pixel ray
rather than sweeping a body-width corridor -- the standard configuration-space
trick, and the reason the planner downstream stays cheap.

Obstacle and drop-off are kept as separate layers all the way through: both
block the path, but they earn different warnings and different beep tones.
"""

import cv2
import numpy as np


class OccupancyGrid:
    """Top-down obstacle map, user at the bottom-centre facing +Z (up the grid)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cell = cfg.grid_cell
        self.n_x = int(round(cfg.grid_width / cfg.grid_cell))
        self.n_z = int(round(cfg.grid_depth / cfg.grid_cell))
        self.obstacle = np.zeros((self.n_z, self.n_x), dtype=np.uint8)
        self.dropoff = np.zeros((self.n_z, self.n_x), dtype=np.uint8)
        self.free_obstacle = self.obstacle
        self.free_dropoff = self.dropoff

        r = max(1, int(round(cfg.body_radius / cfg.grid_cell)))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (2 * r + 1, 2 * r + 1))

    # -- construction -------------------------------------------------------

    def build(self, points, obstacle_mask, dropoff_mask):
        """Rasterise classified points into dilated obstacle/drop-off layers."""
        self.obstacle[:] = 0
        self.dropoff[:] = 0

        self._rasterise(points[obstacle_mask], self.obstacle)
        self._rasterise(points[dropoff_mask], self.dropoff)

        # Dilate into configuration space: a cell is unsafe if the user's body
        # would overlap an obstacle when their centre is there.
        self.free_obstacle = cv2.dilate(self.obstacle, self._kernel)
        self.free_dropoff = cv2.dilate(self.dropoff, self._kernel)
        return self

    def _rasterise(self, pts, grid):
        if len(pts) == 0:
            return
        ix, iz = self.to_cell(pts[:, 0], pts[:, 2])
        ok = (ix >= 0) & (ix < self.n_x) & (iz >= 0) & (iz < self.n_z)
        grid[iz[ok], ix[ok]] = 1

    # -- coordinates --------------------------------------------------------

    def to_cell(self, x, z):
        """Metric (X lateral, Z forward) -> integer grid indices."""
        ix = np.floor((np.asarray(x) + self.cfg.grid_width / 2.0) / self.cell)
        iz = np.floor(np.asarray(z) / self.cell)
        return ix.astype(np.int32), iz.astype(np.int32)

    def blocked_at(self, x, z):
        """Is the dilated grid blocked at this metric position? -> (obs, drop)"""
        ix, iz = self.to_cell(x, z)
        if not (0 <= ix < self.n_x and 0 <= iz < self.n_z):
            return False, False
        return bool(self.free_obstacle[iz, ix]), bool(self.free_dropoff[iz, ix])
