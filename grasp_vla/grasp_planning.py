"""Clearance-aware top-down grasp planning.

The simple planner aligns the gripper with the object's bounding rectangle and
opens the fingers fully (4 cm each side). In clutter the open fingers can land
on a neighbouring object and the grasp fails. This planner:

1. opens the fingers only as wide as the object needs, plus a margin; and
2. tries several wrist angles and keeps the one whose finger footprints overlap
   the fewest obstacle points in the height map.

For boxes only the two angles aligned with the faces are tried (a diagonal
grasp on a cube is unstable); round objects can be grasped at any angle.
"""
from __future__ import annotations

import math

import numpy as np

from .perception import MIN_HEIGHT, Detection
from .sim import TabletopSim

MAX_OPEN = 0.04         # Panda: 4 cm per finger
OPEN_MARGIN = 0.012     # clearance between pad and object when open (m)
FINGER_THICK = 0.022    # finger extent along the closing axis (m)
FINGER_HALF_W = 0.012   # half-width of a finger across the closing axis (m)
ROUND_FILL = 0.85       # below this fill ratio the footprint is treated as round


def grasp_yaw(det: Detection) -> float:
    """Wrist yaw aligned with the object's bounding rectangle, wrapped into (-pi/4, pi/4]."""
    yaw = det.yaw
    while yaw > math.pi / 4:
        yaw -= math.pi / 2
    while yaw <= -math.pi / 4:
        yaw += math.pi / 2
    return yaw


def finger_collisions(points_xy: np.ndarray, centre: np.ndarray, yaw: float, opening: float) -> int:
    """Number of obstacle points inside the two open-finger footprints."""
    axis = np.array([-math.sin(yaw), math.cos(yaw)])     # closing direction
    perp = np.array([math.cos(yaw), math.sin(yaw)])
    rel = points_xy - centre
    a = rel @ axis
    b = rel @ perp
    in_finger = (np.abs(a) > opening - 0.003) & (np.abs(a) < opening + FINGER_THICK) & (np.abs(b) < FINGER_HALF_W)
    return int(in_finger.sum())


def plan_grasp(sim: TabletopSim, depth: np.ndarray, det: Detection) -> tuple[float, float, dict]:
    """Return (yaw, finger opening, diagnostics) for a top-down grasp on `det`."""
    pts = sim.deproject(depth)
    obstacle = (pts[..., 2] > MIN_HEIGHT) & ~det.pixels
    obs_xy = pts[obstacle][:, :2]
    opening = min(MAX_OPEN, det.width / 2 + OPEN_MARGIN)
    base = grasp_yaw(det)
    if det.fill_ratio < ROUND_FILL:
        candidates = [base + math.radians(d) for d in range(0, 180, 15)]
    else:
        candidates = [base, base + math.pi / 2]
    scored = [(finger_collisions(obs_xy, det.centroid[:2], y, opening), abs(_wrap(y)), _wrap(y))
              for y in candidates]
    hits, _, yaw = min(scored)
    return yaw, opening, {"collision_points": hits,
                          "baseline_collision_points": finger_collisions(obs_xy, det.centroid[:2], base, MAX_OPEN)}


def _wrap(yaw: float) -> float:
    """Wrap into (-pi/2, pi/2]; the gripper is symmetric under a half turn."""
    while yaw > math.pi / 2:
        yaw -= math.pi
    while yaw <= -math.pi / 2:
        yaw += math.pi
    return yaw
