"""Low-level Panda control: inverse kinematics, Cartesian moves and the gripper."""
from __future__ import annotations

import math

import numpy as np
import pybullet as p

from .sim import PANDA_ARM_JOINTS, PANDA_EE_LINK, PANDA_FINGERS, PANDA_PARK, TabletopSim

# The Panda URDF in pybullet_data has its hand frame offset from the fingertip
# centre; this is the distance from link 11 to the point between the pads.
FINGERTIP_OFFSET = 0.0


class PandaController:
    def __init__(self, sim: TabletopSim):
        self.sim = sim
        self.cid = sim.client
        self.robot = sim.robot
        lower, upper, ranges = [], [], []
        for j in PANDA_ARM_JOINTS:
            info = p.getJointInfo(self.robot, j, physicsClientId=self.cid)
            lower.append(info[8]); upper.append(info[9]); ranges.append(info[9] - info[8])
        self.lower, self.upper, self.ranges = lower, upper, ranges
        self.rest = [0.0, -0.4, 0.0, -2.4, 0.0, 2.0, math.pi / 4]
        # hold the parked pose until the first move
        p.setJointMotorControlArray(self.robot, PANDA_ARM_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=PANDA_PARK, forces=[240.0] * 7,
                                    physicsClientId=self.cid)
        self.gripper_target = 0.04

    def ee_pos(self) -> np.ndarray:
        return np.array(p.getLinkState(self.robot, PANDA_EE_LINK, physicsClientId=self.cid)[4])

    def _ik(self, pos, yaw):
        orn = p.getQuaternionFromEuler([math.pi, 0.0, yaw])
        q = p.calculateInverseKinematics(
            self.robot, PANDA_EE_LINK, pos, orn,
            lowerLimits=self.lower, upperLimits=self.upper, jointRanges=self.ranges,
            restPoses=self.rest, maxNumIterations=200, residualThreshold=1e-5,
            physicsClientId=self.cid)
        return q[:7]

    def move_to(self, pos, yaw: float, steps: int = 240, max_force: float = 240.0) -> float:
        """Move the hand to `pos` with a downward-facing gripper rotated by `yaw`.

        The target is interpolated in Cartesian space so the hand travels in a
        straight line. Returns the final position error in metres.
        """
        start = self.ee_pos()
        goal = np.asarray(pos, dtype=float)
        n_way = max(2, int(np.linalg.norm(goal - start) / 0.01))
        per = max(1, steps // n_way)
        for k in range(1, n_way + 1):
            target = start + (goal - start) * k / n_way
            q = self._ik(target, yaw)
            p.setJointMotorControlArray(self.robot, PANDA_ARM_JOINTS, p.POSITION_CONTROL,
                                        targetPositions=q, forces=[max_force] * 7,
                                        positionGains=[0.08] * 7, physicsClientId=self.cid)
            self._hold_gripper()
            self.sim.step(per)
        # settle at the goal
        for _ in range(60):
            self._hold_gripper()
            self.sim.step(1)
        return float(np.linalg.norm(self.ee_pos() - goal))

    def _hold_gripper(self, force: float = 60.0):
        for f in PANDA_FINGERS:
            p.setJointMotorControl2(self.robot, f, p.POSITION_CONTROL, self.gripper_target,
                                    force=force, physicsClientId=self.cid)

    def open_gripper(self, width: float = 0.04, steps: int = 120):
        self.gripper_target = width
        for _ in range(steps):
            self._hold_gripper()
            self.sim.step(1)

    def close_gripper(self, steps: int = 180) -> float:
        """Close the fingers and return the final opening (metres per finger)."""
        self.gripper_target = 0.0
        for _ in range(steps):
            self._hold_gripper(force=80.0)
            self.sim.step(1)
        return self.finger_width()

    def finger_width(self) -> float:
        return sum(p.getJointState(self.robot, f, physicsClientId=self.cid)[0] for f in PANDA_FINGERS)
