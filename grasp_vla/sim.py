"""PyBullet tabletop scene: a Franka Panda arm, a table, a tray and random objects.

Every scene is generated from a seed, so a trial can be replayed exactly.
Object identities from the simulator (segmentation mask, true poses) are used
only for scoring, never by the perception or planning code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import pybullet_data

# Colours as RGBA. Names are what instructions refer to.
COLORS = {
    "red": (0.85, 0.12, 0.12, 1),
    "green": (0.15, 0.65, 0.20, 1),
    "blue": (0.15, 0.30, 0.85, 1),
    "yellow": (0.95, 0.85, 0.10, 1),
    "purple": (0.55, 0.20, 0.70, 1),
    "orange": (0.98, 0.55, 0.10, 1),
}
SHAPES = ("cube", "cylinder", "ball")

TABLE_TOP_Z = 0.0          # table surface is at z = 0 (robot base sits on it)
WORKSPACE_X = (0.35, 0.65)  # region where objects are spawned (metres)
WORKSPACE_Y = (-0.22, 0.17)
TRAY_POS = (0.50, 0.42, TABLE_TOP_Z)  # tray spans y = 0.24..0.60

PANDA_ARM_JOINTS = list(range(7))
PANDA_FINGERS = (9, 10)
PANDA_EE_LINK = 11
PANDA_HOME = [0.0, -0.4, 0.0, -2.4, 0.0, 2.0, math.pi / 4]
# "Parked" pose: hand at (0.15, -0.35, 0.45), out of the overhead camera's view.
PANDA_PARK = [-0.3555, -0.7185, -0.5929, -2.2833, -0.3779, 1.6499, -0.0081]
PARK_POS = (0.15, -0.35, 0.45)


@dataclass
class ObjectSpec:
    color: str
    shape: str
    size: float  # half-extent (cube) or radius (cylinder, ball), metres

    @property
    def name(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass
class SceneObject:
    spec: ObjectSpec
    body_id: int
    initial_pos: tuple = field(default_factory=tuple)


class TabletopSim:
    """A headless PyBullet tabletop with a Panda arm."""

    def __init__(self, gui: bool = False, time_step: float = 1 / 240):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.time_step = time_step
        self.objects: list[SceneObject] = []
        self.frames: list[np.ndarray] | None = None  # set to a list to record video
        self._frame_every = 8
        self._step_count = 0
        # Sensor imperfections (set per scene by reset): Gaussian depth noise and
        # an unknown camera offset that the perception code is not told about.
        self.depth_noise = 0.0
        self.cam_offset = np.zeros(3)
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ scene
    def reset(self, seed: int, n_objects: int = 5, min_gap: float = 0.03,
              friction=(1.0, 1.0), depth_noise: float = 0.0,
              cam_offset_std: float = 0.0) -> list[SceneObject]:
        """Build a fresh random scene and return its objects.

        min_gap: clearance between neighbouring objects (m).
        friction: range for each object's friction coefficient.
        depth_noise: std of per-pixel depth noise (m).
        cam_offset_std: std of the true camera position error (m), i.e. extrinsic
            calibration error. Perception still assumes the nominal pose.
        """
        cid = self.client
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(self.time_step, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=80, physicsClientId=cid)
        rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(seed + 10_000)
        self.depth_noise = depth_noise
        self.cam_offset = rng.normal(0, cam_offset_std, 3) * np.array([1, 1, 0])

        # Table: a large thin box whose top is at z = 0.
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.02], physicsClientId=cid)
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.02],
                                        rgbaColor=[0.78, 0.70, 0.58, 1], physicsClientId=cid)
        self.table_id = p.createMultiBody(0, table_col, table_vis, [0.45, 0, -0.02], physicsClientId=cid)

        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True,
                                physicsClientId=cid)
        for j, q in zip(PANDA_ARM_JOINTS, PANDA_PARK):
            p.resetJointState(self.robot, j, q, physicsClientId=cid)
        p.setJointMotorControlArray(self.robot, PANDA_ARM_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=PANDA_PARK, forces=[240.0] * 7,
                                    physicsClientId=cid)
        for f in PANDA_FINGERS:
            p.resetJointState(self.robot, f, 0.04, physicsClientId=cid)
            p.changeDynamics(self.robot, f, lateralFriction=1.5, spinningFriction=0.01,
                             physicsClientId=cid)

        self.tray_id = p.loadURDF("tray/traybox.urdf", TRAY_POS, globalScaling=0.6,
                                  useFixedBase=True, physicsClientId=cid)

        self.objects = []
        placed: list[tuple[float, float, float]] = []
        combos = [(c, s) for c in COLORS for s in SHAPES]
        idx = rng.permutation(len(combos))[:n_objects]  # each (colour, shape) at most once
        for k in idx:
            color, shape = combos[k]
            size = float(rng.uniform(0.020, 0.028))
            spec = ObjectSpec(color, shape, size)
            for _ in range(500):  # rejection-sample a free spot
                x = rng.uniform(*WORKSPACE_X)
                y = rng.uniform(*WORKSPACE_Y)
                if all(math.hypot(x - px, y - py) > size + pr + min_gap for px, py, pr in placed):
                    break
            else:
                raise RuntimeError(f"could not place {n_objects} objects (seed {seed})")
            yaw = float(rng.uniform(-math.pi, math.pi))
            body = self._make_object(spec, (x, y), yaw, float(rng.uniform(*friction)),
                                     float(rng.uniform(0.05, 0.20)))
            placed.append((x, y, size))
            self.objects.append(SceneObject(spec, body, (x, y)))

        self.step(240)  # let everything settle
        return self.objects

    def _make_object(self, spec: ObjectSpec, xy, yaw: float, friction: float = 1.0,
                     mass: float = 0.08) -> int:
        cid = self.client
        s = spec.size
        rgba = COLORS[spec.color]
        if spec.shape == "cube":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s, s, s], physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s, s, s], rgbaColor=rgba, physicsClientId=cid)
            z = s
        elif spec.shape == "cylinder":
            h = 2.2 * s
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=s, height=h, physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=s, length=h, rgbaColor=rgba, physicsClientId=cid)
            z = h / 2
        else:  # ball
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=s, physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=s, rgbaColor=rgba, physicsClientId=cid)
            z = s
        body = p.createMultiBody(mass, col, vis, [xy[0], xy[1], TABLE_TOP_Z + z + 0.002],
                                 p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=cid)
        p.changeDynamics(body, -1, lateralFriction=friction, spinningFriction=0.005,
                         rollingFriction=0.001 if spec.shape == "ball" else 0.0,
                         physicsClientId=cid)
        return body

    # ---------------------------------------------------------------- physics
    def step(self, n: int = 1):
        for _ in range(n):
            p.stepSimulation(physicsClientId=self.client)
            self._step_count += 1
            if self.frames is not None and self._step_count % self._frame_every == 0:
                self.frames.append(self.render_view())

    def object_pos(self, body_id: int) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)[0])

    # ----------------------------------------------------------------- camera
    # Fixed overhead camera looking straight down at the workspace.
    CAM_W, CAM_H, CAM_FOV = 320, 320, 60.0
    CAM_EYE = (0.50, 0.0, 0.75)
    CAM_TARGET = (0.50, 0.0, 0.0)
    CAM_NEAR, CAM_FAR = 0.1, 2.0

    def camera_matrices(self):
        """Nominal (calibrated) camera matrices, as the perception code knows them."""
        view = p.computeViewMatrix(self.CAM_EYE, self.CAM_TARGET, [1, 0, 0])
        proj = p.computeProjectionMatrixFOV(self.CAM_FOV, self.CAM_W / self.CAM_H,
                                            self.CAM_NEAR, self.CAM_FAR)
        return view, proj

    def capture(self):
        """Return (rgb uint8 HxWx3, metric depth HxW, segmentation HxW)."""
        eye = np.add(self.CAM_EYE, self.cam_offset)
        target = np.add(self.CAM_TARGET, self.cam_offset)
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), [1, 0, 0])
        _, proj = self.camera_matrices()
        _, _, rgba, zbuf, seg = p.getCameraImage(
            self.CAM_W, self.CAM_H, view, proj, renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client)
        rgb = np.reshape(rgba, (self.CAM_H, self.CAM_W, 4))[..., :3].astype(np.uint8)
        zbuf = np.reshape(zbuf, (self.CAM_H, self.CAM_W))
        n, f = self.CAM_NEAR, self.CAM_FAR
        depth = f * n / (f - (f - n) * zbuf)
        if self.depth_noise > 0:
            depth = depth + self._rng.normal(0, self.depth_noise, depth.shape)
        seg = np.reshape(seg, (self.CAM_H, self.CAM_W))
        return rgb, depth, seg

    def deproject(self, depth: np.ndarray) -> np.ndarray:
        """Turn a depth image into an HxWx3 array of world-frame points."""
        view, proj = self.camera_matrices()
        V = np.array(view).reshape(4, 4, order="F")
        P = np.array(proj).reshape(4, 4, order="F")
        inv = np.linalg.inv(P @ V)
        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        x_ndc = (2 * (us + 0.5) / w) - 1
        y_ndc = 1 - (2 * (vs + 0.5) / h)
        n, f = self.CAM_NEAR, self.CAM_FAR
        z_ndc = (f + n) / (f - n) - 2 * f * n / ((f - n) * depth)
        pts = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(depth)], -1) @ inv.T
        return pts[..., :3] / pts[..., 3:4]

    def render_view(self, w: int = 480, h: int = 360) -> np.ndarray:
        """A three-quarter view for videos."""
        view = p.computeViewMatrix([1.15, -0.55, 0.65], [0.45, 0.05, 0.05], [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(50, w / h, 0.1, 3.0)
        _, _, rgba, _, _ = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                                            physicsClientId=self.client)
        return np.reshape(rgba, (h, w, 4))[..., :3].astype(np.uint8)

    def close(self):
        p.disconnect(self.client)
