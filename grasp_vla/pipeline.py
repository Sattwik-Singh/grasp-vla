"""End-to-end episode: instruction -> plan -> detect -> ground -> grasp -> place.

The robot decides everything from the camera image and the instruction. The
simulator's ground truth is read only afterwards, by `score_*`, to label the
outcome of each step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from .control import PandaController
from .language import Step, parse
from .perception import Detection, detect
from .sim import PARK_POS, TRAY_POS, TabletopSim

LIFT_HEIGHT = 0.10        # object must rise this far above its start to count as lifted
HOLD_STEPS = 240          # and stay up for one simulated second
TRAY_DROP = (TRAY_POS[0], TRAY_POS[1], 0.28)

OUTCOMES = ("success", "parse_fail", "no_detection", "wrong_object",
            "grasp_miss", "drop", "place_miss")


@dataclass
class StepResult:
    target: str
    outcome: str
    chosen_body: int | None = None
    lift: float = 0.0
    info: dict = field(default_factory=dict)


def grasp_yaw(det: Detection) -> float:
    """Wrist yaw for a top-down grasp, wrapped into (-pi/4, pi/4]."""
    yaw = det.yaw
    while yaw > math.pi / 4:
        yaw -= math.pi / 2
    while yaw <= -math.pi / 4:
        yaw += math.pi / 2
    return yaw


def execute_grasp(ctrl: PandaController, det: Detection, approach: float = 0.12) -> float:
    """Pick the detected object and lift it. Returns finger opening after closing."""
    yaw = grasp_yaw(det)
    x, y, z = det.centroid
    ctrl.open_gripper()
    ctrl.move_to([x, y, det.top_z + approach], yaw)
    ctrl.move_to([x, y, z], yaw, steps=180)
    width = ctrl.close_gripper()
    ctrl.move_to([x, y, z + 0.22], yaw, steps=240)
    return width


def matches(step: Step, spec) -> bool:
    return ((step.color is None or spec.color == step.color) and
            (step.shape is None or spec.shape == step.shape))


def nearest_object(sim: TabletopSim, det: Detection):
    return min(sim.objects, key=lambda o: np.linalg.norm(sim.object_pos(o.body_id)[:2] - det.centroid[:2]))


def in_tray(pos) -> bool:
    return abs(pos[0] - TRAY_POS[0]) < 0.17 and abs(pos[1] - TRAY_POS[1]) < 0.17 and pos[2] < 0.12


def run_episode(sim: TabletopSim, instruction: str, grounder, planner=parse,
                truth_step: list[Step] | None = None,
                split_by_color: bool = True) -> list[StepResult]:
    """Run one instruction. `truth_step` (optional) is the intended plan, used for
    scoring when the planner itself is being tested."""
    ctrl = PandaController(sim)
    steps = planner(instruction)
    if not steps:
        return [StepResult(instruction, "parse_fail")]
    results = []
    for i, step in enumerate(steps):
        intended = truth_step[i] if truth_step and i < len(truth_step) else step
        if i > 0:
            ctrl.move_to(PARK_POS, 0.0)  # get out of the camera's view
        rgb, depth, _ = sim.capture()
        dets = detect(sim, rgb, depth, split_by_color=split_by_color)
        if not dets:
            results.append(StepResult(step.target, "no_detection"))
            continue
        scores = np.asarray(grounder.scores(step, dets))
        det = dets[int(np.argmax(scores))]
        obj = nearest_object(sim, det)
        res = StepResult(step.target, "success", obj.body_id,
                         info={"n_detections": len(dets), "chosen": obj.spec.name})
        if not matches(intended, obj.spec):
            res.outcome = "wrong_object"
        z0 = sim.object_pos(obj.body_id)[2]
        width = execute_grasp(ctrl, det)
        res.info["finger_width"] = width
        lifted = sim.object_pos(obj.body_id)[2] - z0
        res.lift = float(lifted)
        if lifted < LIFT_HEIGHT:
            res.outcome = "grasp_miss" if res.outcome == "success" else res.outcome
            ctrl.open_gripper()
            results.append(res)
            continue
        # hold for a second; a slipping object counts as dropped
        ctrl.move_to(ctrl.ee_pos(), grasp_yaw(det), steps=HOLD_STEPS)
        if sim.object_pos(obj.body_id)[2] - z0 < LIFT_HEIGHT:
            res.outcome = "drop" if res.outcome == "success" else res.outcome
            results.append(res)
            continue
        if step.action == "pick_place":
            ctrl.move_to(TRAY_DROP, 0.0, steps=360)
            if sim.object_pos(obj.body_id)[2] - z0 < LIFT_HEIGHT and res.outcome == "success":
                res.outcome = "drop"
            ctrl.open_gripper(steps=180)
            sim.step(240)
            if res.outcome == "success" and not in_tray(sim.object_pos(obj.body_id)):
                res.outcome = "place_miss"
        elif sim.object_pos(obj.body_id)[2] - z0 < LIFT_HEIGHT and res.outcome == "success":
            res.outcome = "drop"
        results.append(res)
    return results
