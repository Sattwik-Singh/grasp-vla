import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from grasp_vla.language import parse  # noqa: E402
from grasp_vla.learned import (GroundingPolicy, cross_instruction_corr, target_to_world,  # noqa: E402
                               tokenize, world_to_target)
from grasp_vla.perception import Detection  # noqa: E402
from grasp_vla.pipeline import grasp_yaw  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("put the red cube in the tray", [("red", "cube")]),
    ("grab the blue tube and drop it in the basket", [("blue", "cylinder")]),
    ("move the green marble to the bin, then the red cylinder", [("green", "ball"), ("red", "cylinder")]),
    ("put the yellow block and the purple sphere in the tray", [("yellow", "cube"), ("purple", "ball")]),
])
def test_parser(text, expected):
    steps = parse(text)
    assert [(s.color, s.shape) for s in steps] == expected
    assert all(s.action == "pick_place" for s in steps)


def test_parser_without_destination_is_pick_only():
    assert parse("pick up the orange can")[0].action == "pick"


def test_parser_rejects_instruction_without_object():
    assert parse("do something useful") == []


def _det(yaw):
    z = np.zeros(1)
    return Detection(np.zeros(3), 0.05, yaw, 0.04, np.zeros((2, 2), bool), np.zeros((2, 2, 3)), z, 0.9, 0.0)


@pytest.mark.parametrize("yaw", np.linspace(-3, 3, 13))
def test_grasp_yaw_is_wrapped_and_equivalent(yaw):
    w = grasp_yaw(_det(yaw))
    assert -math.pi / 4 < w <= math.pi / 4 + 1e-9
    assert abs(((w - yaw) / (math.pi / 2)) - round((w - yaw) / (math.pi / 2))) < 1e-9


def test_target_mapping_round_trips():
    xy = np.array([0.42, -0.11])
    assert np.allclose(target_to_world(world_to_target(xy)), xy)


def test_cross_instruction_corr_limits():
    v = torch.randn(1, 64)
    assert cross_instruction_corr(v.repeat(4, 1)) == pytest.approx(1.0, abs=1e-5)
    eye = torch.eye(8) - 1 / 8  # zero-mean orthogonal-ish rows
    assert cross_instruction_corr(eye) < 0


def _obs(b=3, n=8):
    crops = torch.randint(0, 255, (b, n, 4, 24, 24), dtype=torch.uint8)
    pos = torch.rand(b, n, 2) * 2 - 1
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[:, -2:] = False
    return crops, pos, valid


@pytest.mark.parametrize("pooling", ["mean", "attn", "film_mean", "xattn"])
def test_policy_shapes(pooling):
    m = GroundingPolicy(pooling)
    ids = torch.tensor([tokenize("put the red cube in the tray")] * 3)
    xy, h, w = m(*_obs(), ids, return_all=True)
    assert xy.shape == (3, 2) and h.shape == (3, 128)
    assert (w is None) == (pooling in ("mean", "film_mean"))


@pytest.mark.parametrize("pooling", ["mean", "attn", "film_mean", "xattn"])
def test_padding_slots_are_ignored(pooling):
    torch.manual_seed(0)
    m = GroundingPolicy(pooling).eval()
    crops, pos, valid = _obs(1)
    ids = torch.tensor([tokenize("red cube")])
    a = m(crops, pos, valid, ids)
    crops2, pos2 = crops.clone(), pos.clone()
    crops2[:, -2:] = 0
    pos2[:, -2:] = 5.0
    assert torch.allclose(a, m(crops2, pos2, valid, ids), atol=1e-5)


def _pooled(m, phrase, crops, pos, valid):
    return m.pool(m.vis(crops, pos), m.lang(torch.tensor([tokenize(phrase)])), valid)[0]


@pytest.mark.parametrize("pooling", ["mean", "attn"])
def test_instruction_free_pooling_ignores_instruction(pooling):
    m = GroundingPolicy(pooling).eval()
    obs = _obs(1)
    assert torch.allclose(_pooled(m, "red cube", *obs), _pooled(m, "blue ball", *obs))


@pytest.mark.parametrize("pooling", ["film_mean", "xattn"])
def test_instruction_aware_pooling_depends_on_instruction(pooling):
    torch.manual_seed(0)
    m = GroundingPolicy(pooling).eval()
    obs = _obs(1)
    assert not torch.allclose(_pooled(m, "red cube", *obs), _pooled(m, "blue ball", *obs))


def test_sim_detects_every_object_in_clean_scenes():
    from grasp_vla.perception import detect
    from grasp_vla.sim import TabletopSim

    sim = TabletopSim()
    for seed in range(5):
        objs = sim.reset(seed)
        rgb, depth, _ = sim.capture()
        dets = detect(sim, rgb, depth)
        assert len(dets) == len(objs)
        for o in objs:
            gt = sim.object_pos(o.body_id)[:2]
            assert min(np.linalg.norm(d.centroid[:2] - gt) for d in dets) < 0.01
    sim.close()


def test_clearance_planner_avoids_a_neighbour():
    from grasp_vla.grasp_planning import finger_collisions

    # neighbour 5 cm away along +y: fingers closing along y would hit it
    pts = np.array([[0.0, 0.05 + dx] for dx in np.linspace(0, 0.01, 20)])
    hit_y = finger_collisions(pts, np.zeros(2), 0.0, 0.04)          # closes along y
    hit_x = finger_collisions(pts, np.zeros(2), math.pi / 2, 0.04)  # closes along x
    assert hit_y > 0 and hit_x == 0
