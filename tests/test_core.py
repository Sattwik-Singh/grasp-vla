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


@pytest.mark.parametrize("pooling", ["mean", "attn", "xattn"])
def test_policy_shapes(pooling):
    m = GroundingPolicy(pooling)
    img = torch.randint(0, 255, (3, 4, 96, 96), dtype=torch.uint8)
    ids = torch.tensor([tokenize("put the red cube in the tray")] * 3)
    xy, h, w = m(img, ids, return_all=True)
    assert xy.shape == (3, 2) and h.shape == (3, 128)
    assert (w is None) == (pooling == "mean")


def test_mean_pooling_ignores_instruction_before_fusion():
    m = GroundingPolicy("mean").eval()
    img = torch.randint(0, 255, (1, 4, 96, 96), dtype=torch.uint8)
    t = m.vis(img)
    a, _ = m.pool(t, m.lang(torch.tensor([tokenize("red cube")])))
    b, _ = m.pool(t, m.lang(torch.tensor([tokenize("blue ball")])))
    assert torch.allclose(a, b)


def test_xattn_pooling_depends_on_instruction():
    torch.manual_seed(0)
    m = GroundingPolicy("xattn").eval()
    img = torch.randint(0, 255, (1, 4, 96, 96), dtype=torch.uint8)
    t = m.vis(img)
    a, _ = m.pool(t, m.lang(torch.tensor([tokenize("red cube")])))
    b, _ = m.pool(t, m.lang(torch.tensor([tokenize("blue ball")])))
    assert not torch.allclose(a, b)


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
