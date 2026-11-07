"""Experiment 3: the learned grounding policies inside the full robot loop.

Replaces the hand-written attribute grounder with each trained policy (seed 0
checkpoint) and runs pick-and-place on fresh cluttered scenes (seeds 4000+).

    python experiments/learned_in_loop.py --episodes 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_vla.grounding import AttributeGrounder  # noqa: E402
from grasp_vla.learned import POOLINGS, GroundingPolicy, LearnedGrounder  # noqa: E402
from grasp_vla.pipeline import run_episode  # noqa: E402
from grasp_vla.sim import TabletopSim  # noqa: E402

from grasp_eval import CONDITIONS, make_instruction, wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--condition", default="clutter")
    ap.add_argument("--seed0", type=int, default=4000)
    a = ap.parse_args()
    torch.set_num_threads(2)
    sim = TabletopSim()
    grounders = {"attribute (hand-written)": AttributeGrounder()}
    for pooling in POOLINGS:
        m = GroundingPolicy(pooling)
        m.load_state_dict(torch.load(f"results/checkpoints/{pooling}_seed0.pt"))
        grounders[pooling] = LearnedGrounder(m, sim)
    out = {}
    for name, g in grounders.items():
        k, wrong = 0, 0
        for seed in range(a.seed0, a.seed0 + a.episodes):
            objects = sim.reset(seed, **CONDITIONS[a.condition])
            text, truth = make_instruction(np.random.default_rng(seed), objects)
            steps = run_episode(sim, text, g, truth_step=truth)
            outcomes = [s.outcome for s in steps] + ["parse_fail"] * (len(truth) - len(steps))
            k += all(o == "success" for o in outcomes)
            wrong += sum(o == "wrong_object" for o in outcomes)
        out[name] = {"episodes": a.episodes, "successes": k, "success_rate": k / a.episodes,
                     "ci95": wilson(k, a.episodes), "wrong_object_steps": wrong}
        print(f"{name:26s} {k / a.episodes:.1%}  wrong-object steps: {wrong}")
    out["_meta"] = {"condition": a.condition, "seeds": [a.seed0, a.seed0 + a.episodes - 1]}
    Path("results/learned_in_loop.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
