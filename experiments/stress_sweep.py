"""Experiment 4: where does the pipeline break?

Sweeps two difficulty knobs one at a time, on fresh scenes (seeds 5000+):
  * camera calibration error (std of the unknown camera offset, mm), 5 objects;
  * object spacing (minimum gap between neighbours, mm), 8 objects.

    python experiments/stress_sweep.py --episodes 100
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_vla.grounding import AttributeGrounder  # noqa: E402
from grasp_vla.pipeline import run_episode  # noqa: E402
from grasp_vla.sim import TabletopSim  # noqa: E402

from grasp_eval import make_instruction, wilson  # noqa: E402

SWEEPS = {
    "calibration_mm": [(v, dict(n_objects=5, min_gap=0.03, cam_offset_std=v / 1000)) for v in (0, 5, 10, 15, 20, 30)],
    "gap_mm": [(v, dict(n_objects=8, min_gap=v / 1000)) for v in (30, 15, 8, 4, 1)],
}


def run(args):
    knob, value, kwargs, seeds, planner = args
    sim, g = TabletopSim(), AttributeGrounder()
    ok, outcomes = 0, {}
    for seed in seeds:
        objects = sim.reset(int(seed), **kwargs)
        text, truth = make_instruction(np.random.default_rng(seed), objects)
        steps = run_episode(sim, text, g, truth_step=truth, grasp_planner=planner)
        outs = [s.outcome for s in steps] + ["parse_fail"] * (len(truth) - len(steps))
        ok += all(o == "success" for o in outs)
        for o in outs:
            outcomes[o] = outcomes.get(o, 0) + 1
    sim.close()
    return knob, value, planner, ok, len(seeds), outcomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    seeds = np.arange(5000, 5000 + a.episodes)
    jobs = [(k, v, kw, seeds, planner) for k, levels in SWEEPS.items() for v, kw in levels
            for planner in ("clearance", "simple")]
    with Pool(a.workers) as pool:
        res = pool.map(run, jobs)
    out = {}
    for knob, value, planner, ok, n, outcomes in res:
        out.setdefault(knob, {}).setdefault(planner, []).append(
            {"value": value, "success_rate": ok / n, "ci95": wilson(ok, n), "episodes": n, "step_outcomes": outcomes})
        print(f"{knob:15s} {value:4} {planner:9s} {ok / n:.0%} {outcomes}")
    Path("results/stress_sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
