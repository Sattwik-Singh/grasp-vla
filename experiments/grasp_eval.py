"""Experiment 1: end-to-end pick-and-place success under five conditions.

Each episode samples a fresh scene and a templated instruction naming one or two
objects that are in the scene. An episode succeeds only if every named object
ends up in the tray. Evaluation seeds start at 2000. Seeds 0-39 were used to
develop the code and seeds 1000-1029 for a pilot run that exposed the merged-blob
failure, so neither set is reused here.

    python experiments/grasp_eval.py --episodes 200 --grounder attribute
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_vla.language import SHAPE_WORDS, Step  # noqa: E402
from grasp_vla.pipeline import OUTCOMES, run_episode  # noqa: E402
from grasp_vla.sim import TabletopSim  # noqa: E402

CONDITIONS = {
    "clean":        dict(n_objects=5, min_gap=0.030),
    "clutter":      dict(n_objects=8, min_gap=0.008),
    "sensor_noise": dict(n_objects=5, min_gap=0.030, depth_noise=0.002, cam_offset_std=0.006),
    "low_friction": dict(n_objects=5, min_gap=0.030, friction=(0.3, 0.6)),
    "combined":     dict(n_objects=8, min_gap=0.008, depth_noise=0.002, cam_offset_std=0.006,
                         friction=(0.4, 1.0)),
}

TEMPLATES_ONE = [
    "put the {a} in the tray",
    "pick up the {a} and place it in the bin",
    "move the {a} into the tray",
    "grab the {a} and drop it in the basket",
]
TEMPLATES_TWO = [
    "put the {a} and the {b} in the tray",
    "move the {a} to the bin, then the {b}",
]
SYNONYMS = {s: [w for w, v in SHAPE_WORDS.items() if v == s] for s in ("cube", "cylinder", "ball")}


def make_instruction(rng, objects):
    k = 2 if rng.random() < 0.25 else 1
    chosen = [objects[i] for i in rng.choice(len(objects), k, replace=False)]
    phrases = [f"{o.spec.color} {rng.choice(SYNONYMS[o.spec.shape])}" for o in chosen]
    tmpl = rng.choice(TEMPLATES_TWO if k == 2 else TEMPLATES_ONE)
    text = tmpl.format(a=phrases[0], b=phrases[1] if k == 2 else "")
    truth = [Step("pick_place", o.spec.name, o.spec.color, o.spec.shape) for o in chosen]
    return text, truth


def _grounder(name):
    if name == "clip":
        from grasp_vla.grounding import ClipGrounder
        return ClipGrounder()
    from grasp_vla.grounding import AttributeGrounder
    return AttributeGrounder()


def run_condition(args):
    cond, seeds, grounder_name, split, planner = args
    sim = TabletopSim()
    grounder = _grounder(grounder_name)
    rows = []
    for seed in seeds:
        objects = sim.reset(seed, **CONDITIONS[cond])
        rng = np.random.default_rng(seed)
        text, truth = make_instruction(rng, objects)
        steps = run_episode(sim, text, grounder, truth_step=truth, split_by_color=split,
                            grasp_planner=planner)
        outcomes = [s.outcome for s in steps]
        if len(steps) < len(truth):  # planner found fewer targets than intended
            outcomes += ["parse_fail"] * (len(truth) - len(steps))
        rows.append({"condition": cond, "seed": int(seed), "instruction": text,
                     "n_targets": len(truth), "step_outcomes": outcomes,
                     "success": all(o == "success" for o in outcomes),
                     "chosen": [s.info.get("chosen") for s in steps],
                     "lift_m": [round(s.lift, 4) for s in steps]})
    sim.close()
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def summarise(rows):
    out = {}
    for cond in CONDITIONS:
        r = [x for x in rows if x["condition"] == cond]
        k = sum(x["success"] for x in r)
        step_counts = Counter(o for x in r for o in x["step_outcomes"])
        n_steps = sum(step_counts.values())
        out[cond] = {
            "episodes": len(r), "successes": k, "success_rate": k / len(r),
            "ci95": wilson(k, len(r)),
            "steps": n_steps, "step_success_rate": step_counts["success"] / n_steps,
            "step_outcomes": {o: step_counts.get(o, 0) for o in OUTCOMES},
            "single_target_rate": float(np.mean([x["success"] for x in r if x["n_targets"] == 1])),
            "two_target_rate": float(np.mean([x["success"] for x in r if x["n_targets"] == 2]))
            if any(x["n_targets"] == 2 for x in r) else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--grounder", choices=["attribute", "clip"], default="attribute")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-color-split", action="store_true",
                    help="ablation: depth-only segmentation")
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--grasp-planner", choices=["clearance", "simple"], default="clearance")
    a = ap.parse_args()

    seeds = np.arange(a.seed0, a.seed0 + a.episodes)
    tag = "_".join([a.grounder, a.grasp_planner] + (["nosplit"] if a.no_color_split else []) + [f"s{a.seed0}"])
    jobs = [(c, chunk.tolist(), a.grounder, not a.no_color_split, a.grasp_planner) for c in CONDITIONS
            for chunk in np.array_split(seeds, a.workers)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = [r for part in pool.map(run_condition, jobs) for r in part]
    summary = summarise(rows)
    summary["_meta"] = {"grounder": a.grounder, "color_split": not a.no_color_split, "grasp_planner": a.grasp_planner, "episodes_per_condition": a.episodes,
                        "seeds": [int(seeds[0]), int(seeds[-1])],
                        "wall_seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / f"grasp_eval_{tag}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"grasp_eval_{tag}_episodes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for cond in CONDITIONS:
        s = summary[cond]
        lo, hi = s["ci95"]
        print(f"{cond:13s} {s['success_rate']:.1%}  [{lo:.1%}, {hi:.1%}]  {s['step_outcomes']}")


if __name__ == "__main__":
    main()
