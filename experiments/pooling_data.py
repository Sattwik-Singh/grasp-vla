"""Generate the dataset for the pooling experiment.

Each scene has 4-8 objects (spacing from cluttered to loose). Every object in a
scene becomes one example: (image, instruction naming it, its x-y position).
Train scenes use seeds 10000+, test scenes 20000+; neither overlaps the
pick-and-place evaluation (2000+).

    python experiments/pooling_data.py --train 2000 --test 300
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_vla.learned import observation, tokenize, world_to_target  # noqa: E402
from grasp_vla.sim import TabletopSim  # noqa: E402

from grasp_eval import SYNONYMS, TEMPLATES_ONE  # noqa: E402


def make_scenes(seeds):
    sim = TabletopSim()
    imgs, scene_idx, ids, canon_ids, targets, names, all_xy = [], [], [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 9))
        gap = float(rng.uniform(0.008, 0.03))
        objects = sim.reset(int(seed), n_objects=n, min_gap=gap)
        imgs.append(observation(sim))
        pos = {o.body_id: sim.object_pos(o.body_id)[:2] for o in objects}
        all_xy.append(np.array([world_to_target(pos[o.body_id]) for o in objects]
                               + [[np.nan, np.nan]] * (8 - n), dtype=np.float32))
        for o in objects:
            phrase = f"{o.spec.color} {rng.choice(SYNONYMS[o.spec.shape])}"
            text = rng.choice(TEMPLATES_ONE).format(a=phrase)
            scene_idx.append(len(imgs) - 1)
            ids.append(tokenize(text))
            canon_ids.append(tokenize(f"put the {o.spec.name} in the tray"))
            targets.append(world_to_target(pos[o.body_id]))
            names.append(o.spec.name)
    sim.close()
    return dict(imgs=np.stack(imgs), scene=np.array(scene_idx), ids=np.array(ids),
                canon_ids=np.array(canon_ids), target=np.array(targets, dtype=np.float32),
                name=np.array(names), objects_xy=np.stack(all_xy))


def build(seeds, workers):
    parts = Pool(workers).map(make_scenes, np.array_split(seeds, workers))
    out, offset = {k: [] for k in parts[0]}, 0
    for part in parts:
        for k, v in part.items():
            out[k].append(v + offset if k == "scene" else v)
        offset += len(part["imgs"])
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for split, seed0, n in (("train", 10000, a.train), ("test", 20000, a.test)):
        d = build(np.arange(seed0, seed0 + n), a.workers)
        np.savez_compressed(out / f"pooling_{split}.npz", **d)
        print(split, {k: v.shape for k, v in d.items()})


if __name__ == "__main__":
    main()
