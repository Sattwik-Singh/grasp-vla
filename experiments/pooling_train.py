"""Experiment 2: mean vs. attention vs. cross-attention pooling.

Trains GroundingPolicy with each pooling type over several seeds and reports,
on held-out test scenes:
  * cross-instruction correlation of the fused representation (lower = the
    representation changes more with the instruction);
  * grounding accuracy: the object nearest the predicted point is the one named;
  * median localisation error in millimetres.

    python experiments/pooling_train.py --seeds 0 1 2 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_vla.learned import HALF, GroundingPolicy, cross_instruction_corr  # noqa: E402

POOLINGS = ("mean", "attn", "xattn")


def load(split, data_dir="data"):
    d = np.load(Path(data_dir) / f"pooling_{split}.npz")
    return {k: d[k] for k in d.files}


def evaluate(model, d, batch=256):
    model.eval()
    imgs = torch.from_numpy(d["imgs"])
    preds, hs = [], []
    with torch.no_grad():
        for i in range(0, len(d["scene"]), batch):
            sl = slice(i, i + batch)
            xy, h, _ = model(imgs[d["scene"][sl]], torch.from_numpy(d["canon_ids"][sl]), return_all=True)
            preds.append(xy)
            hs.append(h)
    pred = torch.cat(preds).numpy()
    h = torch.cat(hs)
    err = np.linalg.norm(pred - d["target"], axis=1) * HALF * 1000  # mm
    # grounding: is the nearest object in the scene the named one?
    correct = []
    for i, s in enumerate(d["scene"]):
        objs = d["objects_xy"][s]
        objs = objs[~np.isnan(objs[:, 0])]
        nearest = objs[np.argmin(np.linalg.norm(objs - pred[i], axis=1))]
        correct.append(np.allclose(nearest, d["target"][i]))
    corrs = [cross_instruction_corr(h[d["scene"] == s]) for s in np.unique(d["scene"])]
    return {"grounding_acc": float(np.mean(correct)), "median_err_mm": float(np.median(err)),
            "cross_instr_corr": float(np.mean(corrs))}


def train(pooling, seed, tr, te, epochs, batch=64, lr=2e-3, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GroundingPolicy(pooling)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(tr["scene"])
    steps = epochs * (n // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    imgs = torch.from_numpy(tr["imgs"])
    ids = torch.from_numpy(tr["ids"])
    tgt = torch.from_numpy(tr["target"])
    scene = torch.from_numpy(tr["scene"])
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(n // batch):
            b = perm[i * batch:(i + 1) * batch]
            img = imgs[scene[b]]
            if np.random.rand() < 0.5:  # cheap augmentation: brightness jitter on RGB
                img = img.clone()
                img[:, :3] = (img[:, :3].float() * float(np.random.uniform(0.85, 1.15))).clamp(0, 255).byte()
            loss = F.smooth_l1_loss(model(img, ids[b]), tgt[b], beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == epochs - 1:
            m = evaluate(model, te)
            history.append({"epoch": ep + 1, "train_loss": tot / (n // batch), **m})
            log(f"  {pooling} seed {seed} ep {ep+1}: loss {tot/(n//batch):.4f} {m} ({time.time()-t0:.0f}s)")
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--poolings", nargs="+", default=list(POOLINGS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tr, te = load("train"), load("test")
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    res_path = out / "pooling_runs.json"
    runs = json.loads(res_path.read_text()) if res_path.exists() else []
    for pooling in a.poolings:
        for seed in a.seeds:
            model, hist = train(pooling, seed, tr, te, a.epochs)
            torch.save(model.state_dict(), out / "checkpoints" / f"{pooling}_seed{seed}.pt")
            runs = [r for r in runs if not (r["pooling"] == pooling and r["seed"] == seed)]
            runs.append({"pooling": pooling, "seed": seed, "epochs": a.epochs,
                         "final": hist[-1], "history": hist})
            res_path.write_text(json.dumps(runs, indent=2))
    print("done")


if __name__ == "__main__":
    main()
