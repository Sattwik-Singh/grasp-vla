"""Make the report figures from the files in results/.

    python experiments/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_vla.learned import POOLINGS, GroundingPolicy, observation, tokenize  # noqa: E402
from grasp_vla.sim import TabletopSim  # noqa: E402

R = Path("results")
FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical slots (light mode) and text inks
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})
LABEL = {"clean": "Clean", "clutter": "Clutter", "sensor_noise": "Sensor\nnoise",
         "low_friction": "Low\nfriction", "combined": "Combined"}
PLABEL = {"mean": "Mean", "attn": "Attention\n(learned query)", "film_mean": "FiLM + mean",
          "xattn": "Cross-attention"}


def fig_grasp():
    runs = [("grasp_eval_attribute_clearance_s3000.json", "Full pipeline", S1, "o"),
            ("grasp_eval_attribute_simple_s3000.json", "Simple grasp planner", S2, "s"),
            ("grasp_eval_attribute_clearance_nosplit_s3000.json", "Depth-only segmentation", S3, "D")]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    conds = list(LABEL)
    for k, (f, name, col, mk) in enumerate(runs):
        d = json.loads((R / f).read_text())
        x = np.arange(len(conds)) + (k - 1) * 0.22
        y = np.array([d[c]["success_rate"] for c in conds]) * 100
        lo = np.array([d[c]["ci95"][0] for c in conds]) * 100
        hi = np.array([d[c]["ci95"][1] for c in conds]) * 100
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk, color=col, ms=5, lw=1.2, capsize=0,
                    label=name, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(range(len(conds)), [LABEL[c] for c in conds])
    ax.set_ylabel("Episode success (%)")
    ax.set_ylim(80, 101)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig(FIG / "grasp_success.pdf")
    fig.savefig(FIG / "grasp_success.png")


def fig_pooling():
    runs = json.loads((R / "pooling_runs.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    for ax, key, title, scale in ((axes[0], "cross_instr_corr", "Cross-instruction correlation", 1),
                                  (axes[1], "grounding_acc", "Grounding accuracy (%)", 100)):
        for i, p in enumerate(POOLINGS):
            v = np.array([r["final"][key] for r in runs if r["pooling"] == p]) * scale
            ax.bar(i, v.mean(), width=0.62, color=S1 if p == "xattn" else "#9fc2ec",
                   edgecolor="white", linewidth=1)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=9, color=INK, zorder=3)
            ax.text(i, v.mean() + (0.02 if scale == 1 else 2), f"{v.mean():.2f}" if scale == 1 else f"{v.mean():.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(POOLINGS)), [PLABEL[p] for p in POOLINGS], fontsize=7)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.1)
    axes[1].set_ylim(0, 105)
    fig.tight_layout(w_pad=2)
    fig.savefig(FIG / "pooling.pdf")
    fig.savefig(FIG / "pooling.png")


def fig_attention(seed: int = 20003):
    """Cross-attention weights on one test scene for two different instructions."""
    m = GroundingPolicy("xattn")
    m.load_state_dict(torch.load(R / "checkpoints" / "xattn_seed0.pt"))
    m.eval()
    sim = TabletopSim()
    rng = np.random.default_rng(seed)
    n, gap = int(rng.integers(4, 9)), float(rng.uniform(0.008, 0.03))
    objects = sim.reset(seed, n_objects=n, min_gap=gap)
    rgb, _, _ = sim.capture()
    crops, pos, valid, dets = observation(sim)
    names = [objects[0].spec.name, objects[1].spec.name]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7))
    for ax, name in zip(axes, names):
        ids = torch.tensor([tokenize(f"put the {name} in the tray")])
        with torch.no_grad():
            _, _, w = m(torch.from_numpy(crops)[None], torch.from_numpy(pos)[None],
                        torch.from_numpy(valid)[None], ids, return_all=True)
        w = w[0].numpy()
        ax.imshow(rgb[40:280, 40:300])
        for d, wi in zip(dets, w):
            x0, y0, x1, y1 = d.bbox
            cx, cy = (x0 + x1) / 2 - 40, (y0 + y1) / 2 - 40
            r = max(x1 - x0, y1 - y0) / 2 + 5
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, lw=0.6 + 3 * wi, color=INK if wi > 0.2 else INK2,
                                    alpha=0.35 + 0.65 * wi))
            ax.text(cx, cy - r - 3, f"{wi:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_title(f'"put the {name} in the tray"', fontsize=7.5, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "attention.pdf")
    fig.savefig(FIG / "attention.png")
    sim.close()


if __name__ == "__main__":
    fig_grasp()
    if (R / "pooling_runs.json").exists():
        fig_pooling()
    if (R / "checkpoints" / "xattn_seed0.pt").exists():
        fig_attention()
    print("figures in", FIG)
