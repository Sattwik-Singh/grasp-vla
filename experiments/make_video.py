"""Record a demo video: successful episodes followed by a real failure.

    python experiments/make_video.py --out media/demo.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_vla.grounding import AttributeGrounder  # noqa: E402
from grasp_vla.pipeline import run_episode  # noqa: E402
from grasp_vla.sim import TabletopSim  # noqa: E402

from grasp_eval import CONDITIONS  # noqa: E402

# (condition, seed, instruction). The last clip is evaluation episode 2013 of
# the "clutter" condition, one of the four clutter failures.
CLIPS = [
    ("clean", 2010, None),
    ("clutter", 2012, None),
    ("clutter", 2013, None),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def caption(frame, lines, color=(255, 255, 255)):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 15)
    y = 8
    for text in lines:
        d.rectangle([6, y - 2, 12 + d.textlength(text, font=f), y + 18], fill=(0, 0, 0))
        d.text((9, y), text, font=f, fill=color)
        y += 22
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media/demo.mp4")
    a = ap.parse_args()
    import json
    episodes = {(r["condition"], r["seed"]): r for r in
                map(json.loads, open("results/grasp_eval_attribute_simple_s2000_episodes.jsonl"))}
    sim = TabletopSim()
    grounder = AttributeGrounder()
    frames = []
    for cond, seed, text in CLIPS:
        sim.reset(seed, **CONDITIONS[cond])
        text = text or episodes[(cond, seed)]["instruction"]
        sim.frames = []
        sim.step(30)
        results = run_episode(sim, text, grounder)
        sim.step(60)
        clip, sim.frames = sim.frames, None
        outcome = ", ".join(r.outcome.replace("_", " ") for r in results)
        ok = all(r.outcome == "success" for r in results)
        col = (120, 255, 120) if ok else (255, 120, 120)
        for i, fr in enumerate(clip):
            lines = [f'"{text}"', f"condition: {cond}"]
            if i > len(clip) - 40:
                lines.append(f"result: {outcome}")
            frames.append(caption(fr, lines, col if i > len(clip) - 40 else (255, 255, 255)))
        print(cond, seed, text, outcome, len(clip), "frames")
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(f"{tmp}/{i:05d}.png")
        # sim runs at 240 Hz and we keep every 8th step: 30 fps is real time, 45 fps is 1.5x
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "45", "-i", f"{tmp}/%05d.png",
                        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-crf", "23", str(out)], check=True)
    print("wrote", out, f"{len(frames)/45:.1f}s")


if __name__ == "__main__":
    main()
