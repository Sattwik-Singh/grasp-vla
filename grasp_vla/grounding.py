"""Language grounding: choose which detection an instruction refers to.

Two grounders share one interface, `scores(step, detections) -> np.ndarray`:

* AttributeGrounder - hand-written colour and shape features. Works only for the
  colour and shape words it knows. Fully offline.
* ClipGrounder - open-vocabulary: scores each RGB crop against the target phrase
  with a pretrained CLIP model, so it can also handle words it was never given.
"""
from __future__ import annotations

import numpy as np

from .language import Step
from .perception import Detection
from .sim import COLORS

# Thresholds tuned on development scenes (seeds 0-39). The evaluation uses
# seeds from 1000 upwards, so it never sees the scenes the thresholds came from.
CUBE_MIN_FILL = 0.85        # squares fill ~0.90 of their bounding box, discs ~0.79
BALL_MIN_UNEVEN = 0.0028    # balls have a domed top: height std > 2.8 mm


def color_of(det: Detection) -> str:
    """Nearest named colour by chromaticity (robust to shading)."""
    rgb = det.mean_rgb / max(det.mean_rgb.sum(), 1e-6)
    best, best_d = None, np.inf
    for name, rgba in COLORS.items():
        ref = np.array(rgba[:3]) / sum(rgba[:3])
        d = np.linalg.norm(rgb - ref)
        if d < best_d:
            best, best_d = name, d
    return best


def shape_of(det: Detection) -> str:
    if det.top_flatness > BALL_MIN_UNEVEN:
        return "ball"
    return "cube" if det.fill_ratio > CUBE_MIN_FILL else "cylinder"


class AttributeGrounder:
    name = "attribute"

    def scores(self, step: Step, dets: list[Detection]) -> np.ndarray:
        s = np.zeros(len(dets))
        for i, d in enumerate(dets):
            if step.color is not None:
                s[i] += 1.0 if color_of(d) == step.color else 0.0
            if step.shape is not None:
                s[i] += 1.0 if shape_of(d) == step.shape else 0.0
        return s


class ClipGrounder:
    """Open-vocabulary grounding with CLIP (needs the weights from HuggingFace)."""
    name = "clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)

    def scores(self, step: Step, dets: list[Detection]) -> np.ndarray:
        from PIL import Image

        if not dets:
            return np.zeros(0)
        crops = [Image.fromarray(_square(d.crop)) for d in dets]
        text = [f"a photo of a {step.target}"]
        with self.torch.no_grad():
            inp = self.proc(text=text, images=crops, return_tensors="pt", padding=True)
            out = self.model(**inp)
        return out.logits_per_text[0].cpu().numpy()


def _square(img: np.ndarray, size: int = 96) -> np.ndarray:
    """Pad a crop to a square on a table-coloured background, then resize."""
    from PIL import Image

    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), img[0, 0], dtype=np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return np.array(Image.fromarray(canvas).resize((size, size)))
