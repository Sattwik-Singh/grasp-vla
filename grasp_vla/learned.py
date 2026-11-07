"""A small learned language-conditioned grounding policy and its pooling variants.

The policy reads the overhead RGB-D image and the instruction and predicts the
(x, y) table position of the object to pick. Three ways of pooling the image
tokens into one vector are compared; everything else is identical.

* mean   - average of all image tokens. The pooled image vector cannot depend
           on the instruction; language only enters afterwards, by concatenation.
* attn   - attention pooling with a learned query. Same capacity as `xattn`,
           but the query is fixed, so pooling still ignores the instruction.
* xattn  - cross-attention pooling: the query is computed from the instruction,
           so the model looks at different image regions for different commands.

Fused representation h = MLP([pooled image, language]) feeds a linear head.
"Cross-instruction correlation" is measured on h (see `cross_instruction_corr`).
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .language import COLOR_WORDS, SHAPE_WORDS
from .sim import TabletopSim

# Crop of the overhead image that covers the workspace (rows, cols at 320x320).
CROP = (slice(64, 256), slice(75, 267))
IMG = 96
CENTER = np.array([0.50, -0.03])
HALF = 0.25  # metres from the crop centre to its edge (approximately)

EXTRA_WORDS = ("put pick up the and place it in into tray bin move to grab drop basket "
               "then a of photo").split()
VOCAB = ["<pad>", "<unk>"] + sorted(set(list(COLOR_WORDS) + list(SHAPE_WORDS) + EXTRA_WORDS))
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
MAX_LEN = 14


def tokenize(text: str) -> list[int]:
    ids = [WORD2ID.get(w, 1) for w in re.findall(r"[a-z]+", text.lower())][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


def observation(sim: TabletopSim) -> np.ndarray:
    """4 x IMG x IMG uint8: RGB plus height above the table (1 unit = 0.4 mm)."""
    rgb, depth, _ = sim.capture()
    height = np.clip((sim.CAM_EYE[2] - depth) / 0.1, 0, 1) * 255
    obs = np.concatenate([rgb, height[..., None]], -1)[CROP]
    t = torch.from_numpy(obs).permute(2, 0, 1).float()[None]
    t = F.interpolate(t, size=(IMG, IMG), mode="area")[0]
    return t.round().clamp(0, 255).byte().numpy()


def world_to_target(xy) -> np.ndarray:
    return (np.asarray(xy) - CENTER) / HALF


def target_to_world(t) -> np.ndarray:
    return np.asarray(t) * HALF + CENTER


class Encoder(nn.Module):
    """CNN over RGB-D + coordinate channels -> 12x12 grid of D-dim tokens."""

    def __init__(self, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(96, d, 3, 1, 1),
        )
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, IMG), torch.linspace(-1, 1, IMG), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None])

    def forward(self, img):  # img: B x 4 x H x W, uint8
        x = img.float() / 255.0
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], 1)
        f = self.net(x)                          # B x d x 12 x 12
        return f.flatten(2).transpose(1, 2)      # B x 144 x d


class LanguageEncoder(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), 64, padding_idx=0)
        self.gru = nn.GRU(64, d // 2, batch_first=True, bidirectional=True)

    def forward(self, ids):
        lengths = (ids != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(ids), lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)                  # 2 x B x d/2
        return torch.cat([h[0], h[1]], -1)       # B x d


class GroundingPolicy(nn.Module):
    def __init__(self, pooling: str, d: int = 128, heads: int = 4):
        super().__init__()
        assert pooling in ("mean", "attn", "xattn")
        self.pooling = pooling
        self.vis = Encoder(d)
        self.lang = LanguageEncoder(d)
        if pooling != "mean":
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, d) * 0.1)   # used by "attn"
            self.q_proj = nn.Linear(d, d)                            # used by "xattn"
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Linear(d, 2)

    def pool(self, tokens, lang):
        if self.pooling == "mean":
            return tokens.mean(1), None
        if self.pooling == "attn":
            q = self.query.expand(tokens.shape[0], -1, -1)
        else:
            q = self.q_proj(lang)[:, None]
        out, w = self.attn(q, tokens, tokens, need_weights=True)
        return out[:, 0], w[:, 0]

    def forward(self, img, ids, return_all: bool = False):
        tokens = self.vis(img)
        lang = self.lang(ids)
        pooled, weights = self.pool(tokens, lang)
        h = self.fuse(torch.cat([pooled, lang], -1))
        xy = self.head(h)
        if return_all:
            return xy, h, weights
        return xy


def cross_instruction_corr(h: torch.Tensor) -> float:
    """Mean pairwise Pearson correlation between the fused vectors that one scene
    produces under K different instructions (h: K x d). 1 means the instruction
    barely changes the representation; lower means it is instruction-specific."""
    h = h - h.mean(1, keepdim=True)
    h = h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)
    c = h @ h.T
    k = h.shape[0]
    return float((c.sum() - k) / (k * (k - 1)))


class LearnedGrounder:
    """Plugs a trained policy into the pick-and-place pipeline: picks the detection
    closest to the predicted table position."""

    def __init__(self, model: GroundingPolicy, sim: TabletopSim):
        self.model = model.eval()
        self.sim = sim
        self.name = f"learned_{model.pooling}"

    def scores(self, step, dets):
        obs = torch.from_numpy(observation(self.sim))[None]
        ids = torch.tensor([tokenize(f"put the {step.target} in the tray")])
        with torch.no_grad():
            xy = target_to_world(self.model(obs, ids)[0].numpy())
        return np.array([-np.linalg.norm(d.centroid[:2] - xy) for d in dets])
