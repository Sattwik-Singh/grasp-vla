"""A small learned language-conditioned grounding policy and its pooling variants.

The policy sees one token per detected object (a small CNN on the object's
RGB-D crop plus its position) and the instruction, and predicts the (x, y)
table position of the object to pick. Four ways of pooling the object tokens
into one vector are compared; everything else is identical.

* mean       - average of the tokens. The pooled vector cannot depend on the
               instruction; language only enters afterwards, by concatenation.
* attn       - attention pooling with a learned query. Same capacity as
               `xattn`, but the query is fixed, so pooling ignores the instruction.
* film_mean  - stronger baseline: each token is modulated by the instruction
               (FiLM) before mean pooling, so pooling is instruction-aware but
               every object still contributes equally.
* xattn      - cross-attention pooling: the query is computed from the
               instruction, so the model weights objects by the command.

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

MAX_OBJ = 8      # token slots per scene (unused slots are masked)
CROP_PX = 24     # each detection becomes a 24x24 RGB-D crop
CENTER = np.array([0.50, -0.03])
HALF = 0.25      # metres; positions are stored as (xy - CENTER) / HALF

EXTRA_WORDS = ("put pick up the and place it in into tray bin move to grab drop basket "
               "then a of photo").split()
VOCAB = ["<pad>", "<unk>"] + sorted(set(list(COLOR_WORDS) + list(SHAPE_WORDS) + EXTRA_WORDS))
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
MAX_LEN = 14


def tokenize(text: str) -> list[int]:
    ids = [WORD2ID.get(w, 1) for w in re.findall(r"[a-z]+", text.lower())][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


def observation(sim: TabletopSim):
    """Object-centric observation built from the same detector the robot uses.

    Returns (crops MAX_OBJ x 4 x 24 x 24 uint8, positions MAX_OBJ x 2, valid mask).
    Channel 4 is height above the table (1 unit = 0.4 mm).
    """
    from .perception import detect

    rgb, depth, _ = sim.capture()
    dets = detect(sim, rgb, depth)[:MAX_OBJ]
    height = (np.clip((sim.CAM_EYE[2] - depth) / 0.1, 0, 1) * 255).astype(np.uint8)
    rgbd = np.concatenate([rgb, height[..., None]], -1)
    crops = np.zeros((MAX_OBJ, 4, CROP_PX, CROP_PX), np.uint8)
    pos = np.zeros((MAX_OBJ, 2), np.float32)
    valid = np.zeros(MAX_OBJ, bool)
    for i, d in enumerate(dets):
        x0, y0, x1, y1 = d.bbox
        cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, max(x1 - x0, y1 - y0) // 2 + 4
        patch = rgbd[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
        t = torch.from_numpy(patch).permute(2, 0, 1).float()[None]
        crops[i] = F.interpolate(t, size=(CROP_PX, CROP_PX), mode="area")[0].round().clamp(0, 255).byte().numpy()
        pos[i] = world_to_target(d.centroid[:2])
        valid[i] = True
    return crops, pos, valid, dets


def world_to_target(xy) -> np.ndarray:
    return (np.asarray(xy) - CENTER) / HALF


def target_to_world(t) -> np.ndarray:
    return np.asarray(t) * HALF + CENTER


class Encoder(nn.Module):
    """Each detected object -> one D-dim token (small CNN on its crop + its position)."""

    def __init__(self, d: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),     # 12
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),    # 6
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, d))
        self.pos = nn.Sequential(nn.Linear(2, d), nn.ReLU(), nn.Linear(d, d))
        # LayerNorm keeps tokens of different objects distinguishable at
        # initialisation; without it all tokens are nearly equal and the
        # attention weights stay uniform (found while debugging training).
        self.norm = nn.LayerNorm(d)

    def forward(self, crops, pos):  # crops: B x N x 4 x 24 x 24 uint8, pos: B x N x 2
        b, n = crops.shape[:2]
        f = self.cnn(crops.flatten(0, 1).float() / 255.0 - 0.5).view(b, n, -1)
        return self.norm(f + self.pos(pos))


class LanguageEncoder(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), 64, padding_idx=0)
        self.gru = nn.GRU(64, d // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        lengths = (ids != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(ids), lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)                  # 2 x B x d/2
        return self.norm(torch.cat([h[0], h[1]], -1))  # B x d


POOLINGS = ("mean", "attn", "film_mean", "xattn")


class GroundingPolicy(nn.Module):
    def __init__(self, pooling: str, d: int = 128, heads: int = 4):
        super().__init__()
        assert pooling in POOLINGS
        self.pooling = pooling
        self.vis = Encoder(d)
        self.lang = LanguageEncoder(d)
        if pooling in ("attn", "xattn"):
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, d) * 0.1)   # used by "attn"
            self.q_proj = nn.Linear(d, d)                            # used by "xattn"
        if pooling == "film_mean":
            self.film = nn.Linear(d, 2 * d)
            self.tok_mlp = nn.Sequential(nn.ReLU(), nn.Linear(d, d))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Linear(d, 2)

    def pool(self, tokens, lang, valid):
        if self.pooling in ("mean", "film_mean"):
            if self.pooling == "film_mean":
                g, b = self.film(lang).chunk(2, -1)
                tokens = self.tok_mlp(tokens * (1 + g[:, None]) + b[:, None])
            w = valid.float() / valid.float().sum(1, keepdim=True).clamp(min=1)
            return (tokens * w[..., None]).sum(1), None
        q = self.query.expand(tokens.shape[0], -1, -1) if self.pooling == "attn" else self.q_proj(lang)[:, None]
        out, w = self.attn(q, tokens, tokens, key_padding_mask=~valid, need_weights=True)
        return out[:, 0], w[:, 0]

    def forward(self, crops, pos, valid, ids, return_all: bool = False):
        tokens = self.vis(crops, pos)
        lang = self.lang(ids)
        pooled, weights = self.pool(tokens, lang, valid)
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
        crops, pos, valid, _ = observation(self.sim)
        ids = torch.tensor([tokenize(f"put the {step.target} in the tray")])
        with torch.no_grad():
            xy = self.model(torch.from_numpy(crops)[None], torch.from_numpy(pos)[None],
                            torch.from_numpy(valid)[None], ids)[0].numpy()
        xy = target_to_world(xy)
        return np.array([-np.linalg.norm(d.centroid[:2] - xy) for d in dets])
