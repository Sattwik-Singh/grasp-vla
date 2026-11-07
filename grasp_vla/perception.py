"""Class-agnostic object detection from an overhead RGB-D image.

Pipeline: depth -> world points -> remove table and tray -> connected
components -> one Detection per component, with a 3D grasp point, a grasp yaw
and an RGB crop for language grounding. Nothing here reads the simulator's
segmentation mask or true object poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .sim import TRAY_POS, TabletopSim

MIN_HEIGHT = 0.006      # points this far above the table count as objects (m)
MIN_PIXELS = 25         # smaller blobs are treated as noise
TRAY_HALF = 0.20        # half-width of the tray exclusion box (m)
WORK_X = (0.25, 0.80)
WORK_Y = (-0.32, 0.30)


@dataclass
class Detection:
    centroid: np.ndarray      # (x, y, z) of the grasp point, world frame
    top_z: float              # highest point of the object
    yaw: float                # gripper yaw for a top-down grasp
    width: float              # object extent across the fingers (m)
    pixels: np.ndarray        # boolean mask in the image
    crop: np.ndarray          # RGB crop around the object
    mean_rgb: np.ndarray
    fill_ratio: float         # area / bounding-rectangle area (≈1 square, ≈0.79 disc)
    top_flatness: float       # std of heights in the top region (m); small = flat top

    @property
    def bbox(self):
        ys, xs = np.nonzero(self.pixels)
        return xs.min(), ys.min(), xs.max(), ys.max()


def _min_area_rect(xy: np.ndarray):
    """Angle (rad) and side lengths of the smallest enclosing rectangle."""
    c = xy - xy.mean(0)
    best = (math.inf, 0.0, 0.0, 0.0)
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        r = c @ R.T
        w, h = np.ptp(r[:, 0]), np.ptp(r[:, 1])
        if w * h < best[0]:
            best = (w * h, a, w, h)
    return best[1], best[2], best[3]


def _chroma(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(float)
    return rgb / np.maximum(rgb.sum(-1, keepdims=True), 1e-6)


def _split_by_color(blob: np.ndarray, rgb: np.ndarray, min_sep: float = 0.08,
                    depth: int = 0) -> list[np.ndarray]:
    """Split a depth blob whose pixels form two clearly different colour groups.

    Touching objects merge into one depth blob. This runs 2-means on pixel
    chromaticity (no colour names involved) and splits when the two centres are
    more than `min_sep` apart and both halves are object-sized. Recurses so up
    to four touching objects can be separated. Objects of the same colour that
    touch are not separated.
    """
    if depth >= 2 or blob.sum() < 2 * MIN_PIXELS:
        return [blob]
    ys, xs = np.nonzero(blob)
    c = _chroma(rgb[ys, xs])
    centres = c[[np.argmin(c[:, 0]), np.argmax(c[:, 0])]]  # deterministic init
    for _ in range(15):
        assign = np.argmin(((c[:, None, :] - centres[None]) ** 2).sum(-1), 1)
        new = np.array([c[assign == j].mean(0) if np.any(assign == j) else centres[j] for j in range(2)])
        if np.allclose(new, centres):
            break
        centres = new
    if np.linalg.norm(centres[0] - centres[1]) < min_sep:
        return [blob]
    parts = []
    for j in range(2):
        m = np.zeros_like(blob)
        m[ys[assign == j], xs[assign == j]] = True
        m = ndimage.binary_opening(m, iterations=1)
        lab, n = ndimage.label(m)
        for k in range(1, n + 1):
            piece = lab == k
            if piece.sum() >= MIN_PIXELS:
                parts.extend(_split_by_color(piece, rgb, min_sep, depth + 1))
    return parts or [blob]


def detect(sim: TabletopSim, rgb: np.ndarray, depth: np.ndarray,
           split_by_color: bool = True) -> list[Detection]:
    pts = sim.deproject(depth)
    z = pts[..., 2]
    x, y = pts[..., 0], pts[..., 1]
    in_tray = (np.abs(x - TRAY_POS[0]) < TRAY_HALF) & (np.abs(y - TRAY_POS[1]) < TRAY_HALF)
    in_work = (x > WORK_X[0]) & (x < WORK_X[1]) & (y > WORK_Y[0]) & (y < WORK_Y[1])
    mask = (z > MIN_HEIGHT) & in_work & ~in_tray
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, n = ndimage.label(mask)
    blobs = [labels == k for k in range(1, n + 1)]
    if split_by_color:
        blobs = [part for b in blobs for part in _split_by_color(b, rgb)]

    dets = []
    for m in blobs:
        if m.sum() < MIN_PIXELS:
            continue
        p3 = pts[m]
        top_z = float(np.percentile(p3[:, 2], 95))
        xy = p3[:, :2]
        yaw, w, h = _min_area_rect(xy)
        # fill ratio in pixel space: pixel count / smallest enclosing rectangle
        ys_, xs_ = np.nonzero(m)
        _, pw, ph = _min_area_rect(np.stack([xs_, ys_], 1).astype(float))
        fill = float(m.sum() / max((pw + 1) * (ph + 1), 1.0))
        top = p3[p3[:, 2] > top_z - 0.012, 2]
        flat = float(np.std(top))
        centroid = np.array([*xy.mean(0), max(top_z / 2, 0.012)])
        ys, xs = np.nonzero(m)
        pad = 6
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
        dets.append(Detection(centroid, top_z, yaw, float(min(w, h)), m,
                              rgb[y0:y1, x0:x1].copy(), rgb[m].mean(0), fill, flat))
    return dets

