"""Volume integration on regular grids.

Volumes are integrated cell by cell: ``V = sum(dz) * cell_area``. On a DTM
built by rasterising a TIN this is the midpoint rule applied to the TIN, and it
converges to the exact TIN prism volume as the cell shrinks (checked in
``tests/test_volume.py`` and by the resolution study in the report).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .reference import base_surface, rim_mask


@dataclass
class CutFill:
    fill_m3: float      # material above the reference (surface higher)
    cut_m3: float       # void below the reference (surface lower)
    area_m2: float      # area of cells that were integrated
    fill_area_m2: float
    cut_area_m2: float

    @property
    def net_m3(self) -> float:
        return self.fill_m3 - self.cut_m3


def cut_fill(surface: np.ndarray, reference: np.ndarray, cell: float,
             mask: np.ndarray | None = None, threshold: float = 0.0) -> CutFill:
    """Integrate ``surface - reference`` over ``mask``.

    ``threshold`` zeroes differences whose magnitude is below it (a level of
    detection); cells with no data in either grid are skipped.
    """
    dz = surface - reference
    valid = np.isfinite(dz)
    if mask is not None:
        valid &= mask
    dz = np.where(valid, dz, 0.0)
    if threshold > 0:
        dz = np.where(np.abs(dz) >= threshold, dz, 0.0)
    a = cell * cell
    pos, neg = dz > 0, dz < 0
    return CutFill(
        fill_m3=float(dz[pos].sum() * a),
        cut_m3=float(-dz[neg].sum() * a),
        area_m2=float(valid.sum() * a),
        fill_area_m2=float(pos.sum() * a),
        cut_area_m2=float(neg.sum() * a),
    )


def refine_mask(z: np.ndarray, outline: np.ndarray, kind: str, cell: float,
                relief: float = 0.5, method: str = "harmonic", pad_m: float = 2.0,
                min_area_m2: float = 200.0, max_iter: int = 8) -> np.ndarray:
    """Shrink a generous hand-drawn outline to the toe (pile) or crest (pit).

    Starting from ``outline``, a base surface is interpolated from its rim and
    the cells standing more than ``relief`` metres above it (``kind="pile"``)
    or below it (``kind="pit"``) are kept. The kept region is padded by
    ``pad_m`` so its rim sits on the surrounding ground, and the process
    repeats until the mask stops changing. Only the connected components that
    contain the most extreme relief (and any other component larger than
    ``min_area_m2``) survive, and holes are filled, so benches and ramps on the
    feature stay part of it.
    """
    if kind not in ("pile", "pit"):
        raise ValueError("kind must be 'pile' or 'pit'")
    sign = 1.0 if kind == "pile" else -1.0
    pad = max(1, int(round(pad_m / cell)))
    min_cells = min_area_m2 / (cell * cell)
    mask = outline.copy()
    for _ in range(max_iter):
        base = base_surface(z, mask, method, cell)
        rel = sign * (z - base)
        core = (rel > relief) & outline
        lab, n = ndimage.label(core)
        if n == 0:
            return np.zeros_like(outline)
        sizes = ndimage.sum(core, lab, index=np.arange(1, n + 1))
        peak = lab[np.unravel_index(np.nanargmax(np.where(core, rel, -np.inf)), rel.shape)]
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_cells] + [peak])
        keep = ndimage.binary_fill_holes(keep)
        new = ndimage.binary_dilation(keep, iterations=pad) & outline
        if np.array_equal(new, mask):
            break
        mask = new
    return mask


def feature_volume(z: np.ndarray, mask: np.ndarray, kind: str, cell: float,
                   method: str = "harmonic") -> tuple[float, np.ndarray, CutFill]:
    """Volume of a pile (material above base) or pit (void below base).

    Returns ``(volume, base, cutfill)``; ``volume`` is the fill for a pile and
    the cut for a pit, i.e. always positive for a well-formed feature.
    """
    base = base_surface(z, mask, method, cell)
    cf = cut_fill(z, base, cell, mask)
    vol = cf.fill_m3 if kind == "pile" else cf.cut_m3
    return vol, base, cf


def edge_contact(mask: np.ndarray, outline: np.ndarray) -> float:
    """Fraction of the mask's perimeter that runs along the search outline.

    Near zero means the toe/crest was found inside the outline. A large value
    means the feature was cut off by the hand-drawn outline, which should then
    be enlarged.
    """
    edge = mask & ~ndimage.binary_erosion(mask)
    if not edge.any():
        return 0.0
    border = outline & ~ndimage.binary_erosion(outline)
    return float((edge & border).sum() / edge.sum())


def rim_stats(z: np.ndarray, mask: np.ndarray) -> dict:
    r = z[rim_mask(mask)]
    return {"rim_min": float(np.nanmin(r)), "rim_max": float(np.nanmax(r)),
            "rim_median": float(np.nanmedian(r))}
