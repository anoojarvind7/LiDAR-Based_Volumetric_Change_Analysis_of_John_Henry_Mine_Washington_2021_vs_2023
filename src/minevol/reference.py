"""Reference (base) surfaces interpolated from the rim of a masked region.

A volume is only as good as the surface it is measured against. For a
stockpile that surface is the ground under its toe; for a pit it is the ground
that used to span its crest. Neither is observable, so it is interpolated from
the cells that ring the region. Three interpolators with different behaviour
are provided; the spread between them is one term of the volume uncertainty.

* ``harmonic`` - solves Laplace's equation inside the mask with the rim as a
  Dirichlet boundary (a "soap film"). Smooth, has no overshoot, and follows a
  sloping or warped rim. This is the primary estimate.
* ``tin`` - linear interpolation on a Delaunay triangulation of the rim
  cells, the classic surveyor's toe-to-toe base.
* ``spline`` - smoothed thin-plate spline through the rim. Can bulge above or
  below the rim, which is why it is a sensitivity case, not the default.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.sparse.linalg import spsolve

METHODS = ("harmonic", "tin", "spline")

_FOUR = ndimage.generate_binary_structure(2, 1)


def rim_mask(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Cells within ``width`` cells outside ``mask`` (4-connected dilation)."""
    grown = ndimage.binary_dilation(mask, structure=_FOUR, iterations=width)
    return grown & ~mask


def base_surface(z: np.ndarray, mask: np.ndarray, method: str = "harmonic",
                 cell: float = 1.0, max_rim_points: int = 4000) -> np.ndarray:
    """Return a copy of ``z`` with the cells in ``mask`` replaced by a surface
    interpolated from the rim of the mask.

    ``z`` must be finite on the rim. Cells outside ``mask`` are returned as-is.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return z.copy()
    rim = rim_mask(mask)
    if not np.isfinite(z[rim]).all():
        raise ValueError("reference rim contains no-data cells; enlarge the DTM or the polygon")
    if method == "harmonic":
        return _harmonic(z, mask)
    rows, cols = np.nonzero(rim)
    pts = np.column_stack([cols, rows]).astype(float) * cell
    vals = z[rows, cols]
    qr, qc = np.nonzero(mask)
    q = np.column_stack([qc, qr]).astype(float) * cell
    out = z.copy()
    if method == "tin":
        est = LinearNDInterpolator(pts, vals)(q)
        miss = ~np.isfinite(est)
        if miss.any():  # outside the rim's convex hull (concave masks)
            est[miss] = NearestNDInterpolator(pts, vals)(q[miss])
    elif method == "spline":
        if len(pts) > max_rim_points:
            keep = np.linspace(0, len(pts) - 1, max_rim_points).astype(int)
            pts, vals = pts[keep], vals[keep]
        # Smoothing scaled to the number of points keeps the spline from
        # chasing single noisy rim cells.
        rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                              smoothing=0.1 * len(pts), degree=1)
        est = np.concatenate([rbf(chunk) for chunk in np.array_split(q, max(1, len(q) // 50000))])
    else:
        raise ValueError(f"unknown base method {method!r}; choose from {METHODS}")
    out[qr, qc] = est
    return out


def _harmonic(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Solve the discrete Laplace equation on ``mask`` with ``z`` as boundary."""
    ny, nx = z.shape
    idx = -np.ones(z.shape, dtype=np.int64)
    rr, cc = np.nonzero(mask)
    n = len(rr)
    idx[rr, cc] = np.arange(n)

    rows, cols, data = [np.arange(n)], [np.arange(n)], [np.zeros(n)]
    rhs = np.zeros(n)
    diag = np.zeros(n)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r2, c2 = rr + dr, cc + dc
        inside = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
        # Neighbours off the grid are simply dropped (natural boundary).
        diag[inside] += 1.0
        nb = np.full(n, -1, dtype=np.int64)
        nb[inside] = idx[r2[inside], c2[inside]]
        unknown = inside & (nb >= 0)
        rows.append(np.nonzero(unknown)[0])
        cols.append(nb[unknown])
        data.append(-np.ones(unknown.sum()))
        known = inside & (nb < 0)
        rhs[known] += z[r2[known], c2[known]]
    data[0] = diag
    a = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
    out = z.copy()
    out[rr, cc] = spsolve(a, rhs)
    return out
