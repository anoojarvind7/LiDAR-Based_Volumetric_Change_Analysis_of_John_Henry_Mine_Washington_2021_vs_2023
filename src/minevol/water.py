"""Water bodies: extent and level from lidar water returns.

Near-infrared lidar does not penetrate water, so pit lakes are measured only
down to their surface. The DTM is hydro-flattened (each lake set to a single
level) and every volume that touches a lake is reported as "above water"; the
submerged part of a pit is unknown from these data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from .grid import Grid


@dataclass
class WaterBody:
    id: int
    level_m: float       # median water-surface elevation
    level_iqr_m: float   # spread of water returns (wind, noise)
    area_m2: float
    n_points: int
    geometry: object


def find_water(grid: Grid, x, y, z, min_area_m2: float = 500.0,
               close_m: float = 5.0) -> tuple[np.ndarray, list[WaterBody]]:
    """Label lakes from class-9 points. Returns (label raster, bodies)."""
    present = grid.bin_count(x, y) > 0
    it = max(1, int(round(close_m / grid.res)))
    # Close gaps between sparse returns (specular drop-outs), then fill islands of no-return.
    wet = ndimage.binary_fill_holes(ndimage.binary_closing(present, iterations=it))
    wet = ndimage.binary_opening(wet, iterations=1)
    lab, n = ndimage.label(wet)
    c = np.floor((x - grid.x0) / grid.res).astype(int)
    r = np.floor((grid.y1 - y) / grid.res).astype(int)
    ok = (c >= 0) & (c < grid.width) & (r >= 0) & (r < grid.height)
    pid = np.zeros(len(x), int)
    pid[ok] = lab[r[ok], c[ok]]
    bodies, out = [], np.zeros_like(lab)
    k = 0
    for i in range(1, n + 1):
        cells = lab == i
        area = cells.sum() * grid.res**2
        zi = z[pid == i]
        if area < min_area_m2 or len(zi) < 20:
            continue
        k += 1
        out[cells] = k
        q1, med, q3 = np.percentile(zi, [25, 50, 75])
        geom = unary_union([shape(g) for g, v in shapes(cells.astype("uint8"), mask=cells,
                                                         transform=grid.transform) if v])
        bodies.append(WaterBody(k, float(med), float(q3 - q1), float(area), len(zi), geom))
    return out, bodies


def extend_into_gaps(labels: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    """Grow each lake into the no-data holes it borders.

    Water often returns no signal away from the shore (specular reflection),
    which leaves a hole in the middle of a lake. A no-data region that touches
    exactly one lake and not the edge of the grid (the survey boundary) is
    taken to be part of that lake.
    """
    out = labels.copy()
    holes, _ = ndimage.label(nodata)
    edge = set(np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])))
    ny, nx = holes.shape
    for h, sl in enumerate(ndimage.find_objects(holes), start=1):
        if sl is None or h in edge:
            continue
        # Work in the hole's bounding box grown by one cell.
        r0, r1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, ny)
        c0, c1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, nx)
        cells = holes[r0:r1, c0:c1] == h
        border = ndimage.binary_dilation(cells) & ~cells
        touch = np.unique(labels[r0:r1, c0:c1][border])
        touch = touch[touch > 0]
        if len(touch) == 1:
            out[r0:r1, c0:c1][cells] = touch[0]
    return out


def flatten(dtm: np.ndarray, labels: np.ndarray, bodies: list[WaterBody]) -> np.ndarray:
    """Set every lake cell to its lake's level."""
    out = dtm.copy()
    for b in bodies:
        out[labels == b.id] = b.level_m
    return out
