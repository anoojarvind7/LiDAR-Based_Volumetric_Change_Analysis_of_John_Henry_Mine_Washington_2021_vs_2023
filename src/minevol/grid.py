"""A fixed analysis grid so every raster lines up cell for cell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import mapping


@dataclass(frozen=True)
class Grid:
    x0: float   # west edge
    y1: float   # north edge
    width: int
    height: int
    res: float
    crs: str

    @classmethod
    def from_bounds(cls, bounds, res: float, crs: str) -> Grid:
        x0, y0, x1, y1 = bounds
        return cls(x0, y1, int(round((x1 - x0) / res)), int(round((y1 - y0) / res)), res, crs)

    @property
    def transform(self):
        return from_origin(self.x0, self.y1, self.res, self.res)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def bounds(self, buffer: float = 0.0) -> tuple[float, float, float, float]:
        return (self.x0 - buffer, self.y1 - self.height * self.res - buffer,
                self.x0 + self.width * self.res + buffer, self.y1 + buffer)

    def extent(self) -> list[float]:
        """matplotlib ``imshow`` extent."""
        x0, y0, x1, y1 = self.bounds()
        return [x0, x1, y0, y1]

    def bin_count(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = np.floor((x - self.x0) / self.res).astype(np.int64)
        r = np.floor((self.y1 - y) / self.res).astype(np.int64)
        ok = (c >= 0) & (c < self.width) & (r >= 0) & (r < self.height)
        flat = np.bincount(r[ok] * self.width + c[ok], minlength=self.width * self.height)
        return flat.reshape(self.shape).astype(float)

    def mask(self, geom) -> np.ndarray:
        """Cells whose centre falls inside ``geom``."""
        return rasterize([(mapping(geom), 1)], out_shape=self.shape,
                         transform=self.transform, fill=0, dtype="uint8").astype(bool)

    def read(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            if (src.width, src.height) != (self.width, self.height) or \
                    not np.allclose(tuple(src.transform)[:6], tuple(self.transform)[:6]):
                raise ValueError(f"{path} is not on the analysis grid")
            z = src.read(1, masked=True).astype("float64")
        return z.filled(np.nan)

    def write(self, path: Path, arr: np.ndarray, nodata: float = -9999.0) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.where(np.isfinite(arr), arr, nodata).astype("float32")
        with rasterio.open(path, "w", driver="GTiff", width=self.width, height=self.height,
                           count=1, dtype="float32", crs=self.crs, transform=self.transform,
                           nodata=nodata, compress="deflate", tiled=True) as dst:
            dst.write(data, 1)
        return path
