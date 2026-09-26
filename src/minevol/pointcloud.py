"""Point-cloud acquisition and processing with PDAL.

Everything that touches points goes through PDAL pipelines built here, so the
exact processing is visible as JSON (``--dump-pipelines``) and reproducible
with the ``pdal`` command-line tool alone.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pdal
import requests
from pyproj import Transformer

from .grid import Grid

log = logging.getLogger(__name__)

# ASPRS classes used below.
GROUND, WATER = 2, 9
NOISE_CLASSES = (7, 18)  # low noise, high noise


def run(stages: list[dict], dump: Path | None = None) -> pdal.Pipeline:
    """Execute a PDAL pipeline given as a list of stage dicts."""
    if dump is not None:
        dump.parent.mkdir(parents=True, exist_ok=True)
        # Paths are written relative to the working directory so the saved
        # pipelines replay from the repository root on any machine.
        text = json.dumps(stages, indent=2).replace(str(Path.cwd()) + "/", "")
        dump.write_text(text)
    pipe = pdal.Pipeline(json.dumps(stages))
    pipe.execute()
    return pipe


# --------------------------------------------------------------------------- fetch

def fetch_ept(url: str, grid: Grid, out: Path, buffer_m: float = 50.0,
              dump: Path | None = None) -> Path:
    """Read an Entwine Point Tile source (e.g. USGS 3DEP on AWS) for the AOI."""
    info = requests.get(url, timeout=60).json()
    ept_crs = f"EPSG:{info['srs']['horizontal']}"
    x0, y0, x1, y1 = grid.bounds(buffer_m)
    tr = Transformer.from_crs(grid.crs, ept_crs, always_xy=True)
    xs, ys = tr.transform([x0, x0, x1, x1], [y0, y1, y0, y1])
    stages = [
        {"type": "readers.ept", "filename": url, "threads": 8,
         "bounds": f"([{min(xs)},{max(xs)}],[{min(ys)},{max(ys)}])"},
        {"type": "filters.reprojection", "out_srs": grid.crs},
        {"type": "filters.crop", "bounds": f"([{x0},{x1}],[{y0},{y1}])"},
        _laz_writer(out, grid.crs),
    ]
    run(stages, dump)
    return out


def list_s3(endpoint: str, bucket: str, prefix: str) -> list[tuple[str, int]]:
    """List objects under ``prefix`` of a public S3-compatible bucket."""
    keys, token = [], None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(f"{endpoint}/{bucket}", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", ns):
            keys.append((c.find("s3:Key", ns).text, int(c.find("s3:Size", ns).text)))
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)


def fetch_s3_tiles(endpoint: str, bucket: str, prefix: str, out_dir: Path) -> list[Path]:
    """Download every LAS/LAZ tile under ``prefix`` (skips files already present)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for key, size in list_s3(endpoint, bucket, prefix):
        if not key.lower().endswith((".laz", ".las")):
            continue
        dst = out_dir / Path(key).name
        files.append(dst)
        if dst.exists() and dst.stat().st_size == size:
            continue
        log.info("downloading %s (%.1f MB)", key, size / 1e6)
        with requests.get(f"{endpoint}/{bucket}/{quote(key)}", stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f:
                f.writelines(r.iter_content(1 << 20))
            tmp.rename(dst)
    if not files:
        raise FileNotFoundError(f"no LAS/LAZ tiles under s3://{bucket}/{prefix} at {endpoint}")
    return files


def merge_tiles(tiles: list[Path], grid: Grid, out: Path, in_srs: str | None = None,
                dump: Path | None = None) -> Path:
    """Merge downloaded tiles, reproject if needed and crop to the AOI."""
    x0, y0, x1, y1 = grid.bounds()
    readers = [{"type": "readers.las", "filename": str(t),
                **({"override_srs": in_srs} if in_srs else {})} for t in tiles]
    stages = readers + [
        {"type": "filters.merge"},
        {"type": "filters.reprojection", "out_srs": grid.crs},
        {"type": "filters.crop", "bounds": f"([{x0},{x1}],[{y0},{y1}])"},
        _laz_writer(out, grid.crs),
    ]
    run(stages, dump)
    return out


# --------------------------------------------------------------------------- classify

def clean_and_classify(src: Path, out: Path, ground: str = "vendor",
                       dump: Path | None = None) -> Path:
    """Remove noise and (optionally) re-classify ground.

    ``ground="vendor"`` keeps the data provider's classification (reviewed by
    their QA and normally the better choice). ``ground="smrf"`` discards it
    and classifies ground from scratch with the Simple Morphological Filter
    (Pingel et al., 2013), which is used as an independent check.
    """
    stages: list[dict] = [
        {"type": "readers.las", "filename": str(src)},
        # Vendor-flagged noise and withheld points never reach a surface.
        {"type": "filters.expression",
         "expression": "Classification != 7 && Classification != 18 && Withheld == 0"},
        # Catch isolated spikes the vendor missed.
        {"type": "filters.outlier", "method": "statistical", "mean_k": 12, "multiplier": 3.0},
        {"type": "filters.expression", "expression": "Classification != 7"},
    ]
    if ground == "smrf":
        stages += [
            # Keep water labels, reset everything else, then run SMRF on the
            # last returns (first returns in forest are canopy).
            {"type": "filters.assign",
             "value": ["Classification = 1 WHERE Classification != 9"]},
            {"type": "filters.smrf", "ignore": "Classification[9:9]",
             "returns": "last,only", "slope": 0.2, "window": 18.0,
             "threshold": 0.45, "scalar": 1.2, "cell": 1.0},
        ]
    elif ground != "vendor":
        raise ValueError("ground must be 'vendor' or 'smrf'")
    stages.append(_laz_writer(out, None))
    run(stages, dump)
    return out


# --------------------------------------------------------------------------- rasters

def tin_dtm(src: Path, grid: Grid, out: Path, classes=(GROUND, WATER),
            max_edge_m: float = 50.0, thin_m: float | None = None,
            dump: Path | None = None) -> Path:
    """Bare-earth DTM: Delaunay TIN of ground (+water) points, rasterised.

    Rasterising the TIN (``filters.faceraster``) rather than binning points
    means every cell value is an exact linear interpolation between real
    ground returns, including under canopy where returns are sparse.
    Triangles longer than ``max_edge_m`` are left empty (no-data) so large
    gaps are visible instead of silently bridged.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    limits = ",".join(f"Classification[{c}:{c}]" for c in classes)
    stages: list[dict] = [{"type": "readers.las", "filename": str(src)},
                          {"type": "filters.range", "limits": limits}]
    if thin_m:
        stages.append({"type": "filters.sample", "radius": thin_m})
    stages += [
        {"type": "filters.delaunay"},
        {"type": "filters.faceraster", "resolution": grid.res,
         "origin_x": grid.x0, "origin_y": grid.y1 - grid.height * grid.res,
         "width": grid.width, "height": grid.height,
         "max_triangle_edge_length": max_edge_m},
        {"type": "writers.raster", "filename": str(out), "data_type": "float32",
         "gdalopts": "COMPRESS=DEFLATE,TILED=YES,PREDICTOR=3"},
    ]
    run(stages, dump)
    return out


def class_points(src: Path, classes) -> np.ndarray:
    """Return a structured array of the points in the given classes."""
    limits = ",".join(f"Classification[{c}:{c}]" for c in classes)
    pipe = run([{"type": "readers.las", "filename": str(src)},
                {"type": "filters.range", "limits": limits}])
    return np.concatenate(pipe.arrays) if pipe.arrays else np.empty(0)


def density(src: Path, grid: Grid, classes=(GROUND,)) -> np.ndarray:
    """Points per square metre on ``grid`` for the given classes."""
    pts = class_points(src, classes)
    counts = grid.bin_count(pts["X"], pts["Y"])
    return counts / (grid.res * grid.res)


def _laz_writer(out: Path, srs: str | None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    w = {"type": "writers.las", "filename": str(out), "compression": "laszip",
         "minor_version": 4, "extra_dims": "all", "forward": "all"}
    if srs:
        w["a_srs"] = srs
    return w
