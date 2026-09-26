"""End-to-end stages, driven by the YAML config."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from . import pointcloud as pc
from . import uncertainty as unc
from .grid import Grid
from .volume import cut_fill, edge_contact, feature_volume, refine_mask, rim_stats
from .water import extend_into_gaps, find_water, flatten

log = logging.getLogger(__name__)


@dataclass
class Project:
    root: Path
    cfg: dict = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> Project:
        path = Path(path).resolve()
        return cls(path.parent.parent, yaml.safe_load(path.read_text()))

    # paths --------------------------------------------------------------
    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def raw(self, s): return self.p("data", "raw", f"{s}.laz")
    def clean(self, s, g): return self.p("data", "interim", f"{s}_{g}.laz")
    def dtm(self, s, g="vendor", hf=True):
        return self.p("data", "processed", f"{s}_{g}_dtm{'_hf' if hf else ''}.tif")
    def pipelines(self, s, name): return self.p("results", "pipelines", f"{s}_{name}.json")

    def survey(self, s) -> dict:
        return self.cfg["surveys"][s]

    def grid(self, s) -> Grid:
        return Grid.from_bounds(self.cfg["site"]["bounds"], self.survey(s)["resolution"],
                                self.cfg["site"]["crs"])

    def _read(self, key: str) -> gpd.GeoDataFrame:
        df = gpd.read_file(self.p(self.cfg["site"][key]))
        if df.crs is None:
            df = df.set_crs(self.cfg["site"]["crs"])
        return df.to_crs(self.cfg["site"]["crs"])

    def features(self) -> gpd.GeoDataFrame:
        """Hand-drawn search outlines (generous rectangles)."""
        return self._read("features")

    def toes(self) -> gpd.GeoDataFrame:
        """Reviewed toe/crest polygons that define each feature's volume."""
        return self._read("toes")


# --------------------------------------------------------------------------- stages

def fetch(prj: Project, s: str) -> Path:
    src = prj.survey(s)["source"]
    out = prj.raw(s)
    if out.exists():
        log.info("%s already downloaded", out)
        return out
    g = prj.grid(s)
    if src["type"] == "ept":
        return pc.fetch_ept(src["url"], g, out, dump=prj.pipelines(s, "fetch"))
    if src.get("local_dir") and Path(src["local_dir"]).is_dir():
        # Tiles already downloaded by hand (e.g. from OpenTopography's web bucket browser).
        tiles = sorted(Path(src["local_dir"]).glob(src.get("local_glob", "*.la[sz]")))
        if tiles:
            return pc.merge_tiles(tiles, g, out, src.get("in_srs"),
                                  dump=prj.pipelines(s, "fetch"))
    if src["type"] == "s3":
        tiles = pc.fetch_s3_tiles(src["endpoint"], src["bucket"], src["prefix"],
                                  prj.p("data", "raw", s))
        return pc.merge_tiles(tiles, g, out, src.get("in_srs"), dump=prj.pipelines(s, "fetch"))
    if src["type"] == "local":
        tiles = sorted(prj.p(src["dir"]).glob("*.la[sz]"))
        return pc.merge_tiles(tiles, g, out, src.get("in_srs"), dump=prj.pipelines(s, "fetch"))
    raise ValueError(f"unknown source type {src['type']}")


def process(prj: Project, s: str, ground: str = "vendor") -> dict:
    """Clean, (re)classify, build the TIN DTM, find and flatten water."""
    sv, g = prj.survey(s), prj.grid(s)
    clean = prj.clean(s, ground)
    if not clean.exists():
        pc.clean_and_classify(prj.raw(s), clean, ground, dump=prj.pipelines(s, f"clean_{ground}"))
    raw_dtm = prj.dtm(s, ground, hf=False)
    pc.tin_dtm(clean, g, raw_dtm, thin_m=sv.get("thin_m"),
               dump=prj.pipelines(s, f"dtm_{ground}"))
    z = g.read(raw_dtm)

    pts = pc.class_points(clean, (pc.GROUND, pc.WATER))
    w = pts["Classification"] == pc.WATER
    labels, bodies = find_water(g, pts["X"][w], pts["Y"][w], pts["Z"][w])
    labels = extend_into_gaps(labels, ~np.isfinite(z))
    for b in bodies:  # areas and outlines after filling mid-lake gaps
        cells = labels == b.id
        b.area_m2 = float(cells.sum() * g.res**2)
        b.geometry = unary_union([shape(gm) for gm, v in shapes(
            cells.astype("uint8"), mask=cells, transform=g.transform) if v])
    g.write(prj.dtm(s, ground), flatten(z, labels, bodies))
    ground_density = g.bin_count(pts["X"][~w], pts["Y"][~w]) / g.res**2
    g.write(prj.p("data", "processed", f"{s}_{ground}_ground_density.tif"), ground_density)

    out = prj.p("results", s)
    out.mkdir(parents=True, exist_ok=True)
    if bodies:
        gpd.GeoDataFrame(
            [{"id": b.id, "level_m": round(b.level_m, 3), "level_iqr_m": round(b.level_iqr_m, 3),
              "area_m2": round(b.area_m2), "n_points": b.n_points, "geometry": b.geometry}
             for b in bodies], crs=g.crs).to_file(out / "water_bodies.geojson", driver="GeoJSON")
    summary = {
        "survey": s, "ground": ground, "resolution_m": g.res,
        "dtm_valid_fraction": float(np.isfinite(z).mean()),
        "ground_density_median_pts_m2": float(np.median(ground_density[np.isfinite(z)])),
        "water_bodies": len(bodies),
    }
    (out / f"process_{ground}.json").write_text(json.dumps(summary, indent=2))
    return summary


def toes(prj: Project, s: str, ground: str = "vendor") -> gpd.GeoDataFrame:
    """Propose toe polygons from the search outlines (see ``refine_mask``).

    The output is meant to be reviewed against the hillshade and committed; it
    is the auditable definition of each feature. A toe sweep (every relief /
    padding pair in the config) is saved alongside to show how sensitive the
    volume is to where the toe is drawn.
    """
    t = prj.cfg["toes"]
    # Toe placement does not need sub-metre cells, and the iterative solve is
    # the slowest step, so it runs on a block-averaged grid.
    g = Grid.from_bounds(prj.cfg["site"]["bounds"],
                         max(t.get("resolution_m", 1.0), prj.survey(s)["resolution"]),
                         prj.cfg["site"]["crs"])
    z = _resample_to(prj, s, ground, g)
    rows, sweep = [], []
    # Keep the search area (and so its rim) inside the survey's coverage.
    covered = ndimage.binary_erosion(np.isfinite(z), iterations=2)
    for _, f in prj.features().iterrows():
        outline = g.mask(f.geometry) & covered
        for relief, pad in itertools.product(t["sweep_relief_m"], t["sweep_pad_m"]):
            m = refine_mask(z, outline, f["kind"], g.res, relief=relief, pad_m=pad)
            vol, _, _ = feature_volume(z, m, f["kind"], g.res)
            sweep.append({"feature": f["name"], "relief_m": relief, "pad_m": pad,
                          "area_m2": float(m.sum() * g.res**2), "volume_m3": vol,
                          "edge_contact": edge_contact(m, outline)})
        m = refine_mask(z, outline, f["kind"], g.res, relief=t["relief_m"], pad_m=t["pad_m"])
        geom = unary_union([shape(gm) for gm, v in shapes(m.astype("uint8"), mask=m,
                                                          transform=g.transform) if v])
        geom = max(getattr(geom, "geoms", [geom]), key=lambda p: p.area)
        rows.append({"name": f["name"], "kind": f["kind"], "source_survey": s,
                     "relief_m": t["relief_m"], "pad_m": t["pad_m"],
                     "edge_contact": round(edge_contact(m, outline), 3),
                     "geometry": geom.simplify(g.res / 2)})
    out = prj.p("results", s)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sweep).to_csv(out / f"toe_sweep_{ground}.csv", index=False)
    gdf = gpd.GeoDataFrame(rows, crs=g.crs)
    gdf.to_file(out / "toes_proposed.geojson", driver="GeoJSON")
    return gdf


def volumes(prj: Project, s: str, ground: str = "vendor") -> pd.DataFrame:
    """Volume of every feature above (pile) or below (pit) its base surface."""
    g = prj.grid(s)
    z = g.read(prj.dtm(s, ground))
    vcfg = prj.cfg["volumes"]
    methods, buffers = vcfg["base_methods"], vcfg["toe_buffer_m"]
    unc_cfg = _survey_error(prj, s)

    rows = []
    for _, f in prj.toes().iterrows():
        ens = []
        for buf, method in itertools.product(buffers, methods):
            m = g.mask(f.geometry.buffer(buf) if buf else f.geometry)
            vol, _, _ = feature_volume(z, m, f["kind"], g.res, method)
            ens.append({"toe_buffer_m": buf, "method": method, "volume": vol})
        ens = pd.DataFrame(ens)
        mask = g.mask(f.geometry)
        if not np.isfinite(z[mask]).all():
            log.warning("%s: %d no-data cells inside toe", f["name"],
                        int((~np.isfinite(z[mask])).sum()))
        v0, base, cf = feature_volume(z, mask, f["kind"], g.res, methods[0])
        area = float(mask.sum() * g.res**2)
        s_ref = float(ens["volume"].std(ddof=1))
        s_srv = unc.survey_sigma(area, g.res, **unc_cfg)
        dz = z - base
        rows.append({
            "feature": f["name"], "kind": f["kind"], "survey": s, "ground": ground,
            "volume_m3": v0, "sigma_m3": unc.combine(s_ref, s_srv),
            "sigma_reference_m3": s_ref, "sigma_survey_m3": s_srv,
            "ensemble_min_m3": float(ens["volume"].min()),
            "ensemble_max_m3": float(ens["volume"].max()),
            "area_m2": area,
            "max_height_m": float(np.nanmax(np.abs(dz[mask]))),
            "opposite_sign_m3": cf.cut_m3 if f["kind"] == "pile" else cf.fill_m3,
            "ground_pts_per_m2": _density_in(prj, s, ground, g, mask),
            **rim_stats(z, mask),
        })
        ens.to_csv(prj.p("results", s, f"ensemble_{_slug(f['name'])}_{ground}.csv"), index=False)
        g.write(prj.p("data", "processed", f"{s}_{ground}_base_{_slug(f['name'])}.tif"),
                np.where(mask, base, np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(prj.p("results", s, f"feature_volumes_{ground}.csv"), index=False)
    return df


def resolution_study(prj: Project, s: str, factors=(1, 2, 5)) -> pd.DataFrame:
    """Central volumes on coarser, block-averaged copies of the DTM (same toes)."""
    g = prj.grid(s)
    z = g.read(prj.dtm(s))
    method = prj.cfg["volumes"]["base_methods"][0]
    rows = []
    for k in factors:
        if g.height % k or g.width % k:
            continue
        zk = z.reshape(g.height // k, k, g.width // k, k).mean(axis=(1, 3))
        gk = Grid(g.x0, g.y1, g.width // k, g.height // k, g.res * k, g.crs)
        for _, f in prj.toes().iterrows():
            vol, _, _ = feature_volume(zk, gk.mask(f.geometry), f["kind"], gk.res, method)
            rows.append({"feature": f["name"], "resolution_m": gk.res, "volume_m3": vol})
    df = pd.DataFrame(rows)
    df.to_csv(prj.p("results", s, "resolution_study.csv"), index=False)
    return df


def change(prj: Project, ground: str = "vendor") -> dict:
    """Difference two surveys on the coarser grid with a level of detection."""
    from scipy import stats

    old, new = prj.cfg["change"]["pair"]
    res = max(prj.survey(old)["resolution"], prj.survey(new)["resolution"])
    g = Grid.from_bounds(prj.cfg["site"]["bounds"], res, prj.cfg["site"]["crs"])
    z0 = _resample_to(prj, old, ground, g)
    z1 = _resample_to(prj, new, ground, g)
    dz = z1 - z0

    # Stable ground: outside every search outline (padded), not water, gentle.
    feats = prj.features()
    regions = prj.toes() if "toes" in prj.cfg["site"] else feats
    disturbed = g.mask(unary_union(feats.geometry).buffer(25))
    wet = np.zeros(g.shape, bool)
    for s in (old, new):
        wb = prj.p("results", s, "water_bodies.geojson")
        if wb.exists():
            wet |= g.mask(unary_union(gpd.read_file(wb).geometry).buffer(5))
    gy, gx = np.gradient(z1, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    stable = ~disturbed & ~wet
    bias = unc.stable_ground_bias(dz, stable, slope, prj.cfg["change"]["stable_max_slope_deg"])
    dz_c = dz - bias["median_m"]

    lags, gamma = unc.empirical_variogram(
        np.where(stable, dz_c, np.nan), stable & np.isfinite(dz_c) & (np.abs(dz_c) < 1)
        & (slope < prj.cfg["change"]["stable_max_slope_deg"]), res)
    vg = unc.fit_spherical(lags, gamma)
    z_crit = stats.norm.ppf(0.5 + prj.cfg["change"]["lod_confidence"] / 2)
    lod = z_crit * np.hypot(vg["sigma_r"], vg["sigma_c"])
    # Uncertainty of the co-registration shift itself: the stable-ground
    # median is only as good as the number of independent patches it averages.
    n_eff = max(bias["n_cells"] * res**2 / (np.pi * vg["range_m"] ** 2), 1.0)
    sigma_bias = float(np.hypot(vg["sigma_r"], vg["sigma_c"]) / np.sqrt(n_eff))

    g.write(prj.p("data", "processed", f"dod_{old}_{new}.tif"), dz_c)
    rows = []
    regions = [("whole site", np.ones(g.shape, bool) & ~wet)] + \
              [(f["name"], g.mask(f.geometry) & ~wet) for _, f in regions.iterrows()]
    for name, m in regions:
        cf = cut_fill(z1 - bias["median_m"], z0, res, m, threshold=lod)
        a_det = cf.fill_area_m2 + cf.cut_area_m2
        rows.append({
            "region": name, "fill_m3": cf.fill_m3, "cut_m3": cf.cut_m3, "net_m3": cf.net_m3,
            "fill_area_m2": cf.fill_area_m2, "cut_area_m2": cf.cut_area_m2,
            "sigma_fill_m3": unc.survey_sigma(cf.fill_area_m2, res, vg["sigma_r"], vg["sigma_c"],
                                              vg["range_m"], sigma_bias),
            "sigma_cut_m3": unc.survey_sigma(cf.cut_area_m2, res, vg["sigma_r"], vg["sigma_c"],
                                             vg["range_m"], sigma_bias),
            "area_valid_m2": cf.area_m2, "area_detected_m2": a_det,
        })
    df = pd.DataFrame(rows)
    out = prj.p("results", "change")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"change_{old}_{new}.csv", index=False)
    meta = {"pair": [old, new], "resolution_m": res, "coregistration": bias,
            "variogram": vg, "lod_m": float(lod), "sigma_bias_m": float(sigma_bias),
            "variogram_lags_m": lags.tolist(), "variogram_gamma": np.nan_to_num(gamma).tolist()}
    (out / f"change_{old}_{new}.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- helpers

def _survey_error(prj: Project, s: str) -> dict:
    """Survey error parameters: from the change analysis if available, else config."""
    sv = prj.survey(s)
    meta = prj.p("results", "change", "change_{}_{}.json".format(*prj.cfg["change"]["pair"]))
    if meta.exists():
        vg = json.loads(meta.read_text())["variogram"]
        # The DoD variogram mixes both surveys' errors; split evenly per survey.
        return {"sigma_r": vg["sigma_r"] / np.sqrt(2), "sigma_c": vg["sigma_c"] / np.sqrt(2),
                "range_m": vg["range_m"]}
    return {"sigma_r": sv["sigma_z"], "sigma_c": 0.5 * sv["sigma_z"], "range_m": 50.0}


def _density_in(prj: Project, s: str, ground: str, g: Grid, mask: np.ndarray) -> float:
    path = prj.p("data", "processed", f"{s}_{ground}_ground_density.tif")
    return float(np.nanmean(g.read(path)[mask])) if path.exists() else float("nan")


def _resample_to(prj: Project, s: str, ground: str, g: Grid) -> np.ndarray:
    src = prj.grid(s)
    z = src.read(prj.dtm(s, ground))
    k = int(round(g.res / src.res))
    if k == 1:
        return z
    return z.reshape(src.height // k, k, src.width // k, k).mean(axis=(1, 3))


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")

