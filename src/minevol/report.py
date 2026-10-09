"""Figures and the results summary (``results/SUMMARY.md``)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, TwoSlopeNorm

from . import uncertainty as unc
from .workflow import Project, _resample_to, _slug

INK, MUTED, ACCENT = "#1f2328", "#6e7781", "#0969da"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


def hillshade(z: np.ndarray, res: float) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=2, dx=res, dy=res)


def _extent2(g):
    e = g.extent()
    return [e[0], e[1], e[3], e[2]]


def fig_overview(prj: Project, s: str, out: Path) -> None:
    g = prj.grid(s)
    z = g.read(prj.dtm(s))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(hillshade(z, g.res), cmap="gray", extent=g.extent())
    im = ax.imshow(z, cmap="cividis", alpha=0.45, extent=g.extent())
    ax.contour(z, levels=np.arange(150, 400, 10), colors="k", linewidths=0.25, alpha=0.5,
               extent=_extent2(g))
    wb = prj.p("results", s, "water_bodies.geojson")
    if wb.exists():
        gpd.read_file(wb).plot(ax=ax, color="#9ecae1", edgecolor="#3182bd", linewidth=0.6)
    o = prj.toes()
    o.boundary.plot(ax=ax, color="#d1242f", linewidth=1.2)
    for _, r in o.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(r["name"], (c.x, c.y), ha="center", fontsize=8, color=INK,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})
    ax.set_xlabel("Easting (m, UTM 10N)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_title(f"{prj.cfg['site']['name']} — {prj.survey(s)['label']}", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Elevation (m NAVD88)")
    fig.savefig(out)
    plt.close(fig)


def fig_relief(prj: Project, s: str, out: Path) -> None:
    """Height above base surface for every feature, with its toe."""
    g = prj.grid(s)
    z = g.read(prj.dtm(s))
    feats = prj.toes()
    n = len(feats)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(11, 4.8 * rows))
    for ax in np.atleast_1d(axs).flat[n:]:
        ax.set_visible(False)
    for ax, (_, f) in zip(np.atleast_1d(axs).flat, feats.iterrows()):  # noqa: B905
        base = g.read(prj.p("data", "processed", f"{s}_vendor_base_{_slug(f['name'])}.tif"))
        rel = z - base
        x0, y0, x1, y1 = f.geometry.buffer(40).bounds
        ax.imshow(hillshade(z, g.res), cmap="gray", extent=g.extent())
        vmax = np.nanpercentile(np.abs(rel), 99)
        im = ax.imshow(rel, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), alpha=0.8,
                       extent=g.extent())
        gpd.GeoSeries([f.geometry]).boundary.plot(ax=ax, color=INK, linewidth=0.8)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f["name"], loc="left")
        fig.colorbar(im, ax=ax, shrink=0.7, label="Height above base (m)")
    fig.savefig(out)
    plt.close(fig)


def fig_volumes(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("volume_m3")
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.barh(y, d["volume_m3"] / 1e6, color=ACCENT, height=0.55)
    ax.errorbar(d["volume_m3"] / 1e6, y, xerr=1.96 * d["sigma_m3"] / 1e6, fmt="none",
                ecolor=INK, capsize=3, lw=1)
    for yi, (v, sg) in enumerate(zip(d["volume_m3"], d["sigma_m3"], strict=True)):
        ax.text(v / 1e6 + 1.96 * sg / 1e6 + 0.03, yi, f"{v / 1e6:.2f} ± {1.96 * sg / 1e6:.2f}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d["feature"])
    ax.set_xlabel("Volume (million m³), error bar = 95 % interval")
    ax.grid(axis="x", color="#d0d7de", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0, (d["volume_m3"] + 1.96 * d["sigma_m3"]).max() / 1e6 * 1.3)
    fig.savefig(out)
    plt.close(fig)


def fig_toe_sweep(sweep: pd.DataFrame, central: dict, out: Path) -> None:
    """Volume versus toe threshold, one panel per feature, one line per padding."""
    feats = list(dict.fromkeys(sweep["feature"]))
    fig, axs = plt.subplots(1, len(feats), figsize=(3.6 * len(feats), 3.2), sharey=False)
    pads = sorted(sweep["pad_m"].unique())
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(pads)))
    for ax, name in zip(np.atleast_1d(axs), feats, strict=True):
        d = sweep[sweep.feature == name]
        for pad, c in zip(pads, shades, strict=True):
            dd = d[d.pad_m == pad].sort_values("relief_m")
            ax.plot(dd.relief_m, dd.volume_m3 / 1e6, "-o", color=c, ms=3, lw=1.5,
                    label=f"pad {pad:g} m")
        ax.axvline(central["relief_m"], color=MUTED, lw=0.8, ls="--")
        ax.set_title(name, loc="left")
        ax.set_xlabel("Toe threshold (m above base)")
        ax.grid(color="#d0d7de", lw=0.5)
    np.atleast_1d(axs)[0].set_ylabel("Volume (million m³)")
    np.atleast_1d(axs)[-1].legend(frameon=False, fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def fig_change(prj: Project, out: Path) -> None:
    old, new = prj.cfg["change"]["pair"]
    path = prj.p("data", "processed", f"dod_{old}_{new}.tif")
    meta = json.loads(prj.p("results", "change", f"change_{old}_{new}.json").read_text())
    from .grid import Grid

    g = Grid.from_bounds(prj.cfg["site"]["bounds"], meta["resolution_m"], prj.cfg["site"]["crs"])
    dz = g.read(path)
    z = _resample_to(prj, new, "vendor", g)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.imshow(hillshade(z, g.res), cmap="gray", extent=g.extent())
    shown = np.where(np.abs(dz) >= meta["lod_m"], dz, np.nan)
    im = ax.imshow(shown, cmap="RdBu_r", norm=TwoSlopeNorm(0, -10, 10), extent=g.extent())
    prj.toes().boundary.plot(ax=ax, color=INK, linewidth=0.6)
    fig.colorbar(im, ax=ax, shrink=0.7, label=f"Elevation change {old} → {new} (m)")
    ax.set_title(f"Change above the {meta['lod_m']:.2f} m level of detection", loc="left")
    ax.ticklabel_format(style="plain", useOffset=False)
    lags, gam = np.array(meta["variogram_lags_m"]), np.array(meta["variogram_gamma"])
    ax2.plot(lags, np.sqrt(gam), "o", color=ACCENT, ms=4)
    ax2.set_xlabel("Lag (m)")
    ax2.set_ylabel("√semivariance of stable-ground Δz (m)")
    vg = meta["variogram"]
    ax2.set_title(f"σ_r={vg['sigma_r']:.3f} m, σ_c={vg['sigma_c']:.3f} m, "
                  f"range={vg['range_m']:.0f} m", loc="left")
    ax2.grid(color="#d0d7de", lw=0.5)
    fig.savefig(out)
    plt.close(fig)


def total_sigma(prj: Project, s: str, df: pd.DataFrame, ground: str = "vendor") -> float:
    """1 σ of the summed volume of all features (see :func:`unc.total_sigma`)."""
    ens = [prj.p("results", s, f"ensemble_{_slug(n)}_{ground}.csv") for n in df["feature"]]
    if not all(p.exists() for p in ens):
        return float(np.sqrt((df["sigma_m3"] ** 2).sum()))
    return unc.total_sigma([pd.read_csv(p) for p in ens], df["sigma_survey_m3"])


def build_report(prj: Project) -> Path:
    figs = prj.p("results", "figures")
    figs.mkdir(parents=True, exist_ok=True)
    lines = [f"# Results — {prj.cfg['site']['name']}", "",
             "Generated by `minevol report`. Volumes in cubic metres; ± is the 95 % interval "
             "(1.96 σ). See `docs/method.md` for how each number is produced.", ""]
    for s in prj.cfg["surveys"]:
        csv = prj.p("results", s, "feature_volumes_vendor.csv")
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        fig_overview(prj, s, figs / f"{s}_overview.png")
        fig_relief(prj, s, figs / f"{s}_relief.png")
        fig_volumes(df, figs / f"{s}_volumes.png")
        lines += [f"## {prj.survey(s)['label']}", "",
                  ("| Feature | Volume (m³) | ± 95 % (m³) | Footprint (m²) | Max height (m) "
                   "| Ground pts/m² | Base share of σ |"),
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in df.iterrows():
            share = r["sigma_reference_m3"] ** 2 / r["sigma_m3"] ** 2
            lines.append(f"| {r['feature']} | {r['volume_m3']:,.0f} | {1.96 * r['sigma_m3']:,.0f} "
                         f"| {r['area_m2']:,.0f} | {r['max_height_m']:.1f} "
                         f"| {r['ground_pts_per_m2']:.1f} | {share:.0%} |")
        tot, tot_s = df["volume_m3"].sum(), total_sigma(prj, s, df)
        lines += [f"| **Total** | **{tot:,.0f}** | {1.96 * tot_s:,.0f} | | | | |", ""]
        rs = prj.p("results", s, "resolution_study.csv")
        if rs.exists():
            r = pd.read_csv(rs).pivot(index="feature", columns="resolution_m", values="volume_m3")
            lines += ["Grid-resolution check (central volume, m³):", "",
                      "| Feature | " + " | ".join(f"{c:g} m" for c in r.columns) + " |",
                      "|---|" + "---:|" * len(r.columns)]
            lines += [f"| {i} | " + " | ".join(f"{v:,.0f}" for v in row) + " |"
                      for i, row in r.iterrows()]
            lines.append("")
        wb = prj.p("results", s, "water_bodies.geojson")
        if wb.exists():
            w = gpd.read_file(wb).sort_values("area_m2", ascending=False)
            w = w[w["area_m2"] >= 5000]
            lines += ["Water bodies ≥ 0.5 ha (surface level from lidar water returns):", "",
                      "| Area (ha) | Level (m NAVD88) | IQR of returns (m) |", "|---:|---:|---:|"]
            lines += [f"| {r.area_m2 / 1e4:.2f} | {r.level_m:.2f} | {r.level_iqr_m:.2f} |"
                      for r in w.itertuples()]
            lines.append("")
        sw = prj.p("results", s, "toe_sweep_vendor.csv")
        if sw.exists():
            fig_toe_sweep(pd.read_csv(sw), prj.cfg["toes"], figs / f"{s}_toe_sweep.png")
            lines += ["How much the volume depends on where the toe is drawn (not part of "
                      "the ± above; see docs/method.md):", "",
                      f"![toe sweep](figures/{s}_toe_sweep.png)", ""]
        lines += [f"![overview](figures/{s}_overview.png)", "",
                  f"![volumes](figures/{s}_volumes.png)", "",
                  f"![relief](figures/{s}_relief.png)", ""]
    old, new = prj.cfg["change"]["pair"]
    cc = prj.p("results", "change", f"change_{old}_{new}.csv")
    if cc.exists():
        fig_change(prj, figs / "change.png")
        meta = json.loads(prj.p("results", "change", f"change_{old}_{new}.json").read_text())
        df = pd.read_csv(cc)
        lines += [f"## Change {old} → {new}", "",
                  f"Co-registration shift removed: {meta['coregistration']['median_m']:+.3f} m "
                  f"(NMAD {meta['coregistration']['nmad_m']:.3f} m on "
                  f"{meta['coregistration']['n_cells']:,} stable cells). "
                  f"Level of detection: {meta['lod_m']:.2f} m.", "",
                  "| Region | Fill (m³) | Cut (m³) | Net (m³) |", "|---|---:|---:|---:|"]
        lines += [f"| {r.region} | {r.fill_m3:,.0f} ± {1.96 * r.sigma_fill_m3:,.0f} "
                  f"| {r.cut_m3:,.0f} ± {1.96 * r.sigma_cut_m3:,.0f} | {r.net_m3:,.0f} |"
                  for r in df.itertuples()]
        lines += ["", "![change](figures/change.png)", ""]
    out = prj.p("results", "SUMMARY.md")
    out.write_text("\n".join(lines))
    return out
