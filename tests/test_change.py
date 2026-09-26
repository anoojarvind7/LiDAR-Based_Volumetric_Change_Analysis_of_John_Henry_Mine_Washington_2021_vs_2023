"""End-to-end check of the change stage on synthetic surveys with a known answer."""

import json

import numpy as np
import pytest
import yaml

from minevol import workflow as wf
from minevol.grid import Grid


def test_change_recovers_known_cut_fill_and_datum_shift(tmp_path):
    bounds = [0, 0, 400, 400]
    cfg = {
        "site": {"name": "synthetic", "crs": "EPSG:6339", "bounds": bounds,
                 "features": "features.geojson"},
        "surveys": {"a": {"resolution": 1.0, "sigma_z": 0.05},
                    "b": {"resolution": 0.5, "sigma_z": 0.05}},
        "change": {"pair": ["a", "b"], "lod_confidence": 0.95, "stable_max_slope_deg": 15},
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "c.yaml").write_text(yaml.safe_dump(cfg))
    square = [[[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]]
    feature = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::6339"}},
        "features": [{"type": "Feature", "properties": {"name": "pile", "kind": "pile"},
                      "geometry": {"type": "Polygon", "coordinates": square}}],
    }
    (tmp_path / "features.geojson").write_text(json.dumps(feature))
    prj = wf.Project.load(tmp_path / "config" / "c.yaml")

    rng = np.random.default_rng(4)
    ga, gb = prj.grid("a"), prj.grid("b")
    xb = gb.x0 + (np.arange(gb.width) + 0.5) * gb.res
    xx, yy = np.meshgrid(xb, gb.y1 - (np.arange(gb.height) + 0.5) * gb.res)
    ground = 200 + 0.02 * xx
    old_b = ground.copy()
    new_b = ground + 0.12  # datum shift between surveys, to be removed
    # 3 m of fill on a 50 x 50 m square and 2 m of cut on a 40 x 40 m square.
    fill = (xx > 120) & (xx < 170) & (yy > 120) & (yy < 170)
    cut = (xx > 260) & (xx < 300) & (yy > 260) & (yy < 300)
    new_b[fill] += 3.0
    new_b[cut] -= 2.0
    old_a = old_b.reshape(ga.height, 2, ga.width, 2).mean(axis=(1, 3))
    Grid.write(ga, prj.dtm("a"), old_a + rng.normal(0, 0.03, ga.shape))
    Grid.write(gb, prj.dtm("b"), new_b + rng.normal(0, 0.03, gb.shape))

    meta = wf.change(prj)
    assert meta["coregistration"]["median_m"] == pytest.approx(0.12, abs=0.01)
    df = __import__("pandas").read_csv(tmp_path / "results" / "change" / "change_a_b.csv")
    site = df[df.region == "whole site"].iloc[0]
    # Over a whole site, ~5 % of unchanged cells exceed a 95 % LoD by chance and
    # add a little to both totals; they cancel in the net.
    assert site.fill_m3 == pytest.approx(3.0 * 50 * 50, rel=0.12)
    assert site.cut_m3 == pytest.approx(2.0 * 40 * 40, rel=0.12)
    assert site.net_m3 == pytest.approx(3.0 * 50 * 50 - 2.0 * 40 * 40, abs=150)
    pile = df[df.region == "pile"].iloc[0]
    assert pile.fill_m3 == pytest.approx(3.0 * 50 * 50, rel=0.02)
    assert pile.cut_m3 == pytest.approx(0, abs=50)
