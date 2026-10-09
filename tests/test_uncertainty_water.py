import numpy as np
import pytest

from minevol.grid import Grid
from minevol.uncertainty import empirical_variogram, fit_spherical, stable_ground_bias, survey_sigma
from minevol.water import find_water, flatten


def test_survey_sigma_random_term_scales_with_sqrt_area():
    s1 = survey_sigma(10_000, 1.0, sigma_r=0.1)
    s4 = survey_sigma(40_000, 1.0, sigma_r=0.1)
    assert s4 == pytest.approx(2 * s1)
    # A tiny area is almost fully correlated: sigma * A.
    assert survey_sigma(1, 1.0, 0.0, sigma_c=0.05, range_m=50) == pytest.approx(0.05, rel=0.02)
    # Continuous where the circle-equivalent radius equals the range (Rolstad eq. 8).
    at = np.pi * 50.0**2
    below = survey_sigma(at * (1 - 1e-9), 1.0, 0.0, sigma_c=0.05, range_m=50)
    above = survey_sigma(at * (1 + 1e-9), 1.0, 0.0, sigma_c=0.05, range_m=50)
    assert below == pytest.approx(above, rel=1e-6)
    assert above == pytest.approx(0.05 * at / np.sqrt(5), rel=1e-6)
    # A larger feature never has a smaller correlated error than a smaller one.
    areas = np.linspace(1_000, 200_000, 50)
    s = [survey_sigma(a, 1.0, 0.0, sigma_c=0.05, range_m=160) for a in areas]
    assert np.all(np.diff(s) > 0)


def test_stable_ground_bias_is_robust_to_outliers():
    rng = np.random.default_rng(1)
    dz = 0.07 + rng.normal(0, 0.05, (200, 200))
    dz[:20] = 5.0  # real change that must be ignored
    out = stable_ground_bias(dz, np.ones_like(dz, bool))
    assert out["median_m"] == pytest.approx(0.07, abs=0.005)
    assert out["nmad_m"] == pytest.approx(0.05, abs=0.005)


def test_variogram_recovers_white_noise_nugget():
    rng = np.random.default_rng(2)
    dz = rng.normal(0, 0.1, (300, 300))
    lags, gam = empirical_variogram(dz, np.ones_like(dz, bool), 1.0, max_lag_m=100)
    assert np.nanmedian(np.sqrt(gam)) == pytest.approx(0.1, rel=0.1)
    fit = fit_spherical(lags, gam)
    assert np.hypot(fit["sigma_r"], fit["sigma_c"]) == pytest.approx(0.1, rel=0.15)


def test_water_level_and_flatten():
    g = Grid(0.0, 100.0, 100, 100, 1.0, "EPSG:6339")
    rng = np.random.default_rng(3)
    x = rng.uniform(20, 60, 5000)
    y = rng.uniform(30, 70, 5000)
    z = 230.5 + rng.normal(0, 0.03, 5000)
    labels, bodies = find_water(g, x, y, z, min_area_m2=100)
    assert len(bodies) == 1
    assert bodies[0].level_m == pytest.approx(230.5, abs=0.01)
    assert bodies[0].area_m2 == pytest.approx(1600, rel=0.1)
    dtm = np.full(g.shape, 229.0)
    flat = flatten(dtm, labels, bodies)
    assert np.allclose(flat[labels == 1], bodies[0].level_m)


def test_lake_grows_into_its_own_gap_but_not_the_survey_edge():
    from minevol.water import extend_into_gaps

    labels = np.zeros((50, 50), int)
    labels[10:30, 10:30] = 1
    nodata = np.zeros((50, 50), bool)
    nodata[15:25, 15:25] = True      # hole in the middle of the lake
    labels[15:25, 15:25] = 0
    nodata[:, 45:] = True            # outside the survey
    out = extend_into_gaps(labels, nodata)
    assert (out[15:25, 15:25] == 1).all()
    assert (out[:, 45:] == 0).all()


def test_total_sigma_sums_ensemble_members_before_spread():
    import pandas as pd

    from minevol.uncertainty import total_sigma

    members = [(b, m) for b in (0.0, -2.0, 2.0) for m in ("harmonic", "tin")]
    shift = {(0.0, "harmonic"): 0, (0.0, "tin"): 5, (-2.0, "harmonic"): -10,
             (-2.0, "tin"): -5, (2.0, "harmonic"): 10, (2.0, "tin"): 15}

    def ens(v0):
        return pd.DataFrame([{"toe_buffer_m": b, "method": m, "volume": v0 + shift[(b, m)]}
                             for b, m in members])

    one = ens(100)["volume"].std(ddof=1)
    # Two features that respond identically to the shared choices: the spread
    # of the total is twice one feature's, not sqrt(2) times.
    assert total_sigma([ens(100), ens(300)], [0.0, 0.0]) == pytest.approx(2 * one)
    assert total_sigma([ens(100)], [3.0]) == pytest.approx(np.hypot(one, 3.0))
