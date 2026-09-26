import numpy as np
import pytest

from minevol.grid import Grid
from minevol.uncertainty import empirical_variogram, fit_spherical, stable_ground_bias, survey_sigma
from minevol.water import find_water, flatten


def test_survey_sigma_random_term_scales_with_sqrt_area():
    s1 = survey_sigma(10_000, 1.0, sigma_r=0.1)
    s4 = survey_sigma(40_000, 1.0, sigma_r=0.1)
    assert s4 == pytest.approx(2 * s1)
    # Fully correlated below the correlation area: sigma * A.
    assert survey_sigma(100, 1.0, 0.0, sigma_c=0.05, range_m=50) == pytest.approx(5.0)


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
