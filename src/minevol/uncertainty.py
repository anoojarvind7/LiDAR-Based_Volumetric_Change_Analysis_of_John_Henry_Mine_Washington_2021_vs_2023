"""Volume uncertainty.

Two independent sources are combined in quadrature (1-sigma):

1. **Reference-surface uncertainty** - how much the answer depends on choices
   nobody can verify: the interpolator used for the base (the config's
   ``base_methods``) and the toe polygon shrunk and grown by the digitising
   tolerance (``toe_buffer_m``). It is the standard deviation of the volume
   over that ensemble. For stockpiles and pits this term usually dominates.

2. **Survey (elevation) uncertainty** - random and spatially correlated DTM
   error propagated over the feature's area A (Rolstad et al., 2009, J.
   Glaciol. 55(192), eq. 8, treating A as a circle of radius r)::

       sigma_V^2 = sigma_r^2 * a * A  +  sigma_c^2 * A^2 * f(r / L)
       f(q) = 1 - q + q^3 / 5     for q <= 1
       f(q) = 1 / (5 q^2)         for q > 1   (i.e. sigma_c^2 * A * pi L^2 / 5)

   with ``a`` the cell area, ``sigma_r`` the uncorrelated error, ``sigma_c``
   and ``L`` the correlated error and its range. A uniform vertical bias does
   **not** enter a feature volume, because the base is interpolated from the
   same survey and shifts with it. It does enter a change (DoD) volume, which
   is why surveys are co-registered on stable ground first
   (:func:`stable_ground_bias`) and the residual bias is carried as
   ``sigma_bias * A``.
"""

from __future__ import annotations

import numpy as np


def survey_sigma(area_m2: float, cell_m: float, sigma_r: float, sigma_c: float = 0.0,
                 range_m: float = 0.0, sigma_bias: float = 0.0) -> float:
    a = cell_m * cell_m
    var = sigma_r**2 * a * area_m2
    if sigma_c and range_m:
        # Rolstad et al. (2009) eq. 8, with the area as a circle of radius r:
        # the fraction of the correlated variance left in the mean elevation.
        q = np.sqrt(area_m2 / np.pi) / range_m
        frac = 1.0 - q + q**3 / 5.0 if q <= 1.0 else 1.0 / (5.0 * q**2)
        var += sigma_c**2 * area_m2**2 * frac
    var += (sigma_bias * area_m2) ** 2
    return float(np.sqrt(var))


def combine(*sigmas: float) -> float:
    return float(np.sqrt(np.sum(np.square(sigmas))))


def total_sigma(ensembles, survey_sigmas) -> float:
    """1 σ of the summed volume of several features.

    The reference-surface term is not independent between features: one
    interpolator and one toe tolerance are applied to every feature at once.
    So each ensemble member (``toe_buffer_m`` x ``method``) is summed over the
    features before its spread is taken; adding the per-feature σ in
    quadrature would understate it. The survey term is treated as independent
    between features.
    """
    import pandas as pd

    totals = pd.concat(ensembles).groupby(["toe_buffer_m", "method"])["volume"].sum()
    return combine(float(totals.std(ddof=1)), *survey_sigmas)


def stable_ground_bias(dz: np.ndarray, stable: np.ndarray, slope_deg: np.ndarray | None = None,
                       max_slope: float = 15.0, clip: float = 1.0) -> dict:
    """Robust statistics of a DoD on terrain assumed not to have changed.

    Returns the median (bias to remove), the NMAD (robust sigma of per-cell
    differences) and the count of cells used. Steep cells are excluded because
    small horizontal offsets turn into large vertical ones there.
    """
    sel = stable & np.isfinite(dz) & (np.abs(dz) < clip)
    if slope_deg is not None:
        sel &= slope_deg < max_slope
    d = dz[sel]
    med = float(np.median(d))
    nmad = float(1.4826 * np.median(np.abs(d - med)))
    return {"median_m": med, "nmad_m": nmad, "n_cells": int(sel.sum())}


def empirical_variogram(dz: np.ndarray, valid: np.ndarray, res: float, max_lag_m: float = 300.0,
                        n_pairs: int = 400_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Semivariance of ``dz`` versus lag, from random cell pairs."""
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(valid)
    i = rng.integers(0, len(rr), n_pairs)
    ang = rng.uniform(0, np.pi, n_pairs)
    lag = rng.uniform(res, max_lag_m, n_pairs) / res
    r2 = np.round(rr[i] + lag * np.sin(ang)).astype(int)
    c2 = np.round(cc[i] + lag * np.cos(ang)).astype(int)
    ok = (r2 >= 0) & (r2 < dz.shape[0]) & (c2 >= 0) & (c2 < dz.shape[1])
    ok[ok] &= valid[r2[ok], c2[ok]]
    h = np.hypot(r2[ok] - rr[i][ok], c2[ok] - cc[i][ok]) * res
    g = 0.5 * (dz[rr[i][ok], cc[i][ok]] - dz[r2[ok], c2[ok]]) ** 2
    edges = np.linspace(0, max_lag_m, 31)
    k = np.digitize(h, edges) - 1
    lags = 0.5 * (edges[1:] + edges[:-1])
    gamma = np.array([np.median(g[k == j]) / 0.4549 if (k == j).any() else np.nan
                      for j in range(len(lags))])  # median-based robust estimator
    return lags, gamma


def fit_spherical(lags: np.ndarray, gamma: np.ndarray) -> dict:
    """Fit nugget + spherical model; returns sigma_r, sigma_c and range."""
    from scipy.optimize import curve_fit

    def model(h, nug, sill, rng):
        s = np.where(h < rng, 1.5 * h / rng - 0.5 * (h / rng) ** 3, 1.0)
        return nug + sill * s

    ok = np.isfinite(gamma)
    p0 = [gamma[ok][0], max(gamma[ok][-1] - gamma[ok][0], 1e-6), lags[ok][len(lags[ok]) // 3]]
    (nug, sill, rng), _ = curve_fit(model, lags[ok], gamma[ok], p0=p0,
                                    bounds=([0, 0, lags[0]], [np.inf, np.inf, lags[-1] * 2]))
    return {"sigma_r": float(np.sqrt(nug)), "sigma_c": float(np.sqrt(sill)), "range_m": float(rng)}
