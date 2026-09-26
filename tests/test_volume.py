"""Analytic checks: the pipeline must recover volumes of shapes we can integrate by hand."""

import numpy as np
import pytest

from minevol.reference import METHODS, base_surface
from minevol.volume import cut_fill, feature_volume, refine_mask


def grid(cell, half=150.0):
    x = np.arange(-half + cell / 2, half, cell)
    xx, yy = np.meshgrid(x, -x)
    return xx, yy


def tilted_plane(xx, yy):
    # A 5 % slope with a twist of cross-slope: natural-ish ground, exactly harmonic.
    return 200.0 + 0.05 * xx - 0.02 * yy


@pytest.mark.parametrize("method", METHODS)
def test_base_reproduces_plane(method):
    xx, yy = grid(1.0)
    z = tilted_plane(xx, yy)
    mask = xx**2 + yy**2 < 60**2
    base = base_surface(z, mask, method)
    assert np.abs(base - z)[mask].max() < 0.02


@pytest.mark.parametrize("cell", [0.5, 1.0, 2.0])
def test_cone_pile_on_slope(cell):
    r, h = 60.0, 15.0
    xx, yy = grid(cell)
    rho = np.hypot(xx, yy)
    z = tilted_plane(xx, yy) + np.clip(h * (1 - rho / r), 0, None)
    mask = rho < r + 3 * cell
    vol, _, _ = feature_volume(z, mask, "pile", cell)
    exact = np.pi * r**2 * h / 3
    assert vol == pytest.approx(exact, rel=0.005)


def test_paraboloid_pit():
    r, d, cell = 80.0, 25.0, 1.0
    xx, yy = grid(cell)
    rho = np.hypot(xx, yy)
    z = tilted_plane(xx, yy) - np.clip(d * (1 - (rho / r) ** 2), 0, None)
    mask = rho < r + 2
    vol, _, _ = feature_volume(z, mask, "pit", cell)
    assert vol == pytest.approx(np.pi * r**2 * d / 2, rel=0.005)


def test_refine_finds_toe_from_generous_outline():
    r, h, cell = 50.0, 12.0, 1.0
    xx, yy = grid(cell)
    rho = np.hypot(xx, yy)
    z = tilted_plane(xx, yy) + np.clip(h * (1 - rho / r), 0, None)
    outline = (np.abs(xx + 10) < 90) & (np.abs(yy - 5) < 80)  # sloppy, off-centre box
    mask = refine_mask(z, outline, "pile", cell, relief=0.25)
    footprint = rho < r
    # The refined mask covers the pile and stays within a few cells of its toe.
    assert mask[footprint].all()
    assert mask.sum() * cell**2 < np.pi * (r + 5) ** 2
    vol, _, _ = feature_volume(z, mask, "pile", cell)
    assert vol == pytest.approx(np.pi * r**2 * h / 3, rel=0.005)


def test_cut_fill_signs_and_threshold():
    a = np.zeros((10, 10))
    b = a.copy()
    b[:5] += 2.0
    b[5:] -= 0.05
    cf = cut_fill(b, a, cell=2.0)
    assert cf.fill_m3 == pytest.approx(50 * 2.0 * 4)
    assert cf.cut_m3 == pytest.approx(50 * 0.05 * 4)
    cf = cut_fill(b, a, cell=2.0, threshold=0.1)
    assert cf.cut_m3 == 0.0
