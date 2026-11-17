"""Tests for the torch implementation of the VEGAS grid"""

import numpy as np
import pytest
import torch

from madnis.integrator.vegas_grid import VegasGrid

DIMS, BINS = 3, 16


def uniform_grid(dims: int = DIMS, bins: int = BINS) -> torch.Tensor:
    return torch.linspace(0, 1, bins + 1, dtype=torch.float64).expand(dims, -1)


def training_function(x: np.ndarray) -> np.ndarray:
    return np.exp(-((x - 0.3) ** 2).sum(axis=1) / 0.01)


@pytest.mark.parametrize("alpha", [0.7, 1.5, 0.0, -1.0])
def test_matches_vegas_package(alpha):
    """mapping, training data and adaptation agree with vegas.AdaptiveMap"""
    vegas = pytest.importorskip("vegas")
    rng = np.random.default_rng(0)
    reference = vegas.AdaptiveMap(grid=[[0, 1]] * DIMS, ninc=BINS)
    grids = VegasGrid(2, DIMS, BINS)
    for iteration in range(8):
        r = rng.random((5000, DIMS))
        # points on the boundary are ignored
        r[:5] = 0.0
        x = np.empty_like(r)
        jac = np.empty(len(r))
        reference.map(r, x, jac)
        x_torch, jac_torch = grids.map(1, torch.from_numpy(r))
        np.testing.assert_allclose(x_torch.numpy(), x, rtol=0, atol=1e-13)
        np.testing.assert_allclose(jac_torch.numpy(), jac, rtol=1e-12)

        f = (training_function(x) * jac) ** 2
        reference.add_training_data(r, f)
        grids.add_training_data(1, torch.from_numpy(r), torch.from_numpy(f))
        reference.adapt(alpha=alpha)
        grids.adapt(1, alpha)
        np.testing.assert_allclose(
            grids.grid[1].numpy(), reference.extract_grid(), rtol=0, atol=1e-12
        )
    # the other grid is not affected
    torch.testing.assert_close(grids.grid[0], uniform_grid(), rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [0.7, 0.0, -1.0])
def test_unchanged_without_training_data(alpha):
    """
    zero training data leave the grid unchanged. For negative alpha, the vegas package divides by
    zero in this case.
    """
    grids = VegasGrid(1, DIMS, BINS)
    r = torch.rand((1000, DIMS), dtype=torch.float64)
    grids.add_training_data(0, r, torch.zeros(1000, dtype=torch.float64))
    grids.adapt(0, alpha)
    torch.testing.assert_close(grids.grid[0], uniform_grid(), rtol=0, atol=0)


def test_single_bin():
    grids = VegasGrid(1, DIMS, 1)
    r = torch.rand((1000, DIMS), dtype=torch.float64)
    grids.add_training_data(0, r, r[:, 0])
    grids.adapt(0, 0.7)
    torch.testing.assert_close(grids.grid[0], uniform_grid(bins=1), rtol=0, atol=0)
    x, jac = grids.map(0, r)
    torch.testing.assert_close(x, r)
    torch.testing.assert_close(jac, torch.ones(1000, dtype=torch.float64))


def test_adaptation_concentrates_bins():
    """after training on a peak, the bins are narrower at the peak and the map stays bijective"""
    torch.manual_seed(0)
    grids = VegasGrid(1, DIMS, BINS)
    for _ in range(5):
        r = torch.rand((20000, DIMS), dtype=torch.float64)
        x, jac = grids.map(0, r)
        f = torch.from_numpy(training_function(x.numpy())) * jac
        grids.add_training_data(0, r, f.square())
        grids.adapt(0, 1.0)
    widths = grids.grid[0].diff(dim=1)
    assert (widths > 0).all()
    peak_bin = (grids.grid[0][:, 1:] > 0.3).int().argmax(dim=1)
    assert (widths.gather(1, peak_bin[:, None])[:, 0] < 0.5 / BINS).all()
    r = torch.rand((1000, DIMS), dtype=torch.float64)
    x, jac = grids.map(0, r)
    assert ((x >= 0) & (x <= 1)).all() and (jac > 0).all()


def test_to_device():
    grids = VegasGrid(2, DIMS, BINS).to(torch.device("meta"))
    assert grids.device.type == "meta"
    assert grids.sum_f.device.type == "meta" and grids.n_f.device.type == "meta"
