"""Tests for the rational quadratic spline transformations"""

import pytest
import torch

from madnis.nn import rational_quadratic_spline, unconstrained_rational_quadratic_spline

BINS = 8


def random_spline_params(n: int, bins: int = BINS):
    gen = torch.Generator().manual_seed(42)
    widths = torch.randn((n, bins), generator=gen, dtype=torch.float64)
    heights = torch.randn((n, bins), generator=gen, dtype=torch.float64)
    derivatives = torch.randn((n, bins + 1), generator=gen, dtype=torch.float64)
    return widths, heights, derivatives


@pytest.mark.parametrize(
    "spline", [rational_quadratic_spline, unconstrained_rational_quadratic_spline]
)
def test_spline_inverse(spline):
    """the inverse spline undoes the spline and the log Jacobians add up to zero"""
    x = torch.rand(1000, dtype=torch.float64)
    params = random_spline_params(1000)
    y, log_jac = spline(x, *params)
    x_rec, log_jac_inv = spline(y, *params, inverse=True)
    torch.testing.assert_close(x_rec, x)
    torch.testing.assert_close(log_jac + log_jac_inv, torch.zeros_like(x))


@pytest.mark.parametrize(
    "spline", [rational_quadratic_spline, unconstrained_rational_quadratic_spline]
)
def test_spline_jacobian(spline):
    """the log Jacobian agrees with the derivative computed with autograd"""
    x = torch.rand(1000, dtype=torch.float64, requires_grad=True)
    params = random_spline_params(1000)
    y, log_jac = spline(x, *params)
    (grad,) = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(log_jac, grad.log())


@pytest.mark.parametrize(
    "bounds", [(0.0, 1.0, 0.0, 1.0), (-2.0, 3.0, -1.0, 5.0)], ids=["unit", "shifted"]
)
def test_spline_maps_interval_monotonically(bounds):
    """the spline maps [left, right] monotonically onto [bottom, top]"""
    left, right, bottom, top = bounds
    n = 1001
    x = torch.linspace(left, right, n, dtype=torch.float64)
    # same spline parameters for all points to check monotonicity along x
    widths, heights, derivatives = (
        p[:1].expand(n, -1) for p in random_spline_params(1)
    )
    y, _ = rational_quadratic_spline(
        x, widths, heights, derivatives, left=left, right=right, bottom=bottom, top=top
    )
    torch.testing.assert_close(y[0], torch.tensor(bottom, dtype=torch.float64))
    torch.testing.assert_close(y[-1], torch.tensor(top, dtype=torch.float64))
    assert (y.diff() > 0).all()


def test_unconstrained_spline_outside_bounds():
    """points outside the bounds are mapped onto themselves with zero log Jacobian"""
    x = torch.tensor([-3.0, -0.5, 1.5, 7.0], dtype=torch.float64)
    params = random_spline_params(4)
    y, log_jac = unconstrained_rational_quadratic_spline(x, *params)
    torch.testing.assert_close(y, x)
    torch.testing.assert_close(log_jac, torch.zeros_like(x))
