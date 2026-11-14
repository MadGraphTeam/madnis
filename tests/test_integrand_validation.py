"""Tests for the checks of the values returned by the integrand"""

import pytest
import torch

from madnis.integrator import Integrand, Integrator

from .test_integrator import (
    TWO_PEAKS_INTEGRAL,
    assert_compatible,
    two_peaks,
    two_peaks_with_prior,
)


def broken_function(value: float, fraction: float = 0.01):
    """two peaks with a given value at a fraction of the points"""

    def function(x):
        f = two_peaks(x)
        return torch.where(x[:, 0] < fraction, torch.full_like(f, value), f)

    return function


def broken_prior(kind: str):
    """multi-channel integrand with invalid prior channel weights at some points"""

    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        broken = x[:, 0] < 0.01
        if kind == "negative":
            alphas = torch.where(broken[:, None], alphas - 0.6, alphas)
        elif kind == "nan":
            alphas = torch.where(broken[:, None], torch.nan, alphas)
        elif kind == "zero":
            alphas = torch.where(broken[:, None], 0.0, alphas)
        return f, alphas

    return Integrand(
        function, input_dim=2, channel_count=2, has_channel_weight_prior=True
    )


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
def test_non_finite_integrand(value):
    torch.manual_seed(0)
    integrator = Integrator(broken_function(value), dims=2, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite values"):
        integrator.train(1)
    with pytest.raises(ValueError, match="non-finite values"):
        integrator.integrate(10000)


@pytest.mark.parametrize(
    "kind, message",
    [
        ("negative", "negative or non-finite prior"),
        ("nan", "negative or non-finite prior"),
        ("zero", "all zero"),
    ],
)
def test_invalid_prior_channel_weights(kind, message):
    torch.manual_seed(0)
    integrator = Integrator(broken_prior(kind), dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        integrator.train(1)


@pytest.mark.parametrize("train_channel_weights", [False, True])
def test_zero_prior_where_integrand_vanishes(train_channel_weights):
    """vanishing prior channel weights are allowed where the integrand is zero"""

    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        cut = x[:, 0] + x[:, 1] < 0.3
        return torch.where(cut, 0.0, f), torch.where(cut[:, None], 0.0, alphas)

    # integral with the cut, from uniform sampling
    torch.manual_seed(0)
    x = torch.rand((4_000_000, 2), dtype=torch.float64)
    expected = function(x, None)[0].mean().item()

    integrand = Integrand(
        function, input_dim=2, channel_count=2, has_channel_weight_prior=True
    )
    integrator = Integrator(
        integrand,
        batch_size=512,
        train_channel_weights=train_channel_weights,
        dtype=torch.float64,
    )
    integrator.train(50)
    assert_compatible(*integrator.integrate(100000), expected)


def test_vegas_checks_integrand():
    pytest.importorskip("vegas")
    from madnis.integrator import VegasPreTraining

    integrator = Integrator(broken_function(torch.nan), dims=2, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite values"):
        VegasPreTraining(integrator).train([1000])


def test_integrand_with_sampling():
    """integrands that sample the points themselves are checked as well"""

    def sampling_function(fail: bool):
        def function(channels):
            n = 10000
            x = torch.rand((n, 2), dtype=torch.float64)
            prob = torch.ones(n, dtype=torch.float64)
            f = two_peaks(x)
            if fail:
                f[0] = torch.nan
            return x, prob, f, None, None, None

        return function

    integrator = Integrator(
        Integrand(
            sampling_function(False), input_dim=2, function_includes_sampling=True
        ),
        dtype=torch.float64,
    )
    assert_compatible(*integrator.integrate(10000), TWO_PEAKS_INTEGRAL)
    integrator = Integrator(
        Integrand(
            sampling_function(True), input_dim=2, function_includes_sampling=True
        ),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="non-finite values"):
        integrator.integrate(10000)
