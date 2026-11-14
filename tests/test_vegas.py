"""Tests for the VEGAS pre-training"""

import pytest
import torch

from madnis.integrator import Integrand, Integrator, VegasPreTraining

from .test_integrator import (
    PEAK_INTEGRAL,
    grouped_integrand,
    SIGMA,
    TWO_PEAKS_INTEGRAL,
    assert_compatible,
    assert_weights_unbiased,
    gauss_1d_integral,
    multichannel_integrand,
    peak,
    two_peaks,
)

pytest.importorskip("vegas")


def make_integrator(multichannel: bool) -> Integrator:
    torch.manual_seed(0)
    integrand = multichannel_integrand() if multichannel else peak
    return Integrator(integrand, dims=2, batch_size=512, dtype=torch.float64)


@pytest.mark.parametrize("multichannel", [False, True])
def test_vegas_training(multichannel):
    integrator = make_integrator(multichannel)
    expected = TWO_PEAKS_INTEGRAL if multichannel else PEAK_INTEGRAL
    vegas = VegasPreTraining(integrator)
    rel_stddev_before = vegas.integration_metrics(50000).rel_stddev
    vegas.train([5000] * 10)
    metrics = vegas.integration_metrics(100000)
    assert metrics.rel_stddev < 0.3 * rel_stddev_before
    assert_compatible(metrics.integral, metrics.error, expected)
    assert_compatible(*vegas.integrate(100000), expected)
    assert_weights_unbiased(vegas.sample(100000).weights, expected)


@pytest.mark.parametrize("multichannel", [False, True])
def test_initialize_integrator(multichannel):
    """the flow initialized with the VEGAS grid is about as good as VEGAS itself"""
    integrator = make_integrator(multichannel)
    expected = TWO_PEAKS_INTEGRAL if multichannel else PEAK_INTEGRAL
    rel_stddev_flow_before = integrator.integration_metrics(50000).rel_stddev
    vegas = VegasPreTraining(integrator)
    vegas.train([5000] * 10)
    rel_stddev_vegas = vegas.integration_metrics(100000).rel_stddev
    vegas.initialize_integrator()
    metrics = integrator.integration_metrics(100000)
    assert metrics.rel_stddev < 0.3 * rel_stddev_flow_before
    assert metrics.rel_stddev < 1.5 * rel_stddev_vegas
    assert_compatible(metrics.integral, metrics.error, expected)


def test_vegas_with_discrete_dimension():
    def function(x):
        return (1 + x[:, 0]) * torch.exp(-((x[:, 1] - 0.5) ** 2) / (2 * SIGMA**2))

    torch.manual_seed(0)
    integrand = Integrand(function, input_dim=2, discrete_dims=[3])
    integrator = Integrator(integrand, batch_size=512, dtype=torch.float64)
    vegas = VegasPreTraining(integrator)
    vegas.train([5000] * 5)
    expected = (1 + 2 + 3) * gauss_1d_integral(0.5)
    assert_compatible(*vegas.integrate(100000), expected)
    vegas.initialize_integrator()
    assert_compatible(*integrator.integrate(100000), expected)


def test_vegas_training_fills_buffer():
    torch.manual_seed(0)
    integrator = Integrator(
        two_peaks, dims=2, batch_size=512, buffer_capacity=20000, dtype=torch.float64
    )
    vegas = VegasPreTraining(integrator)
    vegas.train([5000] * 3)
    assert integrator.buffer.size == 15000
    _, _, q, func_vals, *_ = integrator.buffer
    assert_weights_unbiased(func_vals / q, TWO_PEAKS_INTEGRAL)


@pytest.mark.parametrize("grouping_mode", ["none", "uniform"])
def test_vegas_sample_weights_with_grouping(grouping_mode):
    torch.manual_seed(0)
    integrator = Integrator(
        grouped_integrand(),
        batch_size=512,
        channel_grouping_mode=grouping_mode,
        dtype=torch.float64,
    )
    vegas = VegasPreTraining(integrator)
    vegas.train([5000] * 3)
    assert_compatible(*vegas.integrate(100000), TWO_PEAKS_INTEGRAL)
    assert_weights_unbiased(vegas.sample(100000).weights, TWO_PEAKS_INTEGRAL)
