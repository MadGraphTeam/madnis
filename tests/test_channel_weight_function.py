"""Tests for sampling with channel weights, but without evaluating the integrand"""

import pytest
import torch

from madnis.integrator import Integrand, Integrator
from madnis.integrator.integrator import SampleBatch

from .test_integrator import peak, two_peaks_with_prior


def make_integrand(calls: dict, **kwargs) -> Integrand:
    """multi-channel integrand with a channel weight function that counts the calls"""

    def function(x, channels):
        calls["integrand"] += 1
        return two_peaks_with_prior(x, channels)

    def channel_weight_function(x, channels):
        calls["channel_weights"] += 1
        _, alphas = two_peaks_with_prior(x, channels)
        return torch.full_like(alphas[:, 0], torch.nan), alphas

    return Integrand(
        function,
        input_dim=2,
        channel_count=2,
        has_channel_weight_prior=True,
        channel_weight_function=channel_weight_function,
        **kwargs,
    )


def test_sample_with_channel_weight_function():
    """the channel weights agree with those of a full evaluation of the integrand"""
    torch.manual_seed(0)
    calls = {"integrand": 0, "channel_weights": 0}
    integrator = Integrator(make_integrand(calls), batch_size=256, dtype=torch.float64)
    integrator.train(20)  # train the channel weight network

    calls["integrand"] = 0
    samples = integrator.sample(5000, evaluate_integrand=False)
    assert calls == {"integrand": 0, "channel_weights": 1}
    torch.testing.assert_close(samples.func_vals, torch.ones_like(samples.func_vals))

    func_vals, alphas_prior = two_peaks_with_prior(samples.x, samples.channels)
    full_samples = SampleBatch(
        samples.x,
        None,
        samples.q_sample,
        func_vals,
        samples.channels,
        alphas_prior,
    )
    with torch.no_grad():
        expected_alphas = integrator._get_alphas(full_samples)
    torch.testing.assert_close(samples.alphas, expected_alphas)


def test_channel_weight_function_with_bounds_and_remapping():
    """the channel weight function is wrapped like the integrand"""
    received = []

    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        return f, 2 * x, alphas

    def channel_weight_function(x, channels):
        received.append(x)
        _, alphas = two_peaks_with_prior(x, channels)
        return torch.zeros_like(x[:, 0]), 2 * x, alphas

    integrand = Integrand(
        function,
        input_dim=2,
        channel_count=2,
        remapped_dim=2,
        has_channel_weight_prior=True,
        bounds=[[1.0, 3.0], [-1.0, 0.0]],
        channel_weight_function=channel_weight_function,
    )
    torch.manual_seed(0)
    integrator = Integrator(integrand, dtype=torch.float64)
    samples = integrator.sample(1000, evaluate_integrand=False)
    x = received[0]
    assert (x[:, 0] >= 1).all() and (x[:, 0] <= 3).all()
    assert (x[:, 1] >= -1).all() and (x[:, 1] <= 0).all()
    torch.testing.assert_close(samples.y, 2 * x)


def test_single_channel_weight_function():
    calls = []
    integrand = Integrand(
        peak,
        input_dim=2,
        channel_weight_function=lambda x: calls.append(x) or torch.zeros(len(x)),
    )
    torch.manual_seed(0)
    integrator = Integrator(integrand, dtype=torch.float64)
    samples = integrator.sample(100, evaluate_integrand=False)
    assert len(calls) == 1
    torch.testing.assert_close(samples.weights, 1 / samples.q_sample)


def test_channel_weight_function_checks_prior():
    def channel_weight_function(x, channels):
        _, alphas = two_peaks_with_prior(x, channels)
        return torch.zeros_like(x[:, 0]), -alphas

    integrand = Integrand(
        two_peaks_with_prior,
        input_dim=2,
        channel_count=2,
        has_channel_weight_prior=True,
        channel_weight_function=channel_weight_function,
    )
    integrator = Integrator(integrand, dtype=torch.float64)
    with pytest.raises(ValueError, match="prior channel weights"):
        integrator.sample(100, evaluate_integrand=False)


def test_channel_weight_function_with_sampling_integrand():
    with pytest.raises(ValueError, match="channel_weight_function"):
        Integrand(
            lambda channels: None,
            input_dim=2,
            function_includes_sampling=True,
            channel_weight_function=lambda channels: None,
        )


def test_vegas_sample_without_integrand():
    pytest.importorskip("vegas")
    from madnis.integrator import VegasPreTraining

    torch.manual_seed(0)
    calls = {"integrand": 0, "channel_weights": 0}
    integrator = Integrator(make_integrand(calls), dtype=torch.float64)
    vegas = VegasPreTraining(integrator)
    samples = vegas.sample(1000, evaluate_integrand=False)
    assert calls == {"integrand": 0, "channel_weights": 2}  # one call per channel
    _, expected_alphas = two_peaks_with_prior(samples.x, samples.channels)
    torch.testing.assert_close(samples.alphas, expected_alphas)

    # without channel weight function, the channel weights are zero, with one column per
    # channel even if the number of channels differs from the dimension
    integrand = Integrand(
        lambda x, c: (x[:, 0], torch.full((len(x), 3), 1 / 3)),
        input_dim=2,
        channel_count=3,
        has_channel_weight_prior=True,
    )
    vegas = VegasPreTraining(Integrator(integrand, dtype=torch.float64))
    samples = vegas.sample(300, evaluate_integrand=False)
    assert samples.alphas.shape == (300, 3)
    assert (samples.func_vals == 0).all()
