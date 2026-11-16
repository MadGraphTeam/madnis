"""Tests for dropping zero-integrand samples with rescaled sampling probabilities"""

import pytest
import torch

from madnis.integrator import Integrator, VegasPreTraining
from madnis.integrator.integrator import SampleBatch
from madnis.integrator.losses import stratified_variance

from .integrands import (
    cut_function,
    cut_function_multichannel,
    multichannel_integrand,
    reference_integral,
)


def drop_zeros(integrator: Integrator, batch: SampleBatch) -> SampleBatch:
    """drops zero-integrand samples like Integrator._get_samples does"""
    mask = batch.func_vals != 0
    channels = integrator._batch_channels(batch)
    kept = batch.map(lambda t: t[mask])
    kept.zero_counts = torch.bincount(
        channels[~mask], minlength=integrator.integration_channel_count or 1
    )
    return integrator._rescale_dropped(kept)


def uniform_batch(n: int, multichannel: bool) -> SampleBatch:
    x = torch.rand((n, 2), dtype=torch.float64)
    q = torch.ones(n, dtype=torch.float64)
    if multichannel:
        func_vals, alphas = cut_function_multichannel(x, None)
        channels = (torch.rand(n) < 0.3).long()
        return SampleBatch(x, x, q, func_vals, channels, alphas)
    return SampleBatch(x, None, q, cut_function(x), None)


@pytest.mark.parametrize("rescale", [True, False])
@pytest.mark.parametrize("multichannel", [False, True])
def test_integration_statistics_with_dropped_zeros(rescale, multichannel):
    """means, variances and counts of the kept samples equal those of the full batch"""
    torch.manual_seed(0)
    integrand = multichannel_integrand() if multichannel else cut_function
    integrator = Integrator(
        integrand,
        dims=None if multichannel else 2,
        drop_zero_integrands=True,
        rescale_dropped_zeros=rescale,
        dtype=torch.float64,
    )
    batch = uniform_batch(20000, multichannel)
    kept = drop_zeros(integrator, batch)
    assert kept.x.shape[0] < batch.x.shape[0]

    _, means, variances, counts = integrator._compute_integral(kept)
    _, means_full, variances_full, counts_full = integrator._compute_integral(batch)
    torch.testing.assert_close(means, means_full)
    torch.testing.assert_close(variances, variances_full)
    torch.testing.assert_close(counts, counts_full)


@pytest.mark.parametrize("multichannel", [False, True])
def test_rescaled_weights_are_unbiased(multichannel):
    """the channel-wise mean weight of the kept samples equals that of the full batch"""
    torch.manual_seed(0)
    integrand = multichannel_integrand() if multichannel else cut_function
    integrator = Integrator(
        integrand,
        dims=None if multichannel else 2,
        drop_zero_integrands=True,
        dtype=torch.float64,
    )
    batch = uniform_batch(20000, multichannel)
    kept = drop_zeros(integrator, batch)
    n_channels = integrator.integration_channel_count or 1
    for samples_in, samples_out in [(batch, kept)]:
        weights_full = samples_in.func_vals / samples_in.q_sample
        weights_kept = samples_out.func_vals / samples_out.q_sample
        channels_full = integrator._batch_channels(samples_in)
        channels_kept = integrator._batch_channels(samples_out)
        for channel in range(n_channels):
            torch.testing.assert_close(
                weights_kept[channels_kept == channel].mean(),
                weights_full[channels_full == channel].mean(),
            )


def test_get_samples_rescales_flow_probability():
    """q_sample of the kept samples is the flow probability times total / kept count"""
    torch.manual_seed(0)
    integrator = Integrator(
        cut_function, dims=2, drop_zero_integrands=True, dtype=torch.float64
    )
    # the cut efficiency of the untrained flow is about 0.3, so several sampling rounds
    # with batch_size_threshold = 0.5 are needed and the rescaling must use the total counts
    samples = integrator._get_samples(4096, train=True)
    n_kept = samples.x.shape[0]
    n_total = n_kept + samples.zero_counts.item()
    assert n_total > 4096
    with torch.no_grad():
        flow_prob = integrator.flow.prob(samples.x)
    torch.testing.assert_close(samples.q_sample, flow_prob * n_total / n_kept)


@pytest.mark.parametrize("rescale", [True, False])
def test_integral_during_training_with_dropped_zeros(rescale):
    """the integral estimated during training is unbiased, also in the single-channel case"""
    torch.manual_seed(0)
    integrator = Integrator(
        cut_function,
        dims=2,
        drop_zero_integrands=True,
        rescale_dropped_zeros=rescale,
        dtype=torch.float64,
    )
    integrator.train(50)
    integral, error = integrator.integral()
    assert abs(integral - reference_integral()) < 5 * error


def test_variance_loss_with_rescaling_improves_cut_efficiency():
    """with rescaling, the variance loss rewards a higher cut efficiency as without dropping"""

    def cut_efficiency(integrator):
        with torch.no_grad():
            weights = integrator.sample(100000).weights
        return (weights != 0).double().mean().item()

    torch.manual_seed(0)
    integrator = Integrator(
        cut_function,
        dims=2,
        loss=stratified_variance,
        drop_zero_integrands=True,
        dtype=torch.float64,
    )
    initial_efficiency = cut_efficiency(integrator)
    integrator.train(300)
    assert cut_efficiency(integrator) > initial_efficiency + 0.3


def test_vegas_buffer_with_dropped_zeros():
    """samples stored during the VEGAS pre-training carry rescaled probabilities"""
    pytest.importorskip("vegas")
    torch.manual_seed(0)
    integrator = Integrator(
        cut_function,
        dims=2,
        drop_zero_integrands=True,
        buffer_capacity=50000,
        dtype=torch.float64,
    )
    n_generated = 40000
    VegasPreTraining(integrator).train([10000] * 4)
    x, _, q, func_vals, *_ = integrator.buffer
    assert (func_vals != 0).all()
    weights = func_vals / q
    # The statistical error has to include the samples with zero weight. We undo the
    # rescaling approximately with the overall acceptance to get the plain weights.
    plain_weights = weights * n_generated / len(weights)
    variance = plain_weights.square().sum() / n_generated - weights.mean() ** 2
    error = (variance / n_generated) ** 0.5
    assert abs(weights.mean().item() - reference_integral()) < 5 * error
    # without rescaling, the mean weight would be too large by the inverse acceptance
    assert len(weights) < 0.8 * n_generated


def channel_cut_integrand(efficiency_1: float) -> "Integrand":
    """two channels, channel 1 has the given fraction of non-zero integrand values"""
    from madnis.integrator import Integrand

    def function(x, channels):
        keep = (channels == 0) | (x[:, 0] < efficiency_1)
        return (1 + x[:, 1]) * keep

    return Integrand(function, input_dim=2, channel_count=2)


def test_missing_counts():
    integrator = Integrator(
        channel_cut_integrand(0.5),
        batch_size_threshold=0.5,
        max_oversampling=10,
        dtype=torch.float64,
    )
    missing = integrator._missing_counts(
        target_counts=torch.tensor([100, 100, 100, 100, 0]),
        generated_counts=torch.tensor([100, 100, 100, 950, 0]),
        nonzero_counts=torch.tensor([60, 20, 0, 1, 0]),
    )
    # above threshold: nothing; 20% efficiency: (30 missing + 2 sqrt(30)) / 0.2;
    # no non-zero sample yet: doubled; budget of 1000 samples: 50 left; empty: nothing
    assert missing.tolist() == [0, 205, 100, 50, 0]


@pytest.mark.parametrize("drop_zeros", [True, False])
def test_resampling_only_in_channels_with_missing_samples(drop_zeros):
    """only the channel with a low fraction of non-zero samples is sampled again"""
    torch.manual_seed(0)
    integrator = Integrator(
        channel_cut_integrand(0.1),
        batch_size_threshold=0.5,
        drop_zero_integrands=drop_zeros,
        dtype=torch.float64,
    )
    samples = integrator._get_samples(2048, uniform_channel_ratio=1.0, train=True)
    nonzero = samples.func_vals != 0
    nonzero_counts = torch.bincount(samples.channels[nonzero], minlength=2)
    if drop_zeros:
        generated = torch.bincount(samples.channels, minlength=2) + samples.zero_counts
    else:
        generated = torch.bincount(samples.channels, minlength=2)
    # channel 0 has no zeros and is sampled only once
    assert generated[0] == 1024
    # channel 1 is filled up to at least half of its requested samples, and the number of
    # generated samples stays close to the expected 512 / 0.1
    assert nonzero_counts[1] >= 512
    assert 4000 < generated[1] < 8000


def test_resampling_stops_for_empty_channel():
    """a channel without any non-zero samples is sampled at most max_oversampling times"""
    torch.manual_seed(0)
    integrator = Integrator(
        channel_cut_integrand(0.0),
        drop_zero_integrands=True,
        max_oversampling=8,
        dtype=torch.float64,
    )
    samples = integrator._get_samples(1024, uniform_channel_ratio=1.0, train=True)
    assert (samples.channels == 0).sum() == 512
    assert (samples.channels == 1).sum() == 0
    assert samples.zero_counts.tolist() == [0, 8 * 512]
