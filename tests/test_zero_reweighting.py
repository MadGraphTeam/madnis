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
    # the cut efficiency of the untrained flow is about 0.3, so after the first step, more
    # samples are drawn and the rescaling must use the total counts
    integrator._get_samples(4096, train=True)
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


def test_oversampled_counts():
    from madnis.integrator import Integrand

    integrator = Integrator(
        Integrand(lambda x, channels: x[:, 0], input_dim=2, channel_count=4),
        batch_size_threshold=0.5,
        max_oversampling=10,
        dtype=torch.float64,
    )
    target = torch.tensor([100, 100, 100, 100])
    # without previous steps, the efficiency is assumed to be one
    assert integrator._oversampled_counts(target).tolist() == [100, 100, 100, 100]
    channels = torch.repeat_interleave(torch.arange(4), 100)
    nonzero = (torch.arange(400) % 100) < torch.tensor(
        [80, 25, 0, 50]
    ).repeat_interleave(100)
    integrator._update_efficiencies(channels, nonzero)
    # efficiency 0.25: 0.5 / 0.25 = 2; efficiency 0: capped; at or above threshold: unchanged
    counts = integrator._oversampled_counts(target)
    assert counts[1] == 200
    assert counts[2] == 1000
    assert counts[0] == 100


@pytest.mark.parametrize("drop_zeros", [True, False])
def test_oversampling_only_in_channels_with_low_efficiency(drop_zeros):
    """only the channel with a low fraction of non-zero samples gets more samples"""
    torch.manual_seed(0)
    integrator = Integrator(
        channel_cut_integrand(0.1),
        batch_size_threshold=0.5,
        drop_zero_integrands=drop_zeros,
        dtype=torch.float64,
    )
    # the first call has no efficiency estimate yet
    integrator._get_samples(2048, uniform_channel_ratio=1.0, train=True)
    efficiency = integrator._nonzero_ema / integrator._generated_ema
    assert efficiency[0] == 1.0
    assert 0.05 < efficiency[1] < 0.15
    samples = integrator._get_samples(2048, uniform_channel_ratio=1.0, train=True)
    nonzero = samples.func_vals != 0
    nonzero_counts = torch.bincount(samples.channels[nonzero], minlength=2)
    generated = torch.bincount(samples.channels, minlength=2)
    if drop_zeros:
        generated = generated + samples.zero_counts
    assert generated[0] == 1024
    # efficiency about 0.1 -> about 0.5 / 0.1 = 5 times the requested samples
    assert generated[1] == torch.ceil(1024 * 0.5 / efficiency[1])
    assert 400 < nonzero_counts[1] < 650


def test_oversampling_is_capped_for_empty_channel():
    """a channel without any non-zero samples is sampled at most max_oversampling times"""
    torch.manual_seed(0)
    integrator = Integrator(
        channel_cut_integrand(0.0),
        drop_zero_integrands=True,
        max_oversampling=8,
        dtype=torch.float64,
    )
    integrator._get_samples(1024, uniform_channel_ratio=1.0, train=True)
    samples = integrator._get_samples(1024, uniform_channel_ratio=1.0, train=True)
    assert (samples.channels == 0).sum() == 512
    assert (samples.channels == 1).sum() == 0
    assert samples.zero_counts.tolist() == [0, 8 * 512]


def test_oversampling_keeps_channel_means_unbiased():
    """
    the channel-wise means computed during training are unbiased, also for small batches and
    low cut efficiencies, because the number of samples is fixed before sampling
    """
    torch.manual_seed(0)
    integrator = Integrator(
        channel_cut_integrand(0.3),
        drop_zero_integrands=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        means = []
        for _ in range(3000):
            samples = integrator._get_samples(40, uniform_channel_ratio=1.0, train=True)
            means.append(integrator._compute_integral(samples)[1])
        means = torch.stack(means)
        reference = integrator._compute_integral(
            integrator._get_samples(2_000_000, uniform_channel_ratio=1.0)
        )[1]
    error = means.std(dim=0) / len(means) ** 0.5
    assert ((means.mean(dim=0) - reference).abs() < 4 * error).all()
