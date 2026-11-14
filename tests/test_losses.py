"""Tests for the loss functions"""

import pytest
import torch

from madnis.integrator import (
    kl_divergence,
    kl_divergence_softclip,
    multi_channel_loss,
    rkl_divergence,
    stratified_variance,
    stratified_variance_softclip,
    variance,
)

ALL_LOSSES = [
    variance,
    stratified_variance,
    stratified_variance_softclip,
    kl_divergence,
    kl_divergence_softclip,
    rkl_divergence,
]


def samples(n: int = 10000, seed: int = 0):
    """integrand values and sampling probabilities for a non-trivial importance sampling"""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=gen, dtype=torch.float64)
    f = 3 * x**2 + 0.5  # integral 1.5 on [0, 1]
    q = (
        0.5 + x
    )  # normalized density on [0, 1], sampled with x as importance weight below
    return f, q


@pytest.mark.parametrize("loss", ALL_LOSSES, ids=lambda loss: loss.__name__)
def test_loss_minimal_for_perfect_density(loss):
    """all losses vanish if the density is proportional to the integrand"""
    f, _ = samples()
    q = f / 1.5
    # the soft-clipped KL weights are not renormalized after clipping, which shifts the
    # loss by a small constant of order 1 / threshold^2
    atol = 1e-3 if loss is kl_divergence_softclip else 1e-12
    torch.testing.assert_close(
        loss(f, q, q), torch.tensor(0.0, dtype=torch.float64), atol=atol, rtol=0
    )


@pytest.mark.parametrize("loss", ALL_LOSSES, ids=lambda loss: loss.__name__)
def test_loss_positive_and_differentiable(loss):
    f, q = samples()
    q_test = q.clone().requires_grad_()
    value = loss(f, q_test, q)
    value.backward()
    assert value.item() > 0
    assert torch.isfinite(q_test.grad).all()
    assert q_test.grad.abs().sum() > 0


def test_variance_is_relative_variance_of_weights():
    f, q = samples()
    weights = f / q
    expected = weights.var(unbiased=False) / weights.mean() ** 2
    torch.testing.assert_close(variance(f, q, q), expected, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        stratified_variance(f, q, q), expected, rtol=1e-6, atol=0
    )


def test_kl_divergence_value():
    """for q_test = q_sample, the loss is the KL divergence of the normalized integrand"""
    f, q = samples()
    p = f / (f / q).mean()
    expected = (p / q * (p / q).log()).mean()
    torch.testing.assert_close(kl_divergence(f, q, q), expected)


def test_kl_softclip_threshold():
    f, q = samples()
    torch.testing.assert_close(
        kl_divergence_softclip(f, q, q, threshold=1e6), kl_divergence(f, q, q)
    )
    assert kl_divergence_softclip(f, q, q, threshold=0.1) != kl_divergence(f, q, q)


@pytest.mark.parametrize("multichannel", [False, True])
def test_stratified_variance_softclip_threshold(multichannel):
    f, q = samples()
    channels = torch.arange(len(f)) % 3 if multichannel else None
    torch.testing.assert_close(
        stratified_variance_softclip(f, q, q, channels, threshold=1e6),
        stratified_variance(f, q, q, channels),
    )


@pytest.mark.parametrize("loss", [variance, kl_divergence, rkl_divergence])
def test_multi_channel_loss_sums_channels(loss):
    f, q = samples()
    channels = torch.arange(len(f)) % 3
    expected = sum(loss(f[channels == c], q[channels == c]) for c in range(3))
    torch.testing.assert_close(loss(f, q, q, channels), expected)


def test_multi_channel_loss_decorator_passes_kwargs():
    @multi_channel_loss
    def scaled_mean(f, q_test, q_sample, scale=1.0):
        return scale * (f / q_sample).mean()

    f, q = samples()
    channels = torch.arange(len(f)) % 2
    torch.testing.assert_close(
        scaled_mean(f, q, None, channels, scale=2.0),
        2.0 * sum((f / q)[channels == c].mean() for c in range(2)),
    )


def test_stratified_variance_multichannel():
    """(sum_c sigma_c / sum_c |I_c|)^2 with channel-wise standard deviations and integrals"""
    f, q = samples()
    channels = torch.arange(len(f)) % 3
    f = f * (1 + channels)  # different integrals per channel
    stddevs = torch.stack(
        [(f / q)[channels == c].var(unbiased=False).sqrt() for c in range(3)]
    )
    integrals = torch.stack([(f / q)[channels == c].mean() for c in range(3)])
    expected = (stddevs.sum() / integrals.sum()) ** 2
    torch.testing.assert_close(
        stratified_variance(f, q, q, channels), expected, rtol=1e-6, atol=0
    )


@pytest.mark.parametrize("loss", [stratified_variance, stratified_variance_softclip])
def test_stratified_losses_default_sampling_probability(loss):
    """without q_sample, the samples are assumed to follow q_test"""
    f, q = samples()
    channels = torch.arange(len(f)) % 2
    torch.testing.assert_close(loss(f, q), loss(f, q, q))
    torch.testing.assert_close(loss(f, q, None, channels), loss(f, q, q, channels))
