"""Tests for the partial unweighting of buffered samples"""

import pytest
import torch

from madnis.integrator import Integrator
from madnis.integrator.integrator import SampleBatch

from .integrands import (
    cut_function,
    cut_function_multichannel,
    multichannel_integrand,
    reference_integral,
)

QUANTILE = 0.95


def make_integrator(multichannel: bool, **kwargs) -> Integrator:
    integrand = multichannel_integrand() if multichannel else cut_function
    return Integrator(
        integrand,
        dims=None if multichannel else 2,
        buffer_capacity=10,
        buffer_unweighting_quantile=QUANTILE,
        dtype=torch.float64,
        **kwargs,
    )


def uniform_batch(n: int, multichannel: bool) -> SampleBatch:
    """uniform samples, including those that fail the cut"""
    x = torch.rand((n, 2), dtype=torch.float64)
    q = torch.ones(n, dtype=torch.float64)
    if multichannel:
        func_vals, alphas = cut_function_multichannel(x, None)
        channels = (torch.rand(n) < 0.3).long()
        return SampleBatch(x, x, q, func_vals, channels, alphas)
    return SampleBatch(x, None, q, cut_function(x), None)


def channel_weights(samples: SampleBatch) -> torch.Tensor:
    """integrand times prior channel weight of the sampling channel, divided by q_sample"""
    func_vals = samples.func_vals
    if samples.channels is not None:
        func_vals = (
            func_vals
            * torch.gather(
                samples.alphas_prior, dim=1, index=samples.channels[:, None]
            )[:, 0]
        )
    return func_vals / samples.q_sample


@pytest.mark.parametrize("multichannel", [False, True])
def test_rescaled_probabilities(multichannel):
    """q_sample is multiplied by min(1, |w| / w_max) and the channel-wise total / accepted"""
    torch.manual_seed(0)
    integrator = make_integrator(multichannel)
    batch = uniform_batch(20000, multichannel)
    channels = integrator._batch_channels(batch)
    weights = channel_weights(batch).abs()

    torch.manual_seed(1)
    unweighted = integrator._unweight_for_buffer(batch)
    # repeat the random numbers to find the accepted samples
    torch.manual_seed(1)
    random = torch.rand_like(weights)

    n_channels = integrator.integration_channel_count or 1
    expected_q = torch.zeros_like(weights)
    accepted = torch.zeros_like(weights, dtype=torch.bool)
    for channel in range(n_channels):
        mask = channels == channel
        max_weight = torch.quantile(weights[mask & (weights > 0)], QUANTILE)
        accept_prob = (weights[mask] / max_weight).clip(max=1.0)
        chan_accepted = random[mask] < accept_prob
        accepted[mask] = chan_accepted
        expected_q[mask] = (
            batch.q_sample[mask] * accept_prob * mask.sum() / chan_accepted.sum()
        )

    assert (unweighted.func_vals != 0).all()
    torch.testing.assert_close(unweighted.x, batch.x[accepted])
    torch.testing.assert_close(unweighted.q_sample, expected_q[accepted])


def test_weights_below_maximum_are_constant():
    """after unweighting, all samples below the maximum weight have the same weight"""
    torch.manual_seed(0)
    integrator = make_integrator(False)
    batch = uniform_batch(20000, False)
    weights = channel_weights(batch)
    max_weight = torch.quantile(weights[weights != 0], QUANTILE)
    unweighted = integrator._unweight_for_buffer(batch)
    unweighted_weights = channel_weights(unweighted)
    # q_sample = 1 before unweighting, so the original weights are the function values
    below = unweighted.func_vals < max_weight
    assert below.double().mean() > 0.5
    n_total, n_accepted = batch.x.shape[0], unweighted.x.shape[0]
    torch.testing.assert_close(
        unweighted_weights[below],
        torch.full_like(unweighted_weights[below], max_weight * n_accepted / n_total),
    )
    # samples above the maximum weight keep their weight up to the overall normalization
    torch.testing.assert_close(
        unweighted_weights[~below], unweighted.func_vals[~below] * n_accepted / n_total
    )
    # number of accepted samples agrees with the sum of acceptance probabilities
    accept_prob = (weights / max_weight).clip(max=1.0)
    expected, spread = accept_prob.sum(), (accept_prob * (1 - accept_prob)).sum().sqrt()
    assert abs(n_accepted - expected) < 5 * spread


@pytest.mark.parametrize("multichannel", [False, True])
def test_unweighting_is_unbiased(multichannel):
    """the channel-wise integral estimated from the unweighted samples is unbiased"""
    torch.manual_seed(0)
    integrator = make_integrator(multichannel)
    n_channels = integrator.integration_channel_count or 1

    # channel-wise reference integrals, defined by the uniform channel assignment
    reference = uniform_batch(4_000_000, multichannel)
    ref_channels = integrator._batch_channels(reference)
    ref_weights = channel_weights(reference)
    reference_means = torch.stack(
        [ref_weights[ref_channels == c].mean() for c in range(n_channels)]
    )

    estimates = []
    for _ in range(40):
        unweighted = integrator._unweight_for_buffer(uniform_batch(50000, multichannel))
        channels = integrator._batch_channels(unweighted)
        weights = channel_weights(unweighted)
        estimates.append(
            torch.stack([weights[channels == c].mean() for c in range(n_channels)])
        )
    estimates = torch.stack(estimates)
    errors = estimates.std(dim=0) / len(estimates) ** 0.5
    assert ((estimates.mean(dim=0) - reference_means).abs() < 5 * errors).all()


def test_zero_weights_without_dropping():
    """zero weights do not enter the quantile and are never stored"""
    torch.manual_seed(0)
    integrator = make_integrator(False)
    batch = uniform_batch(20000, False)
    # keep only 2% of the non-zero samples, such that the quantile including zeros would be zero
    keep_nonzero = torch.rand(20000) < 0.02
    batch.func_vals = batch.func_vals * keep_nonzero
    unweighted = integrator._unweight_for_buffer(batch)
    assert unweighted.x.shape[0] > 0
    assert (unweighted.func_vals != 0).all()


@pytest.mark.parametrize("multichannel", [False, True])
def test_training_with_buffer_unweighting(multichannel):
    """buffered training on unweighted samples gives correct integrals"""
    torch.manual_seed(0)
    integrand = multichannel_integrand() if multichannel else cut_function
    integrator = Integrator(
        integrand,
        dims=None if multichannel else 2,
        batch_size=512,
        drop_zero_integrands=True,
        buffer_capacity=20000,
        minimum_buffer_size=1000,
        buffered_steps=3,
        buffer_unweighting_quantile=QUANTILE,
        dtype=torch.float64,
    )
    statuses = []
    integrator.train(200, callback=statuses.append)
    assert sum(status.buffered for status in statuses) > 100
    assert not any(status.loss != status.loss for status in statuses)  # no nans

    integral, error = integrator.integrate(200000)
    assert abs(integral - reference_integral()) < 5 * error
