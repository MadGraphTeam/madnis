"""Tests for the integration and unweighting metrics"""

import math

import pytest
import torch

from madnis.integrator import (
    integration_metrics,
    truncated_max_weight,
    unweighting_metrics,
)


def test_integration_metrics():
    means = torch.tensor([1.0, 3.0], dtype=torch.float64)
    variances = torch.tensor([4.0, 9.0], dtype=torch.float64)
    counts = torch.tensor([100, 400])
    metrics = integration_metrics(means, variances, counts)
    error = math.sqrt(4 / 100 + 9 / 400)
    assert metrics.integral == pytest.approx(4.0)
    assert metrics.count == 500
    assert metrics.error == pytest.approx(error)
    assert metrics.rel_error == pytest.approx(error / 4)
    assert metrics.rel_stddev == pytest.approx(error * math.sqrt(500) / 4)
    # optimal stratification: sigma_tot = sum_c sigma_c
    assert metrics.rel_stddev_opt == pytest.approx((2 + 3) / 4)
    assert metrics.channel_integrals == [1.0, 3.0]
    assert metrics.channel_counts == [100, 400]
    assert metrics.channel_errors == pytest.approx([0.2, 0.15])
    assert metrics.channel_rel_stddevs == pytest.approx([2.0, 1.0])


def test_unweighting_metrics_constant_weights():
    weights = torch.full((10000,), 2.0)
    weights[::4] = 0.0  # cut efficiency 0.75
    metrics = unweighting_metrics(weights)
    assert metrics.cut_eff == pytest.approx(0.75)
    assert metrics.uweff_after_cuts == pytest.approx(1.0)
    assert metrics.uweff_after_cuts_partial == pytest.approx(1.0)
    assert metrics.uweff_before_cuts == pytest.approx(0.75)
    assert metrics.over_weight_rate == 0.0


def reference_max_weight(
    weights: torch.Tensor, max_overweight_truncation: float
) -> float:
    """literal implementation of the MadGraph7 loop, with the number of unweighted events
    of the sample as target count"""
    w_sorted = sorted(weights.abs()[weights != 0].tolist(), reverse=True)
    w_sum = 0.0
    for w in w_sorted:
        w_sum += w
        n_unweighted = sum(min(1.0, wi / w) for wi in w_sorted)
        if w_sum / w > max_overweight_truncation * n_unweighted:
            return w
    return w_sorted[-1]


@pytest.mark.parametrize("truncation", [0.001, 0.01, 0.1])
def test_truncated_max_weight(truncation):
    torch.manual_seed(0)
    weights = torch.exp(0.7 * torch.randn(3000, dtype=torch.float64))
    weights[::5] = 0.0
    weights[1::7] *= -1
    assert truncated_max_weight(weights, truncation) == pytest.approx(
        reference_max_weight(weights, truncation)
    )


def test_unweighting_metrics_uniform_weights():
    """for weights uniform in [0, 1], the efficiency is about mean / max = 1 / 2"""
    torch.manual_seed(0)
    weights = torch.rand(100000, dtype=torch.float64)
    metrics = unweighting_metrics(weights)
    assert metrics.uweff_after_cuts == pytest.approx(0.5, abs=0.01)
    # the weights have a sharp upper edge, so ignoring over-weights changes almost nothing
    assert metrics.uweff_after_cuts_partial == pytest.approx(
        metrics.uweff_after_cuts, rel=1e-3
    )


@pytest.mark.parametrize("truncation", [0.003, 0.01, 0.03])
def test_unweighting_metrics_over_weights(truncation):
    """over-weight events make up the given fraction of the unweighted events"""
    torch.manual_seed(0)
    weights = torch.exp(0.5 * torch.randn(200000, dtype=torch.float64))
    metrics = unweighting_metrics(weights, max_overweight_truncation=truncation)
    max_weight = truncated_max_weight(weights, truncation)
    ratio = weights / max_weight
    n_unweighted = ratio.clip(max=1).sum().item()
    over_weight_content = ratio[ratio >= 1].sum().item()
    assert over_weight_content / n_unweighted == pytest.approx(truncation, rel=0.05)
    assert metrics.over_weight_rate == pytest.approx((ratio > 1).double().mean().item())
    assert metrics.uweff_after_cuts_partial == pytest.approx(
        n_unweighted / len(weights)
    )
    # counting over-weights with their weight increases the efficiency by at most truncation
    ratio_full_partial = metrics.uweff_after_cuts / metrics.uweff_after_cuts_partial
    assert 1 < ratio_full_partial < 1 + truncation


def test_unweighting_metrics_independent_of_sample_size():
    """unlike mean / max, the truncated efficiency converges for large samples"""
    effs = []
    for n in [20000, 200000, 2000000]:
        torch.manual_seed(0)
        weights = torch.exp(0.5 * torch.randn(n, dtype=torch.float64))
        effs.append(unweighting_metrics(weights).uweff_after_cuts_partial)
    assert effs[1] == pytest.approx(effs[2], rel=0.03)
    assert effs[0] == pytest.approx(effs[2], rel=0.1)


def test_unweighting_metrics_multichannel():
    """channels are combined as total unweighted events over total generated events"""
    channels = torch.arange(10000) % 2
    weights = torch.where(channels == 0, 1.0, 3.0).double()
    weights[(channels == 1) & (torch.arange(10000) % 4 == 1)] = 0.0  # cuts in channel 1
    total, per_channel = unweighting_metrics(weights, channels, 2)
    assert per_channel[0].cut_eff == pytest.approx(1.0)
    assert per_channel[1].cut_eff == pytest.approx(0.5)
    assert per_channel[1].uweff_before_cuts_partial == pytest.approx(0.5)
    # absolute integrals 0.5 and 0.75. For 0.4 and 0.6 unweighted events, 0.4 and 1.2
    # events are generated, and 0.4 and 0.6 of them pass the cuts.
    assert total.uweff_before_cuts_partial == pytest.approx(1.0 / 1.6)
    assert total.cut_eff == pytest.approx(1.0 / 1.6)
    assert total.uweff_after_cuts_partial == pytest.approx(1.0)
