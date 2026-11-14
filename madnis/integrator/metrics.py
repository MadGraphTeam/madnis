import math
from dataclasses import dataclass

import torch


@dataclass
class UnweightingMetrics:
    """
    Metrics for the unweighting performance. The maximum weight is determined with the truncation
    prescription of MadGraph7, see ``unweighting_metrics``. Events with a weight above the
    maximum weight are kept as over-weight events with a weight larger than one.

    Args:
        cut_eff: cut efficiency
        uweff_before_cuts: unweighting efficiency before cuts, defined as mean weight divided
            by the maximum weight, such that over-weight events count with their full weight
            (computed as ``uweff_after_cuts * cut_eff``)
        uweff_before_cuts_partial: unweighting efficiency before cuts, defined as the expected
            number of accepted events per generated event, such that over-weight events count
            as one event. This is the unweighting efficiency reported by MadGraph7 (computed as
            ``uweff_after_cuts_partial * cut_eff``)
        uweff_after_cuts: same as ``uweff_before_cuts``, for events passing the cuts
        uweff_after_cuts_partial: same as ``uweff_before_cuts_partial``, for events passing the
            cuts
        over_weight_rate: fraction of the events passing the cuts with a weight above the
            maximum weight
    """

    cut_eff: float
    uweff_before_cuts: float
    uweff_before_cuts_partial: float
    uweff_after_cuts: float
    uweff_after_cuts_partial: float
    over_weight_rate: float


def truncated_max_weight(
    weights: torch.Tensor, max_overweight_truncation: float = 0.01
) -> float:
    r"""
    Determines the maximum weight for unweighting with the truncation prescription of
    MadGraph7. Going through the absolute weights in descending order, the maximum weight is
    the first weight ``w_max`` for which the over-weight events, i.e. those with weights
    ``|w| >= w_max``, add up to more than a fraction ``max_overweight_truncation`` of the
    expected number of unweighted events,

    .. math::

        \sum_{|w_i| \geq w_\text{max}} \frac{|w_i|}{w_\text{max}}
        > \text{max\_overweight\_truncation} \cdot
        \sum_i \min\left(1, \frac{|w_i|}{w_\text{max}}\right) .

    In MadGraph7, the right-hand side is the target number of unweighted events. Here, it is
    the number of unweighted events obtained from the given sample, such that the result does
    not depend on the sample size.

    Args:
        weights: weights of the samples, shape (n, )
        max_overweight_truncation: tolerated fraction of over-weight events
    Returns:
        maximum weight, or zero if all weights are zero
    """
    w = weights.abs()
    w = w[w != 0].sort(descending=True).values.double()
    if len(w) == 0:
        return 0.0
    w_sum = w.cumsum(dim=0)
    index = torch.arange(1, len(w) + 1, device=w.device, dtype=w.dtype)
    # expected number of unweighted events for w_max = w[k]: all events with larger weights
    # are accepted, all events with smaller weights with probability |w_i| / w_max
    n_unweighted = index + (w_sum[-1] - w_sum) / w
    exceeded = torch.nonzero(w_sum / w > max_overweight_truncation * n_unweighted)
    return (w[exceeded[0, 0]] if len(exceeded) > 0 else w[-1]).item()


def unweighting_metrics(
    weights: torch.Tensor,
    channels: torch.Tensor | None = None,
    channel_count: int | None = None,
    max_overweight_truncation: float = 0.01,
) -> UnweightingMetrics | tuple[UnweightingMetrics, list[UnweightingMetrics]]:
    """
    Calculates the unweighting efficiency, with the maximum weight determined separately for
    every channel using the truncation prescription of MadGraph7, see ``truncated_max_weight``.
    In the multi-channel case, the combined metrics assume that unweighted events are generated
    in every channel in proportion to its absolute integral, as in MadGraph7. The combined
    unweighting efficiency is then the total number of unweighted events divided by the total
    number of generated events.

    Args:
        weights: weights of the samples, normalized such that their mean is the integral
        channels: channel indices of the samples
        channel_count: number of channels
        max_overweight_truncation: tolerated fraction of over-weight events
    Returns:
        An UnweightingMetrics object. In the multi-channel case, it also returns a list of
        UnweightingMetrics objects for all channels.
    """

    if channels is None:
        channel_count = 1
    uweffs = []
    abs_integrals = []
    for channel in range(channel_count):
        w = weights if channel_count == 1 else weights[channels == channel]
        n_total = len(w)
        w = w[w != 0.0].abs()
        n_accepted = len(w)
        abs_integrals.append(w.sum().item() / len(weights))

        if n_accepted > 0:
            cut_efficiency = n_accepted / n_total
            ratio = w / truncated_max_weight(w, max_overweight_truncation)
            uweff = ratio.mean().item()
            uweff_partial = ratio.clip(max=1.0).mean().item()
            over_weight_rate = (ratio > 1.0).double().mean().item()
        else:
            cut_efficiency = 0.0
            uweff = 0.0
            uweff_partial = 0.0
            over_weight_rate = 0.0

        uweffs.append(
            UnweightingMetrics(
                cut_eff=cut_efficiency,
                uweff_before_cuts=uweff * cut_efficiency,
                uweff_before_cuts_partial=uweff_partial * cut_efficiency,
                uweff_after_cuts=uweff,
                uweff_after_cuts_partial=uweff_partial,
                over_weight_rate=over_weight_rate,
            )
        )

    if channel_count == 1:
        return uweffs[0]

    # Number of generated events per channel, for a number of unweighted events proportional
    # to the absolute channel integral. All combined metrics are ratios of event counts.
    abs_integral_total = sum(abs_integrals)
    generated, after_cuts, unweighted, unweighted_partial, over_weights = 0, 0, 0, 0, 0
    for abs_integral, uweff in zip(abs_integrals, uweffs):
        if abs_integral == 0.0:
            continue
        n_gen = abs_integral / abs_integral_total / uweff.uweff_before_cuts_partial
        generated += n_gen
        after_cuts += n_gen * uweff.cut_eff
        unweighted += n_gen * uweff.uweff_before_cuts
        unweighted_partial += n_gen * uweff.uweff_before_cuts_partial
        over_weights += n_gen * uweff.cut_eff * uweff.over_weight_rate
    total = UnweightingMetrics(
        cut_eff=after_cuts / generated,
        uweff_before_cuts=unweighted / generated,
        uweff_before_cuts_partial=unweighted_partial / generated,
        uweff_after_cuts=unweighted / after_cuts,
        uweff_after_cuts_partial=unweighted_partial / after_cuts,
        over_weight_rate=over_weights / after_cuts,
    )
    return total, uweffs


@dataclass
class IntegrationMetrics:
    """
    Metrics for the integration performance

    Args:
        integral: total integration results
        count: number of integration samples
        error: Monte Carlo integration error
        rel_error: relative integration error
        rel_stddev: relative standard deviation (does not scale with number of samples)
        rel_stddev_opt: optimal relative standard deviation that would have been possible with
            stratified sampling
        channel_integrals: channel-wise integrals
        channel_counts: channel-wise number of samples
        channel_errors: channel-wise integration errors
        channel_rel_errors: channel-wise relative integration errors
        channel_rel_stddevs: channel-wise relative standard deviations
    """

    integral: float
    count: int
    error: float
    rel_error: float
    rel_stddev: float
    rel_stddev_opt: float
    channel_integrals: list[float]
    channel_counts: list[int]
    channel_errors: list[float]
    channel_rel_errors: list[float]
    channel_rel_stddevs: list[float]


def integration_metrics(
    channel_means: torch.Tensor,
    channel_variances: torch.Tensor,
    channel_counts: torch.Tensor,
) -> IntegrationMetrics:
    """
    Calculate metrics for the integration performance

    Args:
        channel_means: channel-wise integrals
        channel_variances: channel-wise variances
        channel_counts: channel-wise sample counts
    Returns:
        An IntegrationMetrics object
    """
    channel_square_errors = channel_variances / channel_counts
    integral = channel_means.sum().item()
    integral_abs = max(abs(integral), 1e-15)
    count = channel_counts.sum().item()
    error = channel_square_errors.nansum().sqrt().item()
    channel_stddevs = channel_variances.nan_to_num().sqrt()
    channel_errors = channel_square_errors.nan_to_num().sqrt()
    channel_means_abs = channel_means.abs()
    return IntegrationMetrics(
        integral=integral,
        count=count,
        error=error,
        rel_error=error / integral_abs,
        rel_stddev=error * math.sqrt(count) / integral_abs,
        rel_stddev_opt=(channel_stddevs.sum() * channel_stddevs).sum().sqrt().item()
        / integral,
        channel_counts=channel_counts.tolist(),
        channel_integrals=channel_means.tolist(),
        channel_errors=channel_errors.tolist(),
        channel_rel_errors=(channel_errors / channel_means_abs).tolist(),
        channel_rel_stddevs=(channel_stddevs / channel_means_abs).tolist(),
    )
