"""Tests for channel grouping with learned sampling of the channel within the group"""

import math

import pytest
import torch

from madnis.integrator import ChannelGrouping, Integrand, Integrator

SIGMA = 0.1
MUS = [0.2, 0.5, 0.8]


def gauss_1d_integral(mu: float) -> float:
    s = SIGMA * math.sqrt(2)
    return SIGMA * math.sqrt(math.pi / 2) * (math.erf((1 - mu) / s) + math.erf(mu / s))


EXPECTED = sum(gauss_1d_integral(mu) ** 2 for mu in MUS)

GROUPINGS = {
    # two groups with two channels each
    "equal": ([None, None, 0, 1], 2),
    # groups of size 3, 2 and 1, larger indices within a group are not channels
    "unequal": ([None, None, None, 0, 0, 1], 3),
}


def make_integrand(grouping: str, received_channels: list | None = None) -> Integrand:
    """
    three peaks in two continuous dimensions. The first dimension of x is the discrete index of
    the channel within its group. Channel c follows peak c % 3.
    """
    channel_assignment, max_group_size = GROUPINGS[grouping]
    channel_count = len(channel_assignment)
    owners = torch.tensor([c % 3 for c in range(channel_count)])
    owner_counts = torch.bincount(owners, minlength=3).double()

    def function(x, channels):
        if received_channels is not None:
            received_channels.append(channels)
        x_cont = x[:, 1:]
        peaks = torch.stack(
            [
                torch.exp(-(x_cont - mu).square().sum(dim=1) / (2 * SIGMA**2))
                for mu in MUS
            ],
            dim=1,
        )
        prior = (peaks + 1e-6) / (peaks + 1e-6).sum(dim=1, keepdim=True)
        return peaks.sum(dim=1), prior[:, owners] / owner_counts[owners]

    return Integrand(
        function,
        input_dim=3,
        channel_count=channel_count,
        has_channel_weight_prior=True,
        channel_grouping=ChannelGrouping(channel_assignment),
        discrete_dims=[max_group_size],
    )


def make_integrator(grouping: str, **kwargs) -> Integrator:
    return Integrator(
        make_integrand(grouping, kwargs.pop("received_channels", None)),
        batch_size=512,
        channel_grouping_mode="learned",
        dtype=torch.float64,
        **kwargs,
    )


def assert_compatible(value: float, error: float, expected: float = EXPECTED):
    assert error > 0
    assert abs(value - expected) < 5 * error, (value, error, expected)


@pytest.mark.parametrize("train_channel_weights", [False, True])
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_learned_grouping_integration(grouping, train_channel_weights):
    """integration and sampling are unbiased, also with trained channel weights"""
    torch.manual_seed(0)
    received_channels = []
    integrator = make_integrator(
        grouping,
        train_channel_weights=train_channel_weights,
        received_channels=received_channels,
    )
    assert integrator.integration_channel_count == len(
        ChannelGrouping(GROUPINGS[grouping][0]).groups
    )
    integrator.train(150)
    assert_compatible(*integrator.integrate(200000))
    weights = integrator.sample(200000).weights
    assert_compatible(
        weights.mean().item(), (weights.std() / len(weights) ** 0.5).item()
    )
    # the integrand is called with the channel indices, not the group indices
    channel_count = len(GROUPINGS[grouping][0])
    assert sorted(set(torch.cat(received_channels).tolist())) == list(
        range(channel_count)
    )


def test_learned_grouping_cwnet_input():
    """the index of the channel within the group is not an input of the channel weights"""
    torch.manual_seed(0)
    integrator = make_integrator("equal")
    assert integrator.cwnet.layers[0].in_features == 2
    samples = integrator.sample(100)
    torch.testing.assert_close(integrator._cwnet_input(samples), samples.x[:, 1:])


def test_learned_grouping_invalid_index():
    """indices larger than the size of a group give samples with zero weight"""
    torch.manual_seed(0)
    integrator = make_integrator("unequal")
    samples = integrator.sample(20000)
    chan_in_group = samples.x[:, 0].long()
    group_sizes = integrator.channel_group_sizes[samples.integration_channels]
    invalid = chan_in_group >= group_sizes
    assert invalid.any()
    assert (samples.func_vals[invalid] == 0).all()
    assert (samples.func_vals[~invalid] != 0).any()


@pytest.mark.parametrize("discrete_dims", [[], [1]])
def test_learned_grouping_requires_index_dimension(discrete_dims):
    integrand = Integrand(
        lambda x, c: (x[:, -1], torch.full((len(x), 4), 0.25)),
        input_dim=3,
        channel_count=4,
        has_channel_weight_prior=True,
        channel_grouping=ChannelGrouping([None, None, 0, 1]),
        discrete_dims=discrete_dims,
    )
    with pytest.raises(ValueError, match="learned"):
        Integrator(integrand, channel_grouping_mode="learned")


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_vegas_learned_grouping(grouping):
    pytest.importorskip("vegas")
    from madnis.integrator import VegasPreTraining

    torch.manual_seed(0)
    integrator = make_integrator(grouping)
    vegas = VegasPreTraining(integrator)
    vegas.train([5000] * 5)
    assert_compatible(*vegas.integrate(100000))
    weights = vegas.sample(100000).weights
    assert_compatible(
        weights.mean().item(), (weights.std() / len(weights) ** 0.5).item()
    )
    vegas.initialize_integrator()
    assert_compatible(*integrator.integrate(100000))
