"""Tests for the Integrand wrapper and the channel grouping"""

import pytest
import torch

from madnis.integrator import ChannelGrouping, Integrand


def test_channel_grouping():
    # channels 2 and 3 are mapped onto channel 0, channel 4 onto channel 1
    grouping = ChannelGrouping([None, None, 0, 0, 1])
    assert [group.channel_indices for group in grouping.groups] == [[0, 2, 3], [1, 4]]
    assert [group.target_index for group in grouping.groups] == [0, 1]
    assert [channel.group.group_index for channel in grouping.channels] == [
        0,
        1,
        0,
        0,
        1,
    ]
    assert [channel.remapped for channel in grouping.channels] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert [channel.position_in_group for channel in grouping.channels] == [
        0,
        0,
        1,
        2,
        1,
    ]


def test_integrand_remap_channels():
    integrand = Integrand(
        lambda x, c: x[:, 0],
        input_dim=2,
        channel_count=5,
        channel_grouping=ChannelGrouping([None, None, 0, 0, 1]),
    )
    assert integrand.unique_channel_count() == 2
    assert integrand.remap_channels(3) == 0
    torch.testing.assert_close(
        integrand.remap_channels(torch.arange(5)), torch.tensor([0, 1, 0, 0, 1])
    )
    ungrouped = Integrand(lambda x, c: x[:, 0], input_dim=2, channel_count=5)
    assert ungrouped.unique_channel_count() == 5
    assert ungrouped.remap_channels(3) == 3


@pytest.mark.parametrize("remapped", [False, True])
@pytest.mark.parametrize("prior", [False, True])
def test_integrand_signatures(remapped, prior):
    """all supported function signatures are wrapped into (f, y, alpha)"""
    x = torch.rand((10, 2))
    channels = torch.randint(3, (10,))
    f = x.sum(dim=1)
    y = 2 * x
    alpha = torch.full((10, 3), 1 / 3)
    outputs = (f,) + ((y,) if remapped else ()) + ((alpha,) if prior else ())

    def function(x_in, c_in):
        return outputs[0] if len(outputs) == 1 else outputs

    integrand = Integrand(
        function,
        input_dim=2,
        channel_count=3,
        remapped_dim=2 if remapped else None,
        has_channel_weight_prior=prior,
    )
    f_out, y_out, alpha_out = integrand(x, channels)
    torch.testing.assert_close(f_out, f)
    assert (y_out is y) if remapped else (y_out is None)
    assert (alpha_out is alpha) if prior else (alpha_out is None)


def test_single_channel_integrand():
    integrand = Integrand(lambda x: x.prod(dim=1), input_dim=3)
    x = torch.rand((10, 3))
    f, y, alpha = integrand(x, None)
    torch.testing.assert_close(f, x.prod(dim=1))
    assert y is None and alpha is None


def test_integrand_bounds():
    """the integrand is mapped from the given bounds onto the unit hypercube"""
    integrand = Integrand(
        lambda x: x[:, 0] * x[:, 1], input_dim=2, bounds=[[1.0, 3.0], [-1.0, 0.0]]
    )
    torch.manual_seed(0)
    x = torch.rand((400000, 2), dtype=torch.float64)
    f, _, _ = integrand(x, None)
    # integral of x0 * x1 over [1, 3] x [-1, 0] = 4 * (-1/2) = -2
    assert f.mean().item() == pytest.approx(-2.0, rel=5e-3)


def test_update_active_channels_mask_with_grouping():
    received = []
    integrand = Integrand(
        lambda x, c: x[:, 0],
        input_dim=2,
        channel_count=5,
        channel_grouping=ChannelGrouping([None, None, 0, 0, 1]),
        update_active_channels_mask=received.append,
    )
    integrand.update_active_channels_mask(torch.tensor([True, False]))
    torch.testing.assert_close(
        received[0], torch.tensor([True, False, True, True, False])
    )


def test_update_active_channels_mask_without_grouping():
    received = []
    integrand = Integrand(
        lambda x, c: x[:, 0],
        input_dim=2,
        channel_count=3,
        update_active_channels_mask=received.append,
    )
    mask = torch.tensor([True, False, True])
    integrand.update_active_channels_mask(mask)
    torch.testing.assert_close(received[0], mask)
