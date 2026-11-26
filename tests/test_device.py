"""
Tests that MadNIS runs on the device given to the Integrator, without torch.set_default_device.

Without a GPU, the integrator runs on device="cpu" while the default device is set to "meta"
during training, integration and sampling. Every tensor that madnis creates without an explicit
device then lands on "meta" and raises an error, as it would on a GPU. If CUDA is available, the
same scenarios also run on the GPU.
"""

from contextlib import contextmanager

import pytest
import torch
from torch import nn

from madnis.integrator import ChannelGrouping, Integrand, Integrator
from madnis.nn import Flow

SIGMA = 0.1
# Before torch 2.13, Adam creates its step counter on the default device and calls .item() on
# it, so the meta check fails inside torch itself.
META_CHECK = ["meta-check"] if torch.__version__ >= (2, 13) else []
DEVICES = META_CHECK + (["cuda"] if torch.cuda.is_available() else [])


@contextmanager
def default_device(device):
    old_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(old_device)


def peaks(x: torch.Tensor, mus: list[float]) -> torch.Tensor:
    return torch.stack(
        [torch.exp(-(x - mu).square().sum(dim=1) / (2 * SIGMA**2)) for mu in mus],
        dim=1,
    )


def two_peaks_with_prior(x: torch.Tensor, channels: torch.Tensor):
    p = peaks(x, [0.3, 0.7])
    prior = (p + 1e-6) / (p + 1e-6).sum(dim=1, keepdim=True)
    return p.sum(dim=1), prior


def single(device):
    return Integrator(lambda x: peaks(x, [0.5])[:, 0], dims=2, device=device)


def single_with_options(device):
    def function(x):
        return torch.where(x[:, 0] > 0.5, peaks(x, [0.6])[:, 0], 0.0)

    return Integrator(
        function,
        dims=2,
        device=device,
        drop_zero_integrands=True,
        buffer_capacity=2000,
        buffered_steps=2,
        minimum_buffer_size=100,
        buffer_unweighting_quantile=0.95,
        grad_clip_threshold=0.003,
    )


def bounds(device):
    integrand = Integrand(
        lambda x: x[:, 0] * x[:, 1], input_dim=2, bounds=[[1.0, 3.0], [-1.0, 0.0]]
    )
    return Integrator(integrand, device=device)


def multichannel(device):
    integrand = Integrand(
        two_peaks_with_prior,
        input_dim=2,
        channel_count=2,
        has_channel_weight_prior=True,
    )
    return Integrator(
        integrand,
        device=device,
        buffer_capacity=2000,
        buffered_steps=2,
        minimum_buffer_size=100,
        buffer_unweighting_quantile=0.95,
        uniform_channel_ratio=0.5,
        channel_dropping_threshold=0.01,
        channel_dropping_interval=2,
        freeze_cwnet_iteration=4,
        drop_zero_integrands=True,
        batch_size_per_channel=64,
    )


def multichannel_without_prior(device):
    integrand = Integrand(
        lambda x, channels: peaks(x, [0.3, 0.7]).sum(dim=1),
        input_dim=2,
        channel_count=2,
    )
    return Integrator(integrand, device=device)


def remapped(device):
    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        return f, x.square(), alphas

    integrand = Integrand(
        function,
        input_dim=2,
        channel_count=2,
        remapped_dim=2,
        has_channel_weight_prior=True,
    )
    return Integrator(integrand, device=device)


def grouped(mode):
    def make(device):
        def function(x, channels):
            f, alphas = two_peaks_with_prior(x, channels)
            return f, torch.cat((alphas, alphas), dim=1) / 2

        integrand = Integrand(
            function,
            input_dim=2,
            channel_count=4,
            has_channel_weight_prior=True,
            channel_grouping=ChannelGrouping([None, None, 0, 1]),
        )
        return Integrator(integrand, device=device, channel_grouping_mode=mode)

    return make


def discrete(device):
    def function(x):
        return (1 + x[:, 0]) * peaks(x[:, 1:], [0.5])[:, 0]

    def prior(x_discrete, index):
        return torch.tensor([1.0, 2.0, 3.0], device=x_discrete.device).expand(
            len(x_discrete), -1
        )

    integrand = Integrand(
        function, input_dim=2, discrete_dims=[3], discrete_prior_prob_function=prior
    )
    return Integrator(
        integrand,
        discrete_model="transformer",
        discrete_flow_kwargs=dict(
            embedding_dim=16, feedforward_dim=32, heads=2, mlp_units=32
        ),
        device=device,
    )


def discrete_multichannel(device):
    integrand = Integrand(
        lambda x, channels: two_peaks_with_prior(x[:, 1:], channels),
        input_dim=3,
        discrete_dims=[3],
        channel_count=2,
        has_channel_weight_prior=True,
    )
    return Integrator(integrand, device=device)


def learned_grouping(device):
    def function(x, channels):
        owners = torch.tensor([0, 1, 0, 1], device=x.device)
        f, prior = two_peaks_with_prior(x[:, 1:], channels)
        return f, prior[:, owners] / 2

    integrand = Integrand(
        function,
        input_dim=3,
        channel_count=4,
        has_channel_weight_prior=True,
        channel_grouping=ChannelGrouping([None, None, 0, 1]),
        discrete_dims=[2],
    )
    return Integrator(integrand, device=device, channel_grouping_mode="learned")


SCENARIOS = {
    "single": single,
    "single_with_options": single_with_options,
    "bounds": bounds,
    "multichannel": multichannel,
    "multichannel_without_prior": multichannel_without_prior,
    "remapped": remapped,
    "grouping_none": grouped("none"),
    "grouping_uniform": grouped("uniform"),
    "grouping_learned": learned_grouping,
    "discrete": discrete,
    "discrete_multichannel": discrete_multichannel,
}


def run_integrator(integrator: Integrator):
    integrator.train(6)
    integral, error = integrator.integrate(2000, batch_size=1000)
    integrator.integration_metrics(2000, batch_size=1000)
    integrator.unweighting_metrics(2000, batch_size=1000)
    samples = integrator.sample(500)
    integrator.load_state_dict(integrator.state_dict())
    return integral, samples


def run_vegas(integrator: Integrator):
    from madnis.integrator import VegasPreTraining

    vegas = VegasPreTraining(integrator)
    vegas.train([500] * 2)
    integral, error = vegas.integrate(2000)
    vegas.integration_metrics(2000)
    vegas.unweighting_metrics(2000)
    samples = vegas.sample(500)
    vegas.sample(100, evaluate_integrand=False)
    vegas.initialize_integrator()
    integrator.train(2)
    return integral, samples


def make_and_run(name, device_name, run):
    torch.manual_seed(0)
    make = SCENARIOS[name]
    if device_name == "meta-check":
        device = torch.device("cpu")
        integrator = make(device)
        with default_device("meta"):
            integral, samples = run(integrator)
    else:
        device = torch.device(device_name)
        integrator = make(device)
        integral, samples = run(integrator)
    assert torch.isfinite(torch.tensor(integral))
    assert samples.x.device.type == device.type
    assert samples.weights.device.type == device.type


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("name", SCENARIOS)
def test_integrator_device(name, device_name):
    make_and_run(name, device_name, run_integrator)


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("name", SCENARIOS)
def test_vegas_device(name, device_name):
    make_and_run(name, device_name, run_vegas)


def test_flow_condition_masks_follow_module():
    """
    The coupling masks move with the flow, but are not part of the state dict, which stays
    compatible with older checkpoints. The layer sizes are integers, not tensors.
    """
    flow = Flow(dims_in=3)
    assert "condition_masks" in dict(flow.named_buffers())
    assert "condition_masks" not in flow.state_dict()
    flow.to(torch.float64)
    assert flow.condition_masks.dtype == torch.bool
    for module in flow.modules():
        if isinstance(module, nn.Linear):
            assert type(module.in_features) is int
            assert type(module.out_features) is int
    multichannel_flow = Flow(dims_in=3, channels=2)
    for subnet in multichannel_flow.subnets:
        assert type(subnet.features_out) is int
