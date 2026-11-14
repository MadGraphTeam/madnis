"""Tests for the normalizing flow: invertibility, normalization and consistency"""

import pytest
import torch

from madnis.nn import Flow


def perturb(module: torch.nn.Module, scale: float = 0.1, seed: int = 0):
    """
    Adds noise to all parameters. The last layers of the flow subnets are initialized with
    zeros, so the flow is the identity before this.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            if param.requires_grad:
                param.add_(
                    scale * torch.randn(param.shape, generator=gen, dtype=param.dtype)
                )
    return module


FLOW_CONFIGS = {
    "1d": dict(dims_in=1),
    "2d": dict(dims_in=2),
    "5d_log": dict(dims_in=5),
    "5d_exchange": dict(dims_in=5, permutations="exchange", blocks=4),
    "5d_random": dict(dims_in=5, permutations="random", blocks=4),
    "3d_gaussian": dict(dims_in=3, uniform_latent=False),
}


def make_flow(config: str, **kwargs) -> Flow:
    torch.manual_seed(0)
    flow = Flow(**FLOW_CONFIGS[config], **kwargs).double()
    return perturb(flow)


def test_untrained_flow_is_identity():
    torch.manual_seed(0)
    flow = Flow(dims_in=3).double()
    x = torch.rand((100, 3), dtype=torch.float64)
    z, log_jac = flow.transform(x)
    torch.testing.assert_close(z, x)
    torch.testing.assert_close(log_jac, torch.zeros(100, dtype=torch.float64))


@pytest.mark.parametrize("config", FLOW_CONFIGS)
def test_transform_inverse(config):
    flow = make_flow(config)
    dims = flow.dims_in
    x = torch.rand((1000, dims), dtype=torch.float64)
    if not flow.uniform_latent:
        x = 2 * x - 1
    with torch.no_grad():
        z, log_jac = flow.transform(x)
        x_rec, log_jac_inv = flow.transform(z, inverse=True)
    assert not torch.allclose(z, x)
    torch.testing.assert_close(x_rec, x)
    torch.testing.assert_close(log_jac + log_jac_inv, torch.zeros_like(log_jac))


@pytest.mark.parametrize("config", FLOW_CONFIGS)
def test_sample_prob_consistent(config):
    """the probability returned by sample agrees with prob and log_prob"""
    flow = make_flow(config)
    torch.manual_seed(1)
    with torch.no_grad():
        x, log_prob, prob = flow.sample(
            1000, return_log_prob=True, return_prob=True, dtype=torch.float64
        )
        torch.testing.assert_close(prob, flow.prob(x))
        torch.testing.assert_close(log_prob, flow.log_prob(x))
    torch.testing.assert_close(prob.log(), log_prob)
    if flow.uniform_latent:
        assert ((x >= 0) & (x <= 1)).all()


@pytest.mark.parametrize("config", ["2d", "5d_log", "5d_random"])
def test_flow_normalized_on_unit_hypercube(config):
    """a flow with uniform latent space is a normalized density on the unit hypercube"""
    flow = make_flow(config)
    x = torch.rand((100000, flow.dims_in), dtype=torch.float64)
    with torch.no_grad():
        prob = flow.prob(x)
    error = prob.std() / len(prob) ** 0.5
    assert abs(prob.mean().item() - 1.0) < 5 * error
    assert prob.std() > 0.05  # non-trivial density


def test_conditional_flow():
    torch.manual_seed(0)
    flow = perturb(Flow(dims_in=3, dims_c=2).double())
    c = torch.rand((1000, 2), dtype=torch.float64)
    with torch.no_grad():
        x, prob = flow.sample(c=c, return_prob=True)
        torch.testing.assert_close(prob, flow.prob(x, c=c))
        # the density depends on the condition
        assert not torch.allclose(flow.prob(x, c=c), flow.prob(x, c=c.flip(0)))


def multichannel_probs(flow: Flow, x: torch.Tensor) -> torch.Tensor:
    """probabilities for equally sized channels, evaluated channel by channel"""
    n = x.shape[0] // flow.channels
    with torch.no_grad():
        return torch.cat(
            [flow.prob(x[i * n : (i + 1) * n], channel=i) for i in range(flow.channels)]
        )


def test_multichannel_flow_channel_formats():
    """channels given as index tensor, list of sizes and int give the same results"""
    torch.manual_seed(0)
    flow = perturb(Flow(dims_in=3, channels=3).double())
    x = torch.rand((300, 3), dtype=torch.float64)
    channels = torch.arange(3).repeat_interleave(100)
    perm = torch.randperm(300)
    prob_int = multichannel_probs(flow, x)
    with torch.no_grad():
        prob_list = flow.prob(x, channel=[100, 100, 100])
        prob_tensor = flow.prob(x[perm], channel=channels[perm])
        prob_other_channel = flow.prob(x[:100], channel=1)
    torch.testing.assert_close(prob_list, prob_int)
    torch.testing.assert_close(prob_tensor, prob_int[perm])
    # channels are independent flows
    assert not torch.allclose(prob_int[:100], prob_other_channel)


def test_multichannel_flow_equal_channel_sizes():
    """channel=None means that all channels contain the same number of samples"""
    torch.manual_seed(0)
    flow = perturb(Flow(dims_in=3, channels=3).double())
    x = torch.rand((300, 3), dtype=torch.float64)
    with torch.no_grad():
        prob_uniform = flow.prob(x)
    torch.testing.assert_close(prob_uniform, multichannel_probs(flow, x))


def test_multichannel_flow_sample_consistent():
    torch.manual_seed(0)
    flow = perturb(Flow(dims_in=2, channels=4).double())
    channels = torch.randint(4, (1000,))
    with torch.no_grad():
        x, prob = flow.sample(
            1000, channel=channels, return_prob=True, dtype=torch.float64
        )
        torch.testing.assert_close(prob, flow.prob(x, channel=channels))


def test_mapping():
    """a mapping is applied after sampling and included in the probability"""

    def mapping(x, inverse):
        # sampling direction (inverse=False): u -> u^2 with log|dx/du|
        # density direction (inverse=True): x -> sqrt(x) with log|du/dx|
        if inverse:
            return x.sqrt(), -(2 * x.sqrt()).log().sum(dim=1)
        else:
            return x.square(), (2 * x).log().sum(dim=1)

    torch.manual_seed(0)
    flow = Flow(dims_in=2, mapping=mapping).double()
    with torch.no_grad():
        x, prob = flow.sample(10000, return_prob=True, dtype=torch.float64)
        torch.testing.assert_close(prob, flow.prob(x))
    # identity flow + mapping x = u^2 gives density 1 / (2 sqrt(x)) per dimension
    torch.testing.assert_close(prob, 1 / (4 * x.sqrt().prod(dim=1)))


@pytest.mark.parametrize("multichannel", [False, True])
def test_init_with_grid(multichannel):
    """after initialization with a grid, every grid bin has probability 1 / bins"""
    torch.manual_seed(0)
    bins = 10
    channels = 2 if multichannel else None
    flow = Flow(dims_in=2, bins=bins, channels=channels).double()
    # grid with narrow bins near 0 in dimension 0 and near 1 in dimension 1
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    grid = torch.stack([edges**3, 1 - (1 - edges) ** 3])
    if multichannel:
        grid = torch.stack([grid, grid.flip(0)])
    flow.init_with_grid(grid)

    for channel in range(2 if multichannel else 1):
        chan_grid = grid[channel] if multichannel else grid
        with torch.no_grad():
            x = flow.sample(
                200000,
                channel=channel if multichannel else None,
                dtype=torch.float64,
            )
        for dim in range(2):
            counts = torch.histogram(x[:, dim], chan_grid[dim]).hist
            fractions = counts / len(x)
            torch.testing.assert_close(
                fractions, torch.full_like(fractions, 1 / bins), atol=4e-3, rtol=0
            )
