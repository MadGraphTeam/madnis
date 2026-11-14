"""Tests for the discrete autoregressive flows and the mixed discrete-continuous flow"""

import itertools

import pytest
import torch

from madnis.nn import DiscreteMADE, DiscreteTransformer, MaskedMLP, MixedFlow

from .test_flow_properties import perturb

DIMS = [2, 3, 4]


def all_configurations(dims: list[int]) -> torch.Tensor:
    return torch.tensor(list(itertools.product(*(range(d) for d in dims))))


def make_discrete(model: str, dims_c: int = 0, **kwargs):
    torch.manual_seed(0)
    if model == "made":
        flow = DiscreteMADE(DIMS, dims_c=dims_c, **kwargs)
    else:
        flow = DiscreteTransformer(
            DIMS,
            dims_c=dims_c,
            embedding_dim=16,
            feedforward_dim=32,
            heads=2,
            transformer_layers=2,
            mlp_units=32,
            **kwargs,
        )
    return perturb(flow.double(), scale=0.3)


def prior_prob_function(x: torch.Tensor, i: int) -> torch.Tensor:
    """prior that prefers large indices and depends on the previous index"""
    probs = torch.arange(1, DIMS[i] + 1, dtype=torch.float64).expand(x.shape[0], -1)
    if i > 0:
        probs = probs + x[:, i - 1 : i].double()
    return probs.clone()


@pytest.mark.parametrize("model", ["made", "transformer"])
@pytest.mark.parametrize("prior", [False, True], ids=["no_prior", "prior"])
def test_discrete_flow_normalized(model, prior):
    """the probabilities of all configurations add up to one"""
    flow = make_discrete(
        model, prior_prob_function=prior_prob_function if prior else None
    )
    x = all_configurations(DIMS)
    with torch.no_grad():
        prob = flow.log_prob(x).exp()
    torch.testing.assert_close(prob.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert prob.std() > 0.1 * prob.mean()  # non-trivial distribution


@pytest.mark.parametrize("model", ["made", "transformer"])
@pytest.mark.parametrize("prior", [False, True], ids=["no_prior", "prior"])
def test_discrete_flow_sampling(model, prior):
    """sampled probabilities agree with log_prob and with the sampled frequencies"""
    flow = make_discrete(
        model, prior_prob_function=prior_prob_function if prior else None
    )
    torch.manual_seed(1)
    n = 100000
    with torch.no_grad():
        x, log_prob, prob = flow.sample(
            n, return_log_prob=True, return_prob=True, dtype=torch.float64
        )
        torch.testing.assert_close(log_prob, flow.log_prob(x))
        configs = all_configurations(DIMS)
        expected = flow.log_prob(configs).exp()
    torch.testing.assert_close(prob, log_prob.exp())
    assert ((x >= 0) & (x < torch.tensor(DIMS))).all()
    # frequencies of all configurations, compared with binomial uncertainties
    index = (x * torch.tensor([12, 4, 1])).sum(dim=1)
    frequencies = torch.bincount(index, minlength=len(configs)) / n
    errors = (expected * (1 - expected) / n).sqrt()
    assert ((frequencies - expected).abs() < 5 * errors + 1e-4).all()


@pytest.mark.parametrize("model", ["made", "transformer"])
def test_conditional_discrete_flow(model):
    """for every condition, the probabilities are normalized and agree with sampling"""
    flow = make_discrete(model, dims_c=2)
    x = all_configurations(DIMS)
    torch.manual_seed(1)
    for c_value in torch.rand((3, 2), dtype=torch.float64):
        c = c_value.expand(len(x), -1)
        with torch.no_grad():
            prob = flow.log_prob(x, c=c).exp()
        torch.testing.assert_close(prob.sum(), torch.tensor(1.0, dtype=torch.float64))
    c = torch.rand((1000, 2), dtype=torch.float64)
    with torch.no_grad():
        x_sample, log_prob = flow.sample(c=c, return_log_prob=True)
        torch.testing.assert_close(log_prob, flow.log_prob(x_sample, c=c))


def test_multichannel_discrete_made():
    torch.manual_seed(0)
    flow = perturb(DiscreteMADE(DIMS, channels=3).double(), scale=0.3)
    x = all_configurations(DIMS)
    with torch.no_grad():
        probs = [flow.log_prob(x, channel=i).exp() for i in range(3)]
        channels = torch.randint(3, (1000,))
        x_sample, log_prob = flow.sample(
            1000, channel=channels, return_log_prob=True, dtype=torch.float64
        )
        torch.testing.assert_close(log_prob, flow.log_prob(x_sample, channel=channels))
    for prob in probs:
        torch.testing.assert_close(prob.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert not torch.allclose(probs[0], probs[1])


@pytest.mark.parametrize("channels", [None, 2])
def test_masked_mlp_autoregressive(channels):
    """output block i only depends on the input blocks 0, ..., i"""
    input_dims, output_dims = [2, 3, 1], [3, 1, 4]
    torch.manual_seed(0)
    net = perturb(MaskedMLP(input_dims, output_dims, channels=channels).double())
    x = torch.rand((5, sum(input_dims)), dtype=torch.float64, requires_grad=True)
    channel = None if channels is None else 1
    jacobian = torch.autograd.functional.jacobian(
        lambda inp: net(inp, channel).sum(dim=0), x
    ).sum(dim=1)
    in_blocks = torch.repeat_interleave(torch.arange(3), torch.tensor(input_dims))
    out_blocks = torch.repeat_interleave(torch.arange(3), torch.tensor(output_dims))
    allowed = out_blocks[:, None] >= in_blocks[None, :]
    assert (jacobian[~allowed] == 0).all()
    assert (jacobian[allowed] != 0).all()


def test_masked_mlp_equal_channel_sizes():
    """channel=None means that all channels contain the same number of samples"""
    torch.manual_seed(0)
    net = perturb(MaskedMLP([2, 3], [3, 4], channels=3).double())
    x = torch.rand((6, 5), dtype=torch.float64)
    with torch.no_grad():
        expected = torch.cat([net(x[2 * i : 2 * i + 2], i) for i in range(3)])
        torch.testing.assert_close(net(x), expected)


@pytest.mark.parametrize("channels", [None, 2])
def test_masked_mlp_cached_forward(channels):
    """evaluating the network block by block with a cache gives the full result"""
    input_dims, output_dims = [2, 3, 1], [3, 1, 4]
    torch.manual_seed(0)
    net = perturb(MaskedMLP(input_dims, output_dims, channels=channels).double())
    x = torch.rand((5, sum(input_dims)), dtype=torch.float64)
    channel = None if channels is None else 1
    with torch.no_grad():
        full = net(x, channel)
        cache = None
        outputs = []
        for i, x_block in enumerate(x.split(input_dims, dim=1)):
            y, cache = net.forward_cached(x_block, i, cache, channel)
            outputs.append(y)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full)


@pytest.mark.parametrize(
    "position, model",
    [
        ("first", "made"),
        ("last", "made"),
        ("first", "transformer"),
        # with discrete dimensions last, the discrete flow is conditional
        ("last", "transformer"),
    ],
)
def test_mixed_flow(position, model):
    """sampled probabilities agree with log_prob and the density is normalized"""
    torch.manual_seed(0)
    discrete_kwargs = (
        {}
        if model == "made"
        else dict(embedding_dim=16, feedforward_dim=32, heads=2, mlp_units=32)
    )
    flow = MixedFlow(
        dims_in_continuous=2,
        dims_in_discrete=[2, 3],
        discrete_dims_position=position,
        discrete_model=model,
        discrete_kwargs=discrete_kwargs,
    ).double()
    perturb(flow)
    torch.manual_seed(1)
    with torch.no_grad():
        x, log_prob = flow.sample(10000, return_log_prob=True, dtype=torch.float64)
        torch.testing.assert_close(log_prob, flow.log_prob(x))

    # normalization: sum over discrete configurations, MC integral over continuous dims
    discrete = all_configurations([2, 3]).double()
    n = 20000
    total = 0.0
    for config in discrete:
        x_cont = torch.rand((n, 2), dtype=torch.float64)
        x_disc = config.expand(n, -1)
        x_full = (
            torch.cat((x_disc, x_cont), dim=1)
            if position == "first"
            else torch.cat((x_cont, x_disc), dim=1)
        )
        with torch.no_grad():
            total += flow.log_prob(x_full).exp().mean().item()
    assert abs(total - 1.0) < 0.03
