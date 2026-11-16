"""End-to-end tests for the Integrator: integration, training, sampling and state handling"""

import io
import math
import os
import signal

import pytest
import torch

from madnis.integrator import ChannelGrouping, Integrand, Integrator, SampleBatch
from madnis.integrator.losses import (
    kl_divergence,
    kl_divergence_softclip,
    rkl_divergence,
    stratified_variance,
    variance,
)
from madnis.nn import Flow

SIGMA = 0.1


def gauss_1d_integral(mu: float, sigma: float = SIGMA) -> float:
    """integral of exp(-(x - mu)^2 / (2 sigma^2)) over [0, 1]"""
    s = sigma * math.sqrt(2)
    return sigma * math.sqrt(math.pi / 2) * (math.erf((1 - mu) / s) + math.erf(mu / s))


def peak(x: torch.Tensor, mu: float = 0.5) -> torch.Tensor:
    return torch.exp(-(x - mu).square().sum(dim=1) / (2 * SIGMA**2))


PEAK_INTEGRAL = gauss_1d_integral(0.5) ** 2


def two_peaks(x: torch.Tensor) -> torch.Tensor:
    return peak(x, 0.25) + peak(x, 0.75)


TWO_PEAKS_INTEGRAL = gauss_1d_integral(0.25) ** 2 + gauss_1d_integral(0.75) ** 2


def two_peaks_with_prior(x: torch.Tensor, channels: torch.Tensor):
    """two-channel integrand, the prior channel weights follow the two peaks"""
    p0, p1 = peak(x, 0.25), peak(x, 0.75)
    alpha_0 = (p0 + 1e-6) / (p0 + p1 + 2e-6)
    return p0 + p1, torch.stack([alpha_0, 1 - alpha_0], dim=1)


def multichannel_integrand(**kwargs) -> Integrand:
    return Integrand(
        two_peaks_with_prior,
        input_dim=2,
        channel_count=2,
        has_channel_weight_prior=True,
        **kwargs,
    )


def assert_compatible(value: float, error: float, expected: float, n_sigma: float = 5):
    assert error > 0
    assert abs(value - expected) < n_sigma * error, (value, error, expected)


def assert_weights_unbiased(weights: torch.Tensor, expected: float):
    error = (weights.std() / len(weights) ** 0.5).item()
    assert_compatible(weights.mean().item(), error, expected)


@pytest.mark.parametrize(
    "loss",
    [
        kl_divergence,
        kl_divergence_softclip,
        rkl_divergence,
        variance,
        stratified_variance,
    ],
    ids=lambda loss: loss.__name__,
)
def test_single_channel_training(loss):
    """training reduces the variance, and integration results are unbiased"""
    torch.manual_seed(0)
    integrator = Integrator(
        peak, dims=2, loss=loss, batch_size=512, dtype=torch.float64
    )
    rel_stddev_before = integrator.integration_metrics(50000).rel_stddev
    integrator.train(200)
    rel_stddev_after = integrator.integration_metrics(50000).rel_stddev
    assert rel_stddev_after < 0.4 * rel_stddev_before
    assert_compatible(*integrator.integrate(100000), PEAK_INTEGRAL)
    assert_compatible(*integrator.integral(), PEAK_INTEGRAL)


def test_float32():
    torch.manual_seed(0)
    integrator = Integrator(peak, dims=2, batch_size=512)
    assert integrator.dummy.dtype == torch.float32
    integrator.train(50)
    assert_compatible(*integrator.integrate(100000), PEAK_INTEGRAL)


@pytest.mark.parametrize("train_channel_weights", [True, False])
def test_multichannel_training(train_channel_weights):
    torch.manual_seed(0)
    integrator = Integrator(
        multichannel_integrand(),
        batch_size=512,
        train_channel_weights=train_channel_weights,
        dtype=torch.float64,
    )
    assert (integrator.cwnet is not None) == train_channel_weights
    rel_stddev_before = integrator.integration_metrics(50000).rel_stddev
    integrator.train(200)
    metrics = integrator.integration_metrics(100000)
    assert metrics.rel_stddev < 0.4 * rel_stddev_before
    assert_compatible(metrics.integral, metrics.error, TWO_PEAKS_INTEGRAL)
    # both channels contribute half of the integral
    for channel_integral, channel_error in zip(
        metrics.channel_integrals, metrics.channel_errors
    ):
        assert_compatible(channel_integral, channel_error, TWO_PEAKS_INTEGRAL / 2)


def test_multichannel_without_prior():
    """without prior channel weights, the channel weight network starts uniform"""
    torch.manual_seed(0)
    integrand = Integrand(lambda x, c: two_peaks(x), input_dim=2, channel_count=2)
    integrator = Integrator(integrand, batch_size=512, dtype=torch.float64)
    integrator.train(100)
    assert_compatible(*integrator.integrate(100000), TWO_PEAKS_INTEGRAL)


@pytest.mark.parametrize("multichannel", [False, True])
@pytest.mark.parametrize("mode", ["variance", "mean", "uniform"])
def test_sample_weights_unbiased(multichannel, mode):
    torch.manual_seed(0)
    integrand = multichannel_integrand() if multichannel else two_peaks
    integrator = Integrator(integrand, dims=2, batch_size=512, dtype=torch.float64)
    integrator.train(50)
    samples = integrator.sample(200000, channel_weight_mode=mode)
    assert samples.x.shape == (200000, 2)
    assert_weights_unbiased(samples.weights, TWO_PEAKS_INTEGRAL)


def test_sample_without_integrand():
    torch.manual_seed(0)
    calls = []
    integrator = Integrator(
        lambda x: calls.append(1) or peak(x), dims=2, dtype=torch.float64
    )
    samples = integrator.sample(1000, evaluate_integrand=False)
    assert len(calls) == 0
    assert samples.x.shape == (1000, 2)
    assert (samples.q_sample > 0).all()


def test_integrand_with_bounds():
    """integral of x0 * x1 over [1, 3] x [-1, 0]"""
    torch.manual_seed(0)
    integrand = Integrand(
        lambda x: x[:, 0] * x[:, 1], input_dim=2, bounds=[[1.0, 3.0], [-1.0, 0.0]]
    )
    integrator = Integrator(integrand, batch_size=512, dtype=torch.float64)
    integrator.train(50)
    assert_compatible(*integrator.integrate(100000), -2.0)


def test_remapped_integrand():
    """the channel weight network gets the remapped point y as input"""

    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        return f, x.square(), alphas

    torch.manual_seed(0)
    integrand = Integrand(
        function,
        input_dim=2,
        channel_count=2,
        remapped_dim=2,
        has_channel_weight_prior=True,
    )
    integrator = Integrator(integrand, batch_size=512, dtype=torch.float64)
    integrator.train(100)
    assert_compatible(*integrator.integrate(100000), TWO_PEAKS_INTEGRAL)


def grouped_integrand() -> Integrand:
    """four channels, channels 2 and 3 are grouped with channels 0 and 1"""

    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        return f, torch.cat((alphas, alphas), dim=1) / 2

    return Integrand(
        function,
        input_dim=2,
        channel_count=4,
        has_channel_weight_prior=True,
        channel_grouping=ChannelGrouping([None, None, 0, 1]),
    )


@pytest.mark.parametrize("grouping_mode", ["none", "uniform"])
def test_channel_grouping_integration(grouping_mode):
    torch.manual_seed(0)
    integrator = Integrator(
        grouped_integrand(),
        batch_size=512,
        channel_grouping_mode=grouping_mode,
        dtype=torch.float64,
    )
    assert integrator.integration_channel_count == (4 if grouping_mode == "none" else 2)
    integrator.train(100)
    assert_compatible(*integrator.integrate(100000), TWO_PEAKS_INTEGRAL)


@pytest.mark.parametrize("grouping_mode", ["none", "uniform"])
def test_channel_grouping_sample_weights(grouping_mode):
    torch.manual_seed(0)
    integrator = Integrator(
        grouped_integrand(),
        batch_size=512,
        channel_grouping_mode=grouping_mode,
        dtype=torch.float64,
    )
    integrator.train(50)
    samples = integrator.sample(200000)
    assert_weights_unbiased(samples.weights, TWO_PEAKS_INTEGRAL)


@pytest.mark.parametrize("grouping_mode", ["none", "uniform"])
def test_active_channels_mask_ungrouped(grouping_mode):
    """the mask passed to the integrand always has one entry per channel"""
    integrator = Integrator(grouped_integrand(), channel_grouping_mode=grouping_mode)
    if grouping_mode == "none":
        integrator.active_channels_mask[:] = torch.tensor([True, False, True, False])
    else:
        integrator.active_channels_mask[:] = torch.tensor([True, False])
    assert integrator._active_channels_mask_ungrouped().tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_channel_dropping():
    """a channel with a negligible contribution is dropped"""

    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        alphas = torch.cat((alphas, torch.full_like(alphas[:, :1], 1e-7)), dim=1)
        return f, alphas / alphas.sum(dim=1, keepdim=True)

    torch.manual_seed(0)
    integrand = Integrand(
        function, input_dim=2, channel_count=3, has_channel_weight_prior=True
    )
    integrator = Integrator(
        integrand,
        batch_size=512,
        channel_dropping_threshold=0.01,
        channel_dropping_interval=50,
        dtype=torch.float64,
    )
    statuses = []
    integrator.train(100, callback=statuses.append)
    assert [
        status.dropped_channels for status in statuses if status.dropped_channels
    ] == [1]
    assert integrator.active_channels_mask.tolist() == [True, True, False]
    samples = integrator.sample(10000)
    assert (samples.channels != 2).all()
    assert_compatible(*integrator.integrate(100000), TWO_PEAKS_INTEGRAL)


def test_freeze_cwnet():
    torch.manual_seed(0)
    integrator = Integrator(
        multichannel_integrand(),
        batch_size=256,
        freeze_cwnet_iteration=10,
        dtype=torch.float64,
    )
    integrator.train(11)
    frozen = [param.clone() for param in integrator.cwnet.parameters()]
    integrator.train(10)
    assert not any(param.requires_grad for param in integrator.cwnet.parameters())
    for param, frozen_param in zip(integrator.cwnet.parameters(), frozen):
        torch.testing.assert_close(param, frozen_param)


@pytest.mark.parametrize("multichannel", [False, True])
def test_buffered_training(multichannel):
    """after the buffer is filled, every online step is followed by buffered_steps steps"""
    torch.manual_seed(0)
    integrand = multichannel_integrand() if multichannel else two_peaks
    integrator = Integrator(
        integrand,
        dims=2,
        batch_size=512,
        buffer_capacity=5000,
        minimum_buffer_size=1000,
        buffered_steps=2,
        max_stored_channel_weights=1 if multichannel else None,
        dtype=torch.float64,
    )
    statuses = []
    integrator.train(30, callback=statuses.append)
    buffered = [status.buffered for status in statuses]
    # the buffer exceeds the minimum size after two online steps
    assert buffered[:6] == [False, False, True, True, False, True]
    assert buffered[3:] == ([True, False, True] * 9)
    assert integrator.buffer.size == 5000
    if multichannel:
        assert integrator.buffer.buffer5.shape == (
            5000,
            1,
        )  # compressed channel weights
    assert_compatible(*integrator.integrate(100000), TWO_PEAKS_INTEGRAL)


def test_scheduler():
    torch.manual_seed(0)
    integrator = Integrator(
        peak,
        dims=2,
        batch_size=256,
        learning_rate=1e-3,
        scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5),
        dtype=torch.float64,
    )
    statuses = []
    integrator.train(12, callback=statuses.append)
    assert [status.learning_rate for status in statuses[::5]] == pytest.approx(
        [1e-3, 5e-4, 2.5e-4]
    )


@pytest.mark.parametrize("multichannel", [False, True])
def test_state_dict_roundtrip(multichannel):
    def make():
        integrand = multichannel_integrand() if multichannel else two_peaks
        return Integrator(integrand, dims=2, batch_size=256, dtype=torch.float64)

    torch.manual_seed(0)
    integrator = make()
    integrator.train(30)
    stream = io.BytesIO()
    torch.save(integrator.state_dict(), stream)
    stream.seek(0)

    loaded = make()
    loaded.load_state_dict(torch.load(stream))
    x = torch.rand((1000, 2), dtype=torch.float64)
    channels = torch.randint(2, (1000,)) if multichannel else None
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.flow.prob(x, channel=channels),
            integrator.flow.prob(x, channel=channels),
        )
    assert loaded.integral() == integrator.integral()


def test_metrics_methods():
    torch.manual_seed(0)
    integrator = Integrator(
        multichannel_integrand(), batch_size=512, dtype=torch.float64
    )
    integrator.train(50)
    metrics = integrator.integration_metrics(20000)
    assert metrics.count == 20000
    assert sum(metrics.channel_counts) == 20000
    # in the multi-channel case, the channel-wise metrics are returned as well
    unweighting, channel_unweighting = integrator.unweighting_metrics(20000)
    assert len(channel_unweighting) == 2
    assert 0 < unweighting.uweff_after_cuts <= 1
    assert unweighting.cut_eff == pytest.approx(1.0)


def discrete_prior(x_discrete: torch.Tensor, index: int) -> torch.Tensor:
    """prior for a single discrete dimension with three options, preferring larger values"""
    return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).expand(
        len(x_discrete), -1
    )


@pytest.mark.parametrize("position", ["first", "last"])
@pytest.mark.parametrize("model", ["made", "transformer"])
@pytest.mark.parametrize("prior", [False, True], ids=["no_prior", "prior"])
def test_discrete_integrand(position, model, prior):
    """sum over a discrete dimension times an integral over a continuous dimension"""
    discrete_index = 0 if position == "first" else 1

    def function(x):
        k, x_cont = x[:, discrete_index], x[:, 1 - discrete_index]
        return (1 + k) * torch.exp(-((x_cont - 0.5) ** 2) / (2 * SIGMA**2))

    torch.manual_seed(0)
    integrand = Integrand(
        function,
        input_dim=2,
        discrete_dims=[3],
        discrete_dims_position=position,
        discrete_prior_prob_function=discrete_prior if prior else None,
    )
    discrete_flow_kwargs = (
        {}
        if model == "made"
        else dict(embedding_dim=16, feedforward_dim=32, heads=2, mlp_units=32)
    )
    integrator = Integrator(
        integrand,
        batch_size=512,
        discrete_model=model,
        discrete_flow_kwargs=discrete_flow_kwargs,
        dtype=torch.float64,
    )
    integrator.train(50)
    expected = (1 + 2 + 3) * gauss_1d_integral(0.5)
    assert_compatible(*integrator.integrate(100000), expected)
    if prior:
        # before training, the discrete samples follow the prior
        untrained = Integrator(
            integrand,
            batch_size=512,
            discrete_model=model,
            discrete_flow_kwargs=discrete_flow_kwargs,
            dtype=torch.float64,
        )
        k = untrained.sample(60000).x[:, discrete_index].long()
        frequencies = torch.bincount(k, minlength=3).double() / len(k)
        torch.testing.assert_close(
            frequencies,
            torch.tensor([1 / 6, 2 / 6, 3 / 6], dtype=torch.float64),
            atol=0.01,
            rtol=0,
        )


def test_multichannel_discrete_integrand():
    """multi-channel integration with a discrete dimension and channel weights"""

    def function(x, channels):
        f, alphas = two_peaks_with_prior(x[:, 1:], channels)
        return (1 + x[:, 0]) * f, alphas

    torch.manual_seed(0)
    integrand = Integrand(
        function,
        input_dim=3,
        channel_count=2,
        has_channel_weight_prior=True,
        discrete_dims=[3],
    )
    integrator = Integrator(integrand, batch_size=512, dtype=torch.float64)
    integrator.train(100)
    assert_compatible(*integrator.integrate(100000), 6 * TWO_PEAKS_INTEGRAL)


def test_purely_discrete_integrand():
    """the discrete flow can learn the target distribution exactly"""

    def function(x):
        # the discrete flow passes integer indices, the integrand has to return floats
        assert x.dtype == torch.int64
        return ((1 + x[:, 0]) * (1 + x[:, 1]) ** 2).double()

    torch.manual_seed(0)
    integrand = Integrand(function, input_dim=2, discrete_dims=[3, 4])
    integrator = Integrator(integrand, batch_size=512, dtype=torch.float64)
    rel_stddev_before = integrator.integration_metrics(20000).rel_stddev
    integrator.train(300)
    metrics = integrator.integration_metrics(100000)
    expected = (1 + 2 + 3) * (1 + 4 + 9 + 16)
    assert_compatible(metrics.integral, metrics.error, expected)
    assert metrics.rel_stddev < 0.2 * rel_stddev_before


def test_channel_dropping_removes_buffered_samples():
    def function(x, channels):
        f, alphas = two_peaks_with_prior(x, channels)
        alphas = torch.cat((alphas, torch.full_like(alphas[:, :1], 1e-7)), dim=1)
        return f, alphas / alphas.sum(dim=1, keepdim=True)

    torch.manual_seed(0)
    integrand = Integrand(
        function, input_dim=2, channel_count=3, has_channel_weight_prior=True
    )
    integrator = Integrator(
        integrand,
        batch_size=512,
        buffer_capacity=20000,
        minimum_buffer_size=100000,  # no buffered steps, only storage
        channel_dropping_threshold=0.01,
        channel_dropping_interval=20,
        dtype=torch.float64,
    )
    integrator.train(19)
    buffered = SampleBatch(*integrator.buffer)
    assert (buffered.channels == 2).any()
    integrator.train(1)
    buffered = SampleBatch(*integrator.buffer)
    assert integrator.buffer.size > 0
    assert (buffered.channels != 2).all()


def test_sample_single_channel():
    torch.manual_seed(0)
    integrator = Integrator(multichannel_integrand(), dtype=torch.float64)
    samples = integrator.sample(1000, channel=1)
    assert (samples.channels == 1).all()


def test_custom_optimizer_and_device():
    torch.manual_seed(0)
    integrator = Integrator(
        peak,
        dims=2,
        batch_size=256,
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
        device="cpu",
        dtype=torch.float64,
    )
    assert isinstance(integrator.optimizer, torch.optim.SGD)
    assert integrator.dummy.device.type == "cpu"
    integrator.train(5)

    flow = Flow(dims_in=2).double()
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    integrator = Integrator(
        peak, dims=2, flow=flow, optimizer=optimizer, dtype=torch.float64
    )
    assert integrator.optimizer is optimizer
    integrator.train(5)


def test_multichannel_without_prior_and_cwnet():
    """without prior and channel weight network, the channel weights are uniform"""
    torch.manual_seed(0)
    integrand = Integrand(lambda x, c: two_peaks(x), input_dim=2, channel_count=2)
    integrator = Integrator(
        integrand, batch_size=512, train_channel_weights=False, dtype=torch.float64
    )
    samples = integrator.sample(100)
    torch.testing.assert_close(
        integrator._get_alphas(samples), torch.full((100, 2), 0.5, dtype=torch.float64)
    )
    integrator.train(20)
    assert_compatible(*integrator.integrate(100000), TWO_PEAKS_INTEGRAL)


def test_nan_loss_skips_optimization():
    def nan_loss(f_true, q_test, q_sample=None, channels=None):
        return (q_test * torch.nan).mean()

    torch.manual_seed(0)
    integrator = Integrator(peak, dims=2, loss=nan_loss, dtype=torch.float64)
    params = [param.clone() for param in integrator.flow.parameters()]
    with pytest.warns(UserWarning, match="nan batch"):
        integrator.train(2)
    for param, old_param in zip(integrator.flow.parameters(), params):
        torch.testing.assert_close(param, old_param)


def test_capture_keyboard_interrupt():
    """the training stops after the step during which SIGINT was received"""

    def callback(status):
        if status.step == 2:
            os.kill(os.getpid(), signal.SIGINT)

    torch.manual_seed(0)
    integrator = Integrator(peak, dims=2, batch_size=128, dtype=torch.float64)
    old_handler = signal.getsignal(signal.SIGINT)
    integrator.train(10, callback=callback, capture_keyboard_interrupt=True)
    assert integrator.step == 3
    assert signal.getsignal(signal.SIGINT) is old_handler


def test_sample_batch_split():
    x = torch.rand((10, 2))
    batch = SampleBatch(x, None, torch.ones(10), torch.ones(10), None)
    batches = list(batch.split(4))
    assert [len(b.x) for b in batches] == [4, 4, 2]
    torch.testing.assert_close(torch.cat([b.x for b in batches]), x)
    assert all(b.y is None for b in batches)


def test_store_nothing_after_unweighting():
    torch.manual_seed(0)
    integrator = Integrator(
        peak,
        dims=2,
        buffer_capacity=100,
        buffer_unweighting_quantile=0.9,
        dtype=torch.float64,
    )
    x = torch.rand((50, 2), dtype=torch.float64)
    integrator._store_samples(
        SampleBatch(x, None, torch.ones(50), torch.zeros(50), None)
    )
    assert integrator.buffer.size == 0


def test_purely_discrete_transformer():
    def function(x):
        return ((1 + x[:, 0]) * (1 + x[:, 1])).double()

    torch.manual_seed(0)
    integrand = Integrand(function, input_dim=2, discrete_dims=[3, 4])
    integrator = Integrator(
        integrand,
        batch_size=256,
        discrete_model="transformer",
        discrete_flow_kwargs=dict(
            embedding_dim=16, feedforward_dim=32, heads=2, mlp_units=32
        ),
        dtype=torch.float64,
    )
    integrator.train(20)
    assert_compatible(*integrator.integrate(50000), 6 * 10)


@pytest.mark.parametrize("discrete_dims", [[3, 4], [3]], ids=["discrete", "mixed"])
def test_invalid_discrete_options(discrete_dims):
    integrand = Integrand(
        lambda x, c: x[:, -1], input_dim=2, channel_count=2, discrete_dims=discrete_dims
    )
    with pytest.raises(NotImplementedError, match="single-channel"):
        Integrator(integrand, discrete_model="transformer")
    with pytest.raises(ValueError, match="made"):
        Integrator(integrand, discrete_model="unknown")


def test_invalid_grouping_mode():
    with pytest.raises(ValueError, match="grouping mode"):
        Integrator(grouped_integrand(), channel_grouping_mode="unknown")
