"""Tests for the training status, the training statistics and the status printer (issue #7)"""

import io

import pytest
import torch

import madnis.integrator.integrator as integrator_module
import madnis.integrator.monitoring as monitoring
from madnis.integrator import (
    Integrand,
    Integrator,
    StatusPrinter,
    StepTiming,
    TrainingStatus,
    VegasPreTraining,
    integration_metrics,
)

from .integrands import cut_function
from .test_integrator import multichannel_integrand, peak


def cut_multichannel(x: torch.Tensor, channels: torch.Tensor):
    p = torch.stack([peak(x, 0.3), peak(x, 0.7)], dim=1)
    prior = (p + 1e-6) / (p + 1e-6).sum(dim=1, keepdim=True)
    return torch.where(x[:, 0] > 0.2, p.sum(dim=1), 0.0), prior


def make_multichannel(**kwargs) -> Integrator:
    integrand = Integrand(
        cut_multichannel, input_dim=2, channel_count=2, has_channel_weight_prior=True
    )
    return Integrator(integrand, batch_size=512, dtype=torch.float64, **kwargs)


def test_online_status():
    torch.manual_seed(0)
    integrator = Integrator(cut_function, dims=2, batch_size=1024, dtype=torch.float64)
    status = integrator.train_step()
    assert not status.buffered and not status.skipped
    assert status.sample_count >= integrator.batch_size
    assert 0 < status.nonzero_count < status.sample_count
    assert status.cut_efficiency == status.nonzero_count / status.sample_count
    # the zeros enter the integration metrics, so the counts are the generated samples
    assert status.integration.count == status.sample_count
    assert status.integration.integral > 0
    assert status.invalid_count == 0
    assert status.grad_norm > 0 and not status.grad_clipped
    assert status.channel_sample_counts is None and status.active_channels is None
    assert status.buffer_size == 0 and status.buffer_stored == 0
    assert status.unweighting is None
    assert status.negative_weight_fraction == 0.0
    timing = status.timing
    assert min(timing.sampling, timing.integrand, timing.optimization) > 0
    assert timing.total >= timing.sampling + timing.integrand + timing.optimization


def test_single_channel_unweighting_status():
    """with one channel, the cut efficiency of the unweighting metrics is the count ratio"""
    torch.manual_seed(0)
    integrator = Integrator(
        cut_function,
        dims=2,
        batch_size=1024,
        dtype=torch.float64,
        drop_zero_integrands=True,
        unweighting_status=True,
    )
    status = integrator.train_step()
    assert status.channel_unweighting is None
    assert status.unweighting.cut_eff == pytest.approx(status.cut_efficiency)
    assert 0 < status.unweighting.uweff_after_cuts_partial <= 1


@pytest.mark.parametrize(
    "drop_zero_integrands, rescale_dropped_zeros",
    [(True, True), (True, False)],
)
def test_status_independent_of_dropping(drop_zero_integrands, rescale_dropped_zeros):
    """
    The first step draws the same samples with and without dropping zeros, so the metrics of the
    online batch have to agree
    """

    def first_status(**kwargs):
        torch.manual_seed(0)
        integrator = make_multichannel(unweighting_status=True, **kwargs)
        return integrator.train_step()

    reference = first_status(drop_zero_integrands=False)
    status = first_status(
        drop_zero_integrands=drop_zero_integrands,
        rescale_dropped_zeros=rescale_dropped_zeros,
    )
    assert status.sample_count == reference.sample_count
    assert status.channel_nonzero_counts == reference.channel_nonzero_counts
    assert status.integration.integral == pytest.approx(
        reference.integration.integral, rel=1e-12
    )
    assert status.integration.channel_counts == reference.integration.channel_counts
    assert status.unweighting == pytest.approx(reference.unweighting, rel=1e-12)
    for channel, (metrics, metrics_reference) in enumerate(
        zip(status.channel_unweighting, reference.channel_unweighting)
    ):
        assert metrics == pytest.approx(metrics_reference, rel=1e-12)
        assert metrics.cut_eff == pytest.approx(
            status.channel_nonzero_counts[channel]
            / status.channel_sample_counts[channel]
        )


def test_multichannel_and_buffered_status():
    torch.manual_seed(0)
    integrator = make_multichannel(
        buffer_capacity=5000,
        buffered_steps=1,
        minimum_buffer_size=10,
        buffer_unweighting_quantile=0.95,
        unweighting_status=True,
    )
    online = integrator.train_step()
    assert online.active_channels == 2
    assert sum(online.channel_sample_counts) == online.sample_count
    assert sum(online.channel_nonzero_counts) == online.nonzero_count
    assert len(online.integration.channel_integrals) == 2
    assert len(online.channel_unweighting) == 2
    assert 0 < online.buffer_stored <= online.nonzero_count
    assert online.buffer_size == online.buffer_stored

    buffered = integrator.train_step()
    assert buffered.buffered
    assert buffered.integration is None and buffered.unweighting is None
    assert buffered.sample_count == 0 and buffered.cut_efficiency is None
    assert buffered.buffer_stored == 0 and buffered.buffer_size == online.buffer_size
    assert buffered.timing.sampling == 0.0 and buffered.timing.integrand == 0.0
    assert buffered.timing.optimization > 0


def test_negative_weight_fraction():
    torch.manual_seed(0)
    integrator = Integrator(
        lambda x: x[:, 0] - 0.25, dims=2, batch_size=8192, dtype=torch.float64
    )
    status = integrator.train_step()
    assert status.negative_weight_fraction == pytest.approx(0.25, abs=0.05)


def test_invalid_integrand_mode_checked():
    with pytest.raises(ValueError, match="Unknown invalid_integrand mode"):
        Integrator(cut_function, dims=2, invalid_integrand="ignore")


def test_invalid_integrand_zero():
    """invalid integrand values are set to zero and counted"""

    def function(x):
        return torch.where(x[:, 0] < 0.1, torch.nan, peak(x))

    torch.manual_seed(0)
    integrator = Integrator(function, dims=2, batch_size=1024, invalid_integrand="zero")
    status = integrator.train_step()
    assert status.invalid_count == pytest.approx(0.1 * status.sample_count, rel=0.3)
    assert status.nonzero_count == status.sample_count - status.invalid_count
    assert torch.isfinite(torch.tensor(status.loss))
    integrator.integrate(1000)
    statistics = integrator.statistics()
    assert statistics.invalid_samples > status.invalid_count
    assert statistics.training_samples == status.sample_count


def test_invalid_prior_zero():
    """negative, non-finite and all-zero prior channel weights are set to zero and counted"""

    def function(x, channels):
        f, prior = cut_multichannel(x, channels)
        prior = torch.where((x[:, 1] < 0.1)[:, None], -prior, prior)
        prior = torch.where((x[:, 1] > 0.9)[:, None], 0.0, prior)
        return torch.where(x[:, 0] > 0.2, f, 1.0), prior

    integrand = Integrand(
        function, input_dim=2, channel_count=2, has_channel_weight_prior=True
    )
    torch.manual_seed(0)
    integrator = Integrator(
        integrand, batch_size=2048, dtype=torch.float64, invalid_integrand="zero"
    )
    samples = integrator.sample(4096)
    x = samples.x
    expected = ((x[:, 1] < 0.1) | (x[:, 1] > 0.9)).sum().item()
    assert integrator.statistics().invalid_samples == expected
    assert torch.isfinite(samples.weights).all()
    assert (samples.weights[(x[:, 1] < 0.1) | (x[:, 1] > 0.9)] == 0).all()
    status = integrator.train_step()
    assert status.invalid_count > 0


def test_vegas_invalid_zero_and_calls():
    def function(x):
        return torch.where(x[:, 0] < 0.1, torch.nan, peak(x))

    torch.manual_seed(0)
    integrator = Integrator(
        function, dims=2, dtype=torch.float64, invalid_integrand="zero"
    )
    vegas = VegasPreTraining(integrator)
    vegas.train([1000] * 2)
    vegas.sample(500)
    statistics = integrator.statistics()
    assert statistics.integrand_calls == 2500
    # the grid adapts away from the invalid region, so less than 10% of the points are there
    assert 0 < statistics.invalid_samples < 250


def test_skipped_step(monkeypatch):
    torch.manual_seed(0)
    integrator = Integrator(cut_function, dims=2, batch_size=256)
    monkeypatch.setattr(
        integrator, "loss", lambda f, q_test, **kwargs: (q_test * torch.nan).sum()
    )
    with pytest.warns(UserWarning, match="nan batch"):
        status = integrator.train_step()
    assert status.skipped and status.grad_norm is None
    assert integrator.statistics().skipped_steps == 1


def test_grad_norm_without_clipping():
    torch.manual_seed(0)
    integrator = Integrator(cut_function, dims=2, batch_size=256)
    status = integrator.train_step()
    grads = [param.grad for param in integrator._trained_parameters()]
    expected = torch.cat([grad.flatten() for grad in grads]).norm().item()
    assert status.grad_norm == pytest.approx(expected, rel=1e-5)
    assert not status.grad_clipped


def test_grad_norm_without_trained_parameters(monkeypatch):
    torch.manual_seed(0)
    integrator = Integrator(cut_function, dims=2, batch_size=256)
    monkeypatch.setattr(integrator, "_trained_parameters", lambda: [])
    assert integrator.train_step().grad_norm == 0.0


@pytest.mark.parametrize("threshold, clipped", [(1e-8, True), (1e8, False)])
def test_grad_norm_with_clipping(threshold, clipped):
    torch.manual_seed(0)
    integrator = Integrator(
        cut_function, dims=2, batch_size=256, grad_clip_threshold=threshold
    )
    status = integrator.train_step()
    assert status.grad_clipped == clipped
    params = integrator._trained_parameters()
    norm_after = torch.cat([param.grad.flatten() for param in params]).norm().item()
    if clipped:
        assert status.grad_norm > norm_after
    else:
        assert status.grad_norm == pytest.approx(norm_after, rel=1e-5)


@pytest.mark.parametrize("detailed_timing", [False, True])
def test_detailed_timing(detailed_timing, monkeypatch):
    calls = []
    monkeypatch.setattr(integrator_module, "synchronize", calls.append)
    torch.manual_seed(0)
    integrator = Integrator(
        cut_function, dims=2, batch_size=256, detailed_timing=detailed_timing
    )
    integrator.train_step()
    # outside of training steps, nothing is timed
    integrator.integrate(1000)
    assert len(calls) == (8 if detailed_timing else 0)


def test_synchronize_and_peak_memory(monkeypatch):
    synchronized = []
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronized.append)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1234)
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    monitoring.synchronize(cpu)
    monitoring.synchronize(cuda)
    assert synchronized == [cuda]
    assert monitoring.peak_memory(cpu) is None
    assert monitoring.peak_memory(cuda) == 1234


def test_statistics():
    torch.manual_seed(0)
    integrator = make_multichannel(
        buffer_capacity=5000, buffered_steps=1, minimum_buffer_size=100
    )
    statistics = integrator.statistics()
    assert statistics.integral is None and statistics.cut_efficiency is None
    statuses = [integrator.train_step() for _ in range(6)]
    integrator.integrate(1000)

    statistics = integrator.statistics()
    assert statistics.online_steps == sum(not s.buffered for s in statuses)
    assert statistics.buffered_steps == sum(s.buffered for s in statuses)
    assert statistics.online_steps + statistics.buffered_steps == 6
    assert statistics.training_samples == sum(s.sample_count for s in statuses)
    assert statistics.nonzero_samples == sum(s.nonzero_count for s in statuses)
    assert statistics.integrand_calls == statistics.training_samples + 1000
    assert statistics.cut_efficiency == pytest.approx(
        statistics.nonzero_samples / statistics.training_samples
    )
    assert statistics.timing.total == pytest.approx(
        sum(s.timing.total for s in statuses)
    )
    assert (statistics.integral, statistics.error) == integrator.integral()
    assert len(statistics.channel_cut_efficiencies) == 2
    assert all(0 < eff <= 1 for eff in statistics.channel_cut_efficiencies)
    assert statistics.peak_memory is None
    # the returned object is a copy
    statistics.timing.total = -1.0
    statistics.online_steps = -1
    assert integrator.statistics().timing.total > 0
    assert integrator.statistics().online_steps > 0


def test_training_status_backwards_compatible():
    status = TrainingStatus(0, 1.0, False, None, 0)
    assert status.integration is None and status.cut_efficiency is None
    assert status.timing == StepTiming()


def test_status_printer():
    torch.manual_seed(0)
    output = io.StringIO()
    integrator = make_multichannel(unweighting_status=True)
    integrator.train(10, callback=StatusPrinter(interval=5, file=output))
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("step      5 | loss ")
    for part in ["integral", "rel. std.", "cut eff.", "unw. eff.", "ms/step"]:
        assert part in lines[1]
    assert "invalid" not in lines[1] and "skipped" not in lines[1]


def test_status_printer_special_cases(capsys):
    """buffered and skipped steps, invalid values and batches without variance"""
    printer = StatusPrinter(interval=3)
    timing = StepTiming(total=0.01)
    printer(TrainingStatus(0, float("nan"), True, None, 0, skipped=True, timing=timing))
    printer(TrainingStatus(1, float("nan"), True, None, 0, skipped=True, timing=timing))
    metrics = integration_metrics(
        torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([10])
    )
    printer(
        TrainingStatus(
            2,
            float("nan"),
            False,
            None,
            0,
            skipped=True,
            integration=metrics,
            sample_count=10,
            nonzero_count=10,
            invalid_count=2,
            timing=timing,
        )
    )
    line = capsys.readouterr().out.strip()
    assert line == (
        "step      3 | rel. std. 0.0000 | cut eff. 1.0000 | invalid 2 | skipped 3 | "
        "10.0 ms/step"
    )
