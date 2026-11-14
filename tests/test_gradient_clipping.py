"""Tests for the gradient clipping of MadGraph7, grad_clip_threshold in the Integrator"""

import math

import pytest
import torch

from madnis.integrator import Integrator

from .integrands import gauss
from .test_integrator import (
    PEAK_INTEGRAL,
    assert_compatible,
    multichannel_integrand,
    peak,
)

THRESHOLD = 0.003


def trained_params(integrator: Integrator) -> list[torch.Tensor]:
    return [
        param
        for group in integrator.optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]


def set_gradients(params: list[torch.Tensor], scale: float, generator=None):
    for param in params:
        param.grad = scale * torch.randn(
            param.shape, dtype=param.dtype, generator=generator
        )


def gradient_rms(params: list[torch.Tensor]) -> float:
    grads = torch.cat([param.grad.flatten() for param in params])
    return grads.square().mean().sqrt().item()


def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="must not be negative"):
        Integrator(gauss, dims=2, grad_clip_threshold=-1.0)


@pytest.mark.parametrize("threshold", [None, 0.0])
def test_clipping_disabled(threshold, monkeypatch):
    """None and zero disable clipping, as a zero threshold does in MadGraph7"""
    torch.manual_seed(0)
    integrator = Integrator(
        gauss, dims=2, batch_size=128, grad_clip_threshold=threshold
    )
    assert integrator.grad_clip_threshold is None

    def fail():
        raise AssertionError("gradients must not be clipped")

    monkeypatch.setattr(integrator, "_clip_gradients", fail)
    integrator.train(2)


def test_large_gradients_rescaled():
    """
    as in MadGraph7, the gradients are multiplied by min(1, threshold * sqrt(n) / norm), so the
    direction is kept and the root mean square of the components equals the threshold
    """
    torch.manual_seed(0)
    integrator = Integrator(
        gauss, dims=2, dtype=torch.float64, grad_clip_threshold=THRESHOLD
    )
    params = trained_params(integrator)
    param_count = sum(param.numel() for param in params)
    set_gradients(params, 100 * THRESHOLD, torch.Generator().manual_seed(1))
    grads_before = [param.grad.clone() for param in params]
    norm_before = torch.cat([grad.flatten() for grad in grads_before]).norm()

    norm = integrator._clip_gradients()

    torch.testing.assert_close(norm, norm_before)
    factor = THRESHOLD * math.sqrt(param_count) / norm_before
    assert factor < 1
    for param, grad in zip(params, grads_before):
        torch.testing.assert_close(param.grad, factor * grad, rtol=1e-5, atol=0)
    assert gradient_rms(params) == pytest.approx(THRESHOLD, rel=1e-5)


def test_small_gradients_unchanged():
    torch.manual_seed(0)
    integrator = Integrator(
        gauss, dims=2, dtype=torch.float64, grad_clip_threshold=THRESHOLD
    )
    params = trained_params(integrator)
    set_gradients(params, 0.1 * THRESHOLD, torch.Generator().manual_seed(1))
    grads_before = [param.grad.clone() for param in params]
    integrator._clip_gradients()
    for param, grad in zip(params, grads_before):
        torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)


def test_frozen_parameters_not_counted():
    """
    After freeze_cwnet_iteration, the channel weight network does not count towards the number of
    parameters, and stale gradients of frozen parameters are left untouched
    """
    torch.manual_seed(0)
    integrator = Integrator(
        multichannel_integrand(),
        dtype=torch.float64,
        grad_clip_threshold=THRESHOLD,
    )
    cwnet_params = list(integrator.cwnet.parameters())
    flow_params = list(integrator.flow.parameters())
    for param in cwnet_params:
        param.requires_grad = False
        param.grad = torch.full_like(param, 1.0)
    for param in flow_params:
        param.grad = torch.full_like(param, 1.0)

    integrator._clip_gradients()

    # all components equal, so each one equals the threshold if only the flow is counted
    for param in flow_params:
        torch.testing.assert_close(
            param.grad, torch.full_like(param, THRESHOLD), rtol=1e-5, atol=0
        )
    for param in cwnet_params:
        torch.testing.assert_close(param.grad, torch.full_like(param, 1.0))


@pytest.mark.parametrize("buffered", [False, True])
def test_optimizer_receives_clipped_gradients(buffered, monkeypatch):
    """online and buffered training steps both clip before the optimizer step"""
    torch.manual_seed(0)
    threshold = 1e-6
    integrator = Integrator(
        multichannel_integrand(),
        batch_size=256,
        dtype=torch.float64,
        grad_clip_threshold=threshold,
        buffer_capacity=1000 if buffered else 0,
        minimum_buffer_size=100,
        buffered_steps=2 if buffered else 0,
    )
    params = trained_params(integrator)
    step = integrator.optimizer.step
    rms_values = []

    def recording_step(*args, **kwargs):
        rms_values.append(gradient_rms(params))
        return step(*args, **kwargs)

    monkeypatch.setattr(integrator.optimizer, "step", recording_step)
    statuses = [integrator.train_step() for _ in range(6)]
    assert any(status.buffered for status in statuses) == buffered
    assert len(rms_values) == 6
    for rms in rms_values:
        assert rms == pytest.approx(threshold, rel=1e-5)


def test_training_with_clipping():
    """training with the MadGraph7 threshold still reduces the variance without bias"""
    torch.manual_seed(0)
    integrator = Integrator(
        peak,
        dims=2,
        batch_size=512,
        dtype=torch.float64,
        grad_clip_threshold=THRESHOLD,
    )
    rel_stddev_before = integrator.integration_metrics(50000).rel_stddev
    integrator.train(200)
    rel_stddev_after = integrator.integration_metrics(50000).rel_stddev
    assert rel_stddev_after < 0.4 * rel_stddev_before
    assert_compatible(*integrator.integrate(100000), PEAK_INTEGRAL)
