import math
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

import torch

if TYPE_CHECKING:
    from .integrator import TrainingStatus


def synchronize(device: torch.device):
    """
    Waits until all queued operations on an accelerator device are finished, such that wall
    times measured on the host include them. Does nothing on the CPU.

    Args:
        device: device to synchronize
    """
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def peak_memory(device: torch.device) -> int | None:
    """
    Returns the peak memory allocated by torch on a CUDA device in bytes, or None for other
    devices.

    Args:
        device: device to query
    """
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device)


@dataclass
class StepTiming:
    """
    Wall times of training steps in seconds. Operations on GPUs are queued asynchronously, so
    the split between sampling, integrand and optimization is only exact with
    ``detailed_timing=True`` in the ``Integrator``. The total time is always exact.

    Args:
        sampling: sampling from the flow
        integrand: evaluation of the integrand
        optimization: loss, gradients and optimizer step
        total: total time of the training step
    """

    sampling: float = 0.0
    integrand: float = 0.0
    optimization: float = 0.0
    total: float = 0.0

    def add(self, other: "StepTiming"):
        """Adds the times of another ``StepTiming`` object"""
        self.sampling += other.sampling
        self.integrand += other.integrand
        self.optimization += other.optimization
        self.total += other.total


@dataclass
class TrainingStatistics:
    """
    Totals since the integrator was created, returned by ``Integrator.statistics``.

    Args:
        online_steps: number of training steps with newly generated samples
        buffered_steps: number of training steps with samples from the buffer
        skipped_steps: number of training steps skipped because the loss was nan
        integrand_calls: number of points at which the integrand was evaluated, including the
            calls by ``integrate``, ``integration_metrics``, ``unweighting_metrics`` and
            ``sample``
        training_samples: number of samples generated in online training steps, including
            samples with zero integrand
        nonzero_samples: number of training samples with non-zero integrand
        invalid_samples: number of invalid integrand values that were set to zero, see the
            ``invalid_integrand`` argument of the ``Integrator``
        timing: accumulated wall times of the training steps
        integral: combined estimate of the integral from the integration history, see
            ``Integrator.integral``, or None if the history is empty
        error: Monte Carlo error of ``integral``, or None if the history is empty
        channel_cut_efficiencies: moving averages of the channel-wise cut efficiencies during
            training, which are used to determine the oversampling
        peak_memory: peak memory allocated by torch on a CUDA device in bytes, otherwise None
    """

    online_steps: int = 0
    buffered_steps: int = 0
    skipped_steps: int = 0
    integrand_calls: int = 0
    training_samples: int = 0
    nonzero_samples: int = 0
    invalid_samples: int = 0
    timing: StepTiming = field(default_factory=StepTiming)
    integral: float | None = None
    error: float | None = None
    channel_cut_efficiencies: list[float] | None = None
    peak_memory: int | None = None

    @property
    def cut_efficiency(self) -> float | None:
        """Fraction of training samples with non-zero integrand"""
        if self.training_samples == 0:
            return None
        return self.nonzero_samples / self.training_samples


class StatusPrinter:
    """
    Callback for ``Integrator.train`` that prints a summary of the last ``interval`` training
    steps. The integral is the combination of the online batches in the interval, weighted with
    their inverse variances. The cut efficiency is the fraction of non-zero samples, and the
    remaining quantities are averages over the interval.

    Example::

        integrator.train(1000, callback=StatusPrinter(interval=100))
    """

    def __init__(self, interval: int = 100, file: TextIO | None = None):
        """
        Args:
            interval: number of training steps between two lines
            file: stream to print to, standard output by default
        """
        self.interval = interval
        self.file = file
        self._reset()

    def _reset(self):
        self._steps = 0
        self._losses = []
        self._weighted_integral = 0.0
        self._weight_sum = 0.0
        self._rel_stddevs = []
        self._unweighting_effs = []
        self._samples = 0
        self._nonzero = 0
        self._invalid = 0
        self._skipped = 0
        self._time = 0.0

    def __call__(self, status: "TrainingStatus"):
        self._steps += 1
        self._time += status.timing.total
        if status.skipped:
            self._skipped += 1
        else:
            self._losses.append(status.loss)
        if status.integration is not None:
            variance = status.integration.error**2
            if variance > 0:
                self._weighted_integral += status.integration.integral / variance
                self._weight_sum += 1 / variance
            self._rel_stddevs.append(status.integration.rel_stddev)
        if status.unweighting is not None:
            self._unweighting_effs.append(status.unweighting.uweff_after_cuts_partial)
        self._samples += status.sample_count
        self._nonzero += status.nonzero_count
        self._invalid += status.invalid_count

        if (status.step + 1) % self.interval == 0:
            print(self._summary(status.step), file=self.file or sys.stdout)
            self._reset()

    def _summary(self, step: int) -> str:
        parts = [f"step {step + 1:6d}"]
        if len(self._losses) > 0:
            parts.append(f"loss {sum(self._losses) / len(self._losses):.4e}")
        if self._weight_sum > 0:
            integral = self._weighted_integral / self._weight_sum
            error = 1 / math.sqrt(self._weight_sum)
            parts.append(f"integral {integral:.6e} +- {error:.2e}")
        if len(self._rel_stddevs) > 0:
            rel_stddev = sum(self._rel_stddevs) / len(self._rel_stddevs)
            parts.append(f"rel. std. {rel_stddev:.4f}")
        if self._samples > 0:
            parts.append(f"cut eff. {self._nonzero / self._samples:.4f}")
        if len(self._unweighting_effs) > 0:
            unweighting_eff = sum(self._unweighting_effs) / len(self._unweighting_effs)
            parts.append(f"unw. eff. {unweighting_eff:.4f}")
        if self._invalid > 0:
            parts.append(f"invalid {self._invalid}")
        if self._skipped > 0:
            parts.append(f"skipped {self._skipped}")
        parts.append(f"{1000 * self._time / self._steps:.1f} ms/step")
        return " | ".join(parts)
