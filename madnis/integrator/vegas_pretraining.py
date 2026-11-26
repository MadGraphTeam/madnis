import math
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from .integrator import Integrator, SampleBatch
from .metrics import (
    IntegrationMetrics,
    UnweightingMetrics,
    integration_metrics,
    unweighting_metrics,
)
from .vegas_grid import VegasGrid


@dataclass
class VegasTrainingStatus:
    """
    Contains the VEGAS training status to pass it to a callback function.

    Args:
        step: optimization step
        variance: integration variance in this iteration. Combined for all channels in the
            multi-channel case
    """

    step: int
    variance: float
    # TODO: maybe add more, like channel-wise statistics?


class VegasPreTraining:
    """
    Implements VEGAS pre-training. It wraps around an ``Integrator`` object and uses its integrand,
    sample buffer and integration history. In addition, it also defines the functions sample,
    integrate, integration_metrics and unweighting_metrics to allow for comparisions between VEGAS
    and MadNIS.
    """

    def __init__(
        self,
        integrator: Integrator,
        bins: int = 64,
        damping: float = 0.7,
    ):
        """
        Args:
            integrator: MadNIS integrator that provides the integrand, buffer and integration
                history. In the multi-channel case, one VEGAS grid is built per channel or
                channel group.
            bins: number of VEGAS grid bins per dimension
            damping: damping parameter (``alpha`` in the vegas package) of the grid adaptation
        """
        self.integrator = integrator
        self.integrand = integrator.integrand
        self.damping = damping
        self.input_dim = self.integrand.input_dim

        if self.integrator.multichannel:
            if self.integrand.channel_grouping is None:
                self.grid_channels = [
                    [channel] for channel in range(self.integrand.channel_count)
                ]
            else:
                self.grid_channels = [
                    group.channel_indices
                    for group in self.integrand.channel_grouping.groups
                ]
        else:
            self.grid_channels = [[0]]
        self.grids = VegasGrid(
            len(self.grid_channels),
            self.input_dim,
            bins,
            device=self.integrator.dummy.device,
        )
        self.step = 0

    def _channel_weight(
        self,
        func_vals: torch.Tensor,
        channels: torch.Tensor,
        alphas: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Returns the channel weight of the sampled channel. Without prior channel weights, the
        channel weights are uniform, as in the integrator without channel weight network.
        """
        if alphas is None:
            return torch.full_like(func_vals, 1 / self.integrand.channel_count)
        return torch.gather(alphas, index=channels[:, None], dim=1)[:, 0]

    def _draw(
        self, grid_index: int, n: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draws points from a VEGAS grid on the device of the integrator.

        Returns:
            tuple of the random numbers with shape (n, input_dim), the mapped points in the
            data type of the integrator and the Jacobians in double precision
        """
        self.grids.to(self.integrator.dummy.device)
        r = torch.rand(
            (n, self.input_dim), dtype=torch.float64, device=self.grids.device
        )
        x, jac = self.grids.map(grid_index, r)
        x, jac = self._map_discrete(x.to(self.integrator.dummy.dtype), jac)
        return r, x, jac

    def _random_channels(self, grid_channels: list[int], n: int) -> torch.Tensor:
        """Draws channels uniformly from the channels of a grid"""
        device = self.integrator.dummy.device
        choices = torch.tensor(grid_channels, device=device)
        return choices[torch.randint(len(grid_channels), (n,), device=device)]

    def _map_discrete(
        self, x: torch.Tensor, jac: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disc_dims = len(self.integrand.discrete_dims)
        disc_dims_vec = torch.tensor(self.integrand.discrete_dims, device=x.device)
        if disc_dims == 0:
            return x, jac
        map_disc = lambda x_disc: torch.clip(
            torch.floor(x_disc * disc_dims_vec),
            min=torch.tensor(0.0, device=x_disc.device),
            max=disc_dims_vec - 1,
        )
        if self.integrand.discrete_dims_position == "first":
            x_disc = x[:, :disc_dims]
            x_cont = x[:, disc_dims:]
            x_ret = torch.cat([map_disc(x_disc), x_cont], dim=1)
        else:
            x_disc = x[:, -disc_dims:]
            x_cont = x[:, :-disc_dims]
            x_ret = torch.cat([x_cont, map_disc(x_disc)], dim=1)
        return x_ret, jac * math.prod(self.integrand.discrete_dims)

    def train_step(self, samples_per_channel: int) -> VegasTrainingStatus:
        """
        Performs a single VEGAS training iteration

        Args:
            samples_per_channel: number of training samples per channel
        Returns:
            ``VegasTrainingStatus`` object containing metrics of the training progress
        """
        n_channels = (
            self.integrator.integration_channel_count
            if self.integrator.multichannel
            else 1
        )
        device = self.integrator.dummy.device
        dtype = self.integrator.dummy.dtype
        variances = torch.zeros(n_channels, dtype=torch.float64, device=device)
        counts = torch.zeros(n_channels, dtype=torch.float64, device=device)
        means = torch.zeros(n_channels, dtype=torch.float64, device=device)
        # TODO: maybe implement stratified training instead of uniform channels
        for grid_index, grid_channels in enumerate(self.grid_channels):
            r, x, jac = self._draw(grid_index, samples_per_channel)

            valid_channels = None
            if (
                self.integrator.group_channels
                and not self.integrator.group_channels_uniform
            ):
                # the channel within the group is given by a discrete dimension
                channels, valid_channels = self.integrator._learned_group_channels(
                    x, torch.full((samples_per_channel,), grid_index, device=device)
                )
            elif self.integrator.multichannel:
                # TODO: no need for random numbers here
                channels = self._random_channels(grid_channels, samples_per_channel)
            else:
                channels = None

            func_vals, y, alphas = self.integrand(x, channels)
            self.integrator._statistics.integrand_calls += len(func_vals)
            func_vals, alphas = self.integrator._check_integrand_output(
                x, func_vals, alphas
            )
            if valid_channels is not None:
                func_vals = torch.where(
                    valid_channels, func_vals, torch.zeros_like(func_vals)
                )

            f = jac * func_vals
            if self.integrator.multichannel:
                f = f * self._channel_weight(func_vals, channels, alphas)
            if self.integrator.drop_zero_integrands:
                mask = f != 0.0
                zero_count = torch.count_nonzero(~mask)
                if self.integrator.group_channels:
                    zero_counts = torch.zeros(
                        (self.integrator.integration_channel_count,),
                        dtype=torch.int64,
                        device=device,
                    )
                    zero_counts[grid_index] = zero_count
                elif self.integrator.multichannel:
                    zero_counts = torch.bincount(
                        channels[~mask], minlength=self.integrand.channel_count
                    )
                else:
                    zero_counts = zero_count[None]
                if self.integrator.multichannel:
                    channels = channels[mask]
                    alphas = None if alphas is None else alphas[mask]
                f = f[mask]
                r = r[mask]
                jac = jac[mask]
                x = x[mask]
                y = None if y is None else y[mask]
                func_vals = func_vals[mask]
            else:
                zero_counts = None
            self.grids.add_training_data(grid_index, r, f.square())
            self.grids.adapt(grid_index, self.damping)

            if self.integrator.group_channels:
                integration_channels = torch.full_like(channels, grid_index)
                f_chan = (
                    f * len(grid_channels)
                    if self.integrator.group_channels_uniform
                    else f
                )
                counts_chan = len(f_chan) + (
                    0 if zero_counts is None else zero_counts[grid_index]
                )
                means[grid_index] = f_chan.sum() / counts_chan
                counts[grid_index] = counts_chan
                variances[grid_index] = (
                    f_chan - means[grid_index]
                ).square().sum() / counts_chan
            else:
                integration_channels = None
                # one grid can contain several channels if they are grouped, but integrated
                # separately (channel_grouping_mode="none")
                index = (
                    channels
                    if self.integrator.multichannel
                    else torch.zeros_like(f, dtype=torch.int64)
                )
                counts_all = torch.bincount(index, minlength=n_channels).double()
                if zero_counts is not None:
                    counts_all += zero_counts
                means_all = (
                    torch.bincount(index, weights=f, minlength=n_channels) / counts_all
                )
                variances_all = (
                    torch.bincount(
                        index,
                        weights=(f - means_all[index]).square(),
                        minlength=n_channels,
                    )
                    / counts_all
                )
                chans = torch.tensor(grid_channels, device=device)
                counts[chans] = counts_all[chans]
                means[chans] = means_all[chans]
                variances[chans] = variances_all[chans]

            stored_samples = SampleBatch(
                x,
                y,
                (1 / jac).to(dtype),
                func_vals,
                channels,
                alphas,
                integration_channels=integration_channels,
            )
            if zero_counts is not None:
                # same rescaling of the sampling probabilities as for the online samples
                stored_samples.zero_counts = zero_counts
                stored_samples = self.integrator._rescale_dropped(stored_samples)
            self.integrator._store_samples(stored_samples)
        self.integrator.integration_history.store(
            means[None], variances[None], counts[None]
        )
        status = VegasTrainingStatus(
            step=self.step,
            variance=torch.sqrt(torch.nansum(variances / counts) * counts.sum()).item()
            / means.sum().item(),
        )
        self.step += 1
        return status

    def train(
        self,
        samples_per_channel: list[int],
        callback: Callable[[VegasTrainingStatus], None] | None = None,
        capture_keyboard_interrupt: bool = False,
    ):
        """
        Performs multiple training steps

        Args:
            samples_per_channel: list of the number of samples per channel, with one entry for every
                training iteration
            callback: function that is called after each training step with the training status
                as argument
            capture_keyboard_interrupt: If True, a keyboard interrupt does not raise an exception.
                Instead, the current training step is finished and the training is aborted
                afterwards.
        """
        interrupted = False
        if capture_keyboard_interrupt:

            def handler(sig, frame):
                nonlocal interrupted
                interrupted = True

            old_handler = signal.signal(signal.SIGINT, handler)

        try:
            for sample_count in samples_per_channel:
                status = self.train_step(sample_count)
                if callback is not None:
                    callback(status)
                if interrupted:
                    break
        finally:
            if capture_keyboard_interrupt:
                signal.signal(signal.SIGINT, old_handler)

    def initialize_integrator(self):
        """
        Initializes the flows in the integrator object using the trained VEGAS grid
        """
        grids = self.grids.grid.to(
            device=self.integrator.dummy.device, dtype=self.integrator.dummy.dtype
        )
        if not self.integrator.multichannel:
            grids = grids[0]
        disc_dims = len(self.integrand.discrete_dims)
        if disc_dims > 0:
            if self.integrand.discrete_dims_position == "first":
                grids = grids[..., disc_dims:, :]
            else:
                grids = grids[..., :-disc_dims, :]
            self.integrator.flow.continuous_flow.init_with_grid(grids)
        else:
            self.integrator.flow.init_with_grid(grids)

    def sample(
        self,
        n: int,
        batch_size: int = 100000,
        channel_weight_mode: Literal["uniform", "mean", "variance"] = "variance",
        channel: int | None = None,
        evaluate_integrand: bool = True,
    ) -> SampleBatch:
        """
        Draws samples and computes their integration weight

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
            channel_weight_mode: specifies whether the channels are weighted by their mean,
                variance or uniformly. Note that weighting by mean can lead to problems for
                non-positive functions
            channel: if different from None, samples are only generated for this channel
            evaluate_integrand: If False, the integrand is not evaluated and func_vals, y and alphas are set to dummy values. This can be used when integrand
                is expensive and one only needs the mapping jacobian.
        Returns:
            ``SampleBatch`` object, see its documentation for details
        """
        device = self.integrator.dummy.device
        dtype = self.integrator.dummy.dtype
        if channel is None and self.integrator.multichannel:
            if channel_weight_mode == "uniform":
                uniform_channel_ratio = 1.0
                channel_weight_mode = "variance"
            else:
                uniform_channel_ratio = 0.0
            channel_weights = self.integrator._get_channel_contributions(
                False, channel_weight_mode
            )
            samples_per_channel = self.integrator._get_channels(
                n, channel_weights, uniform_channel_ratio
            )
        else:
            samples_per_channel = torch.zeros(
                (len(self.grid_channels),), dtype=torch.int64, device="cpu"
            )
            samples_per_channel[channel] = n
        # the sample counts determine tensor shapes, so move them to the CPU once
        samples_per_channel = samples_per_channel.cpu()

        samples = []
        for grid_index, grid_channels in enumerate(self.grid_channels):
            n_samples = int(
                samples_per_channel[grid_index]
                if self.integrator.group_channels
                else samples_per_channel[grid_channels].sum()
            )
            if n_samples == 0:
                continue
            _, x_torch, jac = self._draw(grid_index, n_samples)
            integration_channels = None
            valid_channels = None
            if (
                self.integrator.group_channels
                and not self.integrator.group_channels_uniform
            ):
                # the channel within the group is given by a discrete dimension
                integration_channels = torch.full(
                    (n_samples,), grid_index, device=device
                )
                channels, valid_channels = self.integrator._learned_group_channels(
                    x_torch, integration_channels
                )
            elif self.integrator.group_channels:
                channels = self._random_channels(grid_channels, n_samples)
                integration_channels = torch.full(
                    (n_samples,), grid_index, device=device
                )
            elif self.integrator.multichannel:
                channels = torch.cat(
                    [
                        torch.full(
                            (int(samples_per_channel[channel_index]),),
                            channel_index,
                            device=device,
                        )
                        for channel_index in grid_channels
                    ]
                )
            else:
                channels = None

            if evaluate_integrand:
                func_vals, y, alphas = self.integrand(x_torch, channels)
                self.integrator._statistics.integrand_calls += len(func_vals)
                func_vals, alphas = self.integrator._check_integrand_output(
                    x_torch, func_vals, alphas
                )
                if valid_channels is not None:
                    func_vals = torch.where(
                        valid_channels, func_vals, torch.zeros_like(func_vals)
                    )
            else:
                func_vals = torch.zeros((n_samples,), dtype=dtype, device=device)
                y = None
                alphas = None

            # TODO: assign channels if importance-sampled
            if (
                self.integrator.group_channels
                and self.integrator.group_channels_uniform
            ):
                func_vals *= len(grid_channels)

            f = jac * func_vals
            if self.integrator.multichannel:
                f = f * self._channel_weight(func_vals, channels, alphas)

            samples.append(
                SampleBatch(
                    x_torch,
                    y,
                    (1 / jac).to(dtype),
                    func_vals,
                    channels,
                    alphas,
                    None,
                    integration_channels,
                    f.to(dtype),
                    alphas,
                )
            )
        cat_samples = SampleBatch.cat(samples)
        if self.integrator.multichannel:
            # the samples are generated per integration channel (channel group)
            channels = self.integrator._batch_channels(cat_samples)
            norm_factors = len(channels) / torch.bincount(
                channels, minlength=self.integrator.integration_channel_count
            )
            cat_samples.weights *= norm_factors[channels]
        return cat_samples

    def _compute_integral(
        self, samples: SampleBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.integrator.multichannel:
            n_channels = self.integrator.integration_channel_count
            alphas = self._channel_weight(
                samples.func_vals, samples.channels, samples.alphas
            )
            f_true = alphas * samples.func_vals
            f_div_q = f_true.detach() / samples.q_sample
            channels = (
                samples.channels
                if samples.integration_channels is None
                else samples.integration_channels
            )
            counts = torch.bincount(channels, minlength=n_channels)
            means = torch.bincount(
                channels,
                weights=f_div_q,
                minlength=n_channels,
            ) / counts.clip(min=1)
            variances = (
                torch.bincount(
                    channels,
                    weights=(f_div_q - means[channels]).square(),
                    minlength=n_channels,
                )
                / counts
            )
        else:
            means = samples.weights.mean(dim=0, keepdim=True)
            counts = torch.full((1,), samples.weights.shape[0], device=means.device)
            variances = samples.weights.var(dim=0, keepdim=True)
        return means, variances, counts

    def integrate(self, n: int) -> tuple[float, float]:
        """
        Draws samples and computes the integral.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
        Returns:
            tuple with the value of the integral and the MC integration error
        """
        means, variances, counts = self._compute_integral(self.sample(n))
        self.integrator.integration_history.store(
            means[None], variances[None], counts[None]
        )
        integral = means.sum().item()
        error = torch.nansum(variances / counts).sqrt().item()
        return integral, error

    def integration_metrics(
        self, n: int, batch_size: int = 100000
    ) -> IntegrationMetrics:
        """
        Draws samples and computes metrics for the total and channel-wise integration quality.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
        Returns:
            ``IntegrationMetrics`` object, see its documentation for details
        """
        samples = self.sample(n)
        means, variances, counts = self._compute_integral(samples)
        self.integrator.integration_history.store(
            means[None], variances[None], counts[None]
        )
        return integration_metrics(means, variances, counts)

    def unweighting_metrics(
        self,
        n: int,
        batch_size: int = 100000,
        channel_weight_mode: Literal["uniform", "mean", "variance"] = "mean",
        max_overweight_truncation: float = 0.01,
    ) -> UnweightingMetrics | tuple[UnweightingMetrics, list[UnweightingMetrics]]:
        """
        Draws samples and computes metrics for the total and channel-wise integration quality.
        This function is only suitable for functions that are non-negative everywhere.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
            channel_weight_mode: specifies whether the channels are weighted by their mean,
                variance or uniformly.
            max_overweight_truncation: tolerated fraction of over-weight events, used to
                determine the maximum weight as in MadGraph7
        Returns:
            ``UnweightingMetrics`` object, see its documentation for details. In the
            multi-channel case, a list of channel-wise ``UnweightingMetrics`` is returned in
            addition.
        """
        samples = self.sample(n, batch_size, channel_weight_mode)
        return unweighting_metrics(
            samples.weights,
            samples.channels,
            self.integrand.channel_count,
            max_overweight_truncation,
        )
