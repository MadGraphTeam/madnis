from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import Distribution
from .masked_mlp import MaskedMLP


class DiscreteMADE(nn.Module, Distribution):
    """
    Autoregressive distribution for discrete dimensions based on a masked autoencoder (MADE,
    1502.03509). The probabilities of each discrete dimension are predicted by a masked MLP,
    conditional on the previous dimensions and an optional condition. Both conditional and
    non-conditional as well as multi-channel distributions are supported.
    """

    def __init__(
        self,
        dims_in: list[int],
        dims_c: int = 0,
        channels: int | None = None,
        prior_prob_function: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        channel_remap_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **mlp_kwargs,  # TODO: replace this with default arguments for MLP
    ):
        """
        Args:
            dims_in: number of options for each discrete dimension
            dims_c: condition dimension
            channels: If None, build single-channel distribution. If integer, build
                multi-channel distribution with this number of channels.
            prior_prob_function: function that gives prior probabilities, which are multiplied
                with the learned probabilities. It is called with the already sampled discrete
                dimensions, shape (n, i), and the index i of the current dimension, and must return
                unnormalized probabilities with shape (n, dims_in[i]). If None, no prior is used.
            channel_remap_function: function that maps the channel indices passed to the
                distribution to the indices of its channels. It is applied to channel index
                tensors and integers, but not to lists of channel sizes. If None, the channel
                indices are used directly.
            mlp_kwargs: additional keyword arguments passed to the ``MaskedMLP`` constructor
        """
        super().__init__()

        self.autoreg_net = MaskedMLP(
            input_dims=[dims_c, *dims_in[:-1]],
            output_dims=dims_in,
            channels=channels,
            **mlp_kwargs,
        )
        self.dims_in = dims_in
        self.max_dim = max(dims_in)
        self.channels = 1 if channels is None else channels
        self.prior_prob_function = prior_prob_function
        self.channel_remap_function = channel_remap_function

        discrete_indices = []
        one_hot_mask = []
        for i, dim in enumerate(dims_in):
            discrete_indices.extend([i] * dim)
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim - dim))
        self.register_buffer("discrete_indices", torch.tensor(discrete_indices))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask))
        self.register_buffer("dims_in_tensor", torch.tensor(dims_in))
        self.register_buffer("dummy", torch.tensor([0.0]))

    def _sort_channels(
        self,
        channel: torch.Tensor | list[int] | int | None,
        tensors: Iterable[torch.Tensor | None, ...],
    ) -> tuple[
        list[int] | int | None, torch.Tensor | None, tuple[torch.Tensor | None, ...]
    ]:
        if isinstance(channel, torch.Tensor):
            if self.channel_remap_function is not None:
                channel = self.channel_remap_function(channel)
            channel_perm = torch.argsort(channel)
            channel_sizes = channel.bincount(minlength=self.channels).tolist()
            perm_tensors = tuple(
                tensor[channel_perm] if tensor is not None else None
                for tensor in tensors
            )
            return channel_sizes, channel_perm, perm_tensors

        if self.channel_remap_function is not None:
            if isinstance(channel, int):
                channel = self.channel_remap_function(channel)
            elif isinstance(channel, list):
                raise ValueError(
                    "channel_remap_function not supported if called "
                    "with list of channel sizes"
                )
        return channel, None, tensors

    def _unsort_channels(
        self,
        channel_perm: torch.Tensor | None,
        tensors: Iterable[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        if channel_perm is None:
            return tensors
        channel_perm_inv = torch.argsort(channel_perm)
        return tuple(
            tensor[channel_perm_inv] if tensor is not None else None
            for tensor in tensors
        )

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        """
        Computes the log-probabilities of the input data.

        Args:
            x: integer input data, shape (n, len(dims_in))
            c: condition, shape (n, dims_c) or None for an unconditional flow
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
        Returns:
            log-probabilities with shape (n, )
        """
        if self.prior_prob_function is not None:
            prior_probs = torch.cat(
                [
                    self.prior_prob_function(x[:, :i], i)
                    for i, _ in enumerate(self.dims_in)
                ],
                dim=1,
            )
        else:
            prior_probs = None

        x_one_hot = (
            F.one_hot(x, self.max_dim)
            .to(self.dummy.dtype)
            .flatten(start_dim=1)[:, self.one_hot_mask]
        )

        channel, channel_perm, (x_one_hot, c, prior_probs) = self._sort_channels(
            channel, (x_one_hot, c, prior_probs)
        )

        net_input = x_one_hot if c is None else torch.cat((c, x_one_hot), dim=1)
        net_prob = self.autoreg_net(net_input[:, : -self.dims_in[-1]], channel).exp()
        unnorm_prob = (
            net_prob if self.prior_prob_function is None else net_prob * prior_probs
        )

        prob_norms = torch.zeros_like(x, dtype=x_one_hot.dtype).scatter_add_(
            dim=1,
            index=self.discrete_indices[None, :].expand(x.shape[0], -1),
            src=unnorm_prob,
        )
        prob_sums = torch.zeros_like(prob_norms).scatter_add_(
            dim=1,
            index=self.discrete_indices[None, :].expand(x.shape[0], -1),
            src=unnorm_prob * x_one_hot,
        )
        prob = torch.prod(prob_sums / prob_norms, dim=1)

        return self._unsort_channels(channel_perm, (prob.log(),))[0]

    def sample(
        self,
        n: int | None = None,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Draws samples following the distribution

        Args:
            n: number of samples
            c: condition, shape (n, dims_c) or None for an unconditional flow
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
            return_log_prob: if True, also return the log-probabilities
            return_prob: if True, also return the probabilities
            device: device of the returned tensor. Only required if no condition is given.
            dtype: dtype of the returned tensor. Only required if no condition is given.
        Returns:
            integer samples with shape (n, len(dims_in)). Depending on the arguments ``return_log_prob``,
            ``return_prob``, this function should also return the log-probabilities with shape (n, ),
            the probabilities with shape (n, ).
        """
        if n is None:
            n = len(c)
        if c is None:
            options = {"device": device, "dtype": dtype}
            x_in = torch.zeros((n, 0), **options)
        else:
            options = {"device": c.device, "dtype": c.dtype}
            x_in = c

        prob = torch.ones((n,), **options)
        net_cache = None

        channel, channel_perm, (x_in,) = self._sort_channels(channel, (x_in,))

        x_out = torch.zeros(
            (n, len(self.dims_in)),
            device=options["device"],
            dtype=torch.int64,
        )
        for i, dim in enumerate(self.dims_in):
            y, net_cache = self.autoreg_net.forward_cached(x_in, i, net_cache, channel)
            net_probs = y.exp()
            if self.prior_prob_function is None:
                unnorm_probs = net_probs
            else:
                prior_probs = self.prior_prob_function(x_out[:, :i], i)
                unnorm_probs = net_probs * prior_probs
            cdf = unnorm_probs.cumsum(dim=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            r = torch.rand((y.shape[0], 1), **options)
            samples = torch.clip(torch.searchsorted(cdf, r)[:, 0], max=cdf.shape[1] - 1)
            x_out[:, i] = samples
            prob = prob * torch.gather(unnorm_probs, 1, samples[:, None])[:, 0] / norm
            x_in = F.one_hot(samples, dim).to(y.dtype)

        return_list = [x_out]
        if return_log_prob:
            return_list.append(prob.log())
        if return_prob:
            return_list.append(prob)
        return_list = self._unsort_channels(channel_perm, return_list)
        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]
