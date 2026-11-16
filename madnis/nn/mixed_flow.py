from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discrete_made import DiscreteMADE
from .discrete_transformer import DiscreteTransformer
from .flow import Distribution, Flow


class MixedFlow(nn.Module, Distribution):
    """
    Distribution for mixed continuous and discrete dimensions. It combines a normalizing flow
    (``Flow``) for the continuous dimensions with an autoregressive discrete distribution
    (``DiscreteMADE`` or ``DiscreteTransformer``) for the discrete dimensions. Depending on their
    position, either the discrete dimensions are sampled first and the flow is conditioned on
    them (one-hot encoded), or the other way around.
    """

    def __init__(
        self,
        dims_in_continuous: int,
        dims_in_discrete: list[int],
        dims_c: int = 0,
        discrete_dims_position: Literal["first", "last"] = "first",
        discrete_model: Literal["made", "transformer"] = "made",
        channels: int | None = None,
        continuous_kwargs: dict[str, Any] = {},
        discrete_kwargs: dict[str, Any] = {},
    ):
        """
        Args:
            dims_in_continuous: number of continuous dimensions
            dims_in_discrete: number of options for each discrete dimension
            dims_c: condition dimension
            discrete_dims_position: If "first", the discrete dimensions are the first dimensions
                of the input and the continuous flow is conditioned on them. If "last", they are
                the last dimensions and the discrete distribution is conditioned on the continuous
                dimensions.
            discrete_model: model used for the discrete dimensions, "made" for ``DiscreteMADE``
                or "transformer" for ``DiscreteTransformer`` (single-channel only)
            channels: If None, build single-channel distribution. If integer, build
                multi-channel distribution with this number of channels.
            continuous_kwargs: additional keyword arguments passed to the ``Flow`` constructor
            discrete_kwargs: additional keyword arguments passed to the ``DiscreteMADE`` or
                ``DiscreteTransformer`` constructor
        """
        super().__init__()
        if discrete_dims_position == "first":
            self.discrete_dims_first = True
        elif discrete_dims_position == "last":
            self.discrete_dims_first = False
        else:
            raise ValueError("discrete_dims_position must be 'first' or 'last'")
        self.dims_in_continuous = dims_in_continuous
        self.dims_in_discrete = len(dims_in_discrete)
        if self.discrete_dims_first:
            dims_c_discrete = dims_c
            dims_c_continuous = sum(dims_in_discrete) + dims_c
        else:
            dims_c_discrete = dims_c + dims_in_continuous
            dims_c_continuous = dims_c

        if discrete_model == "made":
            self.discrete_flow = DiscreteMADE(
                dims_in=dims_in_discrete,
                dims_c=dims_c_discrete,
                channels=channels,
                **discrete_kwargs,
            )
        elif discrete_model == "transformer":
            if channels is not None:
                raise NotImplementedError(
                    "DiscreteTransformer only supported for single-channel integration"
                )
            self.discrete_flow = DiscreteTransformer(
                dims_in=dims_in_discrete,
                dims_c=dims_c_discrete,
                **discrete_kwargs,
            )
        else:
            raise ValueError("discrete_model must be 'made' or 'transformer'")

        self.continuous_flow = Flow(
            dims_in=dims_in_continuous,
            dims_c=dims_c_continuous,
            channels=channels,
            **continuous_kwargs,
        )

        self.max_dim_discrete = max(dims_in_discrete)
        one_hot_mask = []
        for dim in dims_in_discrete:
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim_discrete - dim))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask))
        self.register_buffer("dummy", torch.tensor([0.0]))

    def encode_discrete(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Computes the concatenated one-hot encoding of the discrete dimensions.

        Args:
            indices: integer tensor with the discrete dimensions, shape (n, len(dims_in_discrete))
        Returns:
            one-hot encoding with shape (n, sum(dims_in_discrete))
        """
        return (
            F.one_hot(indices, self.max_dim_discrete)
            .to(self.dummy.dtype)
            .flatten(start_dim=1)[:, self.one_hot_mask]
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
            x: input data, shape (n, dims_in)
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
        if self.discrete_dims_first:
            x_discrete = x[:, : self.dims_in_discrete].long()
            log_prob_discrete = self.discrete_flow.log_prob(
                x_discrete, c=c, channel=channel
            )
            condition = self.encode_discrete(x_discrete)
            if c is not None:
                condition = torch.cat((c, condition), dim=1)
            log_prob_continuous = self.continuous_flow.log_prob(
                x[:, self.dims_in_discrete :], c=condition, channel=channel
            )
        else:
            x_continuous = x[:, : self.dims_in_continuous]
            log_prob_continuous = self.continuous_flow.log_prob(
                x_continuous, c=c, channel=channel
            )
            condition = (
                x_continuous if c is None else torch.cat((c, x_continuous), dim=1)
            )
            log_prob_discrete = self.discrete_flow.log_prob(
                x[:, self.dims_in_continuous :].long(), c=condition, channel=channel
            )
        return log_prob_discrete + log_prob_continuous

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
            samples with shape (n, dims_in). Depending on the arguments ``return_log_prob``,
            ``return_prob``, this function should also return the log-probabilities with shape (n, ),
            the probabilities with shape (n, ).
        """
        if self.discrete_dims_first:
            x_discrete, prob_discrete = self.discrete_flow.sample(
                n=n,
                c=c,
                channel=channel,
                return_prob=True,
                device=device,
                dtype=dtype,
            )
            condition = self.encode_discrete(x_discrete)
            if c is not None:
                condition = torch.cat((c, condition), dim=1)
            x_continuous, log_prob_continuous = self.continuous_flow.sample(
                c=condition,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            x = torch.cat((x_discrete.to(x_continuous.dtype), x_continuous), dim=1)
        else:
            x_continuous, log_prob_continuous = self.continuous_flow.sample(
                n=n,
                c=c,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            condition = (
                x_continuous if c is None else torch.cat((c, x_continuous), dim=1)
            )
            x_discrete, prob_discrete = self.discrete_flow.sample(
                c=condition,
                channel=channel,
                return_prob=True,
                device=device,
                dtype=dtype,
            )
            x = torch.cat((x_continuous, x_discrete.to(x_continuous.dtype)), dim=1)

        extra_returns = []
        if return_log_prob:
            extra_returns.append(prob_discrete.log() + log_prob_continuous)
        if return_prob:
            extra_returns.append(prob_discrete * log_prob_continuous.exp())
        if len(extra_returns) > 0:
            return x, *extra_returns
        else:
            return x
