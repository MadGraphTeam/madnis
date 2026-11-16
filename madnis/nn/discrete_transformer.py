from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import Distribution
from .mlp import MLP


class DiscreteTransformer(nn.Module, Distribution):
    """
    Autoregressive distribution for discrete dimensions based on a transformer encoder with a
    causal mask. Every discrete dimension is embedded as a token, and the probabilities of each
    dimension are predicted conditional on the previous dimensions and an optional condition. Only
    single-channel distributions are supported.
    """

    def __init__(
        self,
        dims_in: list[int],
        dims_c: int = 0,
        prior_prob_function: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        embedding_dim: int = 64,
        feedforward_dim: int = 256,
        heads: int = 8,
        transformer_layers: int = 3,
        mlp_layers: int = 3,
        mlp_units: int = 256,
        mlp_activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        """
        Args:
            dims_in: number of options for each discrete dimension
            dims_c: condition dimension
            prior_prob_function: function that gives prior probabilities, which are multiplied
                with the learned probabilities. It is called with the already sampled discrete
                dimensions, shape (n, i), and the index i of the current dimension, and must return
                unnormalized probabilities with shape (n, dims_in[i]). If None, no prior is used.
            embedding_dim: dimension of the token embeddings
            feedforward_dim: dimension of the feed-forward networks in the transformer layers
            heads: number of attention heads
            transformer_layers: number of transformer encoder layers
            mlp_layers: number of layers of the MLP that maps the transformer output to the
                log-probabilities
            mlp_units: number of hidden nodes of the output MLP
            mlp_activation: function that builds a nn.Module used as activation function of the
                output MLP
        """
        super().__init__()
        self.dims_in = dims_in
        self.prior_prob_function = prior_prob_function

        self.max_dim = max(dims_in)
        n_dims = len(dims_in)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=0,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )
        self.mlp = MLP(
            features_in=embedding_dim,
            features_out=self.max_dim,
            layers=mlp_layers,
            units=mlp_units,
            activation=mlp_activation,
        )
        self.x_embedding = nn.Embedding(
            num_embeddings=self.max_dim, embedding_dim=embedding_dim
        )
        self.c_embedding = nn.Linear(dims_c, embedding_dim) if dims_c > 0 else None
        self.pos_embedding = nn.Embedding(
            num_embeddings=n_dims, embedding_dim=embedding_dim
        )

        self.register_buffer(
            "causal_mask",
            torch.ones((n_dims, n_dims), dtype=torch.bool).triu(diagonal=1),
        )
        prior_mask = torch.zeros((n_dims, self.max_dim))
        for i, n_opts in enumerate(self.dims_in):
            prior_mask[i, :n_opts] = 1
        self.register_buffer("prior_mask", prior_mask)

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
        x_embed = F.pad(self.x_embedding(x[:, :-1]), (0, 0, 1, 0))
        pos_embed = self.pos_embedding(
            torch.arange(len(self.dims_in), device=x.device)[None, :]
        )
        embedding = x_embed + pos_embed
        if c is not None:
            embedding = embedding + self.c_embedding(c)[:, None, :]
        net_log_probs = self.mlp(self.transformer(embedding, self.causal_mask))

        prior_probs = self.prior_mask.expand(x.shape[0], -1, -1).clone()
        if self.prior_prob_function is not None:
            for i, n_opts in enumerate(self.dims_in):
                prior_probs[:, i, :n_opts] = self.prior_prob_function(x[:, :i], i)

        probs_unnorm = net_log_probs.exp() * prior_probs
        prob = torch.prod(
            torch.gather(probs_unnorm, 2, x[:, :, None])[:, :, 0]
            / probs_unnorm.sum(dim=2),
            dim=1,
        )

        return prob.log()

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
        if c is not None:
            dtype = c.dtype
            device = c.device

        x = torch.zeros((n, 0), dtype=torch.int64, device=device)
        prob = torch.ones((n,), dtype=dtype, device=device)
        embedding = None
        if c is not None:
            c_embedding = self.c_embedding(c)
        for i, dim in enumerate(self.dims_in):
            next_embedding = (
                self.pos_embedding(torch.tensor([i], device=device))
                .expand(n, -1)
                .clone()
            )
            if i > 0:
                next_embedding += self.x_embedding(x[:, i - 1])
            if c is not None:
                next_embedding += c_embedding
            if embedding is None:
                embedding = next_embedding[:, None, :]
            else:
                embedding = torch.cat((embedding, next_embedding[:, None, :]), dim=1)
            net_log_probs = self.mlp(
                self.transformer(embedding, self.causal_mask[: i + 1, : i + 1])[:, -1]
            )

            unnorm_probs = net_log_probs.exp() * self.prior_mask[None, i, :]
            if self.prior_prob_function is not None:
                unnorm_probs[:, :dim] *= self.prior_prob_function(x[:, :i], i)
            cdf = unnorm_probs.cumsum(dim=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            r = torch.rand((n, 1), dtype=dtype, device=device)
            samples = torch.searchsorted(cdf, r)
            x = torch.cat((x, samples), dim=1)
            prob = prob * torch.gather(unnorm_probs, 1, samples)[:, 0] / norm

        return_list = [x]
        if return_log_prob:
            return_list.append(prob.log())
        if return_prob:
            return_list.append(prob)
        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]
