"""Toy integrands with cuts, shared by the integrator tests"""

import torch

from madnis.integrator import Integrand

# cut on x0 + x1, the accepted region is a triangle with area 0.5 * (2 - X_CUT)^2
X_CUT = 1.2


def gauss(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-((x[:, 0] - 0.8) ** 2 + (x[:, 1] - 0.8) ** 2) / 0.5)


def cut_function(x: torch.Tensor) -> torch.Tensor:
    """smooth function times a cut that is not aligned with the axes"""
    return gauss(x) * (x[:, 0] + x[:, 1] > X_CUT)


def cut_function_multichannel(x: torch.Tensor, channels: torch.Tensor):
    """
    same function with two channels. The prior channel weights favor channel 0 for large x0, so
    the two channels have different cut efficiencies once the flow adapts.
    """
    alpha_0 = (1 + x[:, 0]) / (2 + x[:, 0] + x[:, 1])
    return cut_function(x), torch.stack([alpha_0, 1 - alpha_0], dim=1)


def multichannel_integrand() -> Integrand:
    return Integrand(
        cut_function_multichannel,
        input_dim=2,
        channel_count=2,
        has_channel_weight_prior=True,
    )


def reference_integral(n: int = 4_000_000) -> float:
    gen = torch.Generator().manual_seed(1234)
    x = torch.rand((n, 2), generator=gen, dtype=torch.float64)
    return cut_function(x).mean().item()
