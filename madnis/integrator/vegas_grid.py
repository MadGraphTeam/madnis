import torch

# smallest density used by the vegas package, to avoid empty increments
TINY = 1e-257


class VegasGrid:
    """
    Adaptive VEGAS grids for one or several channels, implemented in torch, such that they run on
    any device. The mapping, the accumulation of training data and the grid adaptation follow the
    ``AdaptiveMap`` class of the vegas package.
    """

    def __init__(
        self,
        grid_count: int,
        dims: int,
        bins: int,
        device: torch.device | None = None,
    ):
        """
        Args:
            grid_count: number of independent grids, for example one per channel
            dims: number of dimensions
            bins: number of increments per dimension
            device: device of the grids and the training data
        """
        self.bins = bins
        self.grid = (
            torch.linspace(0, 1, bins + 1, dtype=torch.float64, device=device)
            .expand(grid_count, dims, bins + 1)
            .clone()
        )
        self.clear()

    @property
    def device(self) -> torch.device:
        return self.grid.device

    def to(self, device: torch.device) -> "VegasGrid":
        """
        Moves the grids and the accumulated training data to the given device
        """
        self.grid = self.grid.to(device)
        self.sum_f = self.sum_f.to(device)
        self.n_f = self.n_f.to(device)
        return self

    def clear(self):
        """
        Clears the accumulated training data. As in the vegas package, the counts start at TINY.
        """
        self.sum_f = torch.zeros(
            self.grid.shape[:2] + (self.bins,), dtype=torch.float64, device=self.device
        )
        self.n_f = torch.full_like(self.sum_f, TINY)

    def _bin_index(self, r: torch.Tensor) -> torch.Tensor:
        return (r * self.bins).floor().long().clamp(0, self.bins - 1)

    def map(
        self, grid_index: int, r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps points from the unit hypercube to the grid.

        Args:
            grid_index: index of the grid
            r: points in the unit hypercube, shape (n, dims)
        Returns:
            tuple of the mapped points with shape (n, dims) and the Jacobians with shape (n, )
        """
        grid = self.grid[grid_index]
        index = self._bin_index(r)
        dim_index = torch.arange(grid.shape[0], device=self.device)
        left = grid[dim_index, index]
        width = grid[dim_index, index + 1] - left
        x = left + (r * self.bins - index) * width
        return x, (width * self.bins).prod(dim=1)

    def add_training_data(self, grid_index: int, r: torch.Tensor, f: torch.Tensor):
        """
        Accumulates the training data per increment, ignoring points on the boundary as in the
        vegas package.

        Args:
            grid_index: index of the grid
            r: points in the unit hypercube, shape (n, dims)
            f: training data, usually the squared integration weights, shape (n, )
        """
        dims = r.shape[1]
        valid = ((r > 0) & (r < 1)).to(torch.float64)
        index = self._bin_index(r) + self.bins * torch.arange(dims, device=self.device)
        self.sum_f[grid_index].view(-1).index_add_(
            0, index.flatten(), (f.abs().to(torch.float64)[:, None] * valid).flatten()
        )
        self.n_f[grid_index].view(-1).index_add_(0, index.flatten(), valid.flatten())

    def adapt(self, grid_index: int, alpha: float):
        """
        Adapts the grid to the accumulated training data, with the smoothing, damping and
        redistribution of the vegas package, and clears the training data.

        Args:
            grid_index: index of the grid
            alpha: damping parameter. Larger values lead to faster adaptation, zero leaves the
                grid unchanged, and negative values use the training data without smoothing and
                damping.
        """
        bins = self.bins
        if bins == 1:
            self.clear()
            return
        grid = self.grid[grid_index]
        if alpha == 0:
            avg_f = torch.ones_like(self.sum_f[grid_index])
        else:
            avg_f = self.sum_f[grid_index] / self.n_f[grid_index]
            if alpha > 0:
                smooth_f = torch.empty_like(avg_f)
                smooth_f[:, 0] = (7 * avg_f[:, 0] + avg_f[:, 1]).abs() / 8
                smooth_f[:, -1] = (7 * avg_f[:, -1] + avg_f[:, -2]).abs() / 8
                smooth_f[:, 1:-1] = (
                    6 * avg_f[:, 1:-1] + avg_f[:, :-2] + avg_f[:, 2:]
                ).abs() / 8
                sum_f = smooth_f.sum(dim=1, keepdim=True)
                avg_f = torch.where(
                    sum_f > 0,
                    smooth_f / torch.where(sum_f > 0, sum_f, 1.0) + TINY,
                    TINY,
                )
                damped = (avg_f > 0) & (avg_f <= 0.99999999)
                avg_f = torch.where(
                    damped,
                    (-(1 - avg_f) / torch.where(damped, avg_f, 0.5).log()) ** alpha,
                    avg_f,
                )

        # without any training data in a dimension, the grid stays unchanged there
        avg_f = torch.where(avg_f.sum(dim=1, keepdim=True) > 0, avg_f, 1.0)
        # redistribute the increments such that each contains the same amount of avg_f
        cum_f = avg_f.cumsum(dim=1)
        targets = cum_f[:, -1:] / bins * torch.arange(1, bins, device=self.device)
        j = torch.searchsorted(cum_f, targets).clamp(max=bins - 1)
        widths = grid[:, 1:] - grid[:, :-1]
        inner = grid.gather(1, j + 1) - (cum_f.gather(1, j) - targets) / avg_f.gather(
            1, j
        ) * widths.gather(1, j)
        self.grid[grid_index] = torch.cat((grid[:, :1], inner, grid[:, -1:]), dim=1)
        self.sum_f[grid_index] = 0.0
        self.n_f[grid_index] = TINY
