from collections.abc import Callable, Iterator

import torch
import torch.nn as nn


class Buffer(nn.Module):
    """
    Circular buffer for multiple tensors with different shapes. The class is a torch.nn.Module to
    allow for simple storage.
    """

    def __init__(
        self,
        capacity: int,
        shapes: list[tuple[int, ...]],
        persistent: bool = True,
        dtypes: list[torch.dtype | None] | None = None,
    ):
        """
        Args:
            capacity: maximum number of samples stored in the buffer
            shapes: shapes of the tensors to be stored, without batch dimension. If a shape is
                None, no tensor is stored at that position. This allows for simpler handling of
                optional stored fields.
            persistent: if True, the content of the buffer is part of the module's state_dict
            dtypes: if different from None, specifies the tensors which have a non-standard dtype
        """
        super().__init__()
        self.keys = []
        if dtypes is None:
            dtypes = [None] * len(shapes)
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            key = f"buffer{i}"
            self.register_buffer(
                key,
                None if shape is None else torch.zeros((capacity, *shape), dtype=dtype),
                persistent,
            )
            self.keys.append(key)
        self.capacity = capacity
        self.register_buffer(
            "_size", torch.zeros((), dtype=torch.int64), persistent=persistent
        )
        self.register_buffer(
            "_store_index", torch.zeros((), dtype=torch.int64), persistent=persistent
        )

    @property
    def size(self):
        return int(self._size.item())

    @size.setter
    def size(self, value):
        self._size.fill_(int(value))

    @property
    def store_index(self):
        return int(self._store_index.item())

    @store_index.setter
    def store_index(self, value):
        self._store_index.fill_(int(value))

    def _batch_slices(self, batch_size: int) -> Iterator[slice]:
        """
        Returns slices that split up the buffer into batches of at most ``batch_size``, respecting
        the buffer size and periodic boundary.
        """
        start = self.store_index
        while start < self.size:
            stop = min(start + batch_size, self.size)
            yield slice(start, stop)
            start = stop
        start = 0
        while start < self.store_index:
            stop = min(start + batch_size, self.store_index)
            yield slice(start, stop)
            start = stop

    def _buffer_fields(self) -> Iterator[torch.Tensor | None]:
        """
        Iterates over the buffered tensors, without removing the padding if the buffer is not full.

        Returns:
            The buffered tensors, or None if a tensor was initialized with shape None
        """
        for key in self.keys:
            yield getattr(self, key)

    def __iter__(self) -> Iterator[torch.Tensor | None]:
        """
        Iterates over the buffered tensors

        Returns:
            The buffered tensors, or None if a tensor was initialized with shape None
        """
        for key in self.keys:
            buffer = getattr(self, key)
            yield None if buffer is None else buffer[: self.size]

    def store(self, *tensors: torch.Tensor | None):
        """
        Adds the given tensors to the buffer. If the buffer is full, the oldest stored samples are
        overwritten.

        Args:
            tensors: samples to be stored. The shapes of the tensors after the batch dimension must
                match the shapes given during initialization. The argument can be None if the
                corresponding shape was None during initialization.
        """
        store_slice1 = None
        for buffer, data in zip(self._buffer_fields(), tensors):
            if data is None:
                continue
            if store_slice1 is None:
                size = min(data.shape[0], self.capacity)
                end_index = self.store_index + size
                if end_index < self.capacity:
                    store_slice1 = slice(self.store_index, end_index)
                    store_slice2 = slice(0, 0)
                    load_slice1 = slice(0, size)
                    load_slice2 = slice(0, 0)
                else:
                    store_slice1 = slice(self.store_index, self.capacity)
                    store_slice2 = slice(0, end_index - self.capacity)
                    load_slice1 = slice(0, self.capacity - self.store_index)
                    load_slice2 = slice(self.capacity - self.store_index, size)
                self.store_index = end_index % self.capacity
                self.size = min(self.size + size, self.capacity)
            buffer[store_slice1] = data[load_slice1]
            buffer[store_slice2] = data[load_slice2]

    def filter(
        self,
        predicate: Callable[[tuple[torch.Tensor | None, ...]], torch.Tensor],
        batch_size: int = 100000,
    ):
        """
        Removes samples from the buffer that do not fulfill the criterion given by the predicate
        function.

        Args:
            predicate: function that returns a mask for a batch of samples, given a tuple with
                all the buffered fields as argument
            batch_size: maximal batch size to limit memory usage
        """
        masks = []
        masked_size = 0
        for batch_slice in self._batch_slices(batch_size):
            mask = predicate(
                tuple(
                    None if t is None else t[batch_slice] for t in self._buffer_fields()
                )
            )
            masked_size += torch.count_nonzero(mask)
            masks.append(mask)
        for buffer in self._buffer_fields():
            if buffer is None:
                continue
            buffer[:masked_size] = torch.cat(
                [
                    buffer[batch_slice][mask]
                    for batch_slice, mask in zip(self._batch_slices(batch_size), masks)
                ],
                dim=0,
            )
        self.size = masked_size
        self.store_index = masked_size % self.capacity

    def sample(self, count: int) -> list[torch.Tensor | None]:
        """
        Returns a batch of samples drawn from the buffer without replacement.

        Args:
            count: number of samples
        Returns:
            samples drawn from the buffer
        """
        weights = next(b for b in self._buffer_fields() if b is not None).new_ones(
            self.size
        )
        indices = torch.multinomial(weights, min(count, self.size), replacement=False)
        return [
            None if buffer is None else buffer[indices]
            for buffer in self._buffer_fields()
        ]


class ChannelBuffer(nn.Module):
    """
    Set of circular buffers, one for every channel, for multiple tensors with different shapes.
    Every channel has its own capacity, such that channels with a small contribution are not
    pushed out of the buffer by channels with many samples. The samples of all channels are
    stored in tensors with shape ``(channels, capacity, *shape)``, which allows for vectorized
    storing and sampling. The class is a torch.nn.Module to allow for simple storage.
    """

    def __init__(
        self,
        channels: int,
        capacity: int,
        shapes: list[tuple[int, ...] | None],
        persistent: bool = True,
        dtypes: list[torch.dtype | None] | None = None,
    ):
        """
        Args:
            channels: number of channels
            capacity: maximum number of samples stored for every channel
            shapes: shapes of the tensors to be stored, without batch dimension. If a shape is
                None, no tensor is stored at that position. This allows for simpler handling of
                optional stored fields.
            persistent: if True, the content of the buffer is part of the module's state_dict
            dtypes: if different from None, specifies the tensors which have a non-standard dtype
        """
        super().__init__()
        self.keys = []
        if dtypes is None:
            dtypes = [None] * len(shapes)
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            key = f"buffer{i}"
            self.register_buffer(
                key,
                (
                    None
                    if shape is None
                    else torch.zeros((channels, capacity, *shape), dtype=dtype)
                ),
                persistent,
            )
            self.keys.append(key)
        self.channels = channels
        self.capacity = capacity
        self.register_buffer(
            "sizes", torch.zeros((channels,), dtype=torch.int64), persistent=persistent
        )
        self.register_buffer(
            "store_indices",
            torch.zeros((channels,), dtype=torch.int64),
            persistent=persistent,
        )

    @property
    def size(self) -> int:
        """Total number of stored samples, summed over all channels"""
        return int(self.sizes.sum().item())

    def _buffer_fields(self) -> Iterator[torch.Tensor | None]:
        """
        Iterates over the buffered tensors, including the unused entries.

        Returns:
            The buffered tensors with shape (channels, capacity, ...), or None if a tensor was
            initialized with shape None
        """
        for key in self.keys:
            yield getattr(self, key)

    def __iter__(self) -> Iterator[torch.Tensor | None]:
        """
        Iterates over the buffered tensors. The samples of all channels are concatenated.

        Returns:
            The buffered tensors, or None if a tensor was initialized with shape None
        """
        mask = (
            torch.arange(self.capacity, device=self.sizes.device)[None, :]
            < self.sizes[:, None]
        )
        for buffer in self._buffer_fields():
            yield None if buffer is None else buffer[mask]

    def store(self, channels: torch.Tensor, *tensors: torch.Tensor | None):
        """
        Adds the given tensors to the buffers of their channels. If the buffer of a channel is
        full, its oldest samples are overwritten.

        Args:
            channels: channel index of every sample, shape (n, )
            tensors: samples to be stored. The shapes of the tensors after the batch dimension must
                match the shapes given during initialization. The argument can be None if the
                corresponding shape was None during initialization.
        """
        channels = channels.to(self.sizes.device)
        counts = torch.bincount(channels, minlength=self.channels)
        order = torch.argsort(channels, stable=True)
        sorted_channels = channels[order]
        # index of every sample among the new samples of its channel
        offsets = torch.cumsum(counts, dim=0) - counts
        rank = (
            torch.arange(len(channels), device=channels.device)
            - offsets[sorted_channels]
        )
        # only the newest samples are kept if more than capacity samples are stored at once
        keep = rank >= counts[sorted_channels] - self.capacity
        store_channels = sorted_channels[keep]
        store_positions = (self.store_indices[store_channels] + rank[keep]) % (
            self.capacity
        )
        load_indices = order[keep]
        for buffer, data in zip(self._buffer_fields(), tensors):
            if buffer is None or data is None:
                continue
            buffer[store_channels, store_positions] = data[load_indices].to(
                buffer.dtype
            )
        self.store_indices.copy_((self.store_indices + counts) % self.capacity)
        self.sizes.copy_(torch.clamp(self.sizes + counts, max=self.capacity))

    def clear(self, mask: torch.Tensor):
        """
        Removes all samples of the selected channels.

        Args:
            mask: channels to be cleared, shape (channels, )
        """
        self.sizes[mask] = 0
        self.store_indices[mask] = 0

    def sample(self, counts: torch.Tensor) -> list[torch.Tensor | None]:
        """
        Returns a batch of samples drawn from the buffer without replacement, with a given number
        of samples for every channel. If a channel contains fewer samples than requested, all its
        samples are returned.

        Args:
            counts: number of samples for every channel, shape (channels, )
        Returns:
            samples drawn from the buffer, ordered by channel
        """
        counts = torch.minimum(counts.to(self.sizes.device), self.sizes)
        sample_channels = torch.nonzero(counts > 0)[:, 0]
        sizes = self.sizes[sample_channels]
        max_size = int(sizes.max().item()) if len(sample_channels) > 0 else 0
        max_count = int(counts.max().item())
        # Taking the samples with the smallest random keys is equivalent to drawing without
        # replacement. The unused entries get keys larger than one and are never selected.
        keys = torch.rand((len(sample_channels), max_size), device=self.sizes.device)
        arange = torch.arange(max_size, device=self.sizes.device)
        keys.masked_fill_(arange[None, :] >= sizes[:, None], 2.0)
        positions = torch.topk(keys, max_count, dim=1, largest=False).indices
        mask = (
            torch.arange(max_count, device=self.sizes.device)[None, :]
            < counts[sample_channels, None]
        )
        channels = sample_channels[:, None].expand(-1, max_count)[mask]
        positions = positions[mask]
        return [
            None if buffer is None else buffer[channels, positions]
            for buffer in self._buffer_fields()
        ]
