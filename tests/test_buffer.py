"""Tests for the circular sample buffer"""

import torch

from madnis.integrator import Buffer, ChannelBuffer


def make_buffer(capacity: int = 10, persistent: bool = True) -> Buffer:
    return Buffer(
        capacity,
        [(2,), None, ()],
        persistent=persistent,
        dtypes=[None, None, torch.int64],
    )


def batch(start: int, stop: int):
    index = torch.arange(start, stop)
    return index[:, None].float().expand(-1, 2), None, index


def test_store_and_iterate():
    buffer = make_buffer()
    buffer.store(*batch(0, 4))
    buffer.store(*batch(4, 7))
    x, none_field, index = buffer
    assert buffer.size == 7
    assert none_field is None
    assert index.dtype == torch.int64
    torch.testing.assert_close(index, torch.arange(7))
    torch.testing.assert_close(x[:, 0], torch.arange(7).float())


def test_overwrites_oldest_samples():
    buffer = make_buffer()
    buffer.store(*batch(0, 8))
    buffer.store(*batch(8, 13))
    _, _, index = buffer
    assert buffer.size == 10
    assert buffer.store_index == 3
    # samples 0, 1, 2 were overwritten by 10, 11, 12
    assert sorted(index.tolist()) == list(range(3, 13))


def test_store_more_than_capacity():
    buffer = make_buffer()
    buffer.store(*batch(0, 25))
    _, _, index = buffer
    assert buffer.size == 10
    assert len(set(index.tolist())) == 10


def test_sample_without_replacement():
    buffer = make_buffer()
    buffer.store(*batch(0, 7))
    torch.manual_seed(0)
    x, none_field, index = buffer.sample(5)
    assert none_field is None
    assert len(set(index.tolist())) == 5
    torch.testing.assert_close(x[:, 0], index.float())
    # asking for more samples than stored returns all samples
    _, _, index = buffer.sample(100)
    assert sorted(index.tolist()) == list(range(7))


def test_filter():
    buffer = make_buffer()
    buffer.store(*batch(0, 8))
    buffer.store(*batch(8, 13))  # wraps around
    buffer.filter(lambda fields: fields[2] % 2 == 0, batch_size=3)
    x, _, index = buffer
    assert buffer.size == 5
    assert buffer.store_index == 5
    assert sorted(index.tolist()) == [4, 6, 8, 10, 12]
    torch.testing.assert_close(x[:, 0], index.float())
    # storing after filtering continues behind the kept samples
    buffer.store(*batch(100, 102))
    _, _, index = buffer
    assert sorted(index.tolist()) == [4, 6, 8, 10, 12, 100, 101]


def test_state_dict():
    buffer = make_buffer(persistent=True)
    buffer.store(*batch(0, 6))
    loaded = make_buffer(persistent=True)
    loaded.load_state_dict(buffer.state_dict())
    assert loaded.size == 6
    torch.testing.assert_close(list(loaded)[2], torch.arange(6))
    assert len(make_buffer(persistent=False).state_dict()) == 0


def test_default_dtypes():
    buffer = Buffer(5, [(3,), ()])
    buffer.store(torch.rand((2, 3)), torch.rand(2))
    assert all(field.dtype == torch.get_default_dtype() for field in buffer)
    assert buffer.size == 2


def make_channel_buffer(channels: int = 3, capacity: int = 4) -> ChannelBuffer:
    return ChannelBuffer(
        channels, capacity, [(2,), None, ()], dtypes=[None, None, torch.int64]
    )


def channel_batch(channels: list[int], start: int):
    index = torch.arange(start, start + len(channels))
    return (
        torch.tensor(channels),
        index[:, None].float().expand(-1, 2),
        None,
        index,
    )


def stored_indices(buffer: ChannelBuffer, channel: int) -> list[int]:
    _, _, index = buffer
    channels = torch.repeat_interleave(torch.arange(buffer.channels), buffer.sizes)
    return sorted(index[channels == channel].tolist())


def test_channel_buffer_store_per_channel():
    buffer = make_channel_buffer()
    buffer.store(*channel_batch([0, 1, 0, 2, 0], 0))
    assert buffer.sizes.tolist() == [3, 1, 1]
    assert buffer.size == 5
    assert stored_indices(buffer, 0) == [0, 2, 4]
    assert stored_indices(buffer, 1) == [1]
    x, none_field, index = buffer
    assert none_field is None
    torch.testing.assert_close(x[:, 0], index.float())


def test_channel_buffer_overwrites_oldest_per_channel():
    buffer = make_channel_buffer()
    buffer.store(*channel_batch([0, 0, 0, 1], 0))
    buffer.store(*channel_batch([0, 0], 10))
    # channel 0 is full and its oldest sample was overwritten, channel 1 is untouched
    assert buffer.sizes.tolist() == [4, 1, 0]
    assert stored_indices(buffer, 0) == [1, 2, 10, 11]
    assert stored_indices(buffer, 1) == [3]
    # storing more than the capacity at once keeps the newest samples
    buffer.store(*channel_batch([1] * 7, 20))
    assert stored_indices(buffer, 1) == [23, 24, 25, 26]
    buffer.store(*channel_batch([1], 30))
    assert stored_indices(buffer, 1) == [24, 25, 26, 30]


def test_channel_buffer_sample():
    buffer = make_channel_buffer(capacity=10)
    buffer.store(*channel_batch([0] * 6 + [1] * 2 + [2] * 5, 0))
    torch.manual_seed(0)
    x, none_field, index = buffer.sample(torch.tensor([3, 5, 0]))
    assert none_field is None
    torch.testing.assert_close(x[:, 0], index.float())
    # channel 1 has only two samples, channel 2 is not requested
    assert len(index) == 5
    assert len(set(index.tolist())) == 5
    assert all(i < 6 for i in index[:3].tolist())
    assert sorted(index[3:].tolist()) == [6, 7]
    # empty request
    _, _, index = buffer.sample(torch.zeros(3, dtype=torch.int64))
    assert len(index) == 0


def test_channel_buffer_sample_is_uniform():
    buffer = make_channel_buffer(channels=1, capacity=10)
    buffer.store(*channel_batch([0] * 13, 0))  # wraps around
    torch.manual_seed(0)
    counts = torch.zeros(13)
    for _ in range(2000):
        _, _, index = buffer.sample(torch.tensor([3]))
        counts += torch.bincount(index, minlength=13)
    assert (counts[:3] == 0).all()
    assert ((counts[3:] - 600).abs() < 100).all()


def test_channel_buffer_clear():
    buffer = make_channel_buffer()
    buffer.store(*channel_batch([0, 1, 2, 1], 0))
    buffer.clear(torch.tensor([False, True, False]))
    assert buffer.sizes.tolist() == [1, 0, 1]
    buffer.store(*channel_batch([1], 10))
    assert stored_indices(buffer, 1) == [10]


def test_channel_buffer_state_dict():
    buffer = make_channel_buffer()
    buffer.store(*channel_batch([0, 1, 2, 2], 0))
    loaded = make_channel_buffer()
    loaded.load_state_dict(buffer.state_dict())
    assert loaded.sizes.tolist() == [1, 1, 2]
    torch.testing.assert_close(list(loaded)[2], list(buffer)[2])
