"""Tests for the circular sample buffer"""

import torch

from madnis.integrator import Buffer


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
