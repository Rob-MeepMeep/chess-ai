"""
test_replay.py — ReplayBuffer eviction order across a save/load round trip.

10 Sept 2026 assessment: save() persisted the buffer in raw physical
ring-slot order but never the write cursor (self._pos); load() then
hardcoded self._pos = 0, assuming (via its own comment) that the saved
list was oldest-to-newest -- true only before the ring has wrapped.
Reproduced: capacity-3 buffer, insert 1..5, save, reload, insert 6 ->
evicted 4 (a real, fairly recent entry) instead of 3 (the true oldest).
Fixed by rotating to true chronological order in save() before writing.
"""

import os
import tempfile

import torch

from chessai.replay import ReplayBuffer


def _item(n):
    # dummy state/policy; outcome=n is a marker we can read back after
    # the tensors round-trip through the sparse-policy save format
    return (torch.zeros(1), torch.zeros(4096), float(n))


def _contents(buf):
    return [o for _, _, o in buf._buffer]


def test_wrapped_buffer_evicts_the_true_oldest_entry_after_reload(tmp_path):
    buf = ReplayBuffer(capacity=3)
    for n in [1, 2, 3, 4, 5]:
        buf.extend([_item(n)])
    # ring has wrapped: physical order is scrambled relative to insertion
    assert buf._pos != 0

    path = os.path.join(tmp_path, "buf.pt")
    buf.save(path)

    reloaded = ReplayBuffer(capacity=3)
    reloaded.load(path)

    # save() should have rotated to true chronological order, so load()'s
    # pos=0 reset is now correct -- the buffer should read oldest-to-newest
    assert _contents(reloaded) == [3.0, 4.0, 5.0]
    assert reloaded._pos == 0

    reloaded.extend([_item(6)])
    # 3 is the true oldest of {3,4,5} and must be the one evicted
    assert set(_contents(reloaded)) == {4.0, 5.0, 6.0}


def test_not_yet_wrapped_buffer_round_trips_unchanged(tmp_path):
    buf = ReplayBuffer(capacity=5)
    for n in [10, 20, 30]:
        buf.extend([_item(n)])
    assert buf._pos == 0   # never wrapped

    path = os.path.join(tmp_path, "buf.pt")
    buf.save(path)

    reloaded = ReplayBuffer(capacity=5)
    reloaded.load(path)

    assert _contents(reloaded) == [10.0, 20.0, 30.0]
    assert reloaded._pos == 0


def test_permanent_partition_survives_round_trip(tmp_path):
    buf = ReplayBuffer(capacity=3)
    buf.add_permanent([_item(100), _item(200)])
    for n in [1, 2, 3, 4]:
        buf.extend([_item(n)])

    path = os.path.join(tmp_path, "buf.pt")
    buf.save(path)

    reloaded = ReplayBuffer(capacity=3)
    reloaded.load(path)

    assert {o for _, _, o in reloaded._permanent} == {100.0, 200.0}
