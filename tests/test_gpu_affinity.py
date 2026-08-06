"""Per-clip-worker GPU assignment.

Measured on Kaggle (2×T4, job b86b8c5a): three clips rendered concurrently and
all three ran on GPU 0. These tests pin the round-robin and, more importantly,
that CPU/single-GPU hosts behave exactly as they did before.
"""

import sys
import threading
import types

import pytest

import gpu_affinity


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(gpu_affinity, "_devices_cache", None)
    monkeypatch.setattr(gpu_affinity, "_local", threading.local())
    monkeypatch.delenv("CLIP_GPUS", raising=False)
    yield


def _fake_torch(monkeypatch, count, available=True, fail_set=False):
    selected = []
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        set_device=(lambda i: (_ for _ in ()).throw(RuntimeError("no such device")))
        if fail_set else selected.append,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return selected


def test_no_devices_on_a_cpu_host(monkeypatch):
    _fake_torch(monkeypatch, 0, available=False)
    assert gpu_affinity.available_devices() == []
    assert gpu_affinity.assign_worker(0) is None
    assert gpu_affinity.current_device() is None


def test_missing_torch_is_not_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises
    assert gpu_affinity.available_devices() == []
    assert gpu_affinity.assign_worker(3) is None


def test_sharding_is_off_unless_opted_in(monkeypatch):
    """Default = no affinity. Assigning workers to cuda:1 broke TransNetV2,
    which loads on the default device and then got tensors from another one."""
    _fake_torch(monkeypatch, 2)
    assert gpu_affinity.available_devices() == []
    assert gpu_affinity.assign_worker(0) is None
    assert gpu_affinity.current_device() is None


def test_single_gpu_pins_everything_to_zero(monkeypatch):
    monkeypatch.setenv("CLIP_GPUS", "0")
    _fake_torch(monkeypatch, 1)
    assert [gpu_affinity.assign_worker(i) for i in range(3)] == \
        ["cuda:0", "cuda:0", "cuda:0"]


def test_two_gpus_round_robin_by_clip_index(monkeypatch):
    monkeypatch.setenv("CLIP_GPUS", "0,1")
    selected = _fake_torch(monkeypatch, 2)
    assert [gpu_affinity.assign_worker(i) for i in range(5)] == \
        ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]
    assert selected == [0, 1, 0, 1, 0]


def test_clip_gpus_restricts_the_pool(monkeypatch):
    monkeypatch.setenv("CLIP_GPUS", "1")
    _fake_torch(monkeypatch, 2)
    assert gpu_affinity.available_devices() == [1]
    assert gpu_affinity.assign_worker(0) == "cuda:1"


def test_clip_gpus_ignores_indices_that_do_not_exist(monkeypatch):
    monkeypatch.setenv("CLIP_GPUS", "0,7,garbage")
    _fake_torch(monkeypatch, 2)
    assert gpu_affinity.available_devices() == [0]


def test_assignment_is_per_thread(monkeypatch):
    """Two worker threads must not clobber each other's device."""
    monkeypatch.setenv("CLIP_GPUS", "0,1")
    _fake_torch(monkeypatch, 2)
    seen = {}

    def worker(index):
        gpu_affinity.assign_worker(index)
        seen[index] = gpu_affinity.current_device()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {0: "cuda:0", 1: "cuda:1"}
    # The main thread was never assigned, so it still defers to torch.
    assert gpu_affinity.current_device() is None


def test_a_failed_set_device_degrades_to_the_default(monkeypatch, capsys):
    monkeypatch.setenv("CLIP_GPUS", "0,1")
    _fake_torch(monkeypatch, 2, fail_set=True)
    assert gpu_affinity.assign_worker(1) is None
    assert gpu_affinity.current_device() is None
    assert "GPU affinity" in capsys.readouterr().out


def test_describe_is_silent_unless_there_is_something_to_share(monkeypatch):
    monkeypatch.setenv("CLIP_GPUS", "0")
    _fake_torch(monkeypatch, 1)
    assert gpu_affinity.describe() == ""


def test_describe_names_the_devices_when_sharding(monkeypatch):
    monkeypatch.setenv("CLIP_GPUS", "0,1")
    _fake_torch(monkeypatch, 2)
    text = gpu_affinity.describe()
    assert "cuda:0" in text and "cuda:1" in text
