"""The TransNetV2 device guard (6-aug-2026).

Clip workers carry a thread-local torch device (gpu_affinity, CLIP_GPUS);
the shared TransNetV2 model lives on the device it was loaded on. Inference
from a differently-pinned worker used to die with "Expected all tensors to be
on the same device" and silently fell back to PySceneDetect. The guard must
pin the current device to the model's during the call and restore it after.
"""
import pytest

import scene_detection


def test_guard_pins_model_device_and_restores(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.current_device", lambda: 1)

    def _set_device(dev):
        calls.append(str(dev))

    monkeypatch.setattr("torch.cuda.set_device", _set_device)

    with scene_detection._model_device_guard("cuda:0"):
        assert calls == ["cuda:0"]
    assert calls == ["cuda:0", "1"], \
        "the worker's previous device must be restored on exit"


def test_guard_restores_even_on_exception(monkeypatch):
    calls = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 1)
    monkeypatch.setattr("torch.cuda.set_device",
                        lambda dev: calls.append(str(dev)))

    with pytest.raises(RuntimeError):
        with scene_detection._model_device_guard("cuda:0"):
            raise RuntimeError("boom")
    assert calls == ["cuda:0", "1"]


def test_guard_noops_on_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with scene_detection._model_device_guard("cpu"):
        pass
    # No set_device calls on a CPU-only host.
