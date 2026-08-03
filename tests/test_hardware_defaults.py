"""Hardware-aware concurrency defaults (CPU hosts must not inherit T4-tuned
5/5 concurrency — the round-2 CLIP_WORKERS bump would oversubscribe them)."""
import hardware_defaults as hd


def test_gpu_detected_via_yolo_device(monkeypatch):
    monkeypatch.setenv("YOLO_DEVICE", "0")
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert hd.gpu_configured() is True


def test_gpu_detected_via_whisper_cuda(monkeypatch):
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert hd.gpu_configured() is True


def test_cpu_host_when_no_gpu_env(monkeypatch):
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert hd.gpu_configured() is False


def test_clip_workers_five_on_gpu(monkeypatch):
    monkeypatch.setenv("YOLO_DEVICE", "0")
    assert hd.default_clip_workers() == 5


def test_clip_workers_scaled_to_cores_on_cpu(monkeypatch):
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(hd.os, "cpu_count", lambda: 4)
    assert hd.default_clip_workers() == 3
    monkeypatch.setattr(hd.os, "cpu_count", lambda: 2)
    assert hd.default_clip_workers() == 1
    monkeypatch.setattr(hd.os, "cpu_count", lambda: None)
    assert hd.default_clip_workers() == 1


def test_max_concurrent_jobs_single_on_cpu(monkeypatch):
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert hd.default_max_concurrent_jobs() == 1


def test_max_concurrent_jobs_five_on_gpu(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert hd.default_max_concurrent_jobs() == 5
