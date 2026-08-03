"""LR-ASD adapter: input contract, alignment, and graceful degradation.

The one thing this model must never get wrong is lip/audio ALIGNMENT — a
mismatched trim silently shifts sound against mouths and the scores become
confident noise, which is worse than no ASD at all. Most of these tests are
about that.
"""
import numpy as np
import pytest

asd_worker = pytest.importorskip("asd_worker")

pytestmark = pytest.mark.skipif(
    not asd_worker.available(),
    reason="LR-ASD weights or python_speech_features unavailable",
)

FACE = asd_worker.FACE_SIZE


@pytest.fixture(scope="module")
def scorer():
    return asd_worker.ASDScorer(device="cpu")


def _faces(n, value=128):
    return np.full((n, FACE, FACE), value, dtype=np.uint8)


def _mfcc(n_video_frames):
    # exactly the 4:1 audio:video grid the network was trained on
    return np.zeros((n_video_frames * 4, 13), dtype=np.float32)


def test_crop_face_returns_the_contract_shape():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = asd_worker.crop_face(frame, (900, 300, 150, 190))
    assert out.shape == (FACE, FACE)
    assert out.dtype == np.uint8


def test_crop_face_handles_a_box_off_the_frame_edge():
    """A face at the very edge must pad, not crash or return an empty patch."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for box in [(-80, -60, 150, 190),          # off top-left
                (1880, 1020, 150, 190),        # off bottom-right
                (0, 0, 10, 10)]:               # degenerate
        out = asd_worker.crop_face(frame, box)
        assert out.shape == (FACE, FACE)


def test_score_returns_one_value_per_video_frame(scorer):
    n = 30
    out = scorer.score(_faces(n), _mfcc(n), durations=(1,))
    assert out.shape == (n,)
    assert np.isfinite(out).all()


def test_score_trims_to_the_shorter_stream(scorer):
    """Video longer than audio must trim to the audio, not invent frames."""
    out = scorer.score(_faces(40), _mfcc(25), durations=(1,))
    assert len(out) == 25


def test_score_trims_audio_to_the_four_to_one_grid(scorer):
    """Ragged audio (not a multiple of 4) must not shift the alignment."""
    mfcc = np.zeros((25 * 4 + 3, 13), dtype=np.float32)   # 3 extra frames
    out = scorer.score(_faces(25), mfcc, durations=(1,))
    assert len(out) == 25


def test_empty_input_is_safe(scorer):
    assert len(scorer.score(np.zeros((0, FACE, FACE), np.uint8),
                            _mfcc(0), durations=(1,))) == 0


def test_audio_too_short_for_one_frame_returns_zeros(scorer):
    out = scorer.score(_faces(10), np.zeros((2, 13), np.float32), durations=(1,))
    assert len(out) == 10
    assert not out.any()


def test_duration_set_is_configurable(monkeypatch):
    monkeypatch.setenv("ASD_DURATION_SET", "2,4")
    assert asd_worker._durations() == (2, 4)
    monkeypatch.setenv("ASD_DURATION_SET", "garbage")
    assert asd_worker._durations() == asd_worker.DEFAULT_DURATIONS
    monkeypatch.delenv("ASD_DURATION_SET")
    assert asd_worker._durations() == asd_worker.DEFAULT_DURATIONS


def test_averaging_windows_agree_on_length(scorer):
    """Multi-window averaging must not change the output length."""
    n = 24
    one = scorer.score(_faces(n), _mfcc(n), durations=(1,))
    multi = scorer.score(_faces(n), _mfcc(n), durations=(1, 3))
    assert len(one) == len(multi) == n


def test_model_runs_on_cpu_without_cuda():
    """Upstream hardcodes .cuda(); this must work on a CPU-only host."""
    s = asd_worker.ASDScorer(device="cpu")
    assert str(s.device) == "cpu"
