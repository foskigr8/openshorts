"""Background-audio removal for delivered clips.

Two very different problems get called "background sound", and they need
different tools — using the wrong one is why naive attempts sound bad:

  * MUSIC / other speakers / ambience that is a real, structured signal.
    No spectral denoiser can remove this; it overlaps speech in frequency and
    time. This needs source separation (Demucs), which is a learned model that
    reconstructs the vocal stem.

  * BROADBAND NOISE — room tone, hiss, air conditioning, handling noise.
    This is what FFmpeg's denoisers (afftdn/anlmdn) are actually for.

So `clean_audio` prefers Demucs when it's installed, and falls back to a
conservative FFmpeg chain when it isn't. The fallback is deliberately gentle:
aggressive spectral subtraction produces the watery, "underwater" artefacts
that sound worse than the noise did, and this pipeline's output is judged on
voice quality.

Everything here is lossless-in / high-bitrate-out until the final AAC encode,
so cleaning never adds a generation of lossy loss.
"""
import os
import shutil
import subprocess
import tempfile

# Gentle, speech-safe repair chain used when full separation isn't available.
#   highpass=80    - removes rumble/handling thumps below the voice band
#   afftdn nf=-25  - mild spectral denoise; -25dB is conservative on purpose,
#                    stronger settings smear consonants and sound "phasey"
#   speechnorm     - evens out level differences between speakers without the
#                    pumping a hard compressor introduces
FFMPEG_SPEECH_CHAIN = (
    "highpass=f=80,"
    "afftdn=nf=-25:nt=w,"
    "speechnorm=e=6.25:r=0.00001:l=1"
)


def demucs_available():
    """True when Demucs can actually run (import + ffmpeg both present)."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        import demucs.separate  # noqa: F401
        return True
    except Exception:
        return False


def _run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _extract_wav(video_path, wav_path):
    # 44.1k stereo float WAV: Demucs' native rate, and lossless into the model.
    _run(["ffmpeg", "-y", "-i", video_path, "-vn",
          "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", wav_path])


def _separate_vocals(wav_path, workdir, model="htdemucs"):
    """Run Demucs and return the path to the isolated vocal stem.

    `--two-stems=vocals` only reconstructs vocals vs. everything-else, which is
    roughly twice as fast as the full 4-stem split and is all we need.
    GPU (6-aug-2026, PART 2): Demucs defaults to CUDA when torch sees a GPU,
    but the device is passed explicitly so a GPU host never falls back to CPU
    silently — the owner's "CPU pegged while both GPUs sit idle" complaint
    covered audio separation too.
    """
    cmd = ["python3", "-m", "demucs.separate", "--two-stems", "vocals",
           "-n", model, "-o", workdir]
    try:
        import torch
        if torch.cuda.is_available():
            cmd += ["--device", "cuda"]
    except Exception:
        pass
    cmd.append(wav_path)
    _run(cmd)
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    vocals = os.path.join(workdir, model, stem, "vocals.wav")
    return vocals if os.path.exists(vocals) else None


def clean_audio(video_path, output_path, mode="auto", model="htdemucs"):
    """Write `video_path` to `output_path` with its background audio removed.

    mode:
      "auto"    - Demucs if available, else the FFmpeg speech chain
      "isolate" - require Demucs (raises RuntimeError if unavailable)
      "denoise" - always the FFmpeg chain (fast, no model download)

    The video stream is always copied, never re-encoded: cleaning audio must
    not cost a generation of video quality.
    Returns output_path on success; raises on failure so callers can fall back
    to the untouched clip rather than shipping silence.
    """
    use_demucs = (mode == "isolate") or (mode == "auto" and demucs_available())
    if mode == "isolate" and not demucs_available():
        raise RuntimeError("demucs is not installed; cannot isolate voice")

    if not use_demucs:
        _run(["ffmpeg", "-y", "-i", video_path,
              "-af", FFMPEG_SPEECH_CHAIN,
              "-c:v", "copy", "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
              output_path])
        return output_path

    workdir = tempfile.mkdtemp(prefix="voiceiso_")
    try:
        wav = os.path.join(workdir, "source.wav")
        _extract_wav(video_path, wav)
        vocals = _separate_vocals(wav, workdir, model=model)
        if not vocals:
            raise RuntimeError("demucs produced no vocal stem")
        # Mux the isolated vocals back over the untouched video. speechnorm
        # after separation because removing a loud music bed leaves the voice
        # quieter than it was; loudnorm at delivery then lands it correctly.
        _run(["ffmpeg", "-y", "-i", video_path, "-i", vocals,
              "-map", "0:v:0", "-map", "1:a:0",
              "-af", "speechnorm=e=6.25:r=0.00001:l=1",
              "-c:v", "copy", "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
              "-shortest", output_path])
        return output_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
