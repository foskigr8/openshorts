from log_view import friendly_log_line, friendly_logs, plain_log_line, plain_logs

# Raw lines captured from a real production job.
RAW_JOB = [
    "Job started by worker.",
    "🎙️  Transcribing video...",
    "🎙️ Transcribing… 25% (3s)",
    "🎙️ Transcribing… 100% (7s)",
    "🎙️ [ASR] parakeet ok: lang=es segments=43",
    "🔥 Found 2 viral clips!",
    "🎬 Processing Clip 1: 0.004s - 35.336s",
    "🎞️ [Encoder] video encoder: h264_nvenc (FFMPEG_ENCODER=auto)",
    "🎬 Processing clip: output/e7f00666/temp_e7f00666_agente-ia_clip_1.mp4",
    "✅ Found 2 scenes.",
    "🧠 Step 2: Preparing Active Tracking...",
    "🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...",
    "✂️ Step 4: Processing video frames...",
    "🔊 Step 5: Extracting audio...",
    "✨ Step 6: Merging...",
    "✅ Clip saved to output/e7f00666/e7f00666_agente-ia_clip_1.mp4",
    "✅ Clip 1 ready: output/e7f00666/e7f00666_agente-ia_clip_1.mp4",
    "Process finished successfully.",
]


def test_real_job_produces_clean_user_view():
    assert friendly_logs(RAW_JOB) == [
        "Job started by worker.",
        "🎙️ Transcribing audio…",
        "🎙️ Transcribing… 25% (3s)",
        "🎙️ Transcribing… 100% (7s)",
        "🔥 Found 2 viral clips!",
        "🎬 Creating clip 1…",
        "✅ Clip 1 ready",
        "Process finished successfully.",
    ]


def test_plain_logs_strip_noise_keep_real_lines():
    entries = [
        {"ts": 1000.0, "text": "🎬 Processing Clip 1: 0s - 35s"},
        {"ts": 1001.0, "text": "[debug] yt-dlp version 2026.8.1"},
        {"ts": 1002.0, "text": "https://rr1---sn-a5oekn7e.googlevideo.com/videoplayback?expire=999999&signature=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
        {"ts": 1003.0, "text": "k8H4tR2m9xQ7pL3vN6wZ0aB5cD1eF2gH3iJ4kL5mN6oP7qR8sT9uV0wX1yZ2aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV3wX4yZ5aB6cD7eF8gH9iJ0kL1mN2oP3qR4sT5uV6wX7yZ8aB9cD0eF1gH2iJ3kL4mN5oP6qR7sT8uV9wX0yZ1aB2cD3eF4gH5iJ6kL7mN8oP9qR0sT1uV2wX3yZ4aB5cD6eF7gH8iJ9kL0mN"},
        {"ts": 1004.0, "text": "🔥 Found 3 viral clips!"},
        {"ts": 1005.0, "text": "[download]  42.3% of 50.00MiB at 8.20MiB/s"},
        {"ts": 1006.0, "text": "[Merger] Merging formats into output.mp4"},
    ]
    out = plain_logs(entries)
    texts = [e["text"] for e in out]
    assert "🎬 Processing Clip 1: 0s - 35s" in texts
    assert "🔥 Found 3 viral clips!" in texts
    # yt-dlp download % progress is genuinely useful — kept.
    assert any(t.startswith("[download]  42.3%") for t in texts)
    assert not any("[debug]" in t for t in texts)
    assert not any("googlevideo.com" in t for t in texts)
    assert not any(len(t) > 140 and " " not in t for t in texts)
    # timestamps are preserved from the real capture.
    assert out[0]["ts"] == 1000.0


def test_plain_log_line_handles_plain_strings():
    # Backward compatibility: callers with raw string logs still work.
    assert plain_log_line("Job started by worker.") == "Job started by worker."
    assert plain_log_line("[debug] foo") is None


def test_friendly_logs_accepts_timestamped_entries():
    # The log shape is now {ts, text}; the cloud-friendly path must read the
    # text field rather than calling .strip() on a dict.
    entries = [
        {"ts": 1.0, "text": "Job started by worker."},
        {"ts": 2.0, "text": "🔥 Found 2 viral clips!"},
        {"ts": 3.0, "text": "/app/scene_detection.py:109: UserWarning: nope"},
    ]
    assert friendly_logs(entries) == [
        "Job started by worker.",
        "🔥 Found 2 viral clips!",
    ]


def test_no_paths_ever_leak():
    for line in friendly_logs(RAW_JOB):
        assert "output/" not in line
        assert ".mp4" not in line


def test_technical_lines_are_hidden():
    assert friendly_log_line("🎙️ [ASR] parakeet ok: lang=es segments=43") is None
    assert friendly_log_line("🎞️ [Encoder] video encoder: h264_nvenc") is None
    assert friendly_log_line("🧠 Step 2: Preparing Active Tracking...") is None
    assert friendly_log_line("yt-dlp downloading format 137") is None


def test_errors_kept_without_paths():
    assert friendly_log_line("Process failed with exit code 1") == \
        "Process failed with exit code 1"
    out = friendly_log_line("❌ Could not read output/abc/clip.mp4")
    assert out is not None and "output/" not in out


def test_consecutive_duplicates_collapse():
    logs = ["🎙️  Transcribing video...", "🎙️  Transcribing audio from: x.mp4"]
    assert friendly_logs(logs) == ["🎙️ Transcribing audio…"]
