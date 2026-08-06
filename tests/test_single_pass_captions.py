"""PART 6.4 (6-aug-2026): captions folded into the render pass.

The reframe used to write the clean clip and then burn_subtitles re-encoded
it — one extra full generation of loss plus a second ffmpeg process. The
render now encodes the clean AND the captioned file from the same reframed
frames; these tests pin the prep (naming convention) and the output args.
"""
import os

import main
import reframe_v2


def _transcript():
    return {"segments": [
        {"start": 0.0, "end": 1.0, "text": "Hello there.",
         "words": [{"word": " Hello", "start": 0.0, "end": 0.4},
                   {"word": " there.", "start": 0.5, "end": 0.9}]},
    ]}


def test_prepare_caption_burn_returns_ass_and_naming(tmp_path):
    clip = tmp_path / "Title_clip_1.mp4"
    clip.write_bytes(b"x")
    prep = main.prepare_caption_burn(
        str(clip), _transcript(), 0.0, 10.0)
    assert prep is not None
    ass_path, out_path, vf = prep
    assert os.path.basename(ass_path).startswith("autosubs_")
    assert os.path.basename(out_path).startswith("subtitled_")
    assert os.path.basename(out_path).endswith("Title_clip_1.mp4")
    assert vf.startswith("ass='")
    assert os.path.exists(ass_path)


def test_prepare_caption_burn_skips_silent_video(tmp_path):
    clip = tmp_path / "Title_clip_1.mp4"
    clip.write_bytes(b"x")
    assert main.prepare_caption_burn(
        str(clip), {"segments": []}, 0.0, 10.0) is None


def test_caption_output_args_map_second_output():
    args = reframe_v2.caption_output_args(
        "ass='/tmp/subs.ass':fontsdir='/fonts'", "/tmp/out_sub.mp4")
    assert args[0:4] == ["-map", "[v]", "-map", "0:a?"]
    assert "-vf" in args
    assert args[args.index("-vf") + 1] == \
        "ass='/tmp/subs.ass':fontsdir='/fonts'"
    assert args[-1] == "/tmp/out_sub.mp4"
    assert "-c:a" in args and "copy" in args
