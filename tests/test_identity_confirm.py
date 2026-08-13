"""Gemini face-identity confirmation — the pure parts (no video, no API)."""

from identity_confirm import _compact_transcript, _parse_mapping


def _seg(start, end, speaker, text):
    return {"start": start, "end": end, "speaker": speaker, "text": text,
            "words": [{"text": w} for w in text.split()]}


class TestParseMapping:
    def test_maps_labels_to_track_ids(self):
        text = '{"mapping": {"A": "F3", "B": "F7"}}'
        tracks = {3: {"frames": [0]}, 7: {"frames": [1]}}
        assert _parse_mapping(text, tracks) == {"A": 3, "B": 7}

    def test_unknown_face_labels_are_dropped(self):
        # A label pointing at a track that doesn't exist is unconfirmed —
        # it must not enter the map (no guessing).
        text = '{"mapping": {"A": "F99", "B": "F2"}}'
        tracks = {2: {"frames": [0]}}
        assert _parse_mapping(text, tracks) == {"B": 2}

    def test_bad_json_returns_empty(self):
        assert _parse_mapping("not json", {}) == {}

    def test_bare_number_face_labels_accepted(self):
        text = '{"mapping": {"A": "3"}}'
        tracks = {3: {"frames": [0]}}
        assert _parse_mapping(text, tracks) == {"A": 3}


class TestCompactTranscript:
    def test_is_clip_relative_and_skips_non_speakers(self):
        segs = [_seg(10, 13, "A", "hello there"),
                _seg(14, 16, None, "no speaker"),
                _seg(20, 22, "B", "hi")]
        out = _compact_transcript(segs, 10, 25)
        assert "speaker A (0.0-3.0s): hello there" in out
        assert "speaker B (10.0-12.0s): hi" in out
        assert "no speaker" not in out
