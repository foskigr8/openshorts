"""Tests for subtitle word merging, SRT generation and style sanitizing."""
import re
from subtitles import (
    merge_continuation_words,
    generate_srt,
    ass_filter_string,
    hex_to_ass_color,
    _sanitize_font_name,
    _clamp_number,
    _clean_caption_word,
    _caption_width_compaction,
    CAPTION_MAX_WIDTH_EM,
)


def test_ass_filter_string_escapes_for_the_filtergraph():
    vf = ass_filter_string("/tmp/subs_1_123.ass")
    assert vf.startswith("ass='/tmp/subs_1_123.ass'")
    assert "fontsdir=" in vf


def _w(text, start, end, speaker=None):
    word = {"word": text, "start": start, "end": end}
    if speaker is not None:
        word["speaker"] = speaker
    return word


class TestMergeContinuationWords:
    def test_merges_compound_fragments(self):
        # faster-whisper splits "YouTube-Kanal." into two tokens; the second
        # one has no leading space and belongs to the first.
        words = [_w(" YouTube", 0.0, 0.5), _w("-Kanal.", 0.5, 0.9), _w(" ist", 1.0, 1.2)]
        merged = merge_continuation_words(words)
        assert [m["word"] for m in merged] == [" YouTube-Kanal.", " ist"]
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 0.9

    def test_keeps_real_word_boundaries(self):
        # Words with a leading space are separate words and must never be glued.
        words = [_w(" ich", 0.0, 0.2), _w(" habe", 0.2, 0.4)]
        merged = merge_continuation_words(words)
        assert [m["word"] for m in merged] == [" ich", " habe"]

    def test_first_word_without_space_stays(self):
        words = [_w("Hallo", 0.0, 0.2), _w(" Welt", 0.2, 0.4)]
        merged = merge_continuation_words(words)
        assert [m["word"] for m in merged] == ["Hallo", " Welt"]

    def test_number_fragments(self):
        words = [_w(" 1", 0.0, 0.2), _w(".200", 0.2, 0.4)]
        merged = merge_continuation_words(words)
        assert [m["word"] for m in merged] == [" 1.200"]

    def test_input_not_mutated(self):
        words = [_w(" a", 0.0, 0.1), _w("-b", 0.1, 0.2)]
        merge_continuation_words(words)
        assert words[0]["word"] == " a"
        assert words[1]["word"] == "-b"


class TestCaptionPunctuation:
    def test_commas_and_semicolons_always_stripped(self):
        assert _clean_caption_word("right,", sentence_final=False) == "right"
        assert _clean_caption_word("wait;", sentence_final=True) == "wait"
        assert _clean_caption_word("a,b,c", sentence_final=False) == "abc"

    def test_sentence_final_punctuation_kept_on_last_word(self):
        assert _clean_caption_word("really.", sentence_final=True) == "really."
        assert _clean_caption_word("why?", sentence_final=True) == "why?"
        assert _clean_caption_word("wow!", sentence_final=True) == "wow!"

    def test_mid_sentence_trailing_punctuation_stripped(self):
        # Token happened to carry a period, but the word is not the end of a
        # segment -> drop it (one-word-at-a-time captions shouldn't end every
        # word with a period).
        assert _clean_caption_word("yes.", sentence_final=False) == "yes"
        assert _clean_caption_word("stop!", sentence_final=False) == "stop"

    def test_punctuation_runs_collapse_to_one(self):
        assert _clean_caption_word("wait!!", sentence_final=True) == "wait!"
        assert _clean_caption_word("no...", sentence_final=True) == "no."

    def test_apostrophes_and_hyphens_survive(self):
        assert _clean_caption_word("don't", sentence_final=False) == "don't"
        assert _clean_caption_word("well-known", sentence_final=True) == "well-known"

    def test_generate_srt_strips_mid_sentence_period(self, tmp_path):
        from subtitles import generate_srt
        out = tmp_path / "subs.srt"
        # Two segments: "yes." is the ONLY word of its segment -> keeps the
        # period; "wait." is followed by more words in its segment -> stripped.
        transcript = {
            "segments": [
                {"start": 0, "end": 99, "text": "", "words": [_w(" yes.", 0.0, 0.5)]},
                {"start": 0, "end": 99, "text": "", "words": [
                    _w(" wait.", 0.6, 1.0), _w(" more", 1.0, 1.4)]},
            ]
        }
        assert generate_srt(transcript, 0, 10, str(out)) is True
        srt = out.read_text(encoding="utf-8-sig")
        assert "yes." in srt
        assert "wait." not in srt
        assert "wait more" in srt


class TestPerWordWidthSizing:
    def test_narrow_word_gets_no_compaction(self):
        assert _caption_width_compaction("NO", "Montserrat ExtraBold", 27) == (None, None)

    def test_wide_word_squeezes_horizontally_only(self):
        # "REALLY" is ~4.1em at Montserrat ExtraBold -> well over the 3.3em
        # envelope, so the word gets a horizontal \fscx < 100 — but NO height
        # change (round-2: per-word height shrink made words look like a
        # different caption style).
        fscx, fsp = _caption_width_compaction("REALLY", "Montserrat ExtraBold", 27)
        assert fscx is not None and fscx < 100 and fscx >= 70
        # Very wide words additionally tighten letter-spacing before the
        # squeeze reaches its floor.
        assert fsp is not None and fsp < 0

    def test_unknown_font_falls_back_to_generic_metrics(self):
        fscx, _fsp = _caption_width_compaction("SUPERCALIFRAGILISTIC", "NotARealFont", 27)
        assert fscx is not None and 70 <= fscx <= 100

    def test_compaction_never_reaches_below_70_percent(self):
        fscx, _fsp = _caption_width_compaction("INTERNATIONALE", "Montserrat ExtraBold", 27)
        assert fscx == 70

    def test_max_width_envelope_constant_is_sane(self):
        # Must be wide enough that normal short words stay full-size but small
        # enough to keep long words inside the frame (9:16 -> PlayResX=162).
        assert 2.5 <= CAPTION_MAX_WIDTH_EM <= 4.5


class TestGenerateSrt:
    def _transcript(self, words):
        return {"segments": [{"start": 0, "end": 99, "text": "", "words": words}]}

    def test_no_orphan_fragments_in_srt(self, tmp_path):
        out = tmp_path / "subs.srt"
        words = [
            _w(" Mein", 0.0, 0.3),
            _w(" YouTube", 0.3, 0.8),
            _w("-Kanal.", 0.8, 1.1),
            _w(" ich", 1.2, 1.4),
            _w(" habe", 1.4, 1.7),
        ]
        assert generate_srt(self._transcript(words), 0, 10, str(out)) is True
        srt = out.read_text(encoding="utf-8-sig")
        # Mid-segment token "YouTube-Kanal." loses its trailing period under
        # the punctuation rule (only sentence-final words keep ./?/!) but the
        # fragment must still be merged into one word.
        assert "YouTube-Kanal" in srt
        assert " -Kanal" not in srt
        assert "ich habe" in srt
        assert "ichhabe" not in srt

    def test_empty_range_returns_false(self, tmp_path):
        out = tmp_path / "subs.srt"
        words = [_w(" spaet", 50.0, 50.5)]
        assert generate_srt(self._transcript(words), 0, 10, str(out)) is False


class TestStyleSanitizing:
    def test_invalid_hex_falls_back_to_white(self):
        assert hex_to_ass_color("#GGGGGG") == hex_to_ass_color("#FFFFFF")
        assert hex_to_ass_color("abc") == hex_to_ass_color("#FFFFFF")
        assert hex_to_ass_color(None) == hex_to_ass_color("#FFFFFF")

    def test_invalid_hex_custom_fallback(self):
        assert hex_to_ass_color("nope", fallback="000000") == hex_to_ass_color("#000000")

    def test_valid_hex_converts(self):
        # #RRGGBB -> &HAABBGGRR
        assert hex_to_ass_color("#FF0000", 1.0) == "&H000000FF"
        assert hex_to_ass_color("00FF00", 1.0) == "&H0000FF00"

    def test_opacity_clamped(self):
        assert hex_to_ass_color("#FFFFFF", 5.0) == hex_to_ass_color("#FFFFFF", 1.0)
        assert hex_to_ass_color("#FFFFFF", -1) == hex_to_ass_color("#FFFFFF", 0.0)

    def test_font_name_injection_stripped(self):
        assert _sanitize_font_name("Arial,Fontsize=99{\\b1}") == "ArialFontsize99b1"
        assert _sanitize_font_name("Comic Sans MS") == "Comic Sans MS"

    def test_font_name_empty_falls_back(self):
        assert _sanitize_font_name("") == "Verdana"
        assert _sanitize_font_name(",,{}") == "Verdana"
        assert _sanitize_font_name(None) == "Verdana"

    def test_clamp_number(self):
        assert _clamp_number(5, 0, 10, 1) == 5
        assert _clamp_number(99, 0, 10, 1) == 10
        assert _clamp_number(-3, 0, 10, 1) == 0
        assert _clamp_number("kaputt", 0, 10, 1) == 1
        assert _clamp_number(None, 0, 10, 1) == 1


class TestGenerateAss:
    from subtitles import generate_ass  # noqa: F401 (import check)

    def _transcript(self, words):
        return {"segments": [{"start": 0, "end": 99, "text": "", "words": words}]}

    def test_karaoke_events_highlight_each_word(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" Erst", 0.0, 0.3), _w(" mal", 0.3, 0.6), _w(" hier", 0.6, 0.9)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            highlight_color="#22C55E", font_color="#FFFFFF") is True
        content = out.read_text(encoding="utf-8-sig")
        # One dialogue event per word, highlight moves through the block
        assert content.count("Dialogue:") == 3
        assert content.count("\\c&H5EC522&") == 3  # #22C55E -> BGR 5EC522
        assert content.count("{\\r}") == 3          # reset to dimmed base style
        assert "Style: Default,Verdana," in content

    def test_general_range_gets_a_bigger_margin_v(self, tmp_path):
        # Regression: captions used a fixed MarginV regardless of reframe
        # mode, so during a GENERAL-layout scene (content shrunk and
        # vertically centered, blurred fill above/below — see
        # reframe_v2.general_filtergraph) the caption floated in the blur
        # band well below the actual content, disconnected from it, reading
        # as a second stacked panel (confirmed on real delivered clips,
        # 31-jul-2026). A word inside a general_ranges window must get a
        # per-line MarginV override pulling it up into the content box;
        # a word outside any range must keep the default line MarginV of 0
        # (inherits the Style's margin).
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" track", 0.0, 0.5), _w(" general", 5.0, 5.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            general_ranges=[(4.0, 6.0)]) is True
        content = out.read_text(encoding="utf-8-sig")
        lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
        assert len(lines) == 2
        track_line, general_line = lines
        # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        track_margin_v = track_line.split(",")[7]
        general_margin_v = general_line.split(",")[7]
        assert track_margin_v == "0"
        assert general_margin_v != "0" and int(general_margin_v) > 0

    def test_general_ranges_defaults_to_no_override(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" hello", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out)) is True
        content = out.read_text(encoding="utf-8-sig")
        line = [l for l in content.splitlines()
                if l.startswith("Dialogue:")][0]
        assert line.split(",")[7] == "0"

    def test_general_floor_keeps_bottom_near_the_bottom(self, tmp_path):
        # 6-aug-2026: the content-box floor used to place "bottom" at ~40%
        # up the frame (reads as MIDDLE). It now clears the blur band by a
        # small inset and then respects the per-job margin.
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" hello", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            alignment="bottom", margin_v=43,
                            general_ranges=[(0.0, 10.0)]) is True
        content = out.read_text(encoding="utf-8-sig")
        line = next(l for l in content.splitlines()
                    if l.startswith("Dialogue:"))
        margin = int(line.split(",")[7])
        assert 40 <= margin <= 55, \
            f"'bottom' must sit just inside the content box, got {margin}"

    def test_karaoke_merges_fragments_too(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" YouTube", 0.0, 0.5), _w("-Kanal.", 0.5, 0.9)]
        assert generate_ass(self._transcript(words), 0, 10, str(out)) is True
        content = out.read_text(encoding="utf-8-sig")
        assert "YouTube-Kanal." in content
        assert content.count("Dialogue:") == 1

    def test_letter_spacing_ratio_reaches_the_style_line(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" hi", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            fontsize=100, letter_spacing_ratio=-0.03) is True
        content = out.read_text(encoding="utf-8-sig")
        style_line = [l for l in content.splitlines() if l.startswith("Style:")][0]
        fields = style_line.split(",")
        # Format: Name,Fontname,Fontsize,Primary,Secondary,Outline,Back,
        # Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,...
        spacing = float(fields[13])
        assert spacing < 0  # tighter than normal, per the negative ratio

    def test_zero_letter_spacing_is_the_default(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" hi", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out)) is True
        content = out.read_text(encoding="utf-8-sig")
        style_line = [l for l in content.splitlines() if l.startswith("Style:")][0]
        assert float(style_line.split(",")[13]) == 0.0


class TestSpeakerColors:
    """Each diarized speaker gets their own caption colour (SPEAKER_CAPTION_
    PALETTE, first-seen order) so a viewer can track who's talking from
    captions alone — opt-in via speaker_colors=True (see AUTO_CAPTION_STYLE)."""

    def _transcript_two_speakers(self):
        return {"segments": [
            {"start": 0, "end": 1, "text": "", "speaker": "A",
             "words": [_w(" hi", 0.0, 0.5)]},
            {"start": 1, "end": 2, "text": "", "speaker": "B",
             "words": [_w(" yo", 1.0, 1.5)]},
        ]}

    def test_disabled_uses_single_highlight_color(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        assert generate_ass(self._transcript_two_speakers(), 0, 10, str(out),
                            highlight_color="#22C55E",
                            speaker_colors=False) is True
        content = out.read_text(encoding="utf-8-sig")
        # Both words use the same (highlight_color) inline colour tag.
        assert content.count("\\c&H5EC522&") == 2

    def test_enabled_gives_each_speaker_a_distinct_color(self, tmp_path):
        from subtitles import generate_ass, SPEAKER_CAPTION_PALETTE, _hex_to_ass_inline_color
        out = tmp_path / "subs.ass"
        assert generate_ass(self._transcript_two_speakers(), 0, 10, str(out),
                            speaker_colors=True) is True
        content = out.read_text(encoding="utf-8-sig")
        first = _hex_to_ass_inline_color(SPEAKER_CAPTION_PALETTE[0])
        second = _hex_to_ass_inline_color(SPEAKER_CAPTION_PALETTE[1])
        assert f"\\c{first}" in content
        assert f"\\c{second}" in content

    def test_same_speaker_always_gets_the_same_color(self, tmp_path):
        from subtitles import generate_ass, SPEAKER_CAPTION_PALETTE, _hex_to_ass_inline_color
        out = tmp_path / "subs.ass"
        transcript = {"segments": [
            {"start": 0, "end": 1, "text": "", "speaker": "A",
             "words": [_w(" one", 0.0, 0.5), _w(" two", 1.0, 1.5)]},
        ]}
        assert generate_ass(transcript, 0, 10, str(out), max_chars=1,
                            speaker_colors=True) is True
        content = out.read_text(encoding="utf-8-sig")
        first = _hex_to_ass_inline_color(SPEAKER_CAPTION_PALETTE[0])
        assert content.count(f"\\c{first}") == 2

    def test_word_with_no_speaker_falls_back_to_highlight_color(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" hello", 0.0, 0.5)]  # no speaker at all
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            highlight_color="#22C55E",
                            speaker_colors=True) is True
        content = out.read_text(encoding="utf-8-sig")
        assert content.count("\\c&H5EC522&") == 1

    def _transcript(self, words):
        return {"segments": [{"start": 0, "end": 99, "text": "", "words": words}]}

    def test_invalid_highlight_falls_back(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" test", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            highlight_color="#NOPE!!") is True
        content = out.read_text(encoding="utf-8-sig")
        assert "\\c&H00D7FF&" in content  # falls back to gold #FFD700

    def test_empty_range_returns_false(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" spaet", 50.0, 50.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out)) is False

    def test_ass_injection_neutralized(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" {\\b1}evil", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out)) is True
        content = out.read_text(encoding="utf-8-sig")
        assert "{\\b1}evil" not in content

    def test_glow_effect_tags(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" neon", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            effect="glow", highlight_color="#00FF88") is True
        content = out.read_text(encoding="utf-8-sig")
        assert "\\blur4" in content
        assert "\\3c&H88FF00&" in content  # glow outline in highlight color

    def test_pop_effect_animates_scale(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" pop", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out), effect="pop") is True
        content = out.read_text(encoding="utf-8-sig")
        # Gentle range: the old 75->112 pop was so wide that a frame caught
        # mid-animation read as a sizing bug rather than a beat.
        assert "\\fscx90\\fscy90" in content
        assert "\\t(0,110,\\fscx108\\fscy108" in content

    def test_pop_effect_animates_stroke_with_glyph(self, tmp_path):
        # The stroke must scale WITH the glyph so round-letter counters don't
        # over-fill with outline mid-pop (plan item 4).
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" o", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            effect="pop", border_width=4) is True
        content = out.read_text(encoding="utf-8-sig")
        # 4 * 0.9 = 3.60 -> 4 * 1.08 = 4.32: the stroke tracks the glyph scale
        # exactly instead of staying fixed.
        assert "\\fscx90\\fscy90\\bord3.60" in content
        assert "\\fscx108\\fscy108\\bord4.32" in content

    def test_wide_word_gets_horizontal_fscx_not_fs(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" really", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            uppercase=True) is True
        content = out.read_text(encoding="utf-8-sig")
        events = content.split("[Events]")[1]
        assert "\\fscx" in events
        # Height override tag \fs<digit> must never appear — \fscx/\fscy/\fsp
        # legitimately contain the "fs" letters, so match the digit form.
        assert not re.search(r"\\fs\d", events)
        assert "REALLY" in events

    def test_short_word_has_no_compaction(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" no", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            uppercase=True) is True
        content = out.read_text(encoding="utf-8-sig")
        events = content.split("[Events]")[1]
        assert "\\fscx" not in events

    def test_pop_compaction_multiplies_not_overwrites(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" really", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            uppercase=True, effect="pop") is True
        content = out.read_text(encoding="utf-8-sig")
        events = content.split("[Events]")[1]
        # The compaction \fscx must be multiplied into the pop's animation
        # range (start ~70-80, end ~80-90) — never a plain 90/108 that would
        # overwrite the squeeze, and never a \fs height change.
        assert "\\fscx7" in events or "\\fscx8" in events
        assert "\\fscy90" in events and "\\fscy108" in events
        assert not re.search(r"\\fs\d", events)

    def test_uppercase_transform(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" hallo", 0.0, 0.5), _w(" welt", 0.5, 1.0)]
        assert generate_ass(self._transcript(words), 0, 10, str(out), uppercase=True) is True
        content = out.read_text(encoding="utf-8-sig")
        assert "HALLO" in content and "WELT" in content
        assert "hallo" not in content.split("[Events]")[1]

    def test_base_opacity_dims_style_color(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" dim", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            font_color="#FFFFFF", base_opacity=0.4) is True
        content = out.read_text(encoding="utf-8-sig")
        # Dimming is fully-opaque scaled RGB (alpha would blend with the black
        # outline into muddy grey): factor 0.5 + 0.5*0.4 = 0.7 -> 0xB2
        assert "&H00B2B2B2" in content
        # no alpha-based dimming anywhere
        assert "\\1a" not in content

    def test_full_opacity_keeps_color_unchanged(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        words = [_w(" voll", 0.0, 0.5)]
        assert generate_ass(self._transcript(words), 0, 10, str(out),
                            font_color="#FFFFFF", base_opacity=1.0) is True
        content = out.read_text(encoding="utf-8-sig")
        assert "&H00FFFFFF" in content  # pure white, no dimming


class TestBurnFilterFonts:
    """The ffmpeg filter must point libass at the bundled fonts dir — without
    it every UI font choice silently falls back to DejaVu (issue #57)."""

    def _captured_cmd(self, monkeypatch, tmp_path, srt_name):
        import subtitles as m
        captured = {}

        class _Ok:
            returncode = 0
            stderr = b""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _Ok()

        monkeypatch.setattr(m.subprocess, "run", fake_run)
        srt = tmp_path / srt_name
        srt.write_text("stub", encoding="utf-8")
        m.burn_subtitles("in.mp4", str(srt), "out.mp4", font_name="Impact")
        return " ".join(str(c) for c in captured["cmd"])

    def test_srt_filter_includes_fontsdir(self, monkeypatch, tmp_path):
        cmd = self._captured_cmd(monkeypatch, tmp_path, "subs.srt")
        assert "fontsdir=" in cmd
        assert "force_style=" in cmd

    def test_ass_filter_includes_fontsdir(self, monkeypatch, tmp_path):
        cmd = self._captured_cmd(monkeypatch, tmp_path, "subs.ass")
        assert "fontsdir=" in cmd
        # ASS carries its own styles; force_style must NOT override them
        assert "force_style" not in cmd


class TestAutoCaptionDefaults:
    """The caption look every clip now ships with (chosen 25-jul-2026)."""

    def test_style_is_complete(self):
        from subtitles import AUTO_CAPTION_STYLE, generate_ass
        # border_width is deliberately NOT here — it's derived from
        # font_size at call time (see auto_stroke_width /
        # CAPTION_STROKE_RATIO), not a fixed field, so it stays proportional
        # if font_size ever changes.
        required = {"alignment", "font_name", "font_size", "font_color",
                    "highlight_color", "border_color",
                    "effect", "base_opacity", "uppercase",
                    "max_chars", "max_duration"}
        assert required <= set(AUTO_CAPTION_STYLE)

    def test_font_is_one_the_image_actually_ships(self):
        # libass falls back to DejaVu SILENTLY when the font is missing (#57),
        # so the default must be a family baked into the image (see
        # fonts/*.ttf + Dockerfile's fc-cache step).
        from subtitles import AUTO_CAPTION_STYLE
        assert AUTO_CAPTION_STYLE["font_name"] in {
            "Anton", "Poppins Black", "Poppins ExtraBold",
            "Montserrat Bold", "Montserrat ExtraBold",
            "Liberation Sans", "Liberation Serif", "DejaVu Sans"}

    def test_single_word_at_a_time(self):
        # Matches research_clips/ptb_4.mp4 (user reference, 1-aug-2026): one
        # word on screen at a time, not a multi-word block with a
        # highlighted active word — max_chars=1 forces this via
        # _collect_word_blocks (any second word always exceeds it).
        from subtitles import AUTO_CAPTION_STYLE as s
        assert s["max_chars"] == 1
        # No distinct highlight FIELD -- speaker_colors is off (see
        # test_pop_effect_and_no_speaker_colors), so highlight and body
        # color must be the same or the "active" word would visibly change
        # color for no reason.
        assert s["highlight_color"].lower() == s["font_color"].lower()

    def test_pop_effect_and_no_speaker_colors(self):
        # Refined twice same day per direct user feedback against rendered
        # samples: (1) the reference's flat/no-animation look read as
        # missing a beat, and its 6px outline read as too heavy — pop
        # restored, outline reduced (now derived, see
        # test_stroke_width_is_proportional). (2) per-speaker colour was
        # tried and explicitly rejected ("i aint look good") — back to one
        # colour for everyone.
        from subtitles import AUTO_CAPTION_STYLE as s
        assert s["effect"] == "pop"
        assert s["speaker_colors"] is False

    def test_stroke_width_is_proportional(self):
        # User spec (1-aug-2026): "8%-12% of font size," then "increase the
        # thickness a bit" the same day -> pinned to the top of that range.
        from subtitles import auto_stroke_width, CAPTION_STROKE_RATIO
        assert 0.10 <= CAPTION_STROKE_RATIO <= 0.16
        for font_size in (16, 32, 64):
            ratio = auto_stroke_width(font_size) / font_size
            assert 0.08 <= ratio <= 0.18  # allow for rounding at small sizes

    def test_letter_spacing_is_tight_and_negative(self):
        # User spec (1-aug-2026): "-2% to -4%," then "feel too apart" the
        # same day -> pulled tighter than that original range.
        from subtitles import CAPTION_LETTER_SPACING_RATIO
        assert -0.10 <= CAPTION_LETTER_SPACING_RATIO <= -0.04

    def test_captions_clear_the_platform_ui(self, tmp_path):
        from subtitles import SAFE_MARGIN_V, generate_ass
        # PlayResY is 288, so the margin must be a meaningful share of it —
        # the old hardcoded 25 (8.7%) sat under TikTok's own bottom chrome.
        assert SAFE_MARGIN_V / 288 >= 0.12
        out = tmp_path / "subs.ass"
        words = [_w(" hola", 0.0, 0.5)]
        assert generate_ass(self._t(words), 0, 10, str(out)) is True
        style_line = [l for l in out.read_text(encoding="utf-8-sig").splitlines()
                      if l.startswith("Style: Default")][0]
        assert f",10,10,{SAFE_MARGIN_V},1" in style_line

    def test_margin_is_overridable(self, tmp_path):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        assert generate_ass(self._t([_w(" hola", 0.0, 0.5)]), 0, 10, str(out),
                            margin_v=90) is True
        assert ",10,10,90,1" in out.read_text(encoding="utf-8-sig")

    @staticmethod
    def _t(words):
        return {"segments": [{"words": words}]}


class TestFilterQuoting:
    """Paths are interpolated INTO a single-quoted ffmpeg filter argument.

    Regression cover for captions silently failing in prod on 29-jul-2026: a
    clip named "Inside Earth's Most Mysterious Temple" produced an .ass path
    carrying that apostrophe, which ends the quoted argument early and kills
    the burn. Apostrophes are constant in English titles.

    The fix is NOT smarter escaping. The shell idiom "'\\''" was tried and is
    worse — ffmpeg's filtergraph parser is not a shell, so it dropped the
    apostrophe and swallowed the following ":fontsdir=" option into the
    filename. The fix is to keep apostrophes out of filter paths entirely.
    """

    def test_generated_subtitle_paths_carry_no_apostrophe(self):
        # Both generators must name their own file, never derive it from a
        # video title. This is the property that actually prevents the bug.
        import re
        src = open("main.py").read()
        m = re.search(r'ass_path = os\.path\.join\(\s*output_dir,\s*f"([^"]+)"', src)
        assert m, "auto-caption .ass path not found"
        assert "{stem}" not in m.group(1), (
            f"auto-caption .ass name derives from the clip stem: {m.group(1)}")

    def test_auto_caption_ass_name_is_unique_per_clip(self):
        # Clips render in parallel; a bare timestamp collides and lets one clip
        # burn another's captions.
        import re
        src = open("main.py").read()
        m = re.search(r'ass_path = os\.path\.join\(\s*output_dir,\s*f"([^"]+)"', src)
        assert "uuid" in m.group(1), f"not unique per clip: {m.group(1)}"

    def test_colon_is_escaped(self):
        from subtitles import _escape_ffmpeg_filter_value
        assert "\\:" in _escape_ffmpeg_filter_value("C:/out/subs.ass")

    def test_plain_path_untouched(self):
        from subtitles import _escape_ffmpeg_filter_value
        assert _escape_ffmpeg_filter_value("/out/subs_0_123.ass") == "/out/subs_0_123.ass"


class TestCaptionPlacement:
    """Per-job caption placement (CAPTION_POSITION / CAPTION_MARGIN_V).

    Position and margin are separate controls: "bottom but lifted off the edge"
    is bottom alignment with a larger MarginV, not a different alignment. The
    ASS style line ends with `...,Alignment,MarginL,MarginR,MarginV,Encoding`.
    """

    def _transcript(self):
        return {"segments": [{"start": 0, "end": 99, "text": "",
                              "words": [_w(" one", 0.0, 0.3), _w(" two", 0.3, 0.6)]}]}

    def _style_line(self, content):
        return next(l for l in content.splitlines() if l.startswith("Style: Default,"))

    def _alignment_and_margin(self, content):
        fields = self._style_line(content).split(",")
        # ...,Alignment,MarginL,MarginR,MarginV,Encoding
        return int(fields[-5]), int(fields[-2])

    def _render(self, tmp_path, **kwargs):
        from subtitles import generate_ass
        out = tmp_path / "subs.ass"
        assert generate_ass(self._transcript(), 0, 10, str(out), **kwargs) is True
        return out.read_text(encoding="utf-8-sig")

    def test_bottom_is_the_default(self, tmp_path):
        from subtitles import SAFE_MARGIN_V
        alignment, margin = self._alignment_and_margin(self._render(tmp_path))
        assert alignment == 2                      # ASS bottom-centre
        assert margin == int(SAFE_MARGIN_V)

    def test_bottom_raised_keeps_alignment_and_lifts_the_margin(self, tmp_path):
        alignment, margin = self._alignment_and_margin(
            self._render(tmp_path, alignment="bottom", margin_v=120))
        assert alignment == 2
        assert margin == 120

    def test_middle_alignment(self, tmp_path):
        alignment, _ = self._alignment_and_margin(
            self._render(tmp_path, alignment="middle"))
        assert alignment == 5

    def test_top_alignment(self, tmp_path):
        alignment, _ = self._alignment_and_margin(
            self._render(tmp_path, alignment="top"))
        assert alignment == 8

    def test_an_invalid_position_falls_back_to_bottom(self, tmp_path):
        alignment, _ = self._alignment_and_margin(
            self._render(tmp_path, alignment="sideways"))
        assert alignment == 2

    def test_margin_is_clamped_into_range(self, tmp_path):
        _, margin = self._alignment_and_margin(
            self._render(tmp_path, alignment="bottom", margin_v=9999))
        assert margin == 200
        _, margin = self._alignment_and_margin(
            self._render(tmp_path, alignment="bottom", margin_v=-50))
        assert margin == 0

    def test_auto_style_reads_the_per_job_env(self, monkeypatch):
        """app.py sets these per job; main.py runs as a fresh subprocess, so
        module-import-time resolution is per job in practice."""
        import importlib
        import subtitles

        monkeypatch.setenv("CAPTION_POSITION", "middle")
        monkeypatch.setenv("CAPTION_MARGIN_V", "120")
        reloaded = importlib.reload(subtitles)
        try:
            assert reloaded.AUTO_CAPTION_STYLE["alignment"] == "middle"
            assert reloaded.AUTO_CAPTION_STYLE["margin_v"] == 120
        finally:
            monkeypatch.delenv("CAPTION_POSITION", raising=False)
            monkeypatch.delenv("CAPTION_MARGIN_V", raising=False)
            importlib.reload(subtitles)

    def test_auto_style_defaults_without_env(self):
        import subtitles
        assert subtitles.AUTO_CAPTION_STYLE["alignment"] == "bottom"
        assert subtitles.AUTO_CAPTION_STYLE["margin_v"] == subtitles.SAFE_MARGIN_V
