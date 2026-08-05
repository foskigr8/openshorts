"""Tests for the viral-clip-finder Stage 3 engine."""

import os
import pathlib

import pytest

import viral_clip_finder as vcf


def _transcript(segments):
    return {"language": "en", "segments": segments}


def _seg(start, end, text, speaker=None, words=None):
    d = {"start": start, "end": end, "text": text}
    if speaker:
        d["speaker"] = speaker
    if words:
        d["words"] = words
    return d


def test_skill_available_with_bundled_package():
    assert vcf.skill_available() is True


def test_format_timestamp():
    assert vcf.format_timestamp(0) == "00:00:00"
    assert vcf.format_timestamp(1234.567) == "00:20:34"
    assert vcf.format_timestamp(3725.0) == "01:02:05"


def test_format_transcript_for_skill_plain():
    tr = _transcript([_seg(18.0, 42.0, "Everyone tracks hours."),
                      _seg(42.0, 50.0, "Nobody tracks decisions.")])
    out = vcf.format_transcript_for_skill(tr)
    assert "[00:00:18] Speaker: Everyone tracks hours." in out
    assert "[00:00:42] Speaker: Nobody tracks decisions." in out


def test_format_transcript_for_skill_diarized_and_named():
    tr = _transcript([_seg(0.0, 14.0, "Welcome back.", speaker="S1"),
                      _seg(14.0, 22.0, "Thanks.", speaker="S2")])
    plain = vcf.format_transcript_for_skill(tr)
    assert "[00:00:00] S1: Welcome back." in plain
    named = vcf.format_transcript_for_skill(tr, face_identities={"S1": "Joe", "S2": "Elon"})
    assert "[00:00:00] Joe: Welcome back." in named
    assert "[00:00:14] Elon: Thanks." in named


def test_format_transcript_for_skill_skips_empty_segments():
    tr = _transcript([_seg(0.0, 1.0, "   "), _seg(1.0, 2.0, "real")])
    assert "real" in vcf.format_transcript_for_skill(tr)
    assert "   " not in vcf.format_transcript_for_skill(tr)


def test_derive_keep_spans_tier_s_trim_suggestion():
    clip = {"tier": "S",
            "trim_suggestion": "Cut from 00:14:34, drop the host's question lead-in"}
    spans = vcf.derive_keep_spans(clip, 14 * 60 + 30, 14 * 60 + 58)
    assert spans == [{"start": 14 * 60 + 34, "end": 14 * 60 + 58}]


def test_derive_keep_spans_tier_s_no_trim_keeps_whole_clip():
    spans = vcf.derive_keep_spans({"tier": "S"}, 10.0, 35.0)
    assert spans == [{"start": 10.0, "end": 35.0}]


def test_derive_keep_spans_tier_l_inverts_cut_list():
    clip = {"tier": "L", "cut_list": [
        {"cut_from": 5.0, "cut_to": 8.0, "reason": "filler"},
        {"cut_from": 15.0, "cut_to": 17.0, "reason": "tangent"},
    ]}
    spans = vcf.derive_keep_spans(clip, 0.0, 20.0)
    assert spans == [{"start": 0.0, "end": 5.0}, {"start": 8.0, "end": 15.0},
                     {"start": 17.0, "end": 20.0}]


def test_derive_keep_spans_tier_l_ignores_single_small_cut():
    # A single short cut on an S-tier clip is a trim, not a cut list.
    clip = {"tier": "S", "cut_list": [{"cut_from": 5.0, "cut_to": 6.0}]}
    spans = vcf.derive_keep_spans(clip, 0.0, 20.0)
    assert spans == [{"start": 0.0, "end": 20.0}]


def test_normalize_clip_maps_fields_and_hook():
    raw = {
        "start": 870.0, "end": 898.0, "tier": "S", "tier_class": "A", "score": 87,
        "primary_pattern": "Pop-the-Balloon",
        "secondary_patterns": ["Concrete Specificity"],
        "caption_text": "YOU'RE NOT BUSY",
        "cold_open_line": "You're not busy.",
        "why_it_hits": "opens a loop and closes it",
        "score_breakdown": {"hook_strength": 14},
        "trim_suggestion": "Cut from 00:14:34, drop the host's question lead-in",
    }
    clip = vcf.normalize_clip(raw, 1200.0)
    assert clip["start"] == 870.0
    assert clip["end"] == 898.0
    assert clip["clip_type"] == "short"
    assert clip["score"] == 87
    assert clip["predicted_score"] == 87
    assert clip["viral_hook_text"] == "YOU'RE NOT BUSY"
    assert clip["keep_spans"] == [{"start": 874.0, "end": 898.0}]
    assert clip["video_title_for_youtube_short"] == "YOU'RE NOT BUSY"


def test_normalize_clip_long_tier():
    raw = {"start": 10.0, "end": 130.0, "tier": "L", "score": 85,
           "cut_list": [{"cut_from": 30.0, "cut_to": 35.0}]}
    clip = vcf.normalize_clip(raw, 300.0)
    assert clip["clip_type"] == "long_context"
    assert clip["tier"] == "L"
    assert clip["keep_spans"] == [{"start": 10.0, "end": 30.0},
                                  {"start": 35.0, "end": 130.0}]


def test_normalize_clip_clamps_out_of_range():
    clip = vcf.normalize_clip({"start": -5, "end": 9999, "score": 60}, 120.0)
    assert clip["start"] == 0.0
    assert clip["end"] == 120.0


def test_normalize_clip_enforces_min_duration():
    clip = vcf.normalize_clip({"start": 50.0, "end": 60.0, "score": 80}, 600.0)
    assert clip["end"] - clip["start"] >= 15.0


def test_normalize_clip_rejects_degenerate():
    assert vcf.normalize_clip({"start": 10.0, "end": 10.5}, 100.0) is None


def test_normalize_response_filters_tier_c_by_default():
    parsed = {
        "clips": [
            {"start": 0, "end": 20, "score": 90, "tier_class": "A"},
            {"start": 30, "end": 50, "score": 60, "tier_class": "C"},
            {"start": 60, "end": 80, "score": 75, "tier_class": "B"},
        ],
        "rejected": [{"start": 1, "end": 5, "anti_pattern": "Fake-Deep Quote"}],
    }
    clips, rejected = vcf.normalize_response(parsed, 100.0, None)
    assert [c["score"] for c in clips] == [90, 75]
    assert len(rejected) == 1
    assert rejected[0]["anti_pattern"] == "Fake-Deep Quote"


def test_normalize_response_honors_clip_count_cap():
    parsed = {"clips": [
        {"start": 0, "end": 20, "score": 70, "tier_class": "B"},
        {"start": 30, "end": 50, "score": 90, "tier_class": "A"},
        {"start": 60, "end": 80, "score": 80, "tier_class": "A"},
    ]}
    clips, _ = vcf.normalize_response(parsed, 100.0, clip_count=2)
    assert [c["score"] for c in clips] == [90, 80]
    # Kept in narrative (start-time) order after capping.
    assert clips[0]["start"] == 30.0


def test_normalize_response_keeps_borderline_when_no_strong():
    parsed = {"clips": [
        {"start": 0, "end": 20, "score": 55, "tier_class": "C"},
    ]}
    clips, _ = vcf.normalize_response(parsed, 100.0, None)
    assert len(clips) == 1


def test_provider_candidates_chain(monkeypatch):
    monkeypatch.delenv("NARRATIVE_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert vcf._provider_candidates() == []

    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "nk")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    cands = vcf._provider_candidates()
    assert [c[0] for c in cands] == ["gemini", "gemini"]
    assert cands[0][1] == "nk"
    assert cands[1][1] == "gk"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "dk")
    monkeypatch.setenv("VCF_ALLOW_DEEPSEEK", "1")
    cands = vcf._provider_candidates()
    assert [c[0] for c in cands] == ["gemini", "deepseek", "gemini"]


def test_build_system_prompt_includes_skill_and_references():
    prompt = vcf.build_system_prompt(include_long_form=False)
    assert "viral-clip-finder" in prompt
    assert "Pop-the-Balloon" in prompt or "Open Loop" in prompt
    assert "Fake-Deep Quote" in prompt
    assert "Hook Strength" in prompt
    assert "=== REFERENCE: SEAMLESS-CUTTING ===" not in prompt


def test_build_system_prompt_long_form_adds_seamless_cutting():
    prompt = vcf.build_system_prompt(include_long_form=True)
    assert "SEAMLESS-CUTTING" in prompt


def test_build_user_prompt_contains_ffmpeg_contract_and_transcript():
    prompt = vcf.build_user_prompt("hello transcript", 600.0, clip_count=4)
    assert "ABSOLUTE SECONDS" in prompt
    assert "hello transcript" in prompt
    assert "exactly up to 4" in prompt


def test_build_user_prompt_long_context_directive():
    prompt = vcf.build_user_prompt("t", 600.0, long_context_count=2)
    assert "Tier L" in prompt
    assert "cut_list" in prompt


def test_select_viral_clips_returns_none_without_provider(monkeypatch):
    monkeypatch.delenv("NARRATIVE_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    tr = _transcript([_seg(0, 10, "hello world")])
    assert vcf.select_viral_clips(tr, 600.0) is None


def test_build_user_prompt_includes_style_variant():
    prompt = vcf.build_user_prompt("t", 600.0, style_variant="high_energy")
    assert "high_energy" in prompt


# --- Cross-module contracts -------------------------------------------------
#
# The engine's output is consumed by main.py, and the shapes have to match
# exactly. The original integration returned keep_spans as [start, end] pairs
# while every consumer in main.py reads them with span.get("start") — which
# made the skill engine silently fall back to the narrative engine on every
# vision-confirmed job, and crash outright when vision was off. These tests
# pin the contract from both sides.


def test_keep_spans_use_the_pipeline_dict_contract():
    clip = vcf.normalize_clip(
        {"start": 10.0, "end": 40.0, "tier": "S", "score": 90}, 600.0)
    assert clip["keep_spans"], "a clip must always carry at least one keep span"
    for span in clip["keep_spans"]:
        # dict, not [start, end] — main.py calls span.get() on these
        assert isinstance(span, dict)
        assert set(span) == {"start", "end"}
        assert span["end"] > span["start"]


def test_keep_spans_match_the_narrative_engine_model():
    """The narrative engine's KeepSpan is the canonical shape."""
    from deepseek_worker import KeepSpan

    clip = vcf.normalize_clip(
        {"start": 0.0, "end": 120.0, "tier": "L", "score": 88,
         "cut_list": [{"cut_from": 30.0, "cut_to": 35.0},
                      {"cut_from": 60.0, "cut_to": 64.0}]}, 300.0)
    for span in clip["keep_spans"]:
        KeepSpan.model_validate(span)  # raises if the shape drifts


def _load_main_functions(*names):
    """Compile named top-level functions out of main.py without importing it.

    main.py pulls in cv2/scenedetect/mediapipe/ultralytics, which a unit-test
    environment has no business needing — but the keep_spans consumers there
    are pure Python over builtins, so we can compile just those and test
    against the REAL source. That keeps this contract honest (it breaks if
    someone edits main.py) instead of re-implementing the consumer here.
    """
    import ast
    import os

    source = pathlib.Path(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "main.py")).read_text()
    tree = ast.parse(source)
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(wanted) == len(names), (
        f"main.py no longer defines all of {names} — the keep_spans contract "
        "test needs updating")
    ns = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "main.py", "exec"), ns)
    return [ns[n] for n in names]


def test_keep_spans_survive_mains_real_consumers():
    """End-to-end against the actual functions in main.py."""
    extend, snap = _load_main_functions(
        "_extend_keep_spans_to_cover_boundaries", "_snap_keep_spans_to_words")

    clip = vcf.normalize_clip(
        {"start": 100.0, "end": 160.0, "tier": "L", "score": 90,
         "cut_list": [{"cut_from": 120.0, "cut_to": 130.0}]}, 600.0)

    # A vision rescue widens the clip; the spans must grow to cover it.
    candidate = dict(clip, start=98.0, end=162.0)
    extend(candidate)
    assert min(s["start"] for s in candidate["keep_spans"]) == 98.0
    assert max(s["end"] for s in candidate["keep_spans"]) == 162.0

    words = [{"w": "w", "s": float(t), "e": t + 0.4} for t in range(98, 163)]
    snapped = snap(candidate["keep_spans"], words,
                   candidate["start"], candidate["end"])
    assert snapped, "snapping must not drop every span"
    assert all(e > s for s, e in snapped)
    # The jump-cut really removes the cut range, and nothing else.
    kept = sum(e - s for s, e in snapped)
    assert 50.0 < kept < 60.0


def test_tier_s_clip_is_not_jump_cut_by_accident():
    """A whole-clip keep span must not trigger the jump-cut re-encode path
    (main.py: do_jump_cut = keep_spans and total_kept < (end - start) * 0.97)."""
    snap, = _load_main_functions("_snap_keep_spans_to_words")
    clip = vcf.normalize_clip(
        {"start": 10.0, "end": 40.0, "tier": "S", "score": 90}, 600.0)
    words = [{"w": "w", "s": float(t), "e": t + 0.4} for t in range(10, 41)]
    snapped = snap(clip["keep_spans"], words, clip["start"], clip["end"])
    total_kept = sum(e - s for s, e in snapped)
    assert total_kept >= (clip["end"] - clip["start"]) * 0.97


def test_normalize_clip_aliases_narrative_summary_for_vision_confirm():
    # main.py's confirm_clip_with_vision reads narrative_summary; without the
    # alias the vision judge reviews the clip with no idea what should resolve.
    clip = vcf.normalize_clip(
        {"start": 10.0, "end": 40.0, "score": 80,
         "why_it_hits": "opens a loop about the rent and closes it",
         "primary_pattern": "Pop-the-Balloon"}, 600.0)
    assert clip["narrative_summary"] == "opens a loop about the rent and closes it"
    assert clip["hook_type"] == "Pop-the-Balloon"


def test_prompt_requests_term_corrections():
    prompt = vcf.build_user_prompt("t", 600.0)
    assert "term_corrections" in prompt


def test_term_corrections_are_part_of_the_response_contract():
    parsed = {"clips": [], "rejected": [],
              "term_corrections": [{"wrong": "Clod", "correct": "Claude"}]}
    vcf.SkillResponse.model_validate(parsed)


def test_overlapping_candidates_keep_the_higher_score():
    parsed = {"clips": [
        {"start": 10.0, "end": 40.0, "score": 72, "tier_class": "B"},
        {"start": 35.0, "end": 65.0, "score": 91, "tier_class": "A"},
        {"start": 90.0, "end": 120.0, "score": 80, "tier_class": "B"},
    ]}
    clips, _ = vcf.normalize_response(parsed, 600.0, None)
    assert [c["start"] for c in clips] == [35.0, 90.0]


def test_lean_references_shrink_the_prompt(monkeypatch):
    monkeypatch.delenv("VCF_REFERENCES", raising=False)
    full = vcf.build_system_prompt()
    monkeypatch.setenv("VCF_REFERENCES", "lean")
    lean = vcf.build_system_prompt()
    assert len(lean) < len(full)
    assert "SCORING-RUBRIC" in lean and "NICHES" not in lean


def test_long_form_reference_is_only_loaded_for_long_clips(monkeypatch):
    monkeypatch.delenv("VCF_REFERENCES", raising=False)
    monkeypatch.delenv("VCF_LONG_FORM_REFS", raising=False)
    assert "SEAMLESS-CUTTING" not in vcf.build_system_prompt(include_long_form=False)
    assert "SEAMLESS-CUTTING" in vcf.build_system_prompt(include_long_form=True)
