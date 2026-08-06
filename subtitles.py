import os
import re
import subprocess
import sys

from ffmpeg_utils import video_encode_args, QUALITY, METADATA_SCRUB
from reframe_v2 import UNIFIED_CONTENT_HEIGHT_RATIO


_STDIO_CONFIGURED = False

# Shared faster-whisper config so both transcription paths (this module and
# main.transcribe_video) behave identically. "small" is meaningfully better at
# German than "base" without being much slower on CPU.
DEFAULT_WHISPER_MODEL = "small"


def get_whisper_config():
    """Return the faster-whisper model config, overridable via env vars."""
    return {
        "model_size": os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        "device": os.environ.get("WHISPER_DEVICE", "cpu"),
        "compute_type": os.environ.get("WHISPER_COMPUTE", "int8"),
    }


# Decode params shared by both transcription paths. condition_on_previous_text
# is off to avoid repetition/hallucination loops; vad_filter drops silence.
WHISPER_TRANSCRIBE_PARAMS = {
    "beam_size": 5,
    "vad_filter": True,
    "condition_on_previous_text": False,
    "word_timestamps": True,
}


def merge_continuation_words(words):
    """Merge faster-whisper continuation fragments into their base word.

    faster-whisper marks a word boundary with a LEADING SPACE on each token.
    Compound-word fragments (e.g. "-Kanal.", ".200") arrive WITHOUT a leading
    space and belong to the preceding word. Without merging, "YouTube" and
    "-Kanal." get space-joined into "YouTube -Kanal." or split across subtitle
    blocks. We concatenate such fragments onto the previous word and extend its
    end time. Normal words keep their leading space, so real word boundaries
    (e.g. "ich habe") are never glued together.

    Returns a new list; the input dicts are not mutated.
    """
    merged = []
    for word in words:
        text = word.get("word", "")
        if merged and isinstance(text, str) and text and not text.startswith(" "):
            prev = merged[-1]
            prev["word"] = f"{prev.get('word', '')}{text}"
            # Continuation fragments carry word-level metadata (e.g. the
            # sentence-final flag stamped by _collect_word_blocks); propagate
            # it so a merged "YouTube-Kanal." keeps its end-of-sentence status.
            if word.get("_sentence_final"):
                prev["_sentence_final"] = True
            if word.get("end") is not None:
                prev["end"] = word["end"]
        else:
            merged.append(dict(word))
    return merged


def _configure_stdio():
    global _STDIO_CONFIGURED
    if _STDIO_CONFIGURED:
        return
    _STDIO_CONFIGURED = True
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _log(message):
    _configure_stdio()
    stream = sys.stdout
    text = str(message)
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text + "\n")
    stream.flush()


def _escape_ffmpeg_filter_value(value):
    """Escape a path/value for use inside a quoted FFmpeg filter argument.

    NOTE: an apostrophe in the path cannot be made safe here. ffmpeg's
    filtergraph parser is not a shell — the shell idiom ``'\\''`` was tried on
    29-jul-2026 and is worse than doing nothing: it drops the apostrophe AND
    swallows the following option, so ``ass='…Earth'\\''s.ass':fontsdir='…'``
    resolved to a filename of "…Earths.ass:fontsdir=…" and failed to open.

    The only reliable answer is to keep apostrophes OUT of any path that is
    interpolated into a filter. Callers generate their own subtitle filenames,
    so they control this: use a neutral name (``subs_<i>_<ts>.ass``), never one
    derived from a video title.
    """
    return value.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")


def _normalize_subtitle_word(value):
    return " ".join(str(value or "").split())


# --- Caption word punctuation (plan item 4) -------------------------------
# Transcript tokens carry noise punctuation ("right," "yes." mid-sentence).
# Tasteful default: commas/semicolons/colons are always stripped; sentence-
# ending ./?/! survive only on the genuinely last word of a segment, where
# they give the one-word-at-a-time caption a real "sentence end" beat.
_PUNCT_ALWAYS_STRIP = ",;:"
_SENTENCE_END_RE = re.compile(r'([.!?])[.!?]*$')


def _clean_caption_word(value, sentence_final):
    """Whitespace-normalize a transcript word and strip punctuation noise."""
    text = " ".join(str(value or "").split())
    for ch in _PUNCT_ALWAYS_STRIP:
        text = text.replace(ch, "")
    if sentence_final:
        # Keep ONE trailing sentence-ending mark; collapse runs like "!!".
        m = _SENTENCE_END_RE.search(text)
        if m:
            text = text[:m.start()] + m.group(1)
    else:
        # Mid-sentence: drop any trailing ./?/! the token happened to carry.
        text = text.rstrip(".?!")
    return text


def _is_sentence_final_word(segment, idx):
    """True when ``idx`` is the last word of its segment (word-level data)."""
    return idx == len(segment.get('words', []) or []) - 1


def transcribe_audio(video_path):
    """
    Transcribe audio from a video file via the configured ASR backend.
    Returns transcript in the same format as main.py for compatibility.
    """
    # Lazy import: transcribe_backends imports helpers from this module.
    from transcribe_backends import transcribe_media

    _log(f"🎙️  Transcribing audio from: {video_path}")
    transcript = transcribe_media(video_path)
    _log(f"✅ Transcription complete. Language: {transcript['language']}")
    return transcript


def generate_srt_from_video(video_path, output_path, max_chars=20, max_duration=2.0,
                            style="classic", **style_opts):
    """
    Transcribe a video and generate a subtitle file directly (SRT, or karaoke
    ASS when style="karaoke"). Used for dubbed videos without a transcript.
    """
    transcript = transcribe_audio(video_path)

    # Get video duration to use as clip_end
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    cap.release()

    if style == "karaoke":
        return generate_ass(transcript, 0, duration, output_path, max_chars, max_duration, **style_opts)
    return generate_srt(transcript, 0, duration, output_path, max_chars, max_duration)


def _collect_word_blocks(transcript, clip_start, clip_end, max_chars=20, max_duration=2.0):
    """
    Flatten transcript words for a clip range and group them into short blocks
    suitable for vertical video. Returns a list of blocks; each block is a list
    of {'word', 'start', 'end'} dicts with times relative to the clip.

    Continuation fragments are merged defensively here too, because transcripts
    from old jobs on disk store unmerged tokens (the leading space is still
    present, so the boundary signal survives).
    """
    flat_words = []
    for segment in transcript.get('segments', []):
        # AssemblyAI/diarized transcripts carry 'speaker' on the SEGMENT, not
        # per word — stamp it onto each word here so it survives flattening,
        # for per-speaker caption colour (see generate_ass).
        seg_speaker = segment.get('speaker')
        seg_words = segment.get('words', []) or []
        for idx, w in enumerate(seg_words):
            word_entry = dict(w, speaker=w.get('speaker', seg_speaker))
            if _is_sentence_final_word(segment, idx):
                word_entry['_sentence_final'] = True
            flat_words.append(word_entry)
    flat_words = merge_continuation_words(flat_words)

    words = []
    for word_info in flat_words:
        if word_info.get('end', 0) > clip_start and word_info.get('start', 0) < clip_end:
            cleaned_word = _clean_caption_word(
                word_info.get('word', ''),
                sentence_final=bool(word_info.get('_sentence_final')))
            if not cleaned_word:
                continue
            words.append({
                'word': cleaned_word,
                'start': max(0, word_info['start'] - clip_start),
                'end': max(0, word_info['end'] - clip_start),
                'speaker': word_info.get('speaker'),
            })

    blocks = []
    current_block = []
    block_start = None

    for word in words:
        if not current_block:
            current_block = [word]
            block_start = word['start']
            continue

        current_text_len = sum(len(w['word']) + 1 for w in current_block)
        duration = word['end'] - block_start

        if current_text_len + len(word['word']) > max_chars or duration > max_duration:
            blocks.append(current_block)
            current_block = [word]
            block_start = word['start']
        else:
            current_block.append(word)

    if current_block:
        blocks.append(current_block)
    return blocks


def generate_srt(transcript, clip_start, clip_end, output_path, max_chars=20, max_duration=2.0):
    """
    Generates an SRT file from the transcript for a specific time range.
    Groups words into short lines suitable for vertical video.
    """
    blocks = _collect_word_blocks(transcript, clip_start, clip_end, max_chars, max_duration)
    if not blocks:
        return False

    srt_content = ""
    for index, block in enumerate(blocks, 1):
        text = " ".join(w['word'] for w in block).strip()
        srt_content += format_srt_block(index, block[0]['start'], block[-1]['end'], text)

    # Write UTF-8 with BOM so Windows/FFmpeg subtitle readers reliably detect Unicode text.
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(srt_content)

    return True


# Vertical margin for burned captions, in PlayResY=288 units (so ~15% of the
# frame height). The old hardcoded 25 (8.7%) put captions underneath TikTok's
# and Reels' own bottom UI — the caption/username block and the music ticker —
# where they were partly covered on the platform even though the exported file
# looked fine.
def _clamp_number(value, lo, hi, default):
    """Coerce value to float and clamp to [lo, hi]; use default if not numeric."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)
    return max(lo, min(hi, num))


SAFE_MARGIN_V = 43

# How far up from the bottom of the (letterboxed) content box captions sit,
# as a fraction of UNIFIED_CONTENT_HEIGHT_RATIO. Raised from 0.15 to 0.36
# (user reference, 1-aug-2026: research_clips/ptb_4.mp4) — that reference
# show's captions sit noticeably higher, roughly two-fifths up the frame
# rather than hugging the very bottom edge.
CAPTION_CONTENT_INSET_RATIO = 0.36


# The caption look applied automatically to every generated clip.
# Base structure matches research_clips/ptb_4.mp4 (user reference,
# 1-aug-2026): one word on screen at a time (max_chars=1 forces single-word
# blocks — see _collect_word_blocks). Superseded the prior 25-jul-2026
# A/B-tested "yellow active word, gentle pop, multi-word block" look
# wholesale per explicit user direction, not a refinement of it.
#
# Iterated twice same day on direct user feedback against rendered samples:
# (1) pop restored (the reference's flat/no-animation look read as missing
# a beat once seen in motion), outline brought down from 6px. (2) per-
# speaker caption colour (added, then explicitly rejected — "i aint look
# good") — REMOVED; back to one colour for everyone. Font switched to
# bundled "Montserrat ExtraBold" (fonts/Montserrat-ExtraBold.ttf,
# OFL-licensed — the Google Fonts repo only ships Montserrat as a variable
# font, so this is a static wght=800 instance cut with fonttools'
# instancer) per explicit spec: stroke = CAPTION_STROKE_RATIO of font size
# (8-12% requested; 10% chosen as the midpoint), letter-spacing =
# CAPTION_LETTER_SPACING_RATIO of font size (-2% to -4% requested; -3%
# chosen as the midpoint) — see their use in generate_ass.
AUTO_CAPTION_STYLE = {
    "style": "karaoke",
    # Where the burned-in captions sit. "bottom" (default), "middle" or "top" —
    # the same three the Subtitle modal offers for restyling a finished clip
    # (see align_map in generate_ass). Settable per deployment with
    # CAPTION_POSITION so a Kaggle/self-host run can choose up front instead of
    # having to restyle every clip afterwards.
    "alignment": (os.environ.get("CAPTION_POSITION", "").strip().lower()
                  or "bottom"),
    # How far up from the frame edge, in ASS MarginV. This is a SEPARATE
    # control from alignment: "bottom but lifted off the edge" is bottom
    # alignment with a larger margin, not a different alignment. Only
    # meaningful for bottom alignment — generate_ass computes a per-line
    # MarginV only when ass_alignment == 2 — so the UI offers it there alone.
    # app.py sets CAPTION_MARGIN_V per job from the submission form.
    "margin_v": _clamp_number(os.environ.get("CAPTION_MARGIN_V", ""),
                              0, 200, SAFE_MARGIN_V),
    "font_name": "Montserrat ExtraBold",
    # 44 was tuned for the old multi-word-block style, where several words
    # shared one line. With a single word filling the whole line now, the
    # same size rendered oversized (ground-truthed 1-aug-2026: "CAN" spanned
    # ~65% of frame width vs. ~40% for a similarly short word in ptb_4.mp4).
    "font_size": 32,
    "font_color": "#FFFFFF",
    "highlight_color": "#FFFFFF",
    "border_color": "#000000",
    "effect": "pop",
    "base_opacity": 1.0,
    "uppercase": True,
    "max_chars": 1,
    "max_duration": 1.4,
    "speaker_colors": False,
}

# Stroke/outline width as a fraction of font size. Original spec (1-aug-2026)
# was "8% to 12%" (10% chosen as midpoint); bumped to the top of that range
# same day per direct feedback ("increase the thickness a bit"). Computed
# dynamically (see auto_stroke_width) rather than a fixed px value, so the
# outline stays proportional if font_size ever changes.
CAPTION_STROKE_RATIO = 0.12

# Letter-spacing as a fraction of font size. Original spec (1-aug-2026) was
# "-2% to -4%" (-3% chosen as midpoint); tightened further same day per
# direct feedback ("feel too apart" — negative Spacing PULLS characters
# together, so "too apart" means not enough pull yet, not too much).
CAPTION_LETTER_SPACING_RATIO = -0.06

# Max rendered width of one caption word, in em (multiples of the base font
# size). Words wider than this auto-compact HORIZONTALLY ONLY (inline \fscx +
# tighter \fsp — never a per-word \fs height change, which made long words
# look like a different caption style entirely; round-2 feedback). Measured
# against the bundled Montserrat ExtraBold advances: "STOP" is 2.87em (~48%
# of a 9:16 frame at the default 32px style), so 3.3em keeps short/medium
# words at full size and only long words (e.g. "REALLY" at 4.1em) squeeze.
CAPTION_MAX_WIDTH_EM = 3.3


def auto_stroke_width(font_size):
    """Outline width for a given font size, per CAPTION_STROKE_RATIO."""
    return max(1, round(font_size * CAPTION_STROKE_RATIO))

# Distinct, saturated colours assigned to diarized speakers in first-seen
# order (see generate_ass's speaker_colors param) — a caption-only viewer
# (sound off, or just skimming) can tell who's talking from the caption
# colour alone, not just from the framing. White first so a single-speaker
# clip (or the first/primary speaker) keeps the plain look; the rest are
# chosen to stay readable on a thick black outline over arbitrary footage.
SPEAKER_CAPTION_PALETTE = [
    "#FFFFFF",  # first speaker — plain white, matches the no-diarization look
    "#FFE066",  # warm yellow
    "#5CD6FF",  # cyan
    "#FF6EC7",  # pink
    "#8CFF7A",  # green
]


def _ass_time(seconds):
    """Format seconds as ASS timestamp H:MM:SS.cc (centiseconds)."""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _hex_to_ass_inline_color(hex_color, fallback="FFFFFF"):
    """Convert #RRGGBB to the &HBBGGRR& form used by inline \\c override tags."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    r = hex_digits[0:2]
    g = hex_digits[2:4]
    b = hex_digits[4:6]
    return f"&H{b}{g}{r}&".upper()


def _escape_ass_text(text):
    """Neutralize characters that would start ASS override blocks."""
    return str(text).replace('\\', '/').replace('{', '(').replace('}', ')')


def _dim_hex_color(hex_color, opacity, fallback="FFFFFF"):
    """Fully-opaque 'dimmed' variant of a color (scaled toward black).

    Dimming via alpha looks muddy in ASS: libass draws the outline as a
    filled shape UNDER the fill, so a semi-transparent white fill blends
    with its own black outline into dark grey. Scaling the RGB instead
    keeps the text crisp on every player."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    # Gentle curve: even strong dimming stays a readable light silver, matching
    # the airy look of browser-alpha dimming over bright video.
    factor = 0.5 + 0.5 * _clamp_number(opacity, 0.05, 1.0, 1.0)
    r = min(255, round(int(hex_digits[0:2], 16) * factor))
    g = min(255, round(int(hex_digits[2:4], 16) * factor))
    b = min(255, round(int(hex_digits[4:6], 16) * factor))
    return f"{r:02X}{g:02X}{b:02X}"


def generate_ass(transcript, clip_start, clip_end, output_path,
                 max_chars=20, max_duration=2.0, alignment='bottom',
                 fontsize=16, font_name="Verdana", font_color="#FFFFFF",
                 border_color="#000000", border_width=2,
                 highlight_color="#FFD700", bg_color="#000000", bg_opacity=0.0,
                 effect="none", base_opacity=1.0, uppercase=False,
                 margin_v=SAFE_MARGIN_V, general_ranges=None,
                 speaker_colors=False, letter_spacing_ratio=0.0):
    """
    Generates a karaoke-style ASS file: each block is shown like the SRT path,
    but the currently spoken word is rendered in highlight_color (modern
    TikTok/CapCut caption look). One dialogue event per word, back to back, so
    the highlight moves with the audio without flicker.

    effect: "none" | "glow" (neon shine around the active word) |
            "pop" (active word scales up) | "box" (thick colored outline).
    base_opacity: opacity of the non-active words — dimmed base text is the
    modern captioneer look (e.g. 0.4).
    speaker_colors: when True and the transcript carries diarized 'speaker'
    labels (see _collect_word_blocks), each speaker's words render in their
    own colour from SPEAKER_CAPTION_PALETTE (assigned by first-seen order)
    instead of the single highlight_color — lets a viewer track who's
    talking from captions alone. A word with no speaker label (or when this
    is False) falls back to highlight_color, unchanged from before.
    letter_spacing_ratio: fraction of fontsize applied as ASS Style
    "Spacing" (negative = tighter). See CAPTION_LETTER_SPACING_RATIO.
    general_ranges: list of (clip-relative start, end) seconds where the
    reframe engine's content box is smaller than the full frame — the
    unified 3:4-consistent crop (reframe_v2.UNIFIED_CROP_RATIO) letterboxes
    into a blurred fill above and below (see reframe_v2.unified_filtergraph),
    for every frame now, so render() passes the whole clip range here.
    The default margin positions captions near the bottom of a FULL-height
    frame, which leaves them floating in the blur band below the actual
    (smaller, centered) content box — disconnected from the footage, reading
    as a second stacked panel (confirmed on real delivered clips,
    31-jul-2026). Events whose start time falls in one of these ranges get a
    per-line MarginV override that lands them in the bottom slice of the
    actual content box instead.
    """
    blocks = _collect_word_blocks(transcript, clip_start, clip_end, max_chars, max_duration)
    if not blocks:
        return False

    # Caption margin for the letterboxed content box: computed as a ratio of
    # PlayResY, so it's resolution-independent like everything else in this
    # file. The content box is vertically centered at
    # UNIFIED_CONTENT_HEIGHT_RATIO of the frame height (see
    # reframe_v2.unified_filtergraph) — this places the caption
    # CAPTION_CONTENT_INSET_RATIO up from the bottom of that box.
    PLAY_RES_Y = 288
    content_bottom_ratio = (1 - UNIFIED_CONTENT_HEIGHT_RATIO) / 2 + UNIFIED_CONTENT_HEIGHT_RATIO
    caption_inset_ratio = CAPTION_CONTENT_INSET_RATIO * UNIFIED_CONTENT_HEIGHT_RATIO
    general_margin_v = round(((1 - content_bottom_ratio) + caption_inset_ratio) * PLAY_RES_Y)
    # Respect a caller who asked for MORE lift than the content box needs.
    # This margin exists to keep captions off the blurred fill during GENERAL
    # scenes, so it is a FLOOR, not a fixed value — before this, a per-job
    # "bottom, raised" choice was silently discarded on every general-layout
    # line, which is what "I set the caption position and only centre happens"
    # looks like from the outside.
    general_margin_v = max(general_margin_v,
                           int(_clamp_number(margin_v, 0, 200, SAFE_MARGIN_V)))
    general_ranges = general_ranges or []

    def _in_general_range(t):
        return any(gstart <= t < gend for gstart, gend in general_ranges)

    # Match the SRT burn path: PlayResY 288 keeps font sizes consistent.
    final_fontsize = int(_clamp_number(fontsize, 10, 200, 16) * 0.85)
    if final_fontsize < 10:
        final_fontsize = 10

    # ASS Spacing is in the same units as font size -- scale letter-spacing
    # off the FINAL (already-scaled) size so it stays proportional to what
    # actually renders, not the pre-scale input.
    letter_spacing = round(final_fontsize * _clamp_number(letter_spacing_ratio, -0.5, 0.5, 0.0), 2)

    align_map = {'top': 8, 'middle': 5, 'bottom': 2}
    ass_alignment = align_map.get(str(alignment).lower(), 2)

    safe_font = _sanitize_font_name(font_name)
    base_opacity = _clamp_number(base_opacity, 0.05, 1.0, 1.0)
    # Dim inactive words via a fully-opaque scaled color (NOT alpha — see
    # _dim_hex_color); the active word overrides the color inline.
    primary_colour = hex_to_ass_color(_dim_hex_color(font_color, base_opacity), 1.0)
    bg_opacity = _clamp_number(bg_opacity, 0.0, 1.0, 0.0)
    border_width = _clamp_number(border_width, 0, 10, 2)

    if bg_opacity > 0:
        border_style = 3
        outline_colour = hex_to_ass_color(bg_color, bg_opacity, fallback="000000")
        outline_width = 1
    else:
        border_style = 1
        outline_colour = hex_to_ass_color(border_color, 1.0, fallback="000000")
        outline_width = max(1, int(border_width))

    back_colour = hex_to_ass_color("#000000", 0.0)

    # Per-speaker colour assignment (first-seen order across the blocks
    # actually being rendered, so it's stable for this clip regardless of
    # what other speakers exist earlier/later in the full source transcript).
    speaker_color_map = {}
    if speaker_colors:
        for block in blocks:
            for word in block:
                spk = word.get('speaker')
                if spk is not None and spk not in speaker_color_map:
                    idx = len(speaker_color_map) % len(SPEAKER_CAPTION_PALETTE)
                    speaker_color_map[spk] = SPEAKER_CAPTION_PALETTE[idx]

    def _active_prefix_for(word, width_compaction=None):
        """Inline override tag for this word's active-state rendering —
        colour depends on its diarized speaker (falls back to
        highlight_color when unknown/disabled). {\\r} after it (added by the
        caller) resets to the (dimmed) base style so the rest of the block
        stays untouched. width_compaction (fscx_pct, fsp_extra) from
        _caption_width_compaction squeezes a wide word HORIZONTALLY ONLY —
        the height is never touched (round-2 feedback). For the pop effect
        the horizontal scale is MULTIPLIED into the animation range so the
        compaction composes with the beat instead of fighting it."""
        color_hex = speaker_color_map.get(word.get('speaker'), highlight_color)
        highlight_inline = _hex_to_ass_inline_color(color_hex, fallback="FFD700")
        width_tag = ""
        if width_compaction is not None:
            fscx_pct, fsp_extra = width_compaction
            width_tag = f"\\fscx{fscx_pct}"
            if fsp_extra:
                width_tag += f"\\fsp{fsp_extra}"
        if effect == "glow":
            glow_bord = max(3, int(outline_width) + 2)
            return (f"{{\\c&HFFFFFF&\\3c{highlight_inline}{width_tag}"
                    f"\\bord{glow_bord}\\blur4}}")
        elif effect == "box":
            box_bord = max(4, int(outline_width) + 3)
            return (f"{{\\c&HFFFFFF&\\3c{highlight_inline}{width_tag}"
                    f"\\bord{box_bord}\\blur0}}")
        elif effect == "pop":
            # Gentle pop. The old 75->112 range started the word so small
            # that any frame caught mid-animation read as a sizing bug
            # rather than a beat. The stroke width animates WITH the glyph
            # (0.9x -> 1.08x) so round-letter counters don't over-fill with
            # outline during the pop — the fixed-\bord version over-thickened
            # relative to the scaled glyph (plan item 4). Fractional \bord is
            # legal in ASS and keeps the ratio exact at small stroke widths
            # where integer rounding would erase the whole animation.
            small_bord = max(1.0, outline_width * 0.9)
            big_bord = max(1.0, outline_width * 1.08)
            if width_compaction is not None:
                fscx_pct, fsp_extra = width_compaction
                # Compose: pop animates the RELATIVE 90->108 scale around the
                # word's compaction width (which stays the same before/after).
                pop_start = max(1, round(fscx_pct * 0.9))
                pop_end = round(fscx_pct * 1.08)
                fsp_tag = f"\\fsp{fsp_extra}" if fsp_extra else ""
                return (f"{{\\c{highlight_inline}{fsp_tag}"
                        f"\\fscx{pop_start}\\fscy90\\bord{small_bord:.2f}"
                        f"\\t(0,110,\\fscx{pop_end}\\fscy108\\bord{big_bord:.2f})}}")
            return (f"{{\\c{highlight_inline}"
                    f"\\fscx90\\fscy90\\bord{small_bord:.2f}"
                    f"\\t(0,110,\\fscx108\\fscy108\\bord{big_bord:.2f})}}")
        else:
            return f"{{\\c{highlight_inline}{width_tag}}}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResY: 288\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{safe_font},{final_fontsize},{primary_colour},{primary_colour},"
        f"{outline_colour},{back_colour},1,0,0,0,100,100,{letter_spacing},0,{border_style},"
        f"{outline_width},0,{ass_alignment},10,10,{int(_clamp_number(margin_v, 0, 200, SAFE_MARGIN_V))},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for block in blocks:
        for i, word in enumerate(block):
            # Event runs until the next word starts (no flicker in gaps);
            # the last word holds until the block ends.
            ev_start = block[0]['start'] if i == 0 else word['start']
            ev_end = block[i + 1]['start'] if i < len(block) - 1 else block[-1]['end']
            if ev_end <= ev_start:
                continue

            parts = []
            for j, other in enumerate(block):
                text = _escape_ass_text(other['word'])
                if uppercase:
                    text = text.upper()
                compaction = _caption_width_compaction(
                    text, font_name, final_fontsize)
                if compaction == (None, None):
                    compaction = None
                if j == i:
                    parts.append(
                        f"{_active_prefix_for(other, width_compaction=compaction)}{text}{{\\r}}")
                else:
                    if compaction is not None:
                        # Wide non-active word: wrap so it squeezes too, then
                        # reset to base style before the next part.
                        fscx_pct, fsp_extra = compaction
                        fsp_tag = f"\\fsp{fsp_extra}" if fsp_extra else ""
                        parts.append(f"{{\\fscx{fscx_pct}{fsp_tag}}}{text}{{\\r}}")
                    else:
                        parts.append(text)

            # Per-line MarginV override: only meaningful for bottom alignment,
            # where MarginV is measured up from the frame's bottom edge.
            line_margin_v = general_margin_v if (ass_alignment == 2 and _in_general_range(ev_start)) else 0

            events.append(
                f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Default,,0,0,{line_margin_v},,{' '.join(parts)}"
            )

    if not events:
        return False

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(header + "\n".join(events) + "\n")

    return True

def format_srt_block(index, start, end, text):
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
    return f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"

_HEX_COLOR_RE = re.compile(r'^[0-9A-Fa-f]{6}$')
_FONT_NAME_RE = re.compile(r'[^A-Za-z0-9 _-]')


def hex_to_ass_color(hex_color, opacity=1.0, fallback="FFFFFF"):
    """Convert #RRGGBB to ASS &HAABBGGRR format. opacity: 0.0=transparent, 1.0=opaque.

    Invalid hex (e.g. "#GGGGGG", None, wrong length) falls back to `fallback`
    instead of raising, so a bad color from the client can't 500 the request.
    """
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    opacity = _clamp_number(opacity, 0.0, 1.0, 1.0)
    r = int(hex_digits[0:2], 16)
    g = int(hex_digits[2:4], 16)
    b = int(hex_digits[4:6], 16)
    alpha = round((1.0 - opacity) * 255)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _sanitize_font_name(name):
    """Strip anything but [A-Za-z0-9 _-] so the font name can't inject extra
    ASS override fields (commas/braces/backslashes) into force_style."""
    cleaned = _FONT_NAME_RE.sub('', str(name or '')).strip()
    return cleaned or "Verdana"


# --- Per-word width-aware font sizing (plan item 4) -----------------------
# One-word captions render the word at the full style font size, so a wide
# word (or a very long one) can stretch across — or overflow — the frame.
# Estimate each word's rendered width from the bundled font's real glyph
# advances (em units) and shrink that word's inline \fs when it would exceed
# CAPTION_MAX_WIDTH_EM. Falls back to a generic uppercase-bold table when the
# font isn't bundled (e.g. a user-selected system font in the subtitle modal).
_FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Rough advances (em) for a typical bold sans in uppercase; used only when
# the requested font has no bundled metrics. Average bold-sans cap is ~0.72.
_GENERIC_ADVANCE_MAP = {
    **{c: 0.72 for c in "ABCDEFGHJKLNOPQRSTUVXYZ"},
    "I": 0.34, "J": 0.55, "M": 0.95, "W": 1.18, "Q": 0.80,
    **{c: 0.55 for c in "abcdefghijklmnopqrstuvwxyz"},
    " ": 0.30, ".": 0.30, "!": 0.30, "?": 0.62, "-": 0.42, "'": 0.20,
    "0": 0.68, "1": 0.42, "2": 0.68, "3": 0.68, "4": 0.68,
    "5": 0.68, "6": 0.68, "7": 0.62, "8": 0.68, "9": 0.68,
}

_font_advance_cache = {}


def _glyph_advance_em_map(font_name):
    """{char: advance width in em} for the bundled font, else generic map."""
    safe = _sanitize_font_name(font_name)
    if safe in _font_advance_cache:
        return _font_advance_cache[safe]
    advance_map = None
    try:
        from fontTools.ttLib import TTFont
        if os.path.isdir(_FONTS_DIR):
            for entry in os.listdir(_FONTS_DIR):
                if not entry.lower().endswith((".ttf", ".otf")):
                    continue
                if os.path.splitext(entry)[0].lower() == safe.lower():
                    font = TTFont(os.path.join(_FONTS_DIR, entry))
                    cmap = font.getBestCmap()
                    hmtx = font["hmtx"]
                    upem = font["head"].unitsPerEm or 1000
                    advance_map = {
                        chr(code): hmtx[glyph][0] / upem
                        for code, glyph in cmap.items()
                        if glyph in hmtx
                    }
                    break
    except Exception:
        advance_map = None
    if advance_map is None:
        advance_map = dict(_GENERIC_ADVANCE_MAP)
    _font_advance_cache[safe] = advance_map
    return advance_map


def _estimated_word_width_em(text, font_name):
    """Summed glyph advances for ``text``, in em of the given font."""
    advances = _glyph_advance_em_map(font_name)
    return sum(advances.get(ch, 0.72) for ch in str(text or ""))


def _caption_width_compaction(text, font_name, base_fontsize,
                              max_width_em=CAPTION_MAX_WIDTH_EM,
                              min_fscx_pct=70):
    """Horizontal-only width compaction for a too-wide word.

    Returns ``(fscx_pct, fsp_extra)`` or ``(None, None)`` when the word fits.
    ``fscx_pct`` is a horizontal scale < 100 (floor-clamped at 70%); the
    word's HEIGHT is never touched — round-2 feedback was that per-word
    ``\\fs`` changes made long words look like a different caption style.
    ``fsp_extra`` is an additional negative letter-spacing (None for mild
    cases) applied before the squeeze reaches its floor, so a slightly-wide
    word is pulled together rather than visibly squashed.
    """
    width_em = _estimated_word_width_em(text, font_name)
    if width_em <= max_width_em:
        return None, None
    ratio = max_width_em / width_em
    fscx_pct = max(min_fscx_pct, round(ratio * 100))
    fsp_extra = -max(1, round(base_fontsize * 0.02)) if fscx_pct < 95 else None
    return fscx_pct, fsp_extra


def burn_subtitles(video_path, srt_path, output_path, alignment=2, fontsize=16,
                   font_name="Verdana", font_color="#FFFFFF",
                   border_color="#000000", border_width=2,
                   bg_color="#000000", bg_opacity=0.0):
    """
    Burns subtitles into the video using FFmpeg.
    Supports two modes:
    - Outline mode (bg_opacity=0): Text with colored outline/border
    - Box mode (bg_opacity>0): Text with semi-transparent background box
    """
    # Position mapping
    ass_alignment = 2
    align_lower = str(alignment).lower()
    if align_lower == 'top':
        ass_alignment = 6
    elif align_lower == 'middle':
        ass_alignment = 10
    elif align_lower == 'bottom':
        ass_alignment = 2

    # Font size scaling for ASS virtual resolution (PlayResY=288 default)
    # For vertical 1080x1920 video, we need larger text for readability
    final_fontsize = int(_clamp_number(fontsize, 10, 200, 16) * 0.85)
    if final_fontsize < 10:
        final_fontsize = 10

    safe_font_name = _sanitize_font_name(font_name)
    bg_opacity = _clamp_number(bg_opacity, 0.0, 1.0, 0.0)
    border_width = _clamp_number(border_width, 0, 10, 2)

    # Path handling for FFmpeg filter syntax
    safe_srt_path = _escape_ffmpeg_filter_value(srt_path)

    # Convert colors to ASS format and build style
    primary_colour = hex_to_ass_color(font_color, 1.0)

    if bg_opacity > 0:
        # Box mode: opaque background box
        border_style = 3
        outline_colour = hex_to_ass_color(bg_color, bg_opacity, fallback="000000")
        outline_width = 1
    else:
        # Outline mode: text border/outline
        border_style = 1
        outline_colour = hex_to_ass_color(border_color, 1.0, fallback="000000")
        outline_width = max(1, int(border_width))

    back_colour = hex_to_ass_color("#000000", 0.0)

    style_string = (
        f"Alignment={ass_alignment},"
        f"Fontname={safe_font_name},"
        f"Fontsize={final_fontsize},"
        f"PrimaryColour={primary_colour},"
        f"OutlineColour={outline_colour},"
        f"BackColour={back_colour},"
        f"BorderStyle={border_style},"
        f"Outline={outline_width},"
        f"Shadow=0,"
        f"MarginV={SAFE_MARGIN_V},"
        f"Bold=1"
    )

    # Let libass see the fonts bundled with the app (e.g. Anton for Impact)
    # even when the system fontconfig has no cache for them.
    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    safe_fonts_dir = _escape_ffmpeg_filter_value(fonts_dir)

    if str(srt_path).lower().endswith('.ass'):
        # ASS files (karaoke style) carry their own styles; force_style would
        # override the per-word color tags.
        vf = f"ass='{safe_srt_path}':fontsdir='{safe_fonts_dir}'"
    else:
        vf = (f"subtitles='{safe_srt_path}':fontsdir='{safe_fonts_dir}'"
              f":charenc=UTF-8:force_style='{style_string}'")

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', vf,
        '-c:a', 'copy',
        *video_encode_args(QUALITY),
        *METADATA_SCRUB,
        '-movflags', '+faststart',
        output_path
    ]

    _log(f"🎬 Burning subtitles: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors='replace')
        _log(f"❌ FFmpeg Subtitle Error: {stderr_text}")
        raise Exception(f"FFmpeg failed: {stderr_text}")

    return True
