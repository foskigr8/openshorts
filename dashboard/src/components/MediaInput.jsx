import React, { useState, useEffect, useRef } from 'react';
import { Link2, Upload, FileVideo, X, Info, Loader2, Youtube, Sparkles } from 'lucide-react';
import { getApiUrl } from '../config';

const SUPPORTED_PLATFORMS = [
    'YouTube', 'Vimeo', 'TikTok', 'X / Twitter', 'Twitch',
    'Facebook', 'Instagram', 'Dailymotion', 'Reddit', 'Streamable',
];

// Caption placement presets. Three named choices rather than a pixel slider:
// this is how the placement problem is actually described ("bottom, but lifted
// off the edge"), and MarginV is meaningless for middle alignment.
// 43 is subtitles.SAFE_MARGIN_V, the server-side default.
const CAPTION_PLACEMENTS = {
    bottom: { label: 'Bottom', position: 'bottom', margin: 43 },
    raised: { label: 'Bottom, raised', position: 'bottom', margin: 120 },
    middle: { label: 'Middle', position: 'middle', margin: null },
};

const YT_HOST_RE = /(^|\.)(youtube\.com|youtu\.be)$/i;

export default function MediaInput({ onProcess, isProcessing }) {
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    // Source tabs (reference UI): file upload / general video URL / YouTube
    // link — all three submit through the same real handler.
    const [mode, setMode] = useState('file'); // 'file' | 'url' | 'youtube'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [acknowledged, setAcknowledged] = useState(false);
    const [outputFormat, setOutputFormat] = useState('vertical'); // vertical | square | horizontal | custom
    const [customW, setCustomW] = useState(1080);
    const [customH, setCustomH] = useState(1350);
    // The clip count is ALWAYS explicit — auto mode was removed. The picker
    // fulfills the requested count at all costs, so the UI never offers
    // "auto" as an option.
    const [clipCount, setClipCount] = useState(8);
    const [longContextClips, setLongContextClips] = useState(0);
    // AI Preferences — each control maps to a REAL per-job effect. Auto Zoom
    // and hook "style" were dropped: framing is already decided
    // automatically per scene, and picking a style before watching the
    // source clip isn't a real choice for the user to make — both stay
    // fully automatic server-side (no override sent, so the default applies).
    const [captions, setCaptions] = useState(true);
    // Caption placement, chosen BEFORE processing. Picking it afterwards in the
    // Subtitle modal re-encodes every clip. Position and margin are separate
    // controls: "bottom, raised" is bottom alignment with a bigger MarginV.
    // MarginV only applies to bottom alignment (generate_ass computes a
    // per-line margin only when ass_alignment == 2), so the raised option
    // exists for bottom only.
    const [captionPlacement, setCaptionPlacement] = useState('bottom');
    const [removeBgAudio, setRemoveBgAudio] = useState(false);
    const [showInfo, setShowInfo] = useState(false);
    const infoRef = useRef(null);

    useEffect(() => {
        if (!showInfo) return;
        const onClick = (e) => {
            if (infoRef.current && !infoRef.current.contains(e.target)) setShowInfo(false);
        };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [showInfo]);

    useEffect(() => {
        fetch(getApiUrl('/api/config'))
            .then((r) => r.ok ? r.json() : null)
            .then((cfg) => {
                if (cfg && cfg.youtubeUrlEnabled === false) {
                    setYoutubeUrlEnabled(false);
                    setMode('file');
                }
            })
            .catch(() => {});
    }, []);

    useEffect(() => {
        let pending = null;
        try {
            pending = localStorage.getItem('os_pending_url');
            if (pending) localStorage.removeItem('os_pending_url');
        } catch { /* ignore */ }
        if (pending) {
            setMode('youtube');
            setUrl(pending);
        }
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged) return;
        if ((mode === 'url' || mode === 'youtube') && url) {
            if (mode === 'youtube' && !YT_HOST_RE.test(new URL(url).hostname)) {
                alert('That does not look like a YouTube link — use the Video URL tab for other sites.');
                return;
            }
            onProcess({
                type: 'url', payload: url, acknowledged: true,
                outputFormat: outputFormat === 'custom' ? 'custom' : outputFormat,
                customWidth: customW, customHeight: customH,
                captions,
                captionPosition: CAPTION_PLACEMENTS[captionPlacement].position,
                captionMargin: CAPTION_PLACEMENTS[captionPlacement].margin,
                // "isolate" = Demucs voice separation, the only mode that
                // actually removes music. "" = leave the audio untouched.
                removeBackgroundAudio: removeBgAudio ? 'isolate' : '',
                clipCount, longContextClips,
            });
        } else if (mode === 'file' && file) {
            onProcess({
                type: 'file', payload: file, acknowledged: true,
                outputFormat: outputFormat === 'custom' ? 'custom' : outputFormat,
                customWidth: customW, customHeight: customH,
                captions,
                captionPosition: CAPTION_PLACEMENTS[captionPlacement].position,
                captionMargin: CAPTION_PLACEMENTS[captionPlacement].margin,
                // "isolate" = Demucs voice separation, the only mode that
                // actually removes music. "" = leave the audio untouched.
                removeBackgroundAudio: removeBgAudio ? 'isolate' : '',
                clipCount, longContextClips,
            });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    const isCustom = outputFormat === 'custom';
    const sliderPct = ((clipCount - 1) / (40 - 1)) * 100;

    return (
        <div className="card card-lit p-4 sm:p-6 animate-fade">
            {/* Source tabs — above the dropzone, underline style (reference) */}
            <div className="flex gap-1 border-b border-rule mb-5">
                {[
                    { id: 'file', label: 'Upload File', Icon: Upload },
                    { id: 'url', label: 'Video URL', Icon: Link2 },
                    ...(youtubeUrlEnabled ? [{ id: 'youtube', label: 'YouTube Link', Icon: Youtube }] : []),
                ].map(({ id, label, Icon }) => (
                    <button
                        key={id}
                        type="button"
                        onClick={() => setMode(id)}
                        className={`flex items-center gap-1.5 pb-2.5 px-1 -mb-px border-b-2 text-xs whitespace-nowrap transition-colors ${
                            mode === id
                                ? 'border-brass text-ink'
                                : 'border-transparent text-muted hover:text-ink2'
                        }`}
                    >
                        <Icon size={14} className={mode === id ? 'text-brass' : ''} />
                        {label}
                    </button>
                ))}
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' || mode === 'youtube' ? (
                    <div className="space-y-4">
                        <div className="relative">
                            <input
                                type="url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder={mode === 'youtube'
                                    ? 'paste a YouTube link (youtube.com / youtu.be)'
                                    : 'https://... paste a video link'}
                                className="input-field pr-11"
                                required
                            />
                            <div className="absolute inset-y-0 right-2 flex items-center" ref={infoRef}>
                                <button
                                    type="button"
                                    onClick={() => setShowInfo((v) => !v)}
                                    aria-label="Supported platforms"
                                    className="p-1.5 text-muted hover:text-brass transition-colors"
                                >
                                    <Info size={16} />
                                </button>
                                {showInfo && (
                                    <div className="absolute right-0 top-full mt-2 w-64 z-20 card p-4 text-left animate-fade">
                                        <p className="eyebrow mb-2">
                                            {mode === 'youtube' ? 'YouTube links only' : 'Paste a link from'}
                                        </p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {SUPPORTED_PLATFORMS.map((p) => (
                                                <span key={p} className="text-xs px-2 py-0.5 rounded-full bg-paper3 text-ink2">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                        {mode === 'youtube' && (
                                            <p className="text-xs text-muted mt-2.5 leading-relaxed">
                                                This tab validates the host is YouTube. Other platforms go through Video URL.
                                            </p>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div
                        className={`border-2 border-dashed rounded-card p-6 sm:p-8 text-center transition-colors ${file ? 'border-brass' : 'border-rule2 hover:border-brass'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-ok min-w-0">
                                <FileVideo size={18} className="shrink-0" />
                                <span className="font-medium truncate">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 text-muted hover:text-ink hover:bg-paper3 rounded-full transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-muted" size={18} />
                                <p className="text-ink2 lowercase">Click to upload or drag and drop</p>
                                <p className="readout mt-2">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                {/* Output format selector — 4 tiles including Custom */}
                <div className="mt-5">
                    <p className="eyebrow mb-2">Output Format</p>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {[
                            { value: 'vertical', label: '9:16', hint: 'Shorts · Reels', w: 18, h: 32 },
                            { value: 'square', label: '1:1', hint: 'Feed posts', w: 28, h: 28 },
                            { value: 'horizontal', label: '16:9', hint: 'Landscape', w: 36, h: 20 },
                            { value: 'custom', label: 'Custom', hint: 'Set your own', w: 24, h: 28 },
                        ].map((f) => {
                            const active = outputFormat === f.value;
                            return (
                                <button
                                    key={f.value}
                                    type="button"
                                    onClick={() => setOutputFormat(f.value)}
                                    className={`py-3 px-2 rounded-input border flex flex-col items-center gap-2 transition-all
                                        ${active ? 'border-[color:var(--color-accent)] text-ink' : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                                    style={active ? {
                                        background: 'linear-gradient(180deg, rgba(239,68,68,0.12) 0%, rgba(239,68,68,0.02) 100%)',
                                        boxShadow: '0 0 18px -6px var(--color-glow)',
                                    } : undefined}
                                >
                                    <span
                                        className="rounded-[3px] border-2 transition-colors"
                                        style={{
                                            width: `${f.w}px`,
                                            height: `${f.h}px`,
                                            borderColor: active ? 'var(--color-accent)' : 'var(--color-rule-2)',
                                            backgroundColor: active ? 'color-mix(in srgb, var(--color-accent) 22%, transparent)' : 'transparent',
                                        }}
                                    />
                                    <span className="block font-mono text-sm leading-none">{f.label}</span>
                                    <span className="block text-[10px] leading-tight text-center text-muted">{f.hint}</span>
                                </button>
                            );
                        })}
                    </div>
                    {isCustom && (
                        <div className="flex items-center gap-3 mt-3">
                            <label className="flex items-center gap-2 text-xs text-muted">
                                width
                                <input
                                    type="number"
                                    min="100"
                                    max="4000"
                                    value={customW}
                                    onChange={(e) => setCustomW(Math.max(100, Math.min(4000, Number(e.target.value) || 1080)))}
                                    className="input-field py-1.5 px-2 w-24 text-sm"
                                />
                            </label>
                            <span className="text-muted">×</span>
                            <label className="flex items-center gap-2 text-xs text-muted">
                                height
                                <input
                                    type="number"
                                    min="100"
                                    max="4000"
                                    value={customH}
                                    onChange={(e) => setCustomH(Math.max(100, Math.min(4000, Number(e.target.value) || 1350)))}
                                    className="input-field py-1.5 px-2 w-24 text-sm"
                                />
                            </label>
                        </div>
                    )}
                </div>

                {/* Clip count: an explicit hard target, always */}
                <div className="mt-5">
                    <div className="flex items-center justify-between mb-3">
                        <p className="eyebrow">Clip Count</p>
                        <span className="readout text-[10px] text-muted">hard target</span>
                    </div>
                    <div className="relative pt-5 pb-1">
                        <span
                            className="absolute top-0 -translate-x-1/2 px-2 py-0.5 rounded-md bg-paper3 border border-brass/40 readout text-[10px] text-ink"
                            style={{ left: `calc(${sliderPct}% + 9px)` }}
                        >
                            {clipCount}
                        </span>
                        <input
                            type="range"
                            min="1"
                            max="40"
                            step="1"
                            value={clipCount}
                            onChange={(e) => setClipCount(Number(e.target.value))}
                            className="slider-filled w-full cursor-pointer"
                            style={{
                                background: `linear-gradient(90deg, var(--color-accent) 0%, var(--color-accent) ${sliderPct}%, var(--color-rule-2) ${sliderPct}%, var(--color-rule-2) 100%)`,
                            }}
                            aria-label="Number of clips to generate"
                        />
                    </div>
                    <div className="flex justify-between text-[10px] text-muted mt-1">
                        <span>1</span>
                        <span className="text-ink2 font-medium">
                            hard target: {clipCount} clips
                        </span>
                        <span>40</span>
                    </div>
                </div>

                {/* AI Preferences — every control maps to a real per-job effect */}
                <div className="mt-5 rounded-input border border-rule p-3 space-y-3">
                    <div className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5 text-xs text-ink2">
                            <Sparkles size={13} className="text-brass" /> AI Preferences
                        </span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <label className="flex items-center gap-2 text-xs text-muted cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={captions}
                                onChange={(e) => setCaptions(e.target.checked)}
                                className="accent-[var(--color-accent)] cursor-pointer"
                            />
                            Add Captions
                        </label>
                        {/* Only meaningful when captions are on. */}
                        <label className={`flex items-center justify-between gap-2 text-xs ${captions ? 'text-muted' : 'text-muted/40'}`}>
                            Caption position
                            <select
                                value={captionPlacement}
                                onChange={(e) => setCaptionPlacement(e.target.value)}
                                disabled={!captions}
                                className="bg-paper2 border border-rule rounded px-2 py-1 text-xs text-ink2 disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed"
                            >
                                {Object.entries(CAPTION_PLACEMENTS).map(([key, opt]) => (
                                    <option key={key} value={key}>{opt.label}</option>
                                ))}
                            </select>
                        </label>
                        <label
                            className="flex items-center gap-2 text-xs text-muted cursor-pointer select-none"
                            title="Separates the voice from music and ambience so only speech remains. Adds roughly 30s per clip."
                        >
                            <input
                                type="checkbox"
                                checked={removeBgAudio}
                                onChange={(e) => setRemoveBgAudio(e.target.checked)}
                                className="accent-[var(--color-accent)] cursor-pointer"
                            />
                            Remove background audio
                        </label>
                        <div className="flex items-center justify-between gap-2 text-xs text-muted">
                            Auto Detect Highlights
                            <span className="px-2 py-1 rounded-full border border-brass/40 text-[10px] readout text-brass">
                                always on
                            </span>
                        </div>
                    </div>
                    {/* Auto Zoom (TRACK/GENERAL) and hook "style" were removed here —
                        both are the AI's job, not a pre-watch guess: framing is
                        already decided automatically per scene, and a style pick
                        made before anyone has seen the source clip isn't a real
                        choice, it's a coin flip. Both stay fully automatic. */}
                </div>

                {/* Long-context clips: opt-in full-arc category */}
                <div className="mt-3 rounded-input border border-rule p-3">
                    <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={longContextClips > 0}
                            onChange={(e) => setLongContextClips(e.target.checked ? 1 : 0)}
                            className="accent-[var(--color-accent)] cursor-pointer"
                        />
                        <span>
                            Include long-context clips
                            <span className="block text-[10px] text-muted mt-0.5">
                                full 1–3 min arcs (setup → tension → resolution), trimmed of dead air
                            </span>
                        </span>
                    </label>
                    {longContextClips > 0 && (
                        <div className="flex items-center gap-2 mt-2.5">
                            <span className="readout text-[10px] text-muted">how many</span>
                            <input
                                type="number"
                                min="1"
                                max="5"
                                value={longContextClips}
                                onChange={(e) => {
                                    const v = Number(e.target.value);
                                    setLongContextClips(Number.isFinite(v) ? Math.max(0, Math.min(5, v)) : 0);
                                }}
                                className="input-field py-1 px-2 w-16 text-sm"
                                aria-label="Number of long-context clips"
                            />
                        </div>
                    )}
                </div>

                <label className="flex items-start gap-2 mt-5 text-xs text-muted cursor-pointer select-none">
                    <input
                        type="checkbox"
                        checked={acknowledged}
                        onChange={(e) => setAcknowledged(e.target.checked)}
                        className="mt-0.5 accent-[var(--color-accent)] cursor-pointer"
                    />
                    <span>
                        I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/#legal" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Terms & Privacy</a>.
                    </span>
                </label>

                <button
                    type="submit"
                    disabled={isProcessing || !acknowledged || ((mode === 'url' || mode === 'youtube') && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-4"
                >
                    {isProcessing ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
