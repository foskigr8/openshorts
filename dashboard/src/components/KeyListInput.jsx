import React, { useState } from 'react';
import { Plus, X, Check, Eye, EyeOff } from 'lucide-react';

// Repeatable list of API keys for one provider — used where a single provider
// can usefully hold more than one key (e.g. a Gemini key pool spread across
// picker + scene-direction calls so no single key's rate limit becomes the
// bottleneck). Renders each key masked with a per-row visibility toggle,
// an "add another key" control, and a Save that persists the whole list.
export default function KeyListInput({ keys, onSave, placeholder = 'AIzaSy...' }) {
    const [draft, setDraft] = useState(keys && keys.length ? keys : ['']);
    const [visible, setVisible] = useState({});
    const [saved, setSaved] = useState(false);

    const setAt = (i, value) => {
        const next = [...draft];
        next[i] = value;
        setDraft(next);
        setSaved(false);
    };

    const addRow = () => setDraft([...draft, '']);

    const removeRow = (i) => {
        const next = draft.filter((_, idx) => idx !== i);
        setDraft(next.length ? next : ['']);
        setSaved(false);
    };

    const handleSave = () => {
        const cleaned = draft.map((k) => k.trim()).filter(Boolean);
        onSave(cleaned);
        setDraft(cleaned.length ? cleaned : ['']);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    return (
        <div className="space-y-3">
            {draft.map((key, i) => (
                <div key={i} className="flex gap-2">
                    <div className="relative flex-1">
                        <input
                            type={visible[i] ? 'text' : 'password'}
                            value={key}
                            onChange={(e) => setAt(i, e.target.value)}
                            placeholder={i === 0 ? placeholder : `${placeholder} (extra key ${i + 1})`}
                            className="input-field pr-10 font-mono"
                        />
                        <button
                            type="button"
                            onClick={() => setVisible((v) => ({ ...v, [i]: !v[i] }))}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                        >
                            {visible[i] ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                    </div>
                    {draft.length > 1 && (
                        <button
                            type="button"
                            onClick={() => removeRow(i)}
                            className="btn-ghost px-3 text-muted hover:text-ink"
                            title="Remove key"
                        >
                            <X size={16} />
                        </button>
                    )}
                </div>
            ))}
            <div className="flex items-center gap-2">
                <button type="button" onClick={addRow} className="btn-quiet py-1.5 px-3 text-xs">
                    <Plus size={12} /> add another key
                </button>
                <button
                    type="button"
                    onClick={handleSave}
                    className={saved ? 'badge-ok px-4 py-1.5 text-xs' : 'btn-quiet py-1.5 px-4 text-xs'}
                >
                    {saved ? <><Check size={12} /> saved</> : 'Save'}
                </button>
            </div>
        </div>
    );
}
