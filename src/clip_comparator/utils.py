import json
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

from .models import ClipSegment

# ----------------------------
# Time helpers
# ----------------------------

def _time_to_ms(val: Any, assume_seconds: bool = False) -> Optional[int]:
    """
    Normalize time values to milliseconds.
    Handles:
    - Already-ms values (large numbers)
    - Seconds (int/float)
    - Optional minutes:seconds encoded as float with decimal seconds (e.g., 96.26 -> 96m 26s) when assume_seconds=False and value looks like mm.ss
    - Strings with ":" (hh:mm:ss or mm:ss)
    When assume_seconds=True, numeric values are treated as seconds (no minutes heuristic).
    """
    if val is None:
        return None

    # Try string formats first
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if ":" in s:
            parts = s.split(":")
            try:
                parts_f = [float(p) for p in parts]
                while len(parts_f) < 3:
                    parts_f.insert(0, 0.0)  # pad to hh:mm:ss
                hours, minutes, seconds = parts_f[-3], parts_f[-2], parts_f[-1]
                total_sec = hours * 3600 + minutes * 60 + seconds
                return int(total_sec * 1000)
            except Exception:
                return None
        try:
            val = float(s)
        except Exception:
            return None

    # Numeric handling
    try:
        num = float(val)
    except Exception:
        return None

    # If it's already a big number, assume ms
    if num >= 100000:
        return int(num)

    # If it's in the range of typical clip durations in ms (>= 1s), but less than 100k, still assume ms
    if num >= 1000:
        return int(num)

    # When assume_seconds=True, treat numeric as seconds directly
    if assume_seconds:
        return int(num * 1000)

    # Heuristic: value like 96.26 likely means 96 minutes 26 seconds (mm.ss), cap to reasonable minutes
    frac = num - int(num)
    if 60 <= num <= 1000 and frac < 0.6:
        minutes = int(num)
        seconds = int(round(frac * 100))
        total_sec = minutes * 60 + seconds
        return int(total_sec * 1000)

    # Otherwise treat as seconds (can be fractional)
    return int(num * 1000)

def _seconds_or_mmss_to_ms(val: Any) -> Optional[int]:
    """
    Interpret numeric as seconds, but if it looks like mm.ss (e.g., 96.26 meaning 96m 26s),
    convert accordingly. This is for Headliner-style floats where end-start would otherwise
    be ~1 second instead of ~60 seconds.
    """
    if val is None:
        return None
    try:
        num = float(val)
    except Exception:
        return _time_to_ms(val, assume_seconds=True)

    frac = num - int(num)
    # If it looks like mm.ss (fraction less than ~0.6), treat as minutes:seconds.
    if num >= 60 and frac < 0.6:
        minutes = int(num)
        seconds = int(round(frac * 100))
        total_sec = minutes * 60 + seconds
        return int(total_sec * 1000)
    # Otherwise, if it's >= 60, treat it as decimal minutes (e.g., 96.75 mins)
    if num >= 60:
        return int(num * 60 * 1000)
    # Fallback: treat as seconds.
    return int(num * 1000)

def _headliner_minutes_to_ms(val: Any) -> Optional[int]:
    """
    Headliner-specific: treat numeric values as minutes with fractional seconds in mm.ss form.
    Examples:
    - 96.26 -> 96 minutes 26 seconds
    - 1.05 -> 1 minute 05 seconds
    Strings with ":" are parsed as hh:mm:ss/mm:ss.
    """
    if val is None:
        return None
    if isinstance(val, str):
        if ":" in val:
            return _time_to_ms(val, assume_seconds=False)
        try:
            val = float(val.strip())
        except Exception:
            return None
    try:
        num = float(val)
    except Exception:
        return None
    minutes = int(num)
    seconds = int((num - minutes) * 100)
    total_sec = minutes * 60 + seconds
    return int(total_sec * 1000)

def _headliner_field_to_ms(val: Any) -> Optional[int]:
    """
    Headliner-specific parser for any time field (including *Millis when they are actually mm.ss).
    - If value is large (>=100000), assume it's already milliseconds.
    - If value is a time string with ":", parse as hh:mm:ss/mm:ss.
    - Otherwise, treat numeric as minutes.seconds (mm.ss).
    """
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if ":" in s:
            return _time_to_ms(s, assume_seconds=False)
        try:
            val = float(s)
        except Exception:
            return None
    try:
        num = float(val)
    except Exception:
        return None
    if num >= 100000:
        return int(num)
    minutes = int(num)
    seconds = int(round((num - minutes) * 100))
    total_sec = minutes * 60 + seconds
    return int(total_sec * 1000)

def _ms_field_to_int(val: Any) -> Optional[int]:
    """
    For fields explicitly labeled as *Millis, treat the value as milliseconds.
    Accepts ints/floats/strings; hh:mm:ss strings are also supported.
    """
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if ":" in s:
            parts = s.split(":")
            try:
                parts_f = [float(p) for p in parts]
                while len(parts_f) < 3:
                    parts_f.insert(0, 0.0)
                hours, minutes, seconds = parts_f[-3], parts_f[-2], parts_f[-1]
                total_sec = hours * 3600 + minutes * 60 + seconds
                return int(total_sec * 1000)
            except Exception:
                return None
        try:
            return int(float(s))
        except Exception:
            return None
    try:
        return int(float(val))
    except Exception:
        return None

# ----------------------------
def ms_to_timestamp(ms: Optional[int]) -> str:
    if ms is None:
        return ""
    total_sec = ms // 1000
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"

# ----------------------------
# AssemblyAI parsing
# ----------------------------

def parse_assembly_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract episode_id, transcript text, and optional word timestamps from AssemblyAI JSON."""
    episode_id = str(data.get("id") or "episode_1")
    transcript = data.get("text") or ""
    words = data.get("words") or []
    return {"episode_id": episode_id, "transcript": transcript, "words": words}

def build_clip_text_from_words(words: List[Dict[str, Any]], start_ms: int, end_ms: int) -> str:
    if not words:
        return ""
    toks: List[str] = []
    for w in words:
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we < start_ms:
            continue
        if ws > end_ms:
            continue
        t = str(w.get("text") or "").strip()
        if t:
            toks.append(t)
    return " ".join(toks)

# ----------------------------
# Headliner clips parsing
# ----------------------------

def _parse_clip_items(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict) and "clips" in raw and isinstance(raw["clips"], list):
        clip_items = raw["clips"]
    elif isinstance(raw, dict) and "segments" in raw and isinstance(raw["segments"], list):
        clip_items = raw["segments"]
    elif isinstance(raw, list):
        clip_items = raw
    else:
        raise ValueError("Unsupported clips JSON format. Expect list or object with 'clips'/'segments'.")
    return [c for c in clip_items if isinstance(c, dict)]

def parse_headliner_clips(
    raw: Any,
    episode_id: str,
    assembly_words: Optional[List[Dict[str, Any]]] = None,
) -> List[ClipSegment]:
    """Parse Headliner clips prioritizing explicit *Millis fields (ms); otherwise treat numbers as seconds."""
    clip_items = _parse_clip_items(raw)
    clips: List[ClipSegment] = []
    for i, c in enumerate(clip_items):

        clip_id = str(c.get("id") or c.get("clip_id") or c.get("clipId") or f"clip_{i+1}")

        # Prefer Headliner-provided millis; otherwise treat numeric as seconds.
        start_ms = _ms_field_to_int(_first_not_none(c.get("startMillis"), c.get("start_ms")))
        if start_ms is None:
            start_ms = _time_to_ms(c.get("start"), assume_seconds=True)

        dur = _ms_field_to_int(_first_not_none(c.get("durationMillis"), c.get("duration_ms")))
        if dur is None:
            dur = _time_to_ms(c.get("duration"), assume_seconds=True)

        # Compute end strictly as start + duration when both exist; otherwise fall back to any provided end.
        end_ms = None
        if start_ms is not None and dur is not None:
            end_ms = start_ms + dur
        else:
            end_ms = _ms_field_to_int(_first_not_none(c.get("endMillis"), c.get("end_ms")))
            if end_ms is None:
                end_ms = _time_to_ms(c.get("end"), assume_seconds=True)

        text = c.get("text") or ""
        if not text and assembly_words and start_ms is not None and end_ms is not None:
            text = build_clip_text_from_words(assembly_words, int(start_ms), int(end_ms))

        clips.append(
            ClipSegment(
                clip_id=str(clip_id),
                start_ms=int(start_ms) if start_ms is not None else None,
                end_ms=int(end_ms) if end_ms is not None else None,
                text=str(text),
                episode_id=episode_id,
                source="",
            )
        )
    return clips

def parse_llm_style_clips(
    raw: Any,
    episode_id: str,
    assembly_words: Optional[List[Dict[str, Any]]] = None,
) -> List[ClipSegment]:
    """
    Parse LLM-style clips where timestamps are expected to be milliseconds (startMillis/endMillis),
    and plain start/end/duration are treated as seconds.
    """
    clip_items = _parse_clip_items(raw)
    clips: List[ClipSegment] = []
    for i, c in enumerate(clip_items):
        clip_id = str(c.get("id") or c.get("clip_id") or c.get("clipId") or f"clip_{i+1}")

        start_ms = _ms_field_to_int(_first_not_none(c.get("startMillis"), c.get("start_ms")))
        if start_ms is None:
            start_ms = _time_to_ms(c.get("start"), assume_seconds=True)

        end_ms = _ms_field_to_int(_first_not_none(c.get("endMillis"), c.get("end_ms")))
        if end_ms is None:
            end_ms = _time_to_ms(c.get("end"), assume_seconds=True)

        dur = _ms_field_to_int(_first_not_none(c.get("durationMillis"), c.get("duration_ms")))
        if dur is None:
            dur = _time_to_ms(c.get("duration"), assume_seconds=True)

        if end_ms is None:
            if start_ms is not None and dur is not None:
                end_ms = start_ms + dur
        elif start_ms is not None and dur is not None and end_ms < start_ms:
            end_ms = start_ms + dur

        text = c.get("text") or ""
        if not text and assembly_words and start_ms is not None and end_ms is not None:
            text = build_clip_text_from_words(assembly_words, int(start_ms), int(end_ms))

        clips.append(
            ClipSegment(
                clip_id=str(clip_id),
                start_ms=int(start_ms) if start_ms is not None else None,
                end_ms=int(end_ms) if end_ms is not None else None,
                text=str(text),
                episode_id=episode_id,
                source="",
            )
        )
    return clips

# ----------------------------
# LLM timestamp attachment by matching text -> words[]
# ----------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def _norm_token(t: str) -> str:
    return t.strip().lower()

def _tokenize_text(s: str) -> List[str]:
    return [_norm_token(x) for x in _WORD_RE.findall(s or "") if x.strip()]

def find_clip_span_in_words(
    words: List[Dict[str, Any]],
    clip_text: str
) -> Optional[Tuple[int, int]]:
    """
    Find (start_ms, end_ms) by locating the clip_text tokens as a contiguous subsequence in words[] tokens.
    Requires AssemblyAI words with start/end.
    """
    if not words:
        return None

    needle = _tokenize_text(clip_text)
    if not needle:
        return None

    hay = []
    starts = []
    ends = []
    for w in words:
        txt = str(w.get("text") or "")
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        tok = _norm_token(txt)
        if not tok:
            continue
        hay.append(tok)
        starts.append(int(ws))
        ends.append(int(we))

    if not hay or len(hay) < len(needle):
        return None

    first = needle[0]
    candidates = [i for i, t in enumerate(hay) if t == first]

    for i in candidates:
        j = 0
        k = i
        while j < len(needle) and k < len(hay) and hay[k] == needle[j]:
            j += 1
            k += 1
        if j == len(needle):
            start_ms = starts[i]
            end_ms = ends[k - 1]
            return (start_ms, end_ms)

    # If exact match fails (rare formatting/tokenization diffs), return None.
    return None

def attach_timestamps_from_words(clips: List[ClipSegment], words: List[Dict[str, Any]]) -> List[ClipSegment]:
    """Mutates/returns clips with start_ms/end_ms filled when possible."""
    for c in clips:
        has_valid = c.start_ms is not None and c.end_ms is not None and c.end_ms > c.start_ms
        if not has_valid:
            span = find_clip_span_in_words(words, c.text or "")
            if span:
                c.start_ms, c.end_ms = span
    return clips

# ----------------------------
# UI safety: dropdown crash fix
# ----------------------------

def safe_clip_picker(label: str, clips: List[ClipSegment], key_prefix: str) -> Optional[ClipSegment]:
    """
    Fixes the crash when the selectbox's prior value is not in current options.
    Keeps behavior the same, but prevents StopIteration.
    """
    if not clips:
        return None
    options = [c.clip_id for c in clips]
    state_key = f"{key_prefix}_selected_id"

    prev = st.session_state.get(state_key)
    if prev not in options:
        prev = options[0]
        st.session_state[state_key] = prev

    picked = st.selectbox(
        f"Browse {label} clips",
        options=options,
        index=options.index(prev),
        key=f"{key_prefix}_selectbox",
    )

    st.session_state[state_key] = picked
    for c in clips:
        if c.clip_id == picked:
            return c
    return clips[0]

def clips_summary_df(clips: List[ClipSegment]) -> pd.DataFrame:
    rows = []
    for c in clips:
        dur = None
        if c.start_ms is not None and c.end_ms is not None:
            dur = max(0, c.end_ms - c.start_ms) / 1000
        rows.append(
            {
                "clip_id": c.clip_id,
                "start": ms_to_timestamp(c.start_ms),
                "end": ms_to_timestamp(c.end_ms),
                "dur_sec": round(dur, 2) if dur is not None else None,
                "words": len((c.text or "").split()),
            }
        )
    return pd.DataFrame(rows)

def pretty_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None
