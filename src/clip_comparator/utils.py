import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

from .models import ClipSegment

# ----------------------------
# Time helpers
# ----------------------------

def _time_to_ms(val: Any, assume_seconds: bool = False) -> Optional[int]:
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
            val = float(s)
        except Exception:
            return None

    try:
        num = float(val)
    except Exception:
        return None

    if num >= 100000:
        return int(num)
    if num >= 1000:
        return int(num)

    if assume_seconds:
        return int(num * 1000)

    frac = num - int(num)
    if 60 <= num <= 1000 and frac < 0.6:
        minutes = int(num)
        seconds = int(round(frac * 100))
        total_sec = minutes * 60 + seconds
        return int(total_sec * 1000)

    return int(num * 1000)


def _ms_field_to_int(val: Any) -> Optional[int]:
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
# Clips parsing
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
    clip_items = _parse_clip_items(raw)
    clips: List[ClipSegment] = []
    for i, c in enumerate(clip_items):
        clip_id = str(c.get("id") or c.get("clip_id") or c.get("clipId") or f"clip_{i+1}")

        start_ms = _ms_field_to_int(_first_not_none(c.get("startMillis"), c.get("start_ms")))
        if start_ms is None:
            start_ms = _time_to_ms(c.get("start"), assume_seconds=True)

        dur = _ms_field_to_int(_first_not_none(c.get("durationMillis"), c.get("duration_ms")))
        if dur is None:
            dur = _time_to_ms(c.get("duration"), assume_seconds=True)

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
# Normalization / tokenization
# ----------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_PUNCT_MAP = {
    "\u2019": "'",  # ’
    "\u2018": "'",  # ‘
    "\u201C": '"',  # “
    "\u201D": '"',  # ”
    "\u2014": "-",  # —
    "\u2013": "-",  # –
    "\u00A0": " ",  # nbsp
}

def _normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    for k, v in _PUNCT_MAP.items():
        s = s.replace(k, v)
    return s

def _norm_token(t: str) -> str:
    return _normalize_text(t).strip().lower()

def _tokenize_text(s: str) -> List[str]:
    s = _normalize_text(s or "")
    return [_norm_token(x) for x in _WORD_RE.findall(s) if x.strip()]

def _tokens_from_word_text(word_text: str) -> List[str]:
    return _tokenize_text(word_text or "")


# ----------------------------
# Build indexed transcript tokens from words[]
# ----------------------------

def build_words_index(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokens: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    pos: Dict[str, List[int]] = {}

    for w in words or []:
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue

        wtoks = _tokens_from_word_text(str(w.get("text") or ""))
        if not wtoks:
            continue

        # replicate timing across split tokens (okay for matching)
        for tok in wtoks:
            idx = len(tokens)
            tokens.append(tok)
            starts.append(int(ws))
            ends.append(int(we))
            pos.setdefault(tok, []).append(idx)

    return {"tokens": tokens, "starts": starts, "ends": ends, "pos": pos}


# ----------------------------
# STRICT contiguous match (compatibility)
# ----------------------------

def find_clip_span_in_words_strict_index(
    words_index: Dict[str, Any],
    clip_text: str,
    *,
    max_candidates: int = 2500,
) -> Optional[Dict[str, Any]]:
    tokens: List[str] = words_index.get("tokens") or []
    starts: List[int] = words_index.get("starts") or []
    ends: List[int] = words_index.get("ends") or []
    pos: Dict[str, List[int]] = words_index.get("pos") or {}

    needle = _tokenize_text(clip_text or "")
    if not needle or len(needle) > len(tokens):
        return None

    best_anchor = None  # (occ, j, tok)
    for j, tok in enumerate(needle[:160]):
        occs = pos.get(tok, [])
        if not occs:
            continue
        cand = (len(occs), j, tok)
        if best_anchor is None or cand < best_anchor:
            best_anchor = cand

    if best_anchor is None:
        return None

    _occ, j_anchor, tok_anchor = best_anchor
    occs = pos.get(tok_anchor, [])
    if not occs:
        return None

    if len(occs) > max_candidates:
        step = max(1, len(occs) // max_candidates)
        occs = occs[::step][:max_candidates]

    for p in occs:
        start_guess = p - j_anchor
        if start_guess < 0:
            continue
        end_guess = start_guess + len(needle) - 1
        if end_guess >= len(tokens):
            continue
        if tokens[start_guess : start_guess + len(needle)] == needle:
            return {
                "start_ms": int(starts[start_guess]),
                "end_ms": int(ends[end_guess]),
                "score": 1.0,
                "method": "strict",
                "span_start_idx": int(start_guess),
                "span_end_idx": int(end_guess),
                "matched_tokens": int(len(needle)),
                "clip_tokens": int(len(needle)),
                "start_guess": int(start_guess),
                "first_match_idx": int(start_guess),
                "lead_misses": 0,
            }

    return None


def find_clip_span_in_words(words: List[Dict[str, Any]], clip_text: str) -> Optional[Tuple[int, int]]:
    if not words:
        return None
    idx = build_words_index(words)
    res = find_clip_span_in_words_strict_index(idx, clip_text)
    if not res:
        return None
    return (int(res["start_ms"]), int(res["end_ms"]))


def attach_timestamps_from_words(clips: List[ClipSegment], words: List[Dict[str, Any]]) -> List[ClipSegment]:
    """
    Compatibility function for older code:
    - fills only if missing/invalid
    - does NOT overwrite valid timestamps
    """
    for c in clips:
        has_valid = c.start_ms is not None and c.end_ms is not None and c.end_ms > c.start_ms
        if has_valid:
            continue
        span = find_clip_span_in_words(words, c.text or "")
        if span:
            c.start_ms, c.end_ms = span
    return clips


# ----------------------------
# FUZZY alignment (PRIMARY)
# ----------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "hers", "our", "their",
    "as", "at", "by", "from", "into", "about", "over", "under", "up", "down", "out",
    "not", "no", "yes", "do", "does", "did", "doing",
    "have", "has", "had",
    "with", "without",
}

def _is_number_token(t: str) -> bool:
    return bool(re.fullmatch(r"\d+", t or ""))

def _edit_distance_leq_1(a: str, b: str) -> bool:
    """
    Fast check: whether Levenshtein distance(a,b) <= 1
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    # same length: allow one substitution
    if la == lb:
        diff = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                diff += 1
                if diff > 1:
                    return False
        return True

    # length differs by 1: allow one insertion/deletion
    if la > lb:
        a, b = b, a
        la, lb = lb, la  # now la < lb

    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1  # skip one char in longer string
    return True  # tail char counts as <=1 edit

def _token_match(transcript_tok: str, clip_tok: str) -> bool:
    if transcript_tok == clip_tok:
        return True
    # typo tolerance for longer words, avoids over-matching short/common tokens
    if len(transcript_tok) >= 6 and len(clip_tok) >= 6 and (not _is_number_token(transcript_tok)) and (not _is_number_token(clip_tok)):
        return _edit_distance_leq_1(transcript_tok, clip_tok)
    return False


def _pick_anchor_tokens(
    clip_tokens: List[str],
    pos_index: Dict[str, List[int]],
    *,
    max_anchors: int = 14,
) -> List[Tuple[int, str]]:
    if not clip_tokens:
        return []

    cands = []
    seen = set()
    for j, tok in enumerate(clip_tokens):
        if tok in seen:
            continue
        seen.add(tok)
        if len(tok) <= 1:
            continue
        occ = pos_index.get(tok, [])
        if not occ:
            continue
        if tok in _STOPWORDS:
            continue
        cands.append((len(occ), j, tok))

    cands.sort(key=lambda x: (x[0], x[1]))
    anchors = [(j, tok) for _occ, j, tok in cands[:max_anchors]]
    if anchors:
        return anchors

    # fallback: allow stopwords if needed
    cands2 = []
    seen2 = set()
    for j, tok in enumerate(clip_tokens):
        if tok in seen2:
            continue
        seen2.add(tok)
        if len(tok) <= 1:
            continue
        occ = pos_index.get(tok, [])
        if not occ:
            continue
        cands2.append((len(occ), j, tok))

    cands2.sort(key=lambda x: (x[0], x[1]))
    return [(j, tok) for _occ, j, tok in cands2[:max_anchors]]


def _score_alignment_greedy(
    transcript_tokens: List[str],
    clip_tokens: List[str],
    start_guess: int,
) -> Optional[Dict[str, Any]]:
    """
    Key change: we return BOTH:
      - first_match_idx (where matching actually first succeeded)
      - start_guess (hypothesized start of the clip)
    We should NOT set clip start to first_match_idx, because early mismatches are common.
    """
    n = len(transcript_tokens)
    if n == 0 or not clip_tokens:
        return None

    max_skip = min(120, max(35, int(len(clip_tokens) * 0.10)))
    max_misses = max(8, int(len(clip_tokens) * 0.22))

    t = max(0, min(start_guess, n - 1))
    matched_positions: List[int] = []
    misses = 0
    lead_misses = 0
    saw_first_match = False
    first_match_idx = None

    for tok in clip_tokens:
        found = False
        for skip in range(0, max_skip + 1):
            k = t + skip
            if k >= n:
                break
            if _token_match(transcript_tokens[k], tok):
                matched_positions.append(k)
                t = k + 1
                found = True
                if not saw_first_match:
                    saw_first_match = True
                    first_match_idx = k
                break

        if not found:
            misses += 1
            if not saw_first_match:
                lead_misses += 1
            if misses > max_misses:
                break

    if not matched_positions:
        return None

    score = len(matched_positions) / max(1, len(clip_tokens))
    return {
        "span_start_idx": matched_positions[0],  # first actual match (debug)
        "span_end_idx": matched_positions[-1],
        "score": score,
        "matched_tokens": len(matched_positions),
        "clip_tokens": len(clip_tokens),
        "start_guess": int(start_guess),
        "first_match_idx": int(first_match_idx if first_match_idx is not None else matched_positions[0]),
        "lead_misses": int(lead_misses),
    }


def find_clip_span_in_words_fuzzy(
    words_index: Dict[str, Any],
    clip_text: str,
    *,
    min_score: float = 0.70,
    top_k_candidates: int = 40,
) -> Optional[Dict[str, Any]]:
    transcript_tokens: List[str] = words_index.get("tokens") or []
    starts: List[int] = words_index.get("starts") or []
    ends: List[int] = words_index.get("ends") or []
    pos: Dict[str, List[int]] = words_index.get("pos") or {}

    raw_clip_tokens = _tokenize_text(clip_text or "")
    if not raw_clip_tokens or not transcript_tokens:
        return None

    filtered = [t for t in raw_clip_tokens if t not in _STOPWORDS]
    clip_tokens = filtered if len(filtered) >= 8 else raw_clip_tokens

    anchors = _pick_anchor_tokens(clip_tokens, pos, max_anchors=14)
    if not anchors:
        return None

    votes: Dict[int, float] = {}
    for j, tok in anchors:
        occs = pos.get(tok, [])
        if not occs:
            continue

        if len(occs) > 300:
            step = max(1, len(occs) // 300)
            occs = occs[::step][:300]

        weight = 1.0 / max(1, min(len(pos.get(tok, [])), 80))
        for p in occs:
            s = p - j
            if s < 0:
                continue
            votes[s] = votes.get(s, 0.0) + weight

    if not votes:
        return None

    candidates = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:top_k_candidates]

    best: Optional[Dict[str, Any]] = None
    for start_guess, _v in candidates:
        res = _score_alignment_greedy(transcript_tokens, clip_tokens, start_guess)
        if not res:
            continue
        if best is None or res["score"] > best["score"]:
            best = res

    if not best or best["score"] < min_score:
        return None

    ei = int(best["span_end_idx"])
    if ei < 0 or ei >= len(ends):
        return None

    # IMPORTANT: start_ms is not decided here anymore.
    # We return indices and let attach_timestamps decide the true start index robustly.
    return {
        "score": float(best["score"]),
        "method": "fuzzy",
        "span_end_idx": int(best["span_end_idx"]),
        "span_start_idx": int(best["span_start_idx"]),   # first match (debug only)
        "start_guess": int(best["start_guess"]),
        "first_match_idx": int(best["first_match_idx"]),
        "lead_misses": int(best["lead_misses"]),
        "matched_tokens": int(best["matched_tokens"]),
        "clip_tokens": int(best["clip_tokens"]),
    }


def attach_timestamps_from_words_fuzzy(
    clips: List[ClipSegment],
    words: List[Dict[str, Any]],
    *,
    overwrite: bool = True,
    min_score: float = 0.70,
    # keep buffers SMALL now; the real fix is start_guess logic
    start_buffer_ms: int = 180,
    end_buffer_ms: int = 180,
    # cap how far back we allow start_guess to pull us (in tokens)
    max_start_back_tokens: int = 60,
    fallback_strict: bool = True,
    keep_old_if_unmatched: bool = False,
) -> Tuple[List[ClipSegment], List[Dict[str, Any]]]:
    debug_rows: List[Dict[str, Any]] = []
    if not clips:
        return clips, debug_rows

    if not words:
        for c in clips:
            debug_rows.append(
                {
                    "clip_id": c.clip_id,
                    "matched": False,
                    "reason": "no_words",
                    "score": None,
                    "lead_misses": None,
                    "old_start": c.start_ms,
                    "old_end": c.end_ms,
                    "new_start": c.start_ms,
                    "new_end": c.end_ms,
                    "method": None,
                }
            )
        return clips, debug_rows

    idx = build_words_index(words)
    starts: List[int] = idx.get("starts") or []
    ends: List[int] = idx.get("ends") or []

    for c in clips:
        old_start, old_end = c.start_ms, c.end_ms

        if not (c.text or "").strip():
            if overwrite and not keep_old_if_unmatched:
                c.start_ms, c.end_ms = None, None
            debug_rows.append(
                {
                    "clip_id": c.clip_id,
                    "matched": False,
                    "reason": "empty_text",
                    "score": None,
                    "lead_misses": None,
                    "old_start": old_start,
                    "old_end": old_end,
                    "new_start": c.start_ms,
                    "new_end": c.end_ms,
                    "method": None,
                }
            )
            continue

        res = find_clip_span_in_words_fuzzy(idx, c.text or "", min_score=min_score)

        if res is None and fallback_strict:
            strict_res = find_clip_span_in_words_strict_index(idx, c.text or "")
            if strict_res:
                # strict gives exact start/end indices
                start_idx = int(strict_res["span_start_idx"])
                end_idx = int(strict_res["span_end_idx"])
                new_start = max(0, int(starts[start_idx]) - int(start_buffer_ms)) if starts else None
                new_end = int(ends[end_idx]) + int(end_buffer_ms) if ends else None
                c.start_ms, c.end_ms = new_start, new_end
                debug_rows.append(
                    {
                        "clip_id": c.clip_id,
                        "matched": True,
                        "reason": None,
                        "score": 1.0,
                        "lead_misses": 0,
                        "old_start": old_start,
                        "old_end": old_end,
                        "new_start": c.start_ms,
                        "new_end": c.end_ms,
                        "method": "strict",
                    }
                )
                continue

        if res:
            # ---- CRITICAL FIX ----
            # Use start_guess (hypothesized start) instead of first matched token.
            start_guess = int(res.get("start_guess", -1))
            first_match_idx = int(res.get("first_match_idx", -1))
            lead_misses = int(res.get("lead_misses", 0))
            end_idx = int(res.get("span_end_idx", -1))

            if not starts or not ends or end_idx < 0 or end_idx >= len(ends):
                if overwrite and not keep_old_if_unmatched:
                    c.start_ms, c.end_ms = None, None
                debug_rows.append(
                    {
                        "clip_id": c.clip_id,
                        "matched": False,
                        "reason": "bad_index",
                        "score": float(res.get("score", 0.0)),
                        "lead_misses": lead_misses,
                        "old_start": old_start,
                        "old_end": old_end,
                        "new_start": c.start_ms,
                        "new_end": c.end_ms,
                        "method": None,
                    }
                )
                continue

            # pick a robust start index:
            # - prefer start_guess
            # - but don't allow it to be absurdly earlier than where matching begins
            start_idx = start_guess if start_guess >= 0 else first_match_idx
            if start_idx < 0:
                start_idx = max(0, end_idx - 10)

            # cap how far back from first_match_idx we allow start_guess to go
            if first_match_idx >= 0 and start_idx < first_match_idx - max_start_back_tokens:
                start_idx = max(0, first_match_idx - max_start_back_tokens)

            # also, if the model had many lead misses, allow a bit more context (but bounded)
            if first_match_idx >= 0 and lead_misses > 0:
                extra = min(max_start_back_tokens, 2 * lead_misses)
                start_idx = max(0, min(start_idx, first_match_idx - extra))

            new_start = max(0, int(starts[start_idx]) - int(start_buffer_ms))
            new_end = int(ends[end_idx]) + int(end_buffer_ms)

            c.start_ms, c.end_ms = int(new_start), int(new_end)

            debug_rows.append(
                {
                    "clip_id": c.clip_id,
                    "matched": True,
                    "reason": None,
                    "score": float(res.get("score", 0.0)),
                    "lead_misses": lead_misses,
                    "old_start": old_start,
                    "old_end": old_end,
                    "new_start": c.start_ms,
                    "new_end": c.end_ms,
                    "method": res.get("method", "fuzzy"),
                }
            )
        else:
            if overwrite and not keep_old_if_unmatched:
                c.start_ms, c.end_ms = None, None
            debug_rows.append(
                {
                    "clip_id": c.clip_id,
                    "matched": False,
                    "reason": "no_match",
                    "score": None,
                    "lead_misses": None,
                    "old_start": old_start,
                    "old_end": old_end,
                    "new_start": c.start_ms,
                    "new_end": c.end_ms,
                    "method": None,
                }
            )

    return clips, debug_rows


def llm_alignment_df(debug_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in debug_rows or []:
        rows.append(
            {
                "clip_id": r.get("clip_id"),
                "matched": r.get("matched"),
                "reason": r.get("reason"),
                "score": r.get("score"),
                "lead_misses": r.get("lead_misses"),
                "old_start": ms_to_timestamp(r.get("old_start")),
                "old_end": ms_to_timestamp(r.get("old_end")),
                "new_start": ms_to_timestamp(r.get("new_start")),
                "new_end": ms_to_timestamp(r.get("new_end")),
                "method": r.get("method"),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# UI safety
# ----------------------------

def safe_clip_picker(label: str, clips: List[ClipSegment], key_prefix: str) -> Optional[ClipSegment]:
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
