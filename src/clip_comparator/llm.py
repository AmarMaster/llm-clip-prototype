import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from .models import ClipSegment

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def call_llm_for_episode_clips(
    transcript_text: str,
    audio_url: str,
    model_name: str,
    api_key: str,
    episode_id: str,
    num_clips: int = 20,
    assembly_words: Optional[List[Dict[str, Any]]] = None,  # kept for signature compatibility; not used here
    target_duration_sec: int = 60,
) -> Tuple[List[ClipSegment], Any]:
    if OpenAI is None:
        raise ImportError("`openai` client not installed.")
    if not api_key.strip():
        raise ValueError("Missing LLM API key.")
    if not audio_url.strip():
        raise ValueError("Missing AUDIO_URL.")
    if not transcript_text.strip():
        raise ValueError("Missing TRANSCRIPT text.")
    if num_clips <= 0:
        raise ValueError("Number of clips must be > 0.")
    if target_duration_sec <= 0:
        raise ValueError("Target duration (sec) must be > 0.")

    client = OpenAI(api_key=api_key)

    target_ms = int(target_duration_sec * 1000)

    example_clips = "\n".join(
        [
            json.dumps(
                {
                    "id": f"clip_{i}",
                    "reason": "Why this moment works well as a short-form clip in 1–2 sentences.",
                    "text": f"VERBATIM text for clip {i} copied directly from the transcript.",
                    "startMillis": target_ms * (i - 1),
                    "endMillis": target_ms * (i - 1) + target_ms,
                }
            )
            for i in range(1, num_clips + 1)
        ]
    )

    example_segments = "\n".join(
        [
            json.dumps(
                {
                    "id": i,
                    "clipId": f"clip_{i}",
                    "startMillis": target_ms * (i - 1),
                    "endMillis": target_ms * (i - 1) + target_ms,
                }
            )
            for i in range(1, num_clips + 1)
        ]
    )

    user_prompt = textwrap.dedent(
        f"""
You are an expert podcast editor and short-form social video strategist.

YOUR TASK
Select EXACTLY {num_clips} short-form clips that would perform well on TikTok/Reels/Shorts.

HARD RULES (MUST FOLLOW)
- Clip text must be copied VERBATIM from the transcript (no paraphrasing, no cleanup).
- Do NOT add speaker names, emojis, captions, titles, commentary, or summaries.
- Each clip must be a contiguous span of the transcript (no stitching).
- Provide timestamps in milliseconds:
  - Clip duration MUST be between {target_ms - 2000} and {target_ms + 2000} milliseconds inclusive.

OUTPUT FORMAT (STRICT)
Return ONLY a SINGLE JSON ARRAY with exactly 2 objects (no extra text, no markdown):

[
  {{
    "type": "analysis",
    "audioUrl": "<repeat audio_url>",
    "episodeSummary": "<1–2 sentence summary>",
    "clips": [
      {example_clips}
    ]
  }},
  {{
    "type": "headliner_style",
    "audioUrl": "<repeat audio_url>",
    "segments": [
      {example_segments}
    ]
  }}
]

Additional requirements:
- First object must contain exactly {num_clips} clips.
- Each clip must include: id, reason, text, startMillis, endMillis.
- Second object must contain exactly {num_clips} segments with startMillis and endMillis.
- Use the transcript to choose exact boundaries; do not guess outside the provided transcript.

AUDIO_URL:
{audio_url}

TRANSCRIPT:
{transcript_text}
        """
    ).strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You strictly output ONLY the required JSON array, with no extra text."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = (resp.choices[0].message.content or "").strip()

    def _extract_json_text(raw: str) -> str:
        s = raw.strip()
        if s.startswith("```") and s.endswith("```"):
            inner = s[s.find("\n") + 1 : s.rfind("```")]
            return inner.strip()
        return s

    parsed_text = _extract_json_text(content)

    output_data = None
    parse_err = None
    try:
        output_data = json.loads(parsed_text)
    except Exception as e:
        parse_err = e
        start = parsed_text.find("[")
        end = parsed_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                output_data = json.loads(parsed_text[start : end + 1])
            except Exception:
                output_data = None

    if output_data is None:
        raise ValueError(f"LLM did not return valid JSON (parse error: {parse_err}); got: {parsed_text[:500]}")

    def _coerce_ms(val: Any) -> Optional[int]:
        try:
            if val is None:
                return None
            return int(val)
        except Exception:
            return None

    # Expect 2-object shape
    if not (isinstance(output_data, list) and len(output_data) == 2 and isinstance(output_data[0], dict)):
        raise ValueError("LLM output not recognized. Expected 2-element array with analysis/headliner_style objects.")

    analysis_block = output_data[0]
    segments_block = output_data[1] if isinstance(output_data[1], dict) else {}

    if not isinstance(analysis_block.get("clips"), list):
        raise ValueError("LLM output[0] missing 'clips' list.")
    analysis_clips: List[Dict[str, Any]] = analysis_block["clips"]

    if len(analysis_clips) != num_clips:
        raise ValueError(f"LLM clips must have exactly {num_clips} items; got {len(analysis_clips)}.")

    # Build times lookup from segments
    times_by_clip: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    times_by_index: List[Tuple[Optional[int], Optional[int]]] = []

    segs = segments_block.get("segments")
    if isinstance(segs, list):
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            seg_start = _coerce_ms(seg.get("startMillis"))
            seg_end = _coerce_ms(seg.get("endMillis"))
            times_by_index.append((seg_start, seg_end))
            cid = str(seg.get("clipId") or seg.get("clip_id") or "")
            if cid:
                times_by_clip[cid] = (seg_start, seg_end)

    clips: List[ClipSegment] = []
    for i, c in enumerate(analysis_clips):
        if not isinstance(c, dict):
            raise ValueError(f"clip {i+1} is not an object.")
        cid = str(c.get("id") or c.get("clipId") or c.get("clip_id") or f"clip_{i+1}")
        txt = str(c.get("text") or "").strip()
        if not txt:
            # Don’t crash the whole run for one empty clip; keep it but mark as empty.
            txt = ""

        start_ms = _coerce_ms(c.get("startMillis"))
        end_ms = _coerce_ms(c.get("endMillis"))

        # If missing, fill from segments by clipId or index
        if (start_ms is None or end_ms is None) and cid in times_by_clip:
            s, e = times_by_clip[cid]
            start_ms = start_ms if start_ms is not None else s
            end_ms = end_ms if end_ms is not None else e

        if (start_ms is None or end_ms is None) and i < len(times_by_index):
            s, e = times_by_index[i]
            start_ms = start_ms if start_ms is not None else s
            end_ms = end_ms if end_ms is not None else e

        clips.append(
            ClipSegment(
                clip_id=cid,
                start_ms=start_ms,
                end_ms=end_ms,
                text=txt,
                episode_id=episode_id,
                source="llm",
            )
        )

    # IMPORTANT:
    # Do NOT do timestamp alignment here.
    # UI will run fuzzy alignment using AssemblyAI words[] as the single source of truth.

    return clips, output_data
