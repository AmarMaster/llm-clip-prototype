from dataclasses import dataclass
from typing import Optional

@dataclass
class ClipSegment:
    """One candidate clip defined by text (+ optional timestamps)."""
    clip_id: str
    start_ms: Optional[int]
    end_ms: Optional[int]
    text: str
    episode_id: str = ""
    source: str = ""  # "headliner" or "llm"

@dataclass
class BlindPair:
    """
    One blind comparison: two clips A/B for the same episode.
    model_a / model_b are "headliner" or "llm".
    """
    clip_id: str
    episode_id: str
    task: str  # "clip_selection"
    clip_text: str
    option_a: str
    option_b: str
    model_a: str
    model_b: str
    headliner_clip_id: Optional[str] = None
    llm_clip_id: Optional[str] = None
