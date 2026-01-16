import hashlib
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

from .models import ClipSegment

try:
    import requests
except ImportError:
    requests = None

try:
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
except ImportError:
    ffmpeg_extract_subclip = None


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def infer_ext_from_url(url: str) -> str:
    """
    Best-effort extension inference from URL path.
    """
    try:
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        return ext if ext else ".mp4"
    except Exception:
        return ".mp4"


def _ensure_moviepy():
    if ffmpeg_extract_subclip is None:
        raise ImportError("moviepy not installed. Run: pip install moviepy")
    return True


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _ffprobe_has_video(media_path: str) -> bool:
    """
    True if file has at least one video stream.
    Uses ffprobe (installed with ffmpeg).
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            media_path,
        ]
        res = _run_cmd(cmd)
        if res.returncode != 0:
            return False
        return bool((res.stdout or "").strip())
    except Exception:
        return False


def _is_audio_ext(path: str) -> bool:
    ext = (os.path.splitext(path)[1] or "").lower()
    return ext in {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}


@st.cache_data(show_spinner=False)
def download_media_to_cache(url: str) -> str:
    """
    Download URL to a stable local path (cached by Streamlit).
    Returns local file path.
    """
    if requests is None:
        raise ImportError("`requests` not installed.")
    if not url.strip():
        raise ValueError("Missing media URL.")

    ext = infer_ext_from_url(url)
    cache_dir = os.path.join(tempfile.gettempdir(), "clip_comparator_media_cache")
    os.makedirs(cache_dir, exist_ok=True)

    out_path = os.path.join(cache_dir, f"{_sha1(url)}{ext}")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    r = requests.get(url, stream=True, timeout=180, allow_redirects=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return out_path


def download_video_to_sources(url: str, sources_dir: Optional[str] = None) -> str:
    """
    Download episode media into sources/ with a stable name.

    IMPORTANT FIX:
    - Preserve extension for direct downloads (mp3 stays mp3)
    - .m3u8 is downloaded via ffmpeg into .mp4
    """
    if not url.strip():
        raise ValueError("Missing media URL.")

    sources_dir = sources_dir or os.path.join(os.getcwd(), "sources")
    _ensure_dir(sources_dir)

    ext = infer_ext_from_url(url)
    if url.lower().endswith(".m3u8") or ext == ".m3u8":
        ext = ".mp4"

    out_path = os.path.join(sources_dir, f"{_sha1(url)}{ext}")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    if url.lower().endswith(".m3u8") or ext == ".m3u8":
        cmd = ["ffmpeg", "-y", "-i", url, "-c:v", "copy", "-c:a", "copy", out_path]
        res = _run_cmd(cmd)
        if res.returncode != 0 or not os.path.exists(out_path):
            tail = (res.stderr or "")[-1200:]
            raise RuntimeError(f"ffmpeg download failed:\n{tail}")
        return out_path

    if requests is None:
        raise ImportError("`requests` not installed.")
    r = requests.get(url, stream=True, timeout=240, allow_redirects=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out_path


def _ffmpeg_reencode_cut(src: str, start_s: float, end_s: float, dest: str):
    """
    Re-encode cut using ffmpeg.

    FIX:
    - If source has NO video stream, do an audio-only cut (-vn).
    - Choose a compatible audio codec for the destination container.
    """
    dur_s = max(0.0, float(end_s) - float(start_s))
    if dur_s <= 0:
        raise ValueError(f"Invalid cut duration: start={start_s}, end={end_s}")

    has_video = _ffprobe_has_video(src)
    dest_ext = (os.path.splitext(dest)[1] or "").lower()

    if not has_video:
        # audio-only cut
        # pick codec based on destination extension
        if dest_ext == ".mp3":
            a_codec = "libmp3lame"
        else:
            # .m4a/.mp4/.aac etc -> aac is broadly compatible
            a_codec = "aac"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_s),
            "-i", src,
            "-t", str(dur_s),
            "-vn",
            "-c:a", a_codec,
            "-movflags", "+faststart",
            dest,
        ]
        res = _run_cmd(cmd)
        if res.returncode != 0:
            tail = (res.stderr or "")[-1200:]
            raise RuntimeError(f"ffmpeg audio cut failed:\n{tail}")
        return

    # video+audio cut
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_s),
        "-i", src,
        "-t", str(dur_s),
        # Map streams safely (audio may not exist in some files, so ? makes it optional)
        "-map", "0:v:0?",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-movflags", "+faststart",
        dest,
    ]
    res = _run_cmd(cmd)
    if res.returncode != 0:
        tail = (res.stderr or "")[-1200:]
        raise RuntimeError(f"ffmpeg video cut failed:\n{tail}")


def render_clips_with_moviepy(
    media_path: str,
    clips: List[ClipSegment],
    out_dir: Optional[str] = None,
    force_reencode: bool = True,
    sanity_check: bool = True,
    min_dur_sec: float = 1.0,
    max_dur_sec: float = 300.0,
) -> List[Tuple[ClipSegment, Optional[str]]]:
    """
    For each clip with start_ms/end_ms, cut the media.

    NOTE:
    - If media is audio-only, outputs will be audio-only and playable in st.audio.
    """
    _ensure_moviepy()

    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Media path not found: {media_path}")

    if out_dir is None:
        out_dir = os.path.join(tempfile.gettempdir(), "clip_comparator_rendered")
    os.makedirs(out_dir, exist_ok=True)

    in_ext = os.path.splitext(media_path)[1].lower() or ".mp4"

    results: List[Tuple[ClipSegment, Optional[str]]] = []
    for c in clips:
        if c.start_ms is None or c.end_ms is None or c.end_ms <= c.start_ms:
            results.append((c, None))
            continue

        start_s = c.start_ms / 1000.0
        end_s = c.end_ms / 1000.0
        dur_s = end_s - start_s

        if sanity_check and (dur_s < min_dur_sec or dur_s > max_dur_sec):
            raise ValueError(
                f"Suspicious duration {dur_s:.3f}s for clip {c.clip_id}: "
                f"{c.start_ms}->{c.end_ms} ms"
            )

        safe_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in c.clip_id)
        stamp = f"{int(c.start_ms)}_{int(c.end_ms)}"
        out_path = os.path.join(out_dir, f"{safe_id}_{stamp}{in_ext}")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            results.append((c, out_path))
            continue

        if force_reencode:
            _ffmpeg_reencode_cut(media_path, start_s, end_s, out_path)
        else:
            ffmpeg_extract_subclip(media_path, start_s, end_s, targetname=out_path)

        results.append((c, out_path))

    return results


def show_rendered_clips_in_order(
    rendered: List[Tuple[ClipSegment, Optional[str]]],
    title: str,
):
    st.subheader(title)

    for idx, (clip, path) in enumerate(rendered, 1):
        st.markdown(f"**{idx}. {clip.clip_id}**  ({clip.source})")
        st.caption(f"{clip.start_ms}–{clip.end_ms} ms")

        if path is None:
            st.info("No timestamps available for this clip, so it cannot be cut.")
            st.text_area("Clip text", value=clip.text or "", height=120, key=f"txt_{title}_{clip.clip_id}")
            st.divider()
            continue

        ext = os.path.splitext(path)[1].lower()
        data = _read_bytes(path)

        with st.expander("Play clip", expanded=False):
            if ext in [".mp4", ".mov", ".mkv", ".webm"]:
                st.video(data)
            else:
                st.audio(data)
            st.caption(clip.text or "")

        st.divider()
