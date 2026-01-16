import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import pandas as pd

from .models import ClipSegment
from .utils import (
    parse_assembly_json,
    parse_headliner_clips,
    parse_llm_style_clips,
    attach_timestamps_from_words,          # compatibility
    attach_timestamps_from_words_fuzzy,    # primary
    llm_alignment_df,
    ms_to_timestamp,
    clips_summary_df,
    pretty_json,
)
from .rss import extract_episodes_from_rss
from .assemblyai import (
    assemblyai_submit_transcript,
    assemblyai_get_transcript,
    assemblyai_try_fetch_words,
)
from .headliner import fetch_headliner_job_result
from .llm import call_llm_for_episode_clips
from .media import (
    download_video_to_sources,
    render_clips_with_moviepy,
)

try:
    import requests
except ImportError:
    requests = None


def _episode_uid_from_audio_url(audio_url: str) -> str:
    s = (audio_url or "").strip()
    if not s:
        return ""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


class ClipComparatorApp:
    def __init__(self):
        st.set_page_config(page_title="Headliner vs LLM – Clip Comparison", layout="wide")
        self.ensure_session_state()
        self.run()

    @staticmethod
    def ensure_session_state():
        ClipComparatorApp._load_dotenv()

        # Episode tracking (to detect podcast changes)
        st.session_state.setdefault("active_episode_uid", "")
        st.session_state.setdefault("rss_meta", None)
        st.session_state.setdefault("rss_episode_uid", "")
        st.session_state.setdefault("rss_episode_title", "")
        st.session_state.setdefault("rss_episode_audio_url", "")
        st.session_state.setdefault("episode_video_url", "")

        # Transcript state
        st.session_state.setdefault("assembly_raw", None)
        st.session_state.setdefault("assembly_parsed", None)
        st.session_state.setdefault("assembly_job_id", None)
        st.session_state.setdefault("assembly_job_status", None)

        # Headliner state
        st.session_state.setdefault("headliner_raw", None)
        st.session_state.setdefault("headliner_job_id", None)
        st.session_state.setdefault("headliner_clips", [])

        # LLM state
        st.session_state.setdefault("llm_clips", [])
        st.session_state.setdefault("llm_raw", None)
        st.session_state.setdefault("llm_alignment_debug", [])
        st.session_state.setdefault("llm_last_transcript_text", "")

        # Shared URLs / UI widget keys that need to be forced on episode change
        st.session_state.setdefault("headliner_audio_url", "")
        st.session_state.setdefault("llm_audio_url", "")
        st.session_state.setdefault("playback_media_url", "")

        # alignment params
        st.session_state.setdefault("llm_align_min_score", 0.70)
        st.session_state.setdefault("llm_start_buffer_ms", 180)
        st.session_state.setdefault("llm_end_buffer_ms", 180)
        st.session_state.setdefault("llm_max_start_back_tokens", 60)

        # Render state
        st.session_state.setdefault("rendered_headliner", None)
        st.session_state.setdefault("rendered_llm", None)

        # Headliner transcript URL field (if you still use it)
        st.session_state.setdefault("transcript_url", os.environ.get("LEX_FRIDMAN_TRSCPT", ""))

    @staticmethod
    def _load_dotenv():
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val

    def _on_episode_change(self, ep: Dict[str, Any], new_uid: str):
        """
        HARD reset relevant session state when the selected episode changes.
        This fixes Streamlit widget state "sticking" (LLM audio URL, transcript text, moviepy URL).
        """
        st.session_state["active_episode_uid"] = new_uid

        audio_url = (ep.get("audio_url") or "").strip()
        video_url = (ep.get("video_url") or "").strip()

        # Update episode meta
        st.session_state["rss_episode_uid"] = new_uid
        st.session_state["rss_episode_title"] = ep.get("title", "")
        st.session_state["rss_episode_audio_url"] = audio_url
        st.session_state["episode_video_url"] = video_url

        # Force widget keys to new defaults
        st.session_state["headliner_audio_url"] = audio_url
        st.session_state["llm_audio_url"] = audio_url
        # MoviePy should default to video enclosure if present; else audio
        st.session_state["playback_media_url"] = video_url or audio_url

        # Clear transcript (because it belongs to previous episode)
        st.session_state["assembly_raw"] = None
        st.session_state["assembly_parsed"] = None
        st.session_state["assembly_job_id"] = None
        st.session_state["assembly_job_status"] = None

        # Clear Headliner outputs
        st.session_state["headliner_raw"] = None
        st.session_state["headliner_job_id"] = None
        st.session_state["headliner_clips"] = []

        # Clear LLM outputs
        st.session_state["llm_clips"] = []
        st.session_state["llm_raw"] = None
        st.session_state["llm_alignment_debug"] = []
        st.session_state["llm_last_transcript_text"] = ""

        # Clear renders
        st.session_state["rendered_headliner"] = None
        st.session_state["rendered_llm"] = None

        # Reset transcript source selection if it becomes invalid later
        # (radio options can change depending on whether transcript exists)
        if "llm_source" in st.session_state:
            st.session_state["llm_source"] = None

    def run(self):
        st.title("Headliner vs LLM – Clip Generation Comparison")
        st.markdown("Generate clips, recompute LLM timestamps from AssemblyAI words[], and render playable snippets.")
        self.section_transcription()
        self.section_clip_generation()
        self.section_render_clips()
        self.section_compare()

    # ----------------------------
    # Transcript acquisition
    # ----------------------------

    def _save_assembly_json(self, data: Dict[str, Any]):
        parsed = parse_assembly_json(data)

        # Use the currently selected episode UID if available, so everything stays consistent.
        rss_uid = (st.session_state.get("rss_episode_uid") or "").strip()
        if rss_uid:
            parsed["episode_id"] = rss_uid

        st.session_state["assembly_raw"] = data
        st.session_state["assembly_parsed"] = parsed

    def section_transcription(self):
        st.header("1) Get transcript (RSS only)")

        default_rss = os.environ.get("LEX_FRIDMAN_RSS", "")
        rss_url = st.text_input("Podcast RSS feed URL", value=default_rss, key="rss_url")

        if st.button("Fetch RSS episodes", width="stretch", key="rss_fetch"):
            if not rss_url.strip():
                st.error("Paste an RSS URL.")
            else:
                try:
                    meta = extract_episodes_from_rss(rss_url.strip())
                    st.session_state["rss_meta"] = meta
                    st.success(f"Loaded feed '{meta['feed_title']}' with {len(meta['episodes'])} episodes.")
                except Exception as e:
                    st.error(f"RSS parse failed: {e}")

        meta = st.session_state.get("rss_meta")
        if not meta:
            return

        df = pd.DataFrame(meta["episodes"])
        st.dataframe(df[["index", "title", "published", "audio_url"]].head(30), width="stretch")

        valid = [e["index"] for e in meta["episodes"] if e.get("audio_url")]
        if not valid:
            st.warning("No episodes with audio URLs in this feed.")
            return

        idx = st.selectbox(
            "Choose episode",
            options=valid,
            format_func=lambda i: meta["episodes"][i]["title"],
            key="rss_pick",
        )
        ep = meta["episodes"][idx]

        # Detect episode change early (before widgets that depend on state are rendered)
        new_uid = _episode_uid_from_audio_url(ep.get("audio_url") or "")
        if new_uid and new_uid != (st.session_state.get("active_episode_uid") or ""):
            self._on_episode_change(ep, new_uid)

        st.write("**Episode:**", ep.get("title", ""))
        st.write("**Audio URL:**", ep.get("audio_url", ""))
        if ep.get("video_url"):
            st.write("**Video enclosure:**", ep.get("video_url", ""))

        aai_key_rss = st.text_input(
            "AssemblyAI API key",
            value=os.environ.get("ASSEMBLY_KEY", os.environ.get("ASSEMBLYAI_API_KEY", "")),
            type="password",
            key="aai_key_rss",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit transcription job", width="stretch", key="rss_submit"):
                if not aai_key_rss.strip():
                    st.error("Missing AssemblyAI API key.")
                else:
                    try:
                        with st.spinner("Submitting transcript job…"):
                            tid = assemblyai_submit_transcript(aai_key_rss.strip(), ep["audio_url"])
                        st.session_state["assembly_job_id"] = tid
                        st.session_state["assembly_job_status"] = "submitted"
                        st.success(f"Submitted. transcript_id={tid}")
                    except Exception as e:
                        st.error(f"Submit failed: {e}")

        with col2:
            tid = st.session_state.get("assembly_job_id") or ""
            st.text_input("Current transcript_id", value=str(tid), key="rss_tid_display")
            if st.button("Poll status (fetch if complete)", width="stretch", key="rss_poll"):
                if not aai_key_rss.strip():
                    st.error("Missing AssemblyAI API key.")
                elif not tid:
                    st.error("No transcript_id yet.")
                else:
                    try:
                        with st.spinner("Polling AssemblyAI…"):
                            data = assemblyai_get_transcript(aai_key_rss.strip(), str(tid))
                        st.session_state["assembly_job_status"] = data.get("status")
                        st.info(f"Status: {data.get('status')}")
                        if data.get("status") == "completed":
                            if "words" not in data or not isinstance(data.get("words"), list):
                                words = assemblyai_try_fetch_words(aai_key_rss.strip(), str(tid))
                                if words:
                                    data["words"] = words
                            self._save_assembly_json(data)
                            parsed = st.session_state.get("assembly_parsed") or {}
                            st.success(
                                f"Transcript loaded. text_chars={len(parsed.get('transcript',''))} · words={len(parsed.get('words',[]))}"
                            )
                        elif data.get("status") == "error":
                            st.error(f"AssemblyAI error: {data.get('error')}")
                    except Exception as e:
                        st.error(f"Poll failed: {e}")

        if st.session_state.get("assembly_parsed"):
            st.markdown("✅ Transcript loaded (LLM tab can use it).")

    # ----------------------------
    # Clip generation
    # ----------------------------

    def section_clip_generation(self):
        st.header("2) Generate clips")

        assembly_parsed = st.session_state.get("assembly_parsed") or {}
        episode_id = assembly_parsed.get("episode_id", "episode_1")
        words = assembly_parsed.get("words", [])

        tab_headliner, tab_llm = st.tabs(["Headliner", "LLM"])

        # ---------------- Headliner ----------------
        with tab_headliner:
            clips_file = st.file_uploader("Headliner clips JSON (optional)", type=["json"], key="headliner_upload")
            if clips_file is not None:
                try:
                    raw = json.load(clips_file)
                    clips = parse_headliner_clips(raw, episode_id=episode_id, assembly_words=words)
                    for c in clips:
                        c.source = "headliner"
                    st.session_state["headliner_raw"] = raw
                    st.session_state["headliner_clips"] = clips
                    st.session_state["rendered_headliner"] = None
                    st.success(f"Parsed {len(clips)} Headliner clips.")
                    st.dataframe(clips_summary_df(clips), width="stretch")
                except Exception as e:
                    st.error(f"Parse failed: {e}")

            api_url = st.text_input("Headliner analysis endpoint URL", value=os.environ.get("HEADLINER_ENDPOINT_URL", ""), key="headliner_api_url")

            # NOTE: Because this is a widget, its value persists. We force it on episode change in _on_episode_change().
            audio_url = st.text_input("audioUrl to send to Headliner", value=st.session_state.get("headliner_audio_url", ""), key="headliner_audio_url")

            transcript_url = st.text_input(
                "transcriptUrl to send to Headliner",
                value=st.session_state.get("transcript_url", ""),
                key="headliner_transcript_url",
            )

            include_transcript = st.checkbox(
                "Attach loaded AssemblyAI transcript JSON",
                value=bool(st.session_state.get("assembly_raw")),
                key="headliner_attach_transcript",
            )

            if st.button("Submit Headliner job", width="stretch", key="headliner_submit"):
                if requests is None:
                    st.error("`requests` not installed.")
                elif not api_url.strip() or not audio_url.strip():
                    st.error("Provide Headliner API URL and audioUrl.")
                else:
                    try:
                        payload: Dict[str, Any] = {"audioUrl": audio_url.strip()}
                        if transcript_url.strip():
                            payload["transcriptUrl"] = transcript_url.strip()
                        if include_transcript and st.session_state.get("assembly_raw"):
                            payload["transcriptJson"] = st.session_state["assembly_raw"]
                            payload["transcriptText"] = (st.session_state.get("assembly_parsed") or {}).get("transcript", "")
                        headers = {"Content-Type": "application/json"}
                        with st.spinner("Submitting Headliner job…"):
                            resp = requests.post(api_url.strip(), headers=headers, json=payload, timeout=60)
                        if resp.status_code >= 400:
                            st.error(f"Headliner error {resp.status_code}: {resp.text[:500]}")
                        else:
                            data = resp.json()
                            st.session_state["headliner_raw"] = data
                            st.session_state["headliner_job_id"] = str(data.get("id") or "")
                            st.success(f"Submitted job_id={st.session_state['headliner_job_id']}")
                    except Exception as e:
                        st.error(f"Submit failed: {e}")

            job_id = st.text_input("Headliner job id", value=str(st.session_state.get("headliner_job_id") or ""), key="headliner_job_id")
            if st.button("Poll Headliner job", width="stretch", key="headliner_poll"):
                if not api_url.strip() or not job_id.strip():
                    st.error("Provide API URL and job id.")
                else:
                    try:
                        data = fetch_headliner_job_result(api_url.strip(), int(job_id.strip()), None)
                        st.session_state["headliner_raw"] = data
                        segs = data.get("segments") or data.get("clips") or data.get("highlights")
                        if isinstance(segs, list) and segs:
                            clips = parse_headliner_clips(segs, episode_id=episode_id, assembly_words=words)
                            for c in clips:
                                c.source = "headliner"
                            st.session_state["headliner_clips"] = clips
                            st.session_state["rendered_headliner"] = None
                            st.success(f"Parsed {len(clips)} Headliner clips.")
                            st.dataframe(clips_summary_df(clips), width="stretch")
                        else:
                            st.info(f"Status: {data.get('status')!r}")
                    except Exception as e:
                        st.error(f"Poll failed: {e}")

        # ---------------- LLM ----------------
        with tab_llm:
            llm_api_key = st.text_input(
                "LLM API key (OpenAI-style)",
                value=os.environ.get("HEADLINER_OPENAI_KEY", os.environ.get("OPENAI_API_KEY", "")),
                type="password",
                key="llm_key",
            )
            llm_model = st.text_input("LLM model name", value="gpt-5.2", key="llm_model")

            # This widget value sticks; we force-update it on episode change in _on_episode_change()
            llm_audio_url = st.text_input("AUDIO_URL (required)", value=st.session_state.get("llm_audio_url", ""), key="llm_audio_url")

            llm_num_clips = st.number_input("Number of clips", min_value=1, value=20, step=1, key="llm_num_clips")
            llm_target_seconds = st.number_input("Target clip duration (seconds)", min_value=5, value=60, step=5, key="llm_target_seconds")

            st.session_state["llm_align_min_score"] = float(
                st.slider("Min match score", 0.50, 0.95, float(st.session_state.get("llm_align_min_score", 0.70)), 0.01)
            )
            st.session_state["llm_start_buffer_ms"] = int(
                st.slider("Start buffer (ms)", 0, 800, int(st.session_state.get("llm_start_buffer_ms", 180)), 10)
            )
            st.session_state["llm_end_buffer_ms"] = int(
                st.slider("End buffer (ms)", 0, 800, int(st.session_state.get("llm_end_buffer_ms", 180)), 10)
            )
            st.session_state["llm_max_start_back_tokens"] = int(
                st.slider("Max start-back tokens", 10, 120, int(st.session_state.get("llm_max_start_back_tokens", 60)), 1)
            )

            # Transcript source (safe even when episode changes)
            has_loaded = st.session_state.get("assembly_parsed") is not None
            choices = []
            if has_loaded:
                choices.append("Use loaded transcript")
            choices += ["Upload transcript JSON file", "Paste transcript JSON", "Paste transcript text"]

            # Ensure current selection is valid after episode change
            if st.session_state.get("llm_source") not in choices:
                st.session_state["llm_source"] = choices[0]
            source = st.radio("Transcript source", choices, key="llm_source")

            transcript_text = ""
            if source == "Use loaded transcript":
                transcript_text = (st.session_state.get("assembly_parsed") or {}).get("transcript", "") or ""
                st.caption(f"Loaded transcript chars: {len(transcript_text)}")
            elif source == "Upload transcript JSON file":
                up = st.file_uploader("Upload AssemblyAI transcript JSON", type=["json"], key="llm_upload_json")
                if up is not None:
                    try:
                        data = json.load(up)
                        self._save_assembly_json(data)
                        transcript_text = (data.get("text") or "").strip()
                        st.success(f"Uploaded JSON. text chars: {len(transcript_text)}")
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                else:
                    transcript_text = st.session_state.get("llm_last_transcript_text", "")
            elif source == "Paste transcript JSON":
                pasted = st.text_area("Paste JSON here", value="", height=260, key="llm_paste_json")
                if pasted.strip():
                    try:
                        data = json.loads(pasted)
                        if isinstance(data, dict):
                            self._save_assembly_json(data)
                            transcript_text = (data.get("text") or "").strip()
                        else:
                            st.warning("JSON must be an object with a 'text' field.")
                    except Exception:
                        st.warning("Not valid JSON (yet).")
            else:
                transcript_text = st.text_area("Paste transcript text here", value="", height=260, key="llm_paste_text")

            if transcript_text:
                st.session_state["llm_last_transcript_text"] = transcript_text

            if st.button("Generate clips with LLM", width="stretch", key="llm_generate"):
                try:
                    words_now = (st.session_state.get("assembly_parsed") or {}).get("words", []) or []

                    clips, raw_out = call_llm_for_episode_clips(
                        transcript_text=transcript_text,
                        audio_url=llm_audio_url,
                        model_name=llm_model,
                        api_key=llm_api_key,
                        episode_id=episode_id,
                        num_clips=int(llm_num_clips),
                        assembly_words=words_now if words_now else None,
                        target_duration_sec=int(llm_target_seconds),
                    )

                    debug_rows: List[Dict[str, Any]] = []
                    if words_now:
                        clips, debug_rows = attach_timestamps_from_words_fuzzy(
                            clips,
                            words_now,
                            overwrite=True,
                            min_score=float(st.session_state.get("llm_align_min_score", 0.70)),
                            start_buffer_ms=int(st.session_state.get("llm_start_buffer_ms", 180)),
                            end_buffer_ms=int(st.session_state.get("llm_end_buffer_ms", 180)),
                            max_start_back_tokens=int(st.session_state.get("llm_max_start_back_tokens", 60)),
                        )
                    else:
                        st.warning("No AssemblyAI words[] available; cannot realign LLM timestamps.")

                    st.session_state["llm_clips"] = clips
                    st.session_state["llm_raw"] = raw_out
                    st.session_state["llm_alignment_debug"] = debug_rows
                    st.session_state["rendered_llm"] = None

                    matched = sum(1 for r in (debug_rows or []) if r.get("matched"))
                    st.success(f"LLM returned {len(clips)} clips. Aligned {matched}/{len(clips)}.")

                except Exception as e:
                    st.error(f"LLM failed: {e}")

            llm_clips_now = st.session_state.get("llm_clips") or []
            if llm_clips_now:
                st.subheader("Current LLM clips")
                st.dataframe(clips_summary_df(llm_clips_now), width="stretch")

                dbg = st.session_state.get("llm_alignment_debug") or []
                if dbg:
                    with st.expander("LLM alignment debug", expanded=False):
                        st.dataframe(llm_alignment_df(dbg), width="stretch")

    # ----------------------------
    # Render clips
    # ----------------------------

    def section_render_clips(self):
        st.header("3) Render clips")

        headliner = st.session_state.get("headliner_clips", []) or []
        llm = st.session_state.get("llm_clips", []) or []

        if not headliner and not llm:
            st.info("Generate clips in Section 2 first.")
            return

        media_url = st.text_input(
            "Media URL used for cutting",
            value=st.session_state.get("playback_media_url", ""),
            key="playback_media_url",
        )

        if st.button("Render playable clips (MoviePy)", width="stretch", key="render_btn"):
            if not media_url.strip():
                st.error("Provide a media URL to cut.")
                return
            try:
                with st.spinner("Downloading media…"):
                    local_media = download_video_to_sources(media_url.strip())

                render_dir = os.path.join(os.getcwd(), "rendered_clips")

                if headliner:
                    with st.spinner("Rendering Headliner clips…"):
                        st.session_state["rendered_headliner"] = render_clips_with_moviepy(
                            local_media, headliner, out_dir=render_dir, force_reencode=True
                        )

                if llm:
                    # Defensive: re-align right before render
                    words_now = (st.session_state.get("assembly_parsed") or {}).get("words", []) or []
                    if words_now:
                        llm, dbg = attach_timestamps_from_words_fuzzy(
                            llm,
                            words_now,
                            overwrite=True,
                            min_score=float(st.session_state.get("llm_align_min_score", 0.70)),
                            start_buffer_ms=int(st.session_state.get("llm_start_buffer_ms", 180)),
                            end_buffer_ms=int(st.session_state.get("llm_end_buffer_ms", 180)),
                            max_start_back_tokens=int(st.session_state.get("llm_max_start_back_tokens", 60)),
                        )
                        st.session_state["llm_clips"] = llm
                        st.session_state["llm_alignment_debug"] = dbg

                    with st.spinner("Rendering LLM clips…"):
                        st.session_state["rendered_llm"] = render_clips_with_moviepy(
                            local_media, llm, out_dir=render_dir, force_reencode=True
                        )

                st.success("Rendering complete. Compare below.")
            except Exception as e:
                st.error(f"Render failed: {e}")

    # ----------------------------
    # Compare
    # ----------------------------

    @staticmethod
    def _dedupe_by_clip_id(clips: List[ClipSegment]) -> List[ClipSegment]:
        seen = set()
        out = []
        for c in clips or []:
            cid = getattr(c, "clip_id", None)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(c)
        return out

    def section_compare(self):
        st.header("4) Compare clips side-by-side")

        headliner = self._dedupe_by_clip_id(st.session_state.get("headliner_clips", []) or [])
        llm = self._dedupe_by_clip_id(st.session_state.get("llm_clips", []) or [])

        rendered_headliner = st.session_state.get("rendered_headliner") or []
        rendered_llm = st.session_state.get("rendered_llm") or []

        def _map_rendered(rendered_list):
            m: Dict[str, str] = {}
            for clip, path in rendered_list:
                if clip and getattr(clip, "clip_id", None) and path:
                    m[clip.clip_id] = path
            return m

        if not headliner or not llm:
            st.info("Need both Headliner and LLM clips to compare.")
            return

        rendered_h = _map_rendered(rendered_headliner)
        rendered_g = _map_rendered(rendered_llm)

        max_pairs = min(len(headliner), len(llm))
        show_n = st.number_input("How many pairs to show", min_value=1, max_value=max_pairs, value=min(10, max_pairs), step=1)

        n = min(len(headliner), len(llm), int(show_n))
        for i in range(n):
            h, g = headliner[i], llm[i]
            st.markdown(f"**Pair {i+1}**")
            c1, c2 = st.columns(2)

            with c1:
                st.caption(f"Headliner · {h.clip_id} · {ms_to_timestamp(h.start_ms)}–{ms_to_timestamp(h.end_ms)}")
                st.text_area("Headliner", value=h.text or "", height=160, key=f"h_{i}_{h.clip_id}")
                p = rendered_h.get(h.clip_id)
                if p:
                    with open(p, "rb") as f:
                        st.audio(f.read())

            with c2:
                st.caption(f"LLM · {g.clip_id} · {ms_to_timestamp(g.start_ms)}–{ms_to_timestamp(g.end_ms)}")
                st.text_area("LLM", value=g.text or "", height=160, key=f"g_{i}_{g.clip_id}")
                p = rendered_g.get(g.clip_id)
                if p:
                    with open(p, "rb") as f:
                        st.audio(f.read())

            st.divider()
