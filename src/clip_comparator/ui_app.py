import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

from .models import ClipSegment
from .utils import (
    parse_assembly_json,
    parse_headliner_clips,
    parse_llm_style_clips,
    attach_timestamps_from_words,
    ms_to_timestamp,
    safe_clip_picker,
    clips_summary_df,
    pretty_json,
)
from .rss import extract_episodes_from_rss
from .assemblyai import (
    assemblyai_upload_file,
    assemblyai_submit_transcript,
    assemblyai_get_transcript,
    assemblyai_try_fetch_words,
)
from .headliner import fetch_headliner_job_result
from .llm import call_llm_for_episode_clips
from .media import (
    download_media_to_cache,
    download_video_to_sources,
    render_clips_with_moviepy,
    show_rendered_clips_in_order,
)

try:
    import requests
except ImportError:
    requests = None


class ClipComparatorApp:
    def __init__(self):
        st.set_page_config(page_title="Headliner vs LLM – Clip Comparison", layout="wide")
        self.ensure_session_state()
        self.run()

    @staticmethod
    def ensure_session_state():
        ClipComparatorApp._load_dotenv()
        st.session_state.setdefault("assembly_raw", None)
        st.session_state.setdefault("assembly_parsed", None)
        st.session_state.setdefault("rss_meta", None)

        st.session_state.setdefault("assembly_job_id", None)
        st.session_state.setdefault("assembly_job_status", None)

        st.session_state.setdefault("headliner_raw", None)
        st.session_state.setdefault("headliner_clips", [])
        st.session_state.setdefault("llm_clips", [])
        st.session_state.setdefault("llm_raw", None)

        st.session_state.setdefault("transcript_url", os.environ.get("LEX_FRIDMAN_TRSCPT", ""))
        st.session_state.setdefault("headliner_job_id", None)

        # rendered clip paths (cached per run)
        st.session_state.setdefault("rendered_headliner", None)
        st.session_state.setdefault("rendered_llm", None)
        st.session_state.setdefault("rendered_uploaded", None)
        st.session_state.setdefault("uploaded_clip_set", [])
        st.session_state.setdefault("episode_video_url", "")

        # keep last used transcript text source so it doesn't "feel empty" after reruns
        st.session_state.setdefault("llm_last_transcript_text", "")

    @staticmethod
    def _load_dotenv():
        """Lightweight .env loader so keys are available without manual input."""
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

    def run(self):
        st.title("Headliner vs LLM – Clip Generation Comparison")

        st.markdown(
            """
Compare Headliner and LLM-generated podcast clips in one place: pull transcripts from RSS, send them to either service, preview returned clips (with timestamps when available), cut playable snippets with MoviePy, and review them side by side.
            """
        )

        self.section_transcription()
        self.section_clip_generation()
        self.section_render_clips()
        self.section_compare()

    # ----------------------------
    # Transcript acquisition
    # ----------------------------

    def _show_loaded_assembly(self):
        raw = st.session_state.get("assembly_raw")
        parsed = st.session_state.get("assembly_parsed")
        if not raw or not parsed:
            return

        st.success(
            f"Transcript loaded. episode_id={parsed['episode_id']} · "
            f"text_chars={len(parsed['transcript'])} · words={len(parsed['words'])}"
        )

        raw_pretty = pretty_json(raw)
        with st.expander("Raw AssemblyAI JSON (copy / download)", expanded=False):
            st.text_area("AssemblyAI JSON", value=raw_pretty, height=260)
            st.download_button(
                "⬇️ Download AssemblyAI JSON",
                data=raw_pretty.encode("utf-8"),
                file_name=f"{parsed['episode_id']}_assemblyai.json",
                mime="application/json",
            )

        with st.expander("Transcript preview (first 2000 chars)", expanded=False):
            st.text_area("Transcript (preview)", value=parsed["transcript"][:2000], height=200)

    def _save_assembly_json(self, data: Dict[str, Any]):
        parsed = parse_assembly_json(data)
        st.session_state["assembly_raw"] = data
        st.session_state["assembly_parsed"] = parsed

    def section_transcription(self):
        st.header("1) Get transcript (RSS only)")

        default_rss = os.environ.get("LEX_FRIDMAN_RSS", "")
        rss_url = st.text_input("Podcast RSS feed URL (pod.link RSS works too)", value=default_rss, key="rss_url")
        if st.button("Fetch RSS episodes", use_container_width=True, key="rss_fetch"):
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
        if meta:
            df = pd.DataFrame(meta["episodes"])
            st.dataframe(df[["index", "title", "published", "audio_url"]].head(30), use_container_width=True)

            valid = [e["index"] for e in meta["episodes"] if e.get("audio_url")]
            if not valid:
                st.warning("No episodes with audio URLs in this feed.")
            else:
                idx = st.selectbox(
                    "Choose episode",
                    options=valid,
                    format_func=lambda i: meta["episodes"][i]["title"],
                    key="rss_pick",
                )
                ep = meta["episodes"][idx]
                st.write("**Episode:**", ep["title"])
                st.write("**Audio URL:**", ep["audio_url"])
                if ep.get("video_url"):
                    st.write("**Video enclosure:**", ep["video_url"])
                    st.session_state["episode_video_url"] = ep["video_url"]

                st.session_state["headliner_audio_url"] = ep["audio_url"]
                st.session_state["llm_audio_url"] = ep["audio_url"]

                aai_key_rss = st.text_input(
                    "AssemblyAI API key",
                    value=os.environ.get("ASSEMBLY_KEY", os.environ.get("ASSEMBLYAI_API_KEY", "")),
                    type="password",
                    key="aai_key_rss",
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Submit transcription job for this episode", use_container_width=True, key="rss_submit"):
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
                    if st.button("Poll status (fetch result if complete)", use_container_width=True, key="rss_poll"):
                        if not aai_key_rss.strip():
                            st.error("Missing AssemblyAI API key.")
                        elif not tid:
                            st.error("No transcript_id yet. Submit first.")
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
                                    st.success("RSS transcript loaded into session (usable in LLM tab).")
                                    self._show_loaded_assembly()
                                elif data.get("status") == "error":
                                    st.error(f"AssemblyAI error: {data.get('error')}")
                                else:
                                    with st.expander("Raw poll JSON", expanded=False):
                                        st.json(data)
                            except Exception as e:
                                st.error(f"Poll failed: {e}")

        if st.session_state.get("assembly_parsed"):
            st.markdown("✅ A transcript is currently loaded in session (LLM tab can use it).")

    # ----------------------------
    # Clip generation
    # ----------------------------

    def section_clip_generation(self):
        st.header("2) Generate clips")

        assembly_parsed = st.session_state.get("assembly_parsed") or {}
        episode_id = assembly_parsed.get("episode_id", "episode_1")
        words = assembly_parsed.get("words", [])

        tab_headliner, tab_llm = st.tabs(["Headliner", "LLM"])

        # ---------------- Headliner tab ----------------
        with tab_headliner:
            st.markdown("Upload Headliner clips JSON OR submit an async analysis job, then fetch results.")
            st.caption("Note: Headliner API currently decides how many clips/segments it returns; there is no documented parameter to request a specific count.")

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
                except Exception as e:
                    st.error(f"Parse failed: {e}")

            api_url = st.text_input("Headliner analysis endpoint URL", value=os.environ.get("HEADLINER_ENDPOINT_URL", ""), key="headliner_api_url")
            audio_default = st.session_state.get("headliner_audio_url") or st.session_state.get("llm_audio_url") or ""
            audio_url = st.text_input("audioUrl to send to Headliner", value=audio_default, key="headliner_audio_url")
            transcript_url = st.text_input("transcriptUrl to send to Headliner (optional)", value=st.session_state.get("transcript_url", ""), key="headliner_transcript_url")
            include_transcript = st.checkbox(
                "Attach loaded AssemblyAI transcript JSON",
                value=bool(st.session_state.get("assembly_raw")),
                key="headliner_attach_transcript",
            )

            if st.session_state.get("headliner_job_id") not in (None, ""):
                st.session_state["headliner_job_id"] = str(st.session_state["headliner_job_id"])

            if st.button("Submit Headliner analysis job", use_container_width=True, key="headliner_submit"):
                if requests is None:
                    st.error("`requests` not installed.")
                elif not api_url.strip():
                    st.error("Provide Headliner API URL.")
                elif not audio_url.strip():
                    st.error("Provide audioUrl.")
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
                            st.error(f"Headliner error {resp.status_code}")
                            with st.expander("Headliner error details", expanded=True):
                                st.write("Status code:", resp.status_code)
                                st.write("Response text:", resp.text)
                                st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json", label="Request payload")
                        else:
                            data = resp.json()
                            st.session_state["headliner_raw"] = data
                            st.session_state["headliner_job_id"] = str(data.get("id") or "")
                            st.success(f"Submitted. job_id={st.session_state['headliner_job_id']}, status={data.get('status')!r}")
                    except Exception as e:
                        st.error(f"Submit failed: {e}")
                        st.exception(e)

            job_id = st.text_input("Headliner job id", value=str(st.session_state.get("headliner_job_id") or ""), key="headliner_job_id")
            if st.button("Poll Headliner job", use_container_width=True, key="headliner_poll"):
                if not api_url.strip():
                    st.error("Provide Headliner API URL.")
                elif not job_id.strip():
                    st.error("Provide job id.")
                else:
                    try:
                        data = fetch_headliner_job_result(api_url.strip(), int(job_id.strip()),None)
                        st.session_state["headliner_raw"] = data
                        segs = data.get("segments") or data.get("clips") or data.get("highlights")
                        st.info(f"Status: {data.get('status')!r}")
                        if isinstance(segs, list) and segs:
                            clips = parse_headliner_clips(segs, episode_id=episode_id, assembly_words=words)
                            for c in clips:
                                c.source = "headliner"
                            st.session_state["headliner_clips"] = clips
                            st.session_state["rendered_headliner"] = None
                            st.success(f"Parsed {len(clips)} Headliner clips.")
                        with st.expander("Raw Headliner JSON", expanded=False):
                            st.json(data)
                    except Exception as e:
                        st.error(f"Poll failed: {e}")
                        st.exception(e)

            # ✅ Always show current Headliner results (persists across reruns)
            headliner_clips = st.session_state.get("headliner_clips") or []
            if headliner_clips:
                st.subheader("Current Headliner clips")
                self._show_clip_set_preview(headliner_clips, "Headliner", key_prefix="headliner_current")

        # ---------------- LLM tab ----------------
        with tab_llm:
            st.markdown("You can **upload the downloaded transcript JSON file** and the app will call the LLM using its transcript text.")

            llm_api_key = st.text_input(
                "LLM API key (OpenAI-style)",
                value=os.environ.get("HEADLINER_OPENAI_KEY", os.environ.get("OPENAI_API_KEY", "")),
                type="password",
                key="llm_key",
            )
            llm_model = st.text_input("LLM model name", value="gpt-4o-mini", key="llm_model")
            llm_audio_url = st.text_input("AUDIO_URL (required)", value=st.session_state.get("headliner_audio_url", ""), key="llm_audio_url")
            llm_num_clips = st.number_input("Number of clips", min_value=1, value=20, step=1, key="llm_num_clips")
            llm_target_seconds = st.number_input("Target clip duration (seconds)", min_value=5, value=60, step=5, key="llm_target_seconds")

            has_loaded = st.session_state.get("assembly_parsed") is not None

            choices = []
            if has_loaded:
                choices.append("Use loaded transcript (from RSS/upload/transcribe)")
            choices += ["Upload transcript JSON file", "Paste transcript JSON", "Paste transcript text"]

            source = st.radio("Transcript source", choices, key="llm_source")
            transcript_text = ""

            if source == "Use loaded transcript (from RSS/upload/transcribe)":
                transcript_text = (st.session_state.get("assembly_parsed") or {}).get("transcript", "") or ""
                st.caption(f"Loaded transcript chars: {len(transcript_text)}")
                with st.expander("Loaded transcript preview", expanded=False):
                    st.text_area("Preview", value=transcript_text[:2000], height=200)

            elif source == "Upload transcript JSON file":
                up = st.file_uploader("Upload AssemblyAI transcript JSON", type=["json"], key="llm_upload_json")
                if up is not None:
                    try:
                        data = json.load(up)
                        self._save_assembly_json(data)
                        transcript_text = (data.get("text") or "").strip()
                        st.success(f"Uploaded JSON. Extracted text chars: {len(transcript_text)}")
                        with st.expander("Transcript preview", expanded=False):
                            st.text_area("Preview", value=transcript_text[:2000], height=200)
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                else:
                    # persist last transcript text so it doesn't feel "empty" after reruns
                    transcript_text = st.session_state.get("llm_last_transcript_text", "")

            elif source == "Paste transcript JSON":
                pasted = st.text_area("Paste JSON here", value="", height=260, key="llm_paste_json")
                if pasted.strip():
                    try:
                        data = json.loads(pasted)
                        if isinstance(data, dict):
                            self._save_assembly_json(data)
                            transcript_text = (data.get("text") or "").strip()
                            st.caption(f"Extracted transcript chars: {len(transcript_text)}")
                        else:
                            st.warning("JSON must be an object with a 'text' field.")
                    except Exception:
                        st.warning("Not valid JSON (yet).")
            else:
                transcript_text = st.text_area("Paste transcript text here", value="", height=260, key="llm_paste_text")

            if transcript_text:
                st.session_state["llm_last_transcript_text"] = transcript_text

            if st.button("Generate clips with LLM", use_container_width=True, key="llm_generate"):
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

                    st.session_state["llm_clips"] = clips
                    st.session_state["llm_raw"] = raw_out
                    st.session_state["rendered_llm"] = None
                    st.success(f"LLM returned {len(clips)} clips.")
                except Exception as e:
                    st.error(f"LLM failed: {e}")

            # ✅ Always show current LLM results (persists across reruns)
            llm_clips_now = st.session_state.get("llm_clips") or []
            if llm_clips_now:
                st.subheader("Current LLM clips")
                self._show_clip_set_preview(llm_clips_now, "LLM", key_prefix="llm_current")

                with st.expander("Raw LLM JSON", expanded=False):
                    if st.session_state.get("llm_raw") is not None:
                        st.json(st.session_state["llm_raw"])
                    else:
                        st.info("No raw LLM JSON stored yet.")

    def _show_clip_set_preview(self, clips: List[ClipSegment], label: str, key_prefix: str):
        if not clips:
            return
        st.dataframe(clips_summary_df(clips), use_container_width=True)

    # ----------------------------
    # Render clips (MoviePy) using timestamps
    # ----------------------------

    def section_render_clips(self):
        st.header("3) Render clips")

        headliner = st.session_state.get("headliner_clips", []) or []
        llm = st.session_state.get("llm_clips", []) or []

        st.markdown("Cut the media into playable clips.")

        uploaded_json = st.file_uploader("Clip JSON with timestamps (optional)", type=["json"], key="render_clip_upload")
        uploaded_format = st.radio("Uploaded JSON format", ["Headliner", "LLM"], horizontal=True, key="uploaded_format")

        if uploaded_json is not None:
            try:
                data = json.load(uploaded_json)
                if uploaded_format == "Headliner":
                    parsed = parse_headliner_clips(data, episode_id="uploaded", assembly_words=None)
                else:
                    parsed = parse_llm_style_clips(data, episode_id="uploaded", assembly_words=None)
                for c in parsed:
                    c.source = "uploaded"
                st.session_state["uploaded_clip_set"] = parsed
                st.success(f"Loaded {len(parsed)} clips from uploaded JSON.")
                # ✅ Do NOT show full clip texts here (per your request)
                st.dataframe(clips_summary_df(parsed), use_container_width=True)
            except Exception as e:
                st.error(f"Could not parse uploaded JSON: {e}")

        uploaded = st.session_state.get("uploaded_clip_set", []) or []

        if not headliner and not llm and not uploaded:
            st.info("Generate clips in Section 2 or upload a clip JSON to render.")
            return

        video_url_hint = st.session_state.get("episode_video_url") or ""
        if video_url_hint:
            st.info(f"Detected video enclosure from RSS: {video_url_hint}")

        media_url = st.text_input(
            "Media URL used for cutting (must match transcript timing)",
            value=video_url_hint or st.session_state.get("llm_audio_url") or st.session_state.get("headliner_audio_url") or "",
            key="playback_media_url",
        )

        # render flags
        do_headliner = bool(headliner)
        do_llm = bool(llm)
        do_uploaded = bool(uploaded)

        # Quick status line (no clip text)
        st.caption(
            f"Ready to render: "
            f"Headliner={len(headliner)} · LLM={len(llm)} · Uploaded={len(uploaded)}"
        )

        if st.button("Render playable clips (MoviePy)", use_container_width=True, key="render_btn"):
            if not media_url.strip():
                st.error("Provide a media URL to cut.")
            else:
                try:
                    with st.spinner("Downloading episode media…"):
                        local_media = download_video_to_sources(media_url.strip())

                    render_dir = os.path.join(os.getcwd(), "rendered_clips")

                    if do_headliner and headliner:
                        with st.spinner("Rendering Headliner clips…"):
                            st.session_state["rendered_headliner"] = render_clips_with_moviepy(
                                local_media, headliner, out_dir=render_dir, force_reencode=True
                            )

                    if do_llm and llm:
                        missing = [c.clip_id for c in llm if c.start_ms is None or c.end_ms is None]
                        if missing:
                            st.warning(
                                "Some LLM clips have no timestamps (cannot be cut). "
                                "This usually means AssemblyAI `words[]` was not available or matching failed.\n"
                                f"Missing: {missing}"
                            )
                        with st.spinner("Rendering LLM clips…"):
                            st.session_state["rendered_llm"] = render_clips_with_moviepy(
                                local_media, llm, out_dir=render_dir, force_reencode=True
                            )

                    if do_uploaded and uploaded:
                        missing = [c.clip_id for c in uploaded if c.start_ms is None or c.end_ms is None]
                        if missing:
                            st.warning("Uploaded clips missing timestamps: " + ", ".join(missing))
                        with st.spinner("Rendering uploaded clips…"):
                            st.session_state["rendered_uploaded"] = render_clips_with_moviepy(
                                local_media, uploaded, out_dir=render_dir, force_reencode=True
                            )

                    st.success("Rendering complete. Compare playback below in Section 4.")
                except Exception as e:
                    st.error(f"Render failed: {e}")

    # ----------------------------
    # Side-by-side comparison
    # ----------------------------

    @staticmethod
    def _dedupe_by_clip_id(clips: List[ClipSegment]) -> List[ClipSegment]:
        """Preserve order, keep first occurrence of each clip_id."""
        seen = set()
        out = []
        for c in clips or []:
            cid = getattr(c, "clip_id", None)
            if cid is None:
                continue
            if cid in seen:
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
            st.info("Add both Headliner and LLM clips (Section 2) to compare them here.")
            return

        max_pairs = min(len(headliner), len(llm))
        show_n = st.number_input(
            "How many pairs to show",
            min_value=1,
            max_value=max_pairs,
            value=min(20, max_pairs),
            step=1,
            key="compare_pairs_n",
        )

        self._show_side_by_side(
            headliner[: int(show_n)],
            llm[: int(show_n)],
            _map_rendered(rendered_headliner),
            _map_rendered(rendered_llm),
        )

    def _show_side_by_side(
        self,
        headliner: List[ClipSegment],
        llm: List[ClipSegment],
        rendered_headliner: Dict[str, str],
        rendered_llm: Dict[str, str],
    ):
        n = min(len(headliner), len(llm))
        for i in range(n):
            h, g = headliner[i], llm[i]
            st.markdown(f"**Pair {i+1}**")
            c1, c2 = st.columns(2)

            with c1:
                st.caption(f"Headliner · {h.clip_id} · {ms_to_timestamp(h.start_ms)}–{ms_to_timestamp(h.end_ms)}")
                st.text_area("Headliner", value=h.text or "", height=160, key=f"h_{i}_{h.clip_id}")
                h_path = rendered_headliner.get(h.clip_id)
                if h_path:
                    ext = os.path.splitext(h_path)[1].lower()
                    with open(h_path, "rb") as f:
                        data = f.read()
                    if ext in [".mp4", ".mov", ".mkv", ".webm"]:
                        st.video(data)
                    else:
                        st.audio(data)

            with c2:
                st.caption(f"LLM · {g.clip_id} · {ms_to_timestamp(g.start_ms)}–{ms_to_timestamp(g.end_ms)}")
                st.text_area("LLM", value=g.text or "", height=160, key=f"g_{i}_{g.clip_id}")
                g_path = rendered_llm.get(g.clip_id)
                if g_path:
                    ext = os.path.splitext(g_path)[1].lower()
                    with open(g_path, "rb") as f:
                        data = f.read()
                    if ext in [".mp4", ".mov", ".mkv", ".webm"]:
                        st.video(data)
                    else:
                        st.audio(data)

            st.divider()
