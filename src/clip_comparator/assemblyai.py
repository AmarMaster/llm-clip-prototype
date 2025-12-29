from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

ASSEMBLYAI_API_BASE = "https://api.assemblyai.com/v2"

def _aai_headers(api_key: str) -> Dict[str, str]:
    return {"authorization": api_key, "content-type": "application/json"}

def assemblyai_upload_file(api_key: str, file_bytes: bytes) -> str:
    if requests is None:
        raise ImportError("`requests` not installed.")
    up_url = f"{ASSEMBLYAI_API_BASE}/upload"
    headers = {"authorization": api_key}
    resp = requests.post(up_url, headers=headers, data=file_bytes, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"AssemblyAI upload failed {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    upload_url = data.get("upload_url")
    if not upload_url:
        raise RuntimeError("AssemblyAI upload returned no upload_url.")
    return upload_url

def assemblyai_submit_transcript(api_key: str, audio_url: str) -> str:
    if requests is None:
        raise ImportError("`requests` not installed.")
    url = f"{ASSEMBLYAI_API_BASE}/transcript"
    payload = {"audio_url": audio_url, "punctuate": True, "format_text": True}
    resp = requests.post(url, headers=_aai_headers(api_key), json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"AssemblyAI submit failed {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    tid = data.get("id")
    if not tid:
        raise RuntimeError("AssemblyAI submit returned no transcript id.")
    return str(tid)

def assemblyai_get_transcript(api_key: str, transcript_id: str) -> Dict[str, Any]:
    if requests is None:
        raise ImportError("`requests` not installed.")
    url = f"{ASSEMBLYAI_API_BASE}/transcript/{transcript_id}"
    resp = requests.get(url, headers=_aai_headers(api_key), timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"AssemblyAI poll failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()

def assemblyai_try_fetch_words(api_key: str, transcript_id: str) -> Optional[List[Dict[str, Any]]]:
    """Best-effort: some plans expose a /words endpoint; if not available, return None."""
    if requests is None:
        return None
    url = f"{ASSEMBLYAI_API_BASE}/transcript/{transcript_id}/words"
    try:
        resp = requests.get(url, headers=_aai_headers(api_key), timeout=60)
        if resp.status_code >= 400:
            return None
        data = resp.json()
        if isinstance(data, dict) and isinstance(data.get("words"), list):
            return data["words"]
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None
