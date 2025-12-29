from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None

def fetch_headliner_job_result(api_url: str, job_id: int, api_key: Optional[str] = None) -> Dict[str, Any]:
    if requests is None:
        raise ImportError("`requests` not installed.")
    status_url = f"{api_url.rstrip('/')}/{job_id}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(status_url, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Headliner poll failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()
