import json
import os
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

import boto3


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_META_KEY_RE = re.compile(r"[^a-z0-9-]+")


def _clean_path_component(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = _SAFE_CHARS_RE.sub("-", s)
    s = s.strip("-")
    return s or "unknown"


def _to_ascii(s: str) -> str:
    """
    S3 user metadata must be ASCII. Normalize + strip non-ASCII.
    Example: '–' becomes '-' or gets removed depending on normalization.
    """
    if s is None:
        return ""
    s = str(s)
    # Normalize accented chars etc.
    s = unicodedata.normalize("NFKD", s)
    # Encode to ASCII, dropping anything not representable
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def _sanitize_meta_key(k: str) -> str:
    """
    Metadata keys become x-amz-meta-<key>. Keep them lowercase ASCII [a-z0-9-].
    """
    k = _to_ascii(k).strip().lower()
    k = k.replace("_", "-")
    k = _META_KEY_RE.sub("-", k)
    k = k.strip("-")
    return k[:64] if k else "meta"


def _sanitize_meta_value(v: str) -> str:
    """
    ASCII-only, trimmed. S3 metadata value limit is effectively 2KB-ish per header,
    but we keep it conservative.
    """
    v = _to_ascii(v).strip()
    # Avoid empty metadata values
    if not v:
        return ""
    return v[:1024]


def _get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Prefer Streamlit secrets if available, else env vars.
    """
    try:
        import streamlit as st  # optional at runtime
        if hasattr(st, "secrets") and name in st.secrets:
            val = st.secrets.get(name)
            if val is not None and str(val).strip() != "":
                return str(val).strip()
    except Exception:
        pass

    val = os.environ.get(name)
    if val is not None and str(val).strip() != "":
        return str(val).strip()

    return default


def _get_int_setting(name: str, default: int) -> int:
    raw = _get_setting(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def get_s3_client() -> Tuple[Any, str, str]:
    """
    Returns (client, bucket, region).
    Expects:
      AWS_ACCESS_KEY_ID
      AWS_SECRET_ACCESS_KEY
      S3_BUCKET
      S3_REGION
      S3_PREFIX (optional)
    """
    bucket = _get_setting("S3_BUCKET")
    region = _get_setting("S3_REGION", "us-east-1")

    if not bucket:
        raise ValueError("Missing S3_BUCKET (set in Streamlit secrets or env vars).")

    access_key = _get_setting("AWS_ACCESS_KEY_ID")
    secret_key = _get_setting("AWS_SECRET_ACCESS_KEY")

    if access_key and secret_key:
        session = boto3.session.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    else:
        session = boto3.session.Session(region_name=region)

    client = session.client("s3")
    return client, bucket, region


def make_transcript_key(episode_id: str) -> str:
    """
    Creates a stable object key like:
      <S3_PREFIX>/<episode_id>/transcript.json
    Default prefix is 'transcripts/'.
    """
    prefix = _get_setting("S3_PREFIX", "transcripts/")
    prefix = (prefix or "").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    ep = _clean_path_component(episode_id)
    return f"{prefix}{ep}/transcript.json"


def publish_transcript_json_to_s3(
    transcript_json: Dict[str, Any],
    episode_id: str,
    *,
    extra_metadata: Optional[Dict[str, str]] = None,
    expires_seconds: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Uploads transcript_json to S3 (private) and returns (s3_key, presigned_url).

    - Overwrites the same key per episode_id (cheap + avoids bucket clutter).
    - Adds SSE-S3 (AES256) encryption (no extra cost).
    - Cache-Control: no-store so you don't see stale versions when debugging.
    - Sanitizes metadata to ASCII (S3 requirement).
    """
    client, bucket, _region = get_s3_client()

    if expires_seconds is None:
        expires_seconds = _get_int_setting("S3_PRESIGN_SECONDS", 24 * 3600)

    # safety: avoid accidental huge uploads
    max_bytes = _get_int_setting("S3_MAX_BYTES", 6 * 1024 * 1024)  # 6MB default

    body_str = json.dumps(transcript_json, ensure_ascii=False, separators=(",", ":"))
    body_bytes = body_str.encode("utf-8")
    if len(body_bytes) > max_bytes:
        raise ValueError(
            f"Transcript JSON is {len(body_bytes)} bytes which exceeds the max {max_bytes} bytes. "
            f"Increase S3_MAX_BYTES if needed."
        )

    key = make_transcript_key(episode_id)

    metadata: Dict[str, str] = {}
    if extra_metadata:
        for k, v in extra_metadata.items():
            if v is None:
                continue
            mk = _sanitize_meta_key(str(k))
            mv = _sanitize_meta_value(str(v))
            if mv:
                metadata[mk] = mv

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body_bytes,
        ContentType="application/json; charset=utf-8",
        CacheControl="no-store",
        Metadata=metadata,
        ServerSideEncryption="AES256",
    )

    presigned = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(expires_seconds),
    )
    return key, presigned
