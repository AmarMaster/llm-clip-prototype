import json
import os
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.config import Config


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
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def _sanitize_meta_key(k: str) -> str:
    k = _to_ascii(k).strip().lower()
    k = k.replace("_", "-")
    k = _META_KEY_RE.sub("-", k)
    k = k.strip("-")
    return k[:64] if k else "meta"


def _sanitize_meta_value(v: str) -> str:
    v = _to_ascii(v).strip()
    if not v:
        return ""
    return v[:1024]


def _get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
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


def _make_session(region: Optional[str]) -> boto3.session.Session:
    access_key = _get_setting("AWS_ACCESS_KEY_ID")
    secret_key = _get_setting("AWS_SECRET_ACCESS_KEY")
    session_token = _get_setting("AWS_SESSION_TOKEN")

    kwargs: Dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token

    return boto3.session.Session(**kwargs)


def get_s3_client() -> Tuple[Any, str, str]:
    """
    Returns (client, bucket, region).

    Expects:
      S3_BUCKET
      S3_REGION (optional; auto-detected if missing)
      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (optional)
      AWS_SESSION_TOKEN (optional)
    """
    bucket = _get_setting("S3_BUCKET")
    if not bucket:
        raise ValueError("Missing S3_BUCKET (set in Streamlit secrets or env vars).")

    region = _get_setting("S3_REGION", None)

    # Force SigV4 everywhere
    cfg = Config(signature_version="s3v4")

    # If region provided, use it directly
    if region:
        session = _make_session(region)
        client = session.client("s3", config=cfg)
        return client, bucket, region

    # No region provided: create a client, then detect bucket region
    session = _make_session("us-east-1")
    client = session.client("s3", config=cfg)

    try:
        loc = client.get_bucket_location(Bucket=bucket) or {}
        bucket_region = loc.get("LocationConstraint") or "us-east-1"
    except Exception:
        bucket_region = "us-east-1"

    # Rebuild client for the detected region (important for presigned URLs)
    if bucket_region != "us-east-1":
        session2 = _make_session(bucket_region)
        client2 = session2.client("s3", config=cfg)
        return client2, bucket, bucket_region

    return client, bucket, "us-east-1"


def make_transcript_key(episode_id: str) -> str:
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
    client, bucket, _region = get_s3_client()

    if expires_seconds is None:
        expires_seconds = _get_int_setting("S3_PRESIGN_SECONDS", 24 * 3600)

    max_bytes = _get_int_setting("S3_MAX_BYTES", 6 * 1024 * 1024)

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
