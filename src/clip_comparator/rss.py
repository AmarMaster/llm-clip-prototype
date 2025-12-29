from typing import Any, Dict

try:
    import feedparser
except ImportError:
    feedparser = None

def extract_episodes_from_rss(feed_url: str) -> Dict[str, Any]:
    if feedparser is None:
        raise ImportError("`feedparser` not installed. Run `pip install feedparser`.")
    feed = feedparser.parse(feed_url)
    if getattr(feed, "bozo", 0):
        raise ValueError(f"Could not parse RSS feed (bozo={feed.bozo}).")

    episodes = []
    for idx, e in enumerate(feed.entries):
        audio_url = None
        video_url = None
        enclosure_type = None

        enclosures = getattr(e, "enclosures", []) or e.get("enclosures", [])
        if enclosures:
            enc = enclosures[0]
            enclosure_type = enc.get("type") or ""
            href = enc.get("href")
            if href:
                if "video" in enclosure_type or href.lower().endswith((".mp4", ".m3u8")):
                    video_url = href
                if "audio" in enclosure_type or href.lower().endswith((".mp3", ".m4a", ".wav", ".aac")):
                    audio_url = href

        if not audio_url:
            for link in getattr(e, "links", []):
                if "audio" in (link.get("type") or ""):
                    audio_url = link.get("href")
                    break

        if not video_url:
            for link in getattr(e, "links", []):
                if "video" in (link.get("type") or "") or (link.get("href") or "").lower().endswith((".mp4", ".m3u8")):
                    video_url = link.get("href")
                    enclosure_type = link.get("type") or enclosure_type
                    break

        episodes.append(
            {
                "index": idx,
                "title": e.get("title", f"Episode {idx+1}"),
                "published": e.get("published", ""),
                "audio_url": audio_url,
                "video_url": video_url,
                "enclosure_type": enclosure_type,
            }
        )

    return {
        "feed_title": feed.feed.get("title", ""),
        "feed_link": feed.feed.get("link", feed_url),
        "feed_url": feed_url,
        "episodes": episodes,
    }
