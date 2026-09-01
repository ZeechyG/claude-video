#!/usr/bin/env python3
"""Native YouTube video understanding via the Gemini API.

The frame pipeline needs pixels, which needs a download. When YouTube refuses
to serve the media (datacenter IPs get a bot check, so CI boxes and cloud
sandboxes are locked out), the visual half of the timeline is simply gone.

Gemini accepts a YouTube URL directly as a `file_data` part — Google's API
reading Google's platform, no download and no credentials — and can describe
what is on screen over time. That restores the visual channel where frames are
impossible.

The tradeoff is real and callers must surface it: these beats are Gemini's
description of the video, not frames Claude looked at. Secondhand seeing. Use
it as a fallback, not as a replacement for frames when frames are available.

Pure stdlib; shares the Gemini request plumbing with whisper.py.
"""
from __future__ import annotations

import json
import re
import sys
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

from whisper import (
    GEMINI_ENDPOINT,
    _coerce_seconds,
    _request_with_retries,
    gemini_model,
    read_setting,
)


# Primary is whatever GEMINI_MODEL resolves to (an alias by default). The rest
# are concrete fallbacks for when that one is busy: a 503 "high demand" is
# common enough on the aliases that waiting out a backoff is worse than asking
# a different model. A fallback that has since been retired 404s and is skipped
# the same way, so a stale entry here degrades instead of breaking.
GEMINI_VIDEO_FALLBACKS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-pro")

# Fewer attempts per model than the audio path: there is somewhere better to go.
ATTEMPTS_PER_MODEL = 2

GEMINI_VIDEO_MAX_OUTPUT_TOKENS = 32768

_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be",
}

VISUAL_PROMPT = (
    "Describe what is VISIBLE on screen in this video, as a timeline.\n\n"
    "Report only the picture: shot changes, what the presenter is physically "
    "doing, diagrams and how they develop, text and graphics burned into the "
    "frame, b-roll, screen recordings, slides, end cards. Read out any on-screen "
    "text verbatim — captions, labels, URLs, lower thirds — since that text is "
    "often the part a transcript cannot capture.\n\n"
    "Do NOT summarize what is said. Speech belongs to a separate transcript; a "
    "beat here should describe the image even when someone is talking over it.\n\n"
    "Emit one beat per distinct visual change, in chronological order. Skip beats "
    "where nothing visible changed — do not pad the timeline with a beat per "
    "second.\n\n"
    "`time` is seconds from the beginning of THIS clip, as a number — the clip you "
    "were given starts at 0. If you were given an excerpt, do not try to add its "
    "offset into the full video; the caller applies that itself."
)

VISUAL_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "time": {"type": "NUMBER"},
            "on_screen": {"type": "STRING"},
        },
        "required": ["time", "on_screen"],
    },
}


_VIDEO_ID = r"([A-Za-z0-9_-]{11})"
_PATH_FORMS = (
    rf"^/watch$",                 # /watch?v=ID
    rf"^/shorts/{_VIDEO_ID}",
    rf"^/embed/{_VIDEO_ID}",
    rf"^/live/{_VIDEO_ID}",
    rf"^/v/{_VIDEO_ID}",
)


def youtube_video_id(source: str) -> str | None:
    """Extract the 11-character video id from any YouTube URL form, else None."""
    try:
        parsed = urlparse(source.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https"):
        return None

    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        return None

    # youtu.be/ID puts the id in the path.
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if re.fullmatch(_VIDEO_ID, candidate) else None

    if parsed.path.rstrip("/") in ("/watch", ""):
        values = parse_qs(parsed.query).get("v") or []
        return values[0] if values and re.fullmatch(_VIDEO_ID, values[0]) else None

    for form in _PATH_FORMS[1:]:
        match = re.match(form, parsed.path)
        if match:
            return match.group(1)
    return None


def canonical_youtube_url(source: str) -> str | None:
    """Rebuild a YouTube URL in the one form Gemini's file_data accepts.

    Share links arrive carrying tracking parameters (`?pp=…&ra=…`), and Gemini
    rejects those with a bare 400 INVALID_ARGUMENT — the host is fine, the extra
    query is not. Reducing to `watch?v=<id>` is what makes a pasted mobile share
    link work.
    """
    video_id = youtube_video_id(source)
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else None


def is_youtube_url(source: str) -> bool:
    """True if `source` is a YouTube link Gemini can ingest by reference.

    Gemini's file_data path takes YouTube links specifically; any other host
    would need a Files API upload first, which needs the bytes we cannot get.
    """
    return youtube_video_id(source) is not None


def video_models() -> list[str]:
    """Configured model first, then distinct fallbacks."""
    ordered = [gemini_model(), *GEMINI_VIDEO_FALLBACKS]
    seen: set[str] = set()
    return [m for m in ordered if not (m in seen or seen.add(m))]


def _offset(seconds: float | None) -> str | None:
    return None if seconds is None else f"{max(0.0, float(seconds)):.0f}s"


def _build_payload(
    url: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> dict:
    # Gemini 400s on share-link tracking params, so only the canonical form goes out.
    part: dict = {"file_data": {"file_uri": canonical_youtube_url(url) or url}}

    # video_metadata clips server-side, so a focused run never pays to have the
    # whole video analyzed.
    window = {}
    if (start := _offset(start_seconds)) is not None:
        window["start_offset"] = start
    if (end := _offset(end_seconds)) is not None:
        window["end_offset"] = end
    if window:
        part["video_metadata"] = window

    return {
        "contents": [{"parts": [{"text": VISUAL_PROMPT}, part]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": VISUAL_SCHEMA,
            "maxOutputTokens": GEMINI_VIDEO_MAX_OUTPUT_TOKENS,
        },
    }


def parse_beats(data: dict) -> list[dict]:
    """Convert a generateContent response into {start, on_screen} beats."""
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        raise SystemExit(
            f"Gemini returned no candidates{f' (blocked: {reason})' if reason else ''}"
        )

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    parts = ((candidate.get("content") or {}).get("parts")) or []
    raw = "".join(part.get("text") or "" for part in parts).strip()

    if finish == "MAX_TOKENS":
        raise SystemExit(
            "Gemini hit its output limit describing this video. Re-run with "
            "--start/--end to narrow the window."
        )
    if not raw:
        if finish and finish != "STOP":
            raise SystemExit(f"Gemini returned no visual timeline (finishReason: {finish})")
        return []

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Gemini returned unparseable JSON: {exc}: {raw[:200]}")
    if not isinstance(items, list):
        raise SystemExit(f"Gemini returned {type(items).__name__}, expected a list of beats")

    beats: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = (item.get("on_screen") or "").strip()
        if not text:
            continue
        beats.append({"start": round(_coerce_seconds(item.get("time")), 2), "on_screen": text})

    beats.sort(key=lambda b: b["start"])
    return beats


def shift_beats(beats: list[dict], offset_seconds: float) -> list[dict]:
    """Move clip-relative beats into absolute source time.

    `video_metadata` clips server-side, so Gemini numbers a focused run from 0.
    Every timestamp this skill reports is absolute, so the window offset goes
    back on here — the same stitching the audio chunker does.
    """
    if not offset_seconds:
        return beats
    return [
        {"start": round(beat["start"] + offset_seconds, 2), "on_screen": beat["on_screen"]}
        for beat in beats
    ]


def usage_tokens(data: dict) -> int:
    return int((data.get("usageMetadata") or {}).get("totalTokenCount") or 0)


def fetch_visual_timeline(
    url: str,
    api_key: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    models: list[str] | None = None,
) -> tuple[list[dict], str, int]:
    """Ask Gemini to describe the video. Returns (beats, model_used, tokens).

    Tries each model in turn: a busy alias (503) or a retired pin (404) moves
    on to the next rather than failing the run.
    """
    body = json.dumps(_build_payload(url, start_seconds, end_seconds)).encode("utf-8")
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    chain = models or video_models()
    last_error: BaseException | None = None
    for index, model in enumerate(chain):
        endpoint = GEMINI_ENDPOINT.format(model=model)
        try:
            data = _request_with_retries(
                lambda: Request(endpoint, data=body, headers=headers, method="POST"),
                f"gemini-video ({model})",
                max_attempts=ATTEMPTS_PER_MODEL,
            )
            beats = shift_beats(parse_beats(data), start_seconds or 0.0)
            return beats, model, usage_tokens(data)
        except SystemExit as exc:
            last_error = exc
            if index < len(chain) - 1:
                print(
                    f"[watch] {model} unavailable for video ({exc}) — trying "
                    f"{chain[index + 1]}…",
                    file=sys.stderr,
                )

    raise SystemExit(f"Gemini video analysis failed on every model tried: {last_error}")


def format_beats(beats: list[dict]) -> str:
    """Render beats as `[MM:SS] description` lines."""
    lines = []
    for beat in beats:
        total = int(beat["start"])
        stamp = (
            f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
            if total >= 3600
            else f"{total // 60:02d}:{total % 60:02d}"
        )
        lines.append(f"[{stamp}] {beat['on_screen']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: gemini_video.py <youtube-url> [start_seconds] [end_seconds]", file=sys.stderr)
        raise SystemExit(2)

    source = sys.argv[1]
    if not is_youtube_url(source):
        raise SystemExit(f"not a YouTube URL: {source}")
    key = read_setting("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")

    start = float(sys.argv[2]) if len(sys.argv) > 2 else None
    end = float(sys.argv[3]) if len(sys.argv) > 3 else None
    result, used, tokens = fetch_visual_timeline(source, key, start, end)
    print(f"# {len(result)} beats via {used} ({tokens} tokens)\n", file=sys.stderr)
    print(format_beats(result))
