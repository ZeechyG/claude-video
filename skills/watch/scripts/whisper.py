#!/usr/bin/env python3
"""Transcribe a video via Groq / OpenAI Whisper, or Google Gemini.

Strategy: extract audio (mono 16kHz mp3, tiny payload), upload to whichever
API has a key. Returns segments in the same shape as transcribe.parse_vtt so
the rest of the pipeline (filter_range, format_transcript) doesn't care where
the transcript came from.

Groq and OpenAI both expose Whisper behind an identical multipart endpoint
that returns timestamped segments natively. Gemini is a different shape: it
takes base64 audio on `generateContent` and returns free-form output, so we
ask for JSON via a response schema and coerce it into the same segment format.

Pure stdlib — no `pip install groq` / `openai` / `google-genai` needed.
"""
from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

OPENAI_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
# Deliberately the moving alias, not a pinned version: pinned Gemini models get
# retired ("no longer available to new users") and would strand installed copies
# of this skill. Override with GEMINI_MODEL for a specific one.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# Both Groq's free tier and OpenAI whisper-1 cap uploads at 25 MB. We target a
# margin under that so multipart framing overhead never pushes a chunk over.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

# Gemini takes audio as base64 inside the JSON request body, and the whole
# request has to stay under 20 MB. Base64 inflates by 4/3, so the raw mp3 has
# to be well under 15 MB.
GEMINI_MAX_UPLOAD_BYTES = 14 * 1024 * 1024

# Gemini is bounded by *output* tokens long before it hits that byte cap: it
# writes the transcript as JSON, so a 30-minute chunk would overrun the
# response limit and come back as truncated (unparseable) JSON. 10 minutes of
# speech lands comfortably inside GEMINI_MAX_OUTPUT_TOKENS.
GEMINI_MAX_CHUNK_SECONDS = 600.0
# Stays under the smallest output limit across the Gemini models worth pointing
# this at (the transcribe-specific ones cap at 32k), and 10 minutes of speech
# is far short of it either way.
GEMINI_MAX_OUTPUT_TOKENS = 32768


def plan_chunks(
    total_seconds: float,
    total_bytes: int,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """Split a duration into contiguous (offset, duration) chunks under max_bytes.

    Size scales linearly with duration (constant-bitrate mono mp3), so an even
    time split yields evenly-sized chunks. Returns a single full-length chunk
    when the audio already fits.

    `max_seconds` adds a second ceiling for backends (Gemini) whose real limit
    is how much transcript they can emit per call, not how much audio they can
    ingest. The chunk count is whichever constraint bites harder.
    """
    fits_bytes = total_bytes <= max_bytes
    fits_time = max_seconds is None or total_seconds <= max_seconds
    if (fits_bytes and fits_time) or total_seconds <= 0:
        return [(0.0, total_seconds)]

    n = math.ceil(total_bytes / max_bytes)
    if max_seconds:
        n = max(n, math.ceil(total_seconds / max_seconds))
    chunk = total_seconds / n
    plan: list[tuple[float, float]] = []
    for i in range(n):
        offset = i * chunk
        # The last chunk absorbs any rounding remainder so durations sum exactly.
        duration = (total_seconds - offset) if i == n - 1 else chunk
        plan.append((round(offset, 3), round(duration, 3)))
    return plan


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Return (backend, api_key). Prefers Groq, then OpenAI, then Gemini.

    Gemini is last in auto-detect order because the Whisper backends return
    timestamps natively; Gemini's are model-generated. Select it explicitly
    with `preferred="gemini"` (`--whisper gemini`) to override that.

    If `preferred` is set, only that backend's key is considered.
    """
    candidates = (
        ("GROQ_API_KEY", "groq"),
        ("OPENAI_API_KEY", "openai"),
        ("GEMINI_API_KEY", "gemini"),
    )
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = read_setting(key_name)
        if value:
            return backend, value

    return None, None


def _dotenv_paths() -> list[Path]:
    # Resolved per call, not at import: the cwd .env depends on where /watch ran.
    return [Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"]


def _from_dotenv(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return value or None
    except OSError:
        return None
    return None


def read_setting(name: str) -> str | None:
    """Return a setting from the environment, falling back to the .env files.

    Shared by key lookup and GEMINI_MODEL so a value written into
    ~/.config/watch/.env works the same way for both.
    """
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for path in _dotenv_paths():
        value = _from_dotenv(path, name)
        if value:
            return value
    return None


def gemini_model() -> str:
    return read_setting("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def extract_audio(video_path: str, out_path: Path) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits any Whisper limit."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


def audio_duration(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")
    fmt = json.loads(result.stdout or "{}").get("format", {})
    return float(fmt.get("duration") or 0.0)


def split_audio(
    full_audio: Path,
    work_dir: Path,
    plan: list[tuple[float, float]],
) -> list[tuple[Path, float]]:
    """Slice full_audio into per-plan chunk files, returning (path, offset) pairs.

    Uses stream copy (`-c copy`) so there is no re-encode and no quality loss;
    mp3 frame boundaries are close enough for transcription's purposes.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    work_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, float]] = []
    for index, (offset, duration) in enumerate(plan):
        out_path = work_dir / f"chunk_{index:03d}.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{offset:.3f}",
            "-i", str(full_audio.resolve()),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(out_path.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise SystemExit(
                f"ffmpeg failed to split audio chunk {index + 1}: {result.stderr.strip()}"
            )
        chunks.append((out_path, offset))
    return chunks


def _build_multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body the Whisper APIs accept.

    Whisper's multipart upload is small and predictable — doing it by hand
    keeps us on pure stdlib instead of pulling requests/groq/openai SDKs.
    """
    boundary = f"----WatchBoundary{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()

    for name, value in fields.items():
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)

    mimetype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mimetype}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(file_path.read_bytes())
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    return buf.getvalue(), boundary


MAX_ATTEMPTS = 4       # initial + 3 retries
MAX_429_RETRIES = 2
RETRY_BASE_DELAY = 2.0


def _request_with_retries(build_request, label: str, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """POST with bounded retries and return the decoded JSON body.

    `build_request` is a zero-arg factory rather than a prebuilt Request so
    each attempt gets a fresh one (a Request's body stream can't be replayed).
    The policy is backend-agnostic — no retry on 4xx except 429, capped 429
    attempts, exponential backoff on 5xx and network errors — so Whisper and
    Gemini share it and only differ in how they build the request.

    `max_attempts` is lowered by callers that have somewhere better to go than
    another backoff: the video path rotates to the next model on a 503 rather
    than waiting out a model that is simply busy.
    """
    context = ssl.create_default_context()
    rate_limit_hits = 0
    last_exc: Exception | None = None
    last_detail = ""

    for attempt in range(max_attempts):
        try:
            with urlopen(build_request(), timeout=300, context=context) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            last_exc, last_detail = exc, detail

            # 4xx other than 429 are client errors — no retry will fix them.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"{label} request failed: {exc}{detail}")

            if exc.code == 429:
                rate_limit_hits += 1
                if rate_limit_hits >= MAX_429_RETRIES:
                    raise SystemExit(f"{label} request failed: {exc}{detail}")
                delay = _retry_after(exc) or RETRY_BASE_DELAY * (2 ** attempt) + 1
            else:
                delay = RETRY_BASE_DELAY * (2 ** attempt)

            if attempt < max_attempts - 1:
                print(
                    f"[watch] {label} HTTP {exc.code} — retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{max_attempts})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc, last_detail = exc, ""
            if attempt < max_attempts - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(
                    f"[watch] {label} network error ({type(exc).__name__}: {exc}) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 2}/{max_attempts})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{label} returned non-JSON response: {exc}: {payload[:200]}")

    raise SystemExit(
        f"{label} request failed after {max_attempts} attempts: {last_exc}{last_detail}"
    )


def _post_whisper(endpoint: str, api_key: str, model: str, audio_path: Path) -> dict:
    fields = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    body, boundary = _build_multipart(fields, audio_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        # Groq sits behind Cloudflare — the default `Python-urllib/3.x` UA
        # trips WAF rule 1010 (403) before auth even runs. Any non-default
        # UA clears it; we identify honestly.
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    return _request_with_retries(
        lambda: Request(endpoint, data=body, headers=headers, method="POST"),
        "whisper",
    )


GEMINI_PROMPT = (
    "Transcribe this audio verbatim.\n\n"
    "Return every spoken word, in the language spoken — do not translate, "
    "summarize, censor, or add commentary, headings, or speaker labels that "
    "are not audible. If a passage is unintelligible, transcribe what you can "
    "and leave the rest out rather than guessing.\n\n"
    "Break the transcript into consecutive segments of roughly 3-8 seconds, "
    "split at natural sentence or clause boundaries. Segments must be in "
    "chronological order and must not overlap.\n\n"
    "`start` and `end` are seconds measured from the beginning of THIS audio "
    "clip (the clip starts at 0), as numbers — not clock times, and not "
    "offsets into any larger recording.\n\n"
    "If the audio contains no intelligible speech, return an empty array."
)

# Structured output: without a schema Gemini narrates ("Here is the
# transcript...") and the timestamps drift into MM:SS strings. Pinning the
# shape is what makes the response parseable instead of prose.
GEMINI_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start": {"type": "NUMBER"},
            "end": {"type": "NUMBER"},
            "text": {"type": "STRING"},
        },
        "required": ["start", "end", "text"],
    },
}


def _post_gemini(api_key: str, model: str, audio_path: Path) -> dict:
    """Send one audio clip to generateContent and return the raw response."""
    mimetype = mimetypes.guess_type(audio_path.name)[0] or "audio/mpeg"
    payload: dict = {
        "contents": [
            {
                "parts": [
                    {"text": GEMINI_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mimetype,
                            "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_RESPONSE_SCHEMA,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
        },
    }

    # No thinkingConfig here on purpose: the knob is spelled differently across
    # Gemini generations (thinkingBudget vs thinking_level) and sending the
    # wrong one is a 400. GEMINI_MODEL can point at any of them, so we stay on
    # the subset of generationConfig every generation accepts.
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }
    endpoint = GEMINI_ENDPOINT.format(model=model)

    return _request_with_retries(
        lambda: Request(endpoint, data=body, headers=headers, method="POST"),
        "gemini",
    )


def _coerce_seconds(value) -> float:
    """Accept a number of seconds, or a "MM:SS"/"HH:MM:SS" string, as seconds.

    The schema asks for a number, but models still occasionally emit clock
    strings; parsing both is cheaper than losing a whole chunk to a ValueError.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        if ":" in text:
            seconds = 0.0
            for part in text.split(":"):
                seconds = seconds * 60 + float(part or 0)
            return seconds
        return float(text)
    return 0.0


def _segments_from_gemini(data: dict) -> list[dict]:
    """Convert a generateContent response into {start, end, text} segments."""
    candidates = data.get("candidates") or []
    if not candidates:
        # No candidate at all usually means the prompt itself was blocked.
        feedback = (data.get("promptFeedback") or {}).get("blockReason")
        raise SystemExit(
            f"Gemini returned no candidates{f' (blocked: {feedback})' if feedback else ''}"
        )

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    parts = ((candidate.get("content") or {}).get("parts")) or []
    raw = "".join(part.get("text") or "" for part in parts).strip()

    if not raw:
        if finish and finish != "STOP":
            raise SystemExit(f"Gemini returned no transcript (finishReason: {finish})")
        return []

    if finish == "MAX_TOKENS":
        raise SystemExit(
            "Gemini hit its output limit mid-transcript — the response is "
            "truncated. Re-run with a shorter --start/--end window."
        )

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Gemini returned unparseable JSON: {exc}: {raw[:200]}")

    if not isinstance(items, list):
        raise SystemExit(f"Gemini returned {type(items).__name__}, expected a list of segments")

    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        start = _coerce_seconds(item.get("start"))
        end = _coerce_seconds(item.get("end"))
        # A model can emit end < start on a bad split; clamping keeps the
        # downstream range filter from silently dropping the segment.
        if end < start:
            end = start
        out.append({"start": round(start, 2), "end": round(end, 2), "text": text})

    return out


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return f" — {body.decode('utf-8', errors='replace')[:400]}"
    except Exception:
        return ""


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def shift_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Return a copy of segments with start/end shifted by offset_seconds.

    Each chunk is transcribed in isolation, so Whisper returns 0-based timestamps
    per chunk; shifting by the chunk's offset stitches them into source time.
    """
    if offset_seconds == 0:
        return segments
    return [
        {
            "start": round(seg["start"] + offset_seconds, 2),
            "end": round(seg["end"] + offset_seconds, 2),
            "text": seg["text"],
        }
        for seg in segments
    ]


def _segments_from_response(data: dict) -> list[dict]:
    """Convert Whisper verbose_json into our {start, end, text} segment format."""
    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start") or 0.0), 2),
            "end": round(float(seg.get("end") or 0.0), 2),
            "text": text,
        })

    if not out:
        full = (data.get("text") or "").strip()
        if full:
            out.append({"start": 0.0, "end": 0.0, "text": full})

    return out


def transcribe_chunks(
    chunks: list[tuple[Path, float]],
    transcribe_one,
) -> list[dict]:
    """Transcribe each chunk, shift its segments by the chunk offset, concatenate.

    A chunk that fails after its own retries is logged and skipped so one bad
    slice doesn't discard the whole transcript. Raises only if every chunk fails.
    """
    segments: list[dict] = []
    failures = 0
    for index, (path, offset) in enumerate(chunks):
        try:
            chunk_segments = transcribe_one(path)
        except SystemExit as exc:
            failures += 1
            print(
                f"[watch] chunk {index + 1}/{len(chunks)} failed — skipping ({exc})",
                file=sys.stderr,
            )
            continue
        segments.extend(shift_segments(chunk_segments, offset))
        print(
            f"[watch] chunk {index + 1}/{len(chunks)} → {len(chunk_segments)} segments",
            file=sys.stderr,
        )

    if failures == len(chunks):
        raise SystemExit("transcription failed on every audio chunk")
    return segments


def upload_limits(backend: str) -> tuple[int, float | None]:
    """Return (max_bytes, max_seconds) for one backend's per-request payload."""
    if backend == "gemini":
        return GEMINI_MAX_UPLOAD_BYTES, GEMINI_MAX_CHUNK_SECONDS
    return MAX_UPLOAD_BYTES, None


def _transcribe_file(backend: str, api_key: str, audio_path: Path) -> list[dict]:
    """Upload one audio file and return its 0-based segments."""
    if backend == "groq":
        return _segments_from_response(_post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path))
    if backend == "openai":
        return _segments_from_response(
            _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
        )
    if backend == "gemini":
        return _segments_from_gemini(_post_gemini(api_key, gemini_model(), audio_path))
    raise SystemExit(f"Unknown transcription backend: {backend}")


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], str]:
    """Run the full flow: extract audio → upload → parse segments.

    Returns (segments, backend_used). Raises SystemExit on any failure.
    """
    if backend is None or api_key is None:
        detected_backend, detected_key = load_api_key()
        backend = backend or detected_backend
        api_key = api_key or detected_key

    if not backend or not api_key:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No transcription API key available. Set GROQ_API_KEY (preferred), "
            "OPENAI_API_KEY, or GEMINI_API_KEY in the environment or in "
            f"~/.config/watch/.env. Run `python3 {setup_py}` to configure."
        )

    print(f"[watch] extracting audio for transcription ({backend})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)
    audio_bytes = audio_path.stat().st_size
    max_bytes, max_seconds = upload_limits(backend)

    def transcribe_one(path: Path) -> list[dict]:
        return _transcribe_file(backend, api_key, path)

    # Gemini is capped by clip length as well as size, so its duration has to
    # be known up front rather than only after a size overrun.
    duration = audio_duration(audio_path) if max_seconds else 0.0
    within_limits = audio_bytes <= max_bytes and (max_seconds is None or duration <= max_seconds)

    if within_limits:
        print(
            f"[watch] audio: {audio_bytes / 1024:.0f} kB — uploading to {backend}…",
            file=sys.stderr,
        )
        segments = transcribe_one(audio_path)
    else:
        if not duration:
            duration = audio_duration(audio_path)
        plan = plan_chunks(duration, audio_bytes, max_bytes, max_seconds)
        reason = (
            f"{audio_bytes / (1024 * 1024):.0f} MB exceeds {max_bytes // (1024 * 1024)} MB"
            if audio_bytes > max_bytes
            else f"{duration / 60:.0f} min exceeds the {max_seconds / 60:.0f} min "
                 f"per-request limit for {backend}"
        )
        print(
            f"[watch] audio: {reason} — splitting into {len(plan)} chunks…",
            file=sys.stderr,
        )
        chunks = split_audio(audio_path, audio_out.parent / "chunks", plan)
        segments = transcribe_chunks(chunks, transcribe_one)

    if not segments:
        raise SystemExit(f"{backend} returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: whisper.py <video-path> [<audio-out.mp3>] [--backend groq|openai|gemini]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    segments, backend = transcribe_video(video, audio_out, backend=backend_override)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
