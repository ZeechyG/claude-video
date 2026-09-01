# Changelog

All notable changes to `/watch` are documented here.

## [0.3.0] — 2026-09-01

### Added
- **Gemini as a third transcription backend.** Set `GEMINI_API_KEY` in `~/.config/watch/.env`, or force it with `--whisper gemini`. Groq and OpenAI both run Whisper, which reports segment timestamps natively; Gemini is a general model asked to transcribe, so its timestamps are model-generated — it is tried last, and the auto-detect result for an existing Groq or OpenAI user is unchanged. Chunks by duration as well as size (over 10 minutes splits and stitches back into source time), because the transcript is Gemini's *output* and a long clip overruns the response limit long before the upload cap. Defaults to the `gemini-flash-latest` alias rather than a pinned version, since pinned models get retired.
- **Gemini's native YouTube path** for the visual timeline (`--youtube-native`, `--no-youtube-native`). YouTube refuses to serve media to datacenter IPs, so on CI runners and cloud sandboxes frames are impossible. Gemini takes a YouTube URL directly, so it can describe what is on screen with nothing downloaded — the report grows a **Visual timeline** of timestamped beats. Runs automatically when a YouTube download fails and a Gemini key is set. `--start`/`--end` clip it server-side. The visuals are Gemini's description rather than frames Claude looked at, and the report says so in its own output so the distinction survives into the answer; frames remain preferred whenever a download works.
- **Timeline-synthesis method in `SKILL.md` Step 4.** Frames and transcript are merged into beats (timestamp, what's on screen, what's spoken, what changed) and read across for structure — how it opens, what holds attention, where it turns, how it closes — with inferences and sampling gaps labelled as such, closing on the three highest-signal observations with timestamps.

### Changed
- The transcript source is reported as `gemini (<model>)` rather than `whisper (gemini)`, which misnamed the model that produced it.
- Retry policy (no retry on 4xx except 429, capped 429 attempts, backoff on 5xx and network errors) is shared by every backend instead of living inside the Whisper client. The video path additionally rotates models: an alias that is busy (503) or a pin that has been retired (404) moves to the next rather than failing the run.
- User-facing wording says "transcription" where it used to say "Whisper", now that not every backend is Whisper.
- `GEMINI_MODEL` resolves through the same environment-then-`.env` path as the API keys, so the line the installer scaffolds is actually read.

## [0.2.0] — 2026-06-29

### Added
- **`--detail` dial** with four modes — `transcript` (captions only, no frames), `efficient` (fast keyframe pass, cap 50), `balanced` (scene-aware, cap 100, default), and `token-burner` (scene-aware, uncapped). Set the default with `WATCH_DETAIL` in `~/.config/watch/.env`.
- **Frame deduplication** (default on; `--no-dedup` to disable). Before the budget cap, a pass downscales each frame to a 16×16 grayscale thumbnail and drops frames whose mean per-pixel difference from the last *kept* frame is within threshold — so the budget goes to distinct content instead of held slides and static recordings. The **Frames** report line shows how many near-duplicates were dropped.
- **Whisper auto-chunking.** Audio over the 25 MB upload cap is split into evenly sized chunks, transcribed per chunk, with segment timestamps shifted back into source time. Partial failures are tolerated — transcription only fails if *every* chunk fails, so length alone no longer breaks it.
- **`--timestamps T1,T2,…`** — grab a frame at each absolute timestamp; reserved against the cap, and the only frames produced under `--detail transcript`.
- **`--no-whisper`** — disable transcription entirely (frames only).
- pytest suite covering config, dedup, download, fixtures, frames, setup, timestamps, watch, and whisper (no network; ffmpeg-synthesized clips).

### Changed
- **Restructured into a self-contained `skills/watch/` package** so `SKILL.md` and its `scripts/` runtime are siblings in one folder. This fixes installs on Codex, Cursor, Copilot, and other Agent Skills hosts: `npx skills add` now copies the skill as a working unit instead of grabbing the root `SKILL.md` without its scripts.
- **Harness-agnostic path resolution** — `SKILL.md` resolves `$SKILL_DIR` from where it was Read instead of the Claude-Code-only `${CLAUDE_SKILL_DIR}`, so script calls work on every host.
- `/watch` is now derived from `SKILL.md` frontmatter; the separate `commands/watch.md` wrapper was dropped to avoid a duplicate slash command.
- `balanced` now full-decodes to detect every scene cut across the whole video. The previous early-exit was faster but kept only the first cuts and dropped the tail of long videos.
- `token-burner` is exempt from the long-video "sparse scan" warning, since it keeps every scene-change frame.
- `--max-frames` is now an override on top of each mode's default cap, rather than a fixed default of 80.

### Fixed
- Non-Claude installs (`npx skills add`) were dead on arrival — the installer copied `SKILL.md` without the `scripts/` it shells out to. The self-contained package layout resolves this.

### Removed
- `V2_PLAN.md` and `V2_CONCERNS.md` planning docs.

## [0.1.3] — 2026-05-09

### Fixed
- Windows: `video.info.json` is read as UTF-8 (#4). Previously `Path.read_text()` defaulted to cp1252 on Windows and crashed on yt-dlp's UTF-8 output, silently dropping Title/Uploader from the report. Same fix applied to `.env` reads/writes in `whisper.py` and `setup.py`.
- `download.py` now logs info.json parse failures to stderr instead of swallowing them.

### Security
- Hardened subprocess argv against option injection (#2): inserted `--` before the URL in the yt-dlp argv, and tightened `is_url` to reject `-`-prefixed sources and require a non-empty netloc. Resolved video/audio paths to absolute via `Path.resolve()` before passing to `ffmpeg`/`ffprobe`, so a relative path starting with `-` can't be misinterpreted as a flag.

## [0.1.2] — 2026-04-24

### Fixed
- Windows console crash: removed the emoji from the long-video warning in `watch.py`; cp1252 consoles couldn't encode it.
- `setup.py` now prints `winget` / `pip` install commands on Windows instead of "unsupported platform" — matches what the README already promised.

### Changed
- `SKILL.md` notes that on Windows the scripts must be invoked with `python`, not `python3` (the latter is the Microsoft Store stub on Windows).

## [0.1.1] — 2026-04-24

### Fixed
- Added `commands/watch.md` shim so `/watch` is callable when installed as a Claude Code plugin. Without it, the plugin loaded but the skill wasn't exposed as a slash command.
- `scripts/build-skill.sh` now strips `commands/` from the claude.ai `.skill` bundle alongside `hooks/` and `.claude-plugin/`.

## [0.1.0] — 2026-04-24

Initial marketplace release.

### Added
- `/watch <url-or-path> [question]` slash command.
- yt-dlp download with native caption extraction (manual + auto-subs).
- ffmpeg frame extraction with auto-scaled fps (≤2 fps, ≤100 frames, duration-aware budget).
- `--start` / `--end` focused mode with denser frame budget and transcript range filtering.
- Whisper fallback (Groq preferred, OpenAI secondary) for videos without captions.
- `setup.py` preflight: silent `--check`, structured `--json`, and installer that auto-runs `brew install` on macOS.
- Session-start hook that prints a one-line status on first run / partial config.
- `.skill` bundle packaging for claude.ai upload via `scripts/build-skill.sh`.
