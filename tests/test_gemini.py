"""Gemini transcription backend: key selection, chunk limits, response parsing.

No network — the Gemini calls are exercised through their request builder and
response parser, which is where all the backend-specific logic lives.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import whisper


MB = 1024 * 1024


def _gemini_response(items, finish_reason="STOP") -> dict:
    """Wrap segment dicts the way generateContent returns them."""
    return {
        "candidates": [
            {
                "finishReason": finish_reason,
                "content": {"parts": [{"text": json.dumps(items)}]},
            }
        ]
    }


class TestLoadApiKey:
    def test_gemini_key_is_detected(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])

        assert whisper.load_api_key() == ("gemini", "g-key")

    def test_whisper_backends_still_win_auto_detect(self, monkeypatch):
        """Gemini is last, so an existing Groq user's behavior is unchanged."""
        monkeypatch.setenv("GROQ_API_KEY", "q-key")
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])

        assert whisper.load_api_key() == ("groq", "q-key")

    def test_preferred_gemini_ignores_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "q-key")
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])

        assert whisper.load_api_key("gemini") == ("gemini", "g-key")

    def test_preferred_backend_without_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "q-key")
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])

        assert whisper.load_api_key("gemini") == (None, None)


class TestGeminiModel:
    def test_defaults_to_the_moving_alias(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])

        assert whisper.gemini_model() == whisper.DEFAULT_GEMINI_MODEL

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])

        assert whisper.gemini_model() == "gemini-3.6-flash"

    def test_dotenv_overrides_default(self, monkeypatch, tmp_path):
        """A GEMINI_MODEL written into ~/.config/watch/.env has to be honored."""
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        env = tmp_path / ".env"
        env.write_text("GEMINI_MODEL=gemini-3.6-flash\n", encoding="utf-8")
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [env])

        assert whisper.gemini_model() == "gemini-3.6-flash"

    def test_env_wins_over_dotenv(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("GEMINI_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [env])
        monkeypatch.setenv("GEMINI_MODEL", "from-env")

        assert whisper.gemini_model() == "from-env"


class TestUploadLimits:
    def test_gemini_has_a_duration_ceiling(self):
        max_bytes, max_seconds = whisper.upload_limits("gemini")
        assert max_bytes == whisper.GEMINI_MAX_UPLOAD_BYTES
        assert max_seconds == whisper.GEMINI_MAX_CHUNK_SECONDS

    def test_whisper_backends_have_no_duration_ceiling(self):
        for backend in ("groq", "openai"):
            max_bytes, max_seconds = whisper.upload_limits(backend)
            assert max_bytes == whisper.MAX_UPLOAD_BYTES
            assert max_seconds is None


class TestPlanChunksWithDuration:
    def test_long_but_small_audio_still_splits_for_gemini(self):
        """35 min of speech is only ~5 MB — under the byte cap, over the time cap."""
        plan = whisper.plan_chunks(2100.0, 5 * MB, *whisper.upload_limits("gemini"))
        assert len(plan) == 4
        assert all(dur <= whisper.GEMINI_MAX_CHUNK_SECONDS for _off, dur in plan)

    def test_same_audio_is_one_chunk_for_whisper(self):
        plan = whisper.plan_chunks(2100.0, 5 * MB, *whisper.upload_limits("groq"))
        assert plan == [(0.0, 2100.0)]

    def test_duration_cap_still_covers_full_timeline(self):
        total = 2100.0
        plan = whisper.plan_chunks(total, 5 * MB, *whisper.upload_limits("gemini"))
        assert plan[0][0] == 0.0
        last_off, last_dur = plan[-1]
        assert last_off + last_dur == pytest.approx(total)

    def test_byte_cap_wins_when_it_is_stricter(self):
        # 60 MB over a 14 MB cap needs 5 chunks; 300s over 600s needs only 1.
        plan = whisper.plan_chunks(300.0, 60 * MB, 14 * MB, 600.0)
        assert len(plan) == 5


class TestCoerceSeconds:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (12.5, 12.5),
            (7, 7.0),
            ("12.5", 12.5),
            ("01:02", 62.0),
            ("1:00:30", 3630.0),
            ("", 0.0),
            (None, 0.0),
            ({}, 0.0),
        ],
    )
    def test_accepts_numbers_and_clock_strings(self, value, expected):
        assert whisper._coerce_seconds(value) == expected


class TestSegmentsFromGemini:
    def test_parses_well_formed_segments(self):
        data = _gemini_response(
            [
                {"start": 0.0, "end": 2.5, "text": "hello"},
                {"start": 2.5, "end": 4.0, "text": "there"},
            ]
        )
        assert whisper._segments_from_gemini(data) == [
            {"start": 0.0, "end": 2.5, "text": "hello"},
            {"start": 2.5, "end": 4.0, "text": "there"},
        ]

    def test_strips_whitespace_and_drops_empty_text(self):
        data = _gemini_response(
            [
                {"start": 0.0, "end": 1.0, "text": "  spaced  "},
                {"start": 1.0, "end": 2.0, "text": "   "},
            ]
        )
        assert whisper._segments_from_gemini(data) == [
            {"start": 0.0, "end": 1.0, "text": "spaced"}
        ]

    def test_coerces_clock_string_timestamps(self):
        data = _gemini_response([{"start": "00:03", "end": "00:05", "text": "clock"}])
        assert whisper._segments_from_gemini(data) == [
            {"start": 3.0, "end": 5.0, "text": "clock"}
        ]

    def test_clamps_end_before_start(self):
        """A bad split must not produce a backwards range the filter would drop."""
        data = _gemini_response([{"start": 9.0, "end": 4.0, "text": "reversed"}])
        assert whisper._segments_from_gemini(data) == [
            {"start": 9.0, "end": 9.0, "text": "reversed"}
        ]

    def test_skips_non_dict_items(self):
        data = _gemini_response([{"start": 0.0, "end": 1.0, "text": "ok"}, "junk", 5])
        assert whisper._segments_from_gemini(data) == [
            {"start": 0.0, "end": 1.0, "text": "ok"}
        ]

    def test_empty_array_means_no_speech(self):
        assert whisper._segments_from_gemini(_gemini_response([])) == []

    def test_raises_on_truncated_response(self):
        data = _gemini_response([{"start": 0, "end": 1, "text": "cut"}], finish_reason="MAX_TOKENS")
        with pytest.raises(SystemExit, match="output limit"):
            whisper._segments_from_gemini(data)

    def test_raises_when_no_candidates(self):
        with pytest.raises(SystemExit, match="no candidates"):
            whisper._segments_from_gemini({"candidates": []})

    def test_reports_prompt_block_reason(self):
        data = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
        with pytest.raises(SystemExit, match="SAFETY"):
            whisper._segments_from_gemini(data)

    def test_raises_on_unparseable_json(self):
        data = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{oops"}]}}]}
        with pytest.raises(SystemExit, match="unparseable JSON"):
            whisper._segments_from_gemini(data)

    def test_raises_when_payload_is_not_a_list(self):
        data = {
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": '{"a": 1}'}]}}
            ]
        }
        with pytest.raises(SystemExit, match="expected a list"):
            whisper._segments_from_gemini(data)

    def test_empty_text_with_bad_finish_reason_raises(self):
        data = {"candidates": [{"finishReason": "RECITATION", "content": {"parts": []}}]}
        with pytest.raises(SystemExit, match="RECITATION"):
            whisper._segments_from_gemini(data)

    def test_joins_multiple_parts(self):
        """Long JSON can arrive split across parts; they concatenate before parsing."""
        payload = json.dumps([{"start": 0, "end": 1, "text": "split"}])
        half = len(payload) // 2
        data = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": payload[:half]}, {"text": payload[half:]}]},
                }
            ]
        }
        assert whisper._segments_from_gemini(data) == [
            {"start": 0.0, "end": 1.0, "text": "split"}
        ]


class TestTranscribeFileDispatch:
    def test_gemini_routes_to_the_gemini_parser(self, monkeypatch):
        sent = {}

        def fake_post(api_key, model, audio_path):
            sent.update(api_key=api_key, model=model)
            return _gemini_response([{"start": 0, "end": 1, "text": "routed"}])

        monkeypatch.setattr(whisper, "_post_gemini", fake_post)
        monkeypatch.setattr(whisper, "gemini_model", lambda: "test-model")

        out = whisper._transcribe_file("gemini", "g-key", Path("a.mp3"))

        assert out == [{"start": 0.0, "end": 1.0, "text": "routed"}]
        assert sent == {"api_key": "g-key", "model": "test-model"}

    def test_unknown_backend_raises(self):
        with pytest.raises(SystemExit, match="Unknown transcription backend"):
            whisper._transcribe_file("bogus", "k", Path("a.mp3"))
