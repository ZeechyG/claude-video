"""Gemini's native YouTube path: URL handling, payload shape, beat parsing.

No network — the live call is exercised through its payload builder, its
response parser, and a stubbed transport for the model-rotation logic.
"""
from __future__ import annotations

import json

import pytest

import gemini_video as gv


def _response(items, finish_reason="STOP", tokens=0) -> dict:
    return {
        "candidates": [
            {"finishReason": finish_reason, "content": {"parts": [{"text": json.dumps(items)}]}}
        ],
        "usageMetadata": {"totalTokenCount": tokens},
    }


class TestYouTubeVideoId:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/watch?v=IjS9eTpmhgk", "IjS9eTpmhgk"),
            ("https://m.youtube.com/watch?v=IjS9eTpmhgk", "IjS9eTpmhgk"),
            ("https://music.youtube.com/watch?v=IjS9eTpmhgk", "IjS9eTpmhgk"),
            ("https://youtu.be/IjS9eTpmhgk", "IjS9eTpmhgk"),
            ("https://youtu.be/IjS9eTpmhgk?si=abc", "IjS9eTpmhgk"),
            ("https://www.youtube.com/shorts/IjS9eTpmhgk", "IjS9eTpmhgk"),
            ("https://www.youtube.com/embed/IjS9eTpmhgk", "IjS9eTpmhgk"),
            ("https://www.youtube.com/live/IjS9eTpmhgk", "IjS9eTpmhgk"),
            ("  https://youtu.be/IjS9eTpmhgk  ", "IjS9eTpmhgk"),
        ],
    )
    def test_extracts_id_from_every_form(self, url, expected):
        assert gv.youtube_video_id(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://vimeo.com/123456",
            "/local/file.mp4",
            "",
            "not a url",
            "ftp://youtube.com/watch?v=IjS9eTpmhgk",
            "https://www.youtube.com/watch?v=tooshort",
            "https://www.youtube.com/watch",
            "https://www.youtube.com/results?search_query=x",
        ],
    )
    def test_rejects_non_youtube_and_malformed(self, url):
        assert gv.youtube_video_id(url) is None

    def test_rejects_lookalike_host(self):
        """Host must match exactly — a suffix trick must not pass."""
        assert gv.youtube_video_id("https://notyoutube.com.evil.co/watch?v=IjS9eTpmhgk") is None
        assert gv.youtube_video_id("https://youtube.com.evil.co/watch?v=IjS9eTpmhgk") is None


class TestCanonicalUrl:
    def test_strips_share_tracking_params(self):
        """Gemini 400s on ?pp=/&ra= — canonicalizing is what makes share links work."""
        messy = "https://m.youtube.com/watch?v=IjS9eTpmhgk&pp=ygUJTWFya2V0aW5n&ra=m"
        assert gv.canonical_youtube_url(messy) == "https://www.youtube.com/watch?v=IjS9eTpmhgk"

    def test_normalizes_short_form(self):
        assert gv.canonical_youtube_url("https://youtu.be/IjS9eTpmhgk") == (
            "https://www.youtube.com/watch?v=IjS9eTpmhgk"
        )

    def test_returns_none_for_non_youtube(self):
        assert gv.canonical_youtube_url("https://vimeo.com/1") is None

    def test_is_youtube_url_agrees_with_extraction(self):
        assert gv.is_youtube_url("https://youtu.be/IjS9eTpmhgk")
        assert not gv.is_youtube_url("https://vimeo.com/1")


class TestBuildPayload:
    def test_sends_canonical_url_not_the_raw_one(self):
        messy = "https://m.youtube.com/watch?v=IjS9eTpmhgk&pp=x&ra=m"
        part = gv._build_payload(messy, None, None)["contents"][0]["parts"][1]
        assert part["file_data"]["file_uri"] == "https://www.youtube.com/watch?v=IjS9eTpmhgk"

    def test_window_becomes_video_metadata_offsets(self):
        part = gv._build_payload("https://youtu.be/IjS9eTpmhgk", 140.0, 200.0)["contents"][0]["parts"][1]
        assert part["video_metadata"] == {"start_offset": "140s", "end_offset": "200s"}

    def test_no_window_omits_video_metadata(self):
        part = gv._build_payload("https://youtu.be/IjS9eTpmhgk", None, None)["contents"][0]["parts"][1]
        assert "video_metadata" not in part

    def test_only_start_sets_only_start_offset(self):
        part = gv._build_payload("https://youtu.be/IjS9eTpmhgk", 30.0, None)["contents"][0]["parts"][1]
        assert part["video_metadata"] == {"start_offset": "30s"}

    def test_requests_json_with_a_schema(self):
        cfg = gv._build_payload("https://youtu.be/IjS9eTpmhgk", None, None)["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["responseSchema"] == gv.VISUAL_SCHEMA


class TestParseBeats:
    def test_parses_and_sorts(self):
        data = _response([
            {"time": 12, "on_screen": "cut to diagram"},
            {"time": 3, "on_screen": "title card"},
        ])
        assert gv.parse_beats(data) == [
            {"start": 3.0, "on_screen": "title card"},
            {"start": 12.0, "on_screen": "cut to diagram"},
        ]

    def test_drops_blank_descriptions(self):
        data = _response([
            {"time": 1, "on_screen": "  real  "},
            {"time": 2, "on_screen": "   "},
        ])
        assert gv.parse_beats(data) == [{"start": 1.0, "on_screen": "real"}]

    def test_coerces_clock_string_times(self):
        data = _response([{"time": "01:05", "on_screen": "beat"}])
        assert gv.parse_beats(data) == [{"start": 65.0, "on_screen": "beat"}]

    def test_skips_non_dict_items(self):
        data = _response([{"time": 1, "on_screen": "ok"}, "junk", 7])
        assert gv.parse_beats(data) == [{"start": 1.0, "on_screen": "ok"}]

    def test_empty_list_is_allowed(self):
        assert gv.parse_beats(_response([])) == []

    def test_raises_on_truncation(self):
        with pytest.raises(SystemExit, match="output limit"):
            gv.parse_beats(_response([{"time": 0, "on_screen": "x"}], finish_reason="MAX_TOKENS"))

    def test_raises_when_blocked(self):
        with pytest.raises(SystemExit, match="SAFETY"):
            gv.parse_beats({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})

    def test_raises_on_bad_json(self):
        data = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{oops"}]}}]}
        with pytest.raises(SystemExit, match="unparseable JSON"):
            gv.parse_beats(data)

    def test_usage_tokens_read_from_metadata(self):
        assert gv.usage_tokens(_response([], tokens=6744)) == 6744
        assert gv.usage_tokens({}) == 0


class TestShiftBeats:
    def test_focused_run_returns_absolute_timestamps(self):
        """Gemini numbers a clipped run from 0; the window offset goes back on."""
        beats = [{"start": 0.0, "on_screen": "a"}, {"start": 34.0, "on_screen": "b"}]
        assert gv.shift_beats(beats, 140.0) == [
            {"start": 140.0, "on_screen": "a"},
            {"start": 174.0, "on_screen": "b"},
        ]

    def test_zero_offset_is_identity(self):
        beats = [{"start": 5.0, "on_screen": "a"}]
        assert gv.shift_beats(beats, 0) is beats


class TestModelRotation:
    def test_configured_model_leads_and_duplicates_collapse(self, monkeypatch):
        monkeypatch.setattr(gv, "gemini_model", lambda: gv.GEMINI_VIDEO_FALLBACKS[0])
        chain = gv.video_models()
        assert chain[0] == gv.GEMINI_VIDEO_FALLBACKS[0]
        assert len(chain) == len(set(chain))

    def test_moves_to_next_model_when_first_is_unavailable(self, monkeypatch):
        tried = []

        def fake_request(build, label, max_attempts=2):
            tried.append(label)
            if len(tried) == 1:
                raise SystemExit("HTTP Error 503: high demand")
            return _response([{"time": 0, "on_screen": "ok"}], tokens=5)

        monkeypatch.setattr(gv, "_request_with_retries", fake_request)
        beats, model, tokens = gv.fetch_visual_timeline(
            "https://youtu.be/IjS9eTpmhgk", "k", models=["busy-model", "good-model"]
        )
        assert beats == [{"start": 0.0, "on_screen": "ok"}]
        assert model == "good-model" and tokens == 5
        assert len(tried) == 2

    def test_applies_window_offset_through_the_public_call(self, monkeypatch):
        monkeypatch.setattr(
            gv, "_request_with_retries",
            lambda build, label, max_attempts=2: _response([{"time": 4, "on_screen": "b"}]),
        )
        beats, _model, _tokens = gv.fetch_visual_timeline(
            "https://youtu.be/IjS9eTpmhgk", "k", start_seconds=600.0, models=["m"]
        )
        assert beats == [{"start": 604.0, "on_screen": "b"}]

    def test_raises_after_every_model_fails(self, monkeypatch):
        def always_fail(build, label, max_attempts=2):
            raise SystemExit("nope")

        monkeypatch.setattr(gv, "_request_with_retries", always_fail)
        with pytest.raises(SystemExit, match="failed on every model"):
            gv.fetch_visual_timeline("https://youtu.be/IjS9eTpmhgk", "k", models=["a", "b"])


class TestFormatBeats:
    def test_renders_mm_ss(self):
        assert gv.format_beats([{"start": 65, "on_screen": "x"}]) == "[01:05] x"

    def test_renders_hours_past_an_hour(self):
        assert gv.format_beats([{"start": 3725, "on_screen": "x"}]) == "[1:02:05] x"
