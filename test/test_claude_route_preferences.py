"""Tests for per-project Claude route model preferences and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from kiro_crew.claude_routes.config import ClaudeConfig, ClaudeRouteConfig
from kiro_crew.claude_routes.preferences import (
    _PREFERENCES,
    clear_project_model_pref,
    get_project_model_pref,
    handle_project_model_pick,
    resolve_slot_project_model,
    set_project_model_pref,
)
from kiro_crew.dashboard.chat_turn.model_fallback import _default_session_model


class TestPreferencesStore:
    def test_set_get_and_clear_isolated_by_route(self, tmp_path: Path) -> None:
        with patch("kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)):
            _PREFERENCES.clear()
            set_project_model_pref("proj-a", "sonnet")
            set_project_model_pref("proj-b", "gpt-terra")

            assert get_project_model_pref("proj-a") == "sonnet"
            assert get_project_model_pref("proj-b") == "gpt-terra"
            assert get_project_model_pref("proj-c") is None

            # Persistence on disk
            pref_file = tmp_path / "claude_project_models.json"
            assert pref_file.is_file()
            disk_data = json.loads(pref_file.read_text(encoding="utf-8"))
            assert disk_data["routes"]["proj-a"]["model"] == "sonnet"
            assert disk_data["routes"]["proj-b"]["model"] == "gpt-terra"

            # Clear preference
            clear_project_model_pref("proj-a")
            assert get_project_model_pref("proj-a") is None
            assert get_project_model_pref("proj-b") == "gpt-terra"

    def test_setting_auto_clears_preference(self, tmp_path: Path) -> None:
        with patch("kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)):
            _PREFERENCES.clear()
            set_project_model_pref("proj-a", "code-medium")
            assert get_project_model_pref("proj-a") == "code-medium"

            set_project_model_pref("proj-a", "auto")
            assert get_project_model_pref("proj-a") is None


class TestResolveSlotProjectModel:
    def test_resolves_project_model_and_retains_independence(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "work-a"
        dir_b = tmp_path / "work-b"
        dir_a.mkdir(parents=True, exist_ok=True)
        dir_b.mkdir(parents=True, exist_ok=True)

        claude_cfg = ClaudeConfig(
            routes={
                "route-a": ClaudeRouteConfig(base_url="http://host-a/v1"),
                "route-b": ClaudeRouteConfig(base_url="http://host-b/v1"),
            },
            project_routes={
                str(dir_a): "route-a",
                str(dir_b): "route-b",
            },
        )

        class DummyConfig:
            claude = claude_cfg

        with patch("kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)):
            _PREFERENCES.clear()
            set_project_model_pref("route-a", "sonnet")
            set_project_model_pref("route-b", "gpt-terra")

            assert resolve_slot_project_model(DummyConfig(), str(dir_a)) == "sonnet"
            assert resolve_slot_project_model(DummyConfig(), str(dir_b)) == "gpt-terra"
            assert resolve_slot_project_model(DummyConfig(), str(tmp_path / "unmapped")) is None

    def test_handle_project_model_pick(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "work"
        proj_dir.mkdir(parents=True, exist_ok=True)

        claude_cfg = ClaudeConfig(
            routes={"route-1": ClaudeRouteConfig(base_url="http://host/v1")},
            project_routes={str(proj_dir): "route-1"},
        )

        class DummyConfig:
            claude = claude_cfg

        with patch("kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)):
            _PREFERENCES.clear()
            handled = handle_project_model_pick(DummyConfig(), str(proj_dir), "code-high")
            assert handled is True
            assert get_project_model_pref("route-1") == "code-high"

            unhandled = handle_project_model_pick(DummyConfig(), str(tmp_path / "other"), "code-high")
            assert unhandled is False


class TestDefaultSessionModelPrecedence:
    def test_project_model_applies_when_slot_model_is_empty(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir(parents=True, exist_ok=True)

        claude_cfg = ClaudeConfig(
            routes={"r1": ClaudeRouteConfig(base_url="http://host/v1")},
            project_routes={str(proj_dir): "r1"},
        )

        class DummyConfig:
            claude = claude_cfg

        slot = MagicMock()
        slot.model = ""
        slot.project = str(proj_dir)

        with patch("kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)):
            _PREFERENCES.clear()
            set_project_model_pref("r1", "gpt-terra")
            resolved = _default_session_model(DummyConfig(), slot, "")
            assert resolved == "gpt-terra"

    def test_explicit_slot_model_pin_retained(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir(parents=True, exist_ok=True)

        claude_cfg = ClaudeConfig(
            routes={"r1": ClaudeRouteConfig(base_url="http://host/v1")},
            project_routes={str(proj_dir): "r1"},
        )

        class DummyConfig:
            claude = claude_cfg

        slot = MagicMock()
        slot.model = "pinned-opus"
        slot.project = str(proj_dir)

        with patch("kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)):
            _PREFERENCES.clear()
            set_project_model_pref("r1", "gpt-terra")
            # If slot.model is already set, _default_session_model returns ""
            # because the slot has its own pin that takes precedence
            resolved = _default_session_model(DummyConfig(), slot, "")
            assert resolved == ""
