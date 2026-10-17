"""Tests for kirocrew claude-route CLI subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kiro_crew.claude_routes.cli import (
    _cmd_bind,
    _cmd_import,
    _cmd_list,
    _cmd_set_key,
    _cmd_unbind,
)
from kiro_crew.claude_routes.config import ClaudeConfig, ClaudeRouteConfig
from kiro_crew.claude_routes.preferences import get_project_model_pref
from kiro_crew.secrets.vault import SecretVault


class TestClaudeRouteCli:
    def test_list_empty_and_json(self, capsys: pytest.CaptureFixture) -> None:
        cfg = ClaudeConfig()

        class DummyConfig:
            claude = cfg

        with patch("kiro_crew.claude_routes.cli.KiroCrewConfig.load", return_value=DummyConfig()):
            rc = _cmd_list(argparse.Namespace(json=False))
            assert rc == 0
            assert "No Claude routes configured" in capsys.readouterr().out

            rc_json = _cmd_list(argparse.Namespace(json=True))
            assert rc_json == 0
            data = json.loads(capsys.readouterr().out)
            assert data["routes"] == {}
            assert data["project_routes"] == {}

    def test_bind_refuses_missing_route(self, capsys: pytest.CaptureFixture) -> None:
        cfg = ClaudeConfig(routes={})

        class DummyConfig:
            claude = cfg

        with patch("kiro_crew.claude_routes.cli.KiroCrewConfig.load", return_value=DummyConfig()):
            rc = _cmd_bind(argparse.Namespace(path="/some/path", route_id="missing-r"))
            assert rc == 1
            assert "does not exist" in capsys.readouterr().err

    def test_bind_and_unbind(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir(parents=True, exist_ok=True)
        cfg = ClaudeConfig(routes={"my-route": ClaudeRouteConfig()})

        class DummyConfig:
            claude = cfg

        captured_updates = []

        def _mock_update(path, *args, **kwargs):
            fn = kwargs.get("mutate") or (args[0] if args else None)
            d = {"claude": {"routes": {"my-route": {}}, "project_routes": {}}}
            if fn:
                fn(d)
            captured_updates.append(d)

        with patch("kiro_crew.claude_routes.cli.KiroCrewConfig.load", return_value=DummyConfig()):
            with patch(
                "kiro_crew.claude_routes.cli.update_config_locked", side_effect=_mock_update
            ):
                # Bind
                rc_bind = _cmd_bind(argparse.Namespace(path=str(proj_dir), route_id="my-route"))
                assert rc_bind == 0
                assert captured_updates[-1]["claude"]["project_routes"][str(proj_dir)] == "my-route"

                # Unbind
                rc_unbind = _cmd_unbind(argparse.Namespace(path=str(proj_dir)))
                assert rc_unbind == 0
                assert str(proj_dir) not in captured_updates[-1]["claude"]["project_routes"]

    def test_set_key_prompts_and_stores_in_vault(self, tmp_path: Path) -> None:
        captured_config = []

        def _mock_update(path, *args, **kwargs):
            fn = kwargs.get("mutate") or (args[0] if args else None)
            d = {"claude": {"routes": {"r1": {}}}}
            if fn:
                fn(d)
            captured_config.append(d)

        with patch("kiro_crew.claude_routes.cli.getpass.getpass", return_value="my-9router-secret"):
            with patch("kiro_crew.claude_routes.cli.get_config_dir", return_value=str(tmp_path)):
                with patch(
                    "kiro_crew.claude_routes.cli.update_config_locked", side_effect=_mock_update
                ):
                    rc = _cmd_set_key(argparse.Namespace(route_id="r1"))
                    assert rc == 0

        # Secret stored in vault
        vault = SecretVault(tmp_path)
        secret = vault.get("claude-route-r1")
        assert secret is not None
        assert secret.reveal() == "my-9router-secret"

        # Route config references secret://
        assert (
            captured_config[-1]["claude"]["routes"]["r1"]["api_key"] == "secret://claude-route-r1"
        )

    def test_import_from_project_settings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        proj = tmp_path / "imported-repo"
        claude_dir = proj / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        (claude_dir / "settings.local.json").write_text(
            json.dumps(
                {
                    "model": "code-medium",
                    "env": {
                        "ANTHROPIC_BASE_URL": "http://gw.local:20128/v1",
                        "ANTHROPIC_API_KEY": "sk-imported-test-token",
                        "ANTHROPIC_DEFAULT_OPUS_MODEL": "code-high",
                        "ANTHROPIC_DEFAULT_SONNET_MODEL": "code-medium",
                    },
                }
            ),
            encoding="utf-8",
        )

        captured_config = []

        def _mock_update(path, *args, **kwargs):
            fn = kwargs.get("mutate") or (args[0] if args else None)
            d = {"claude": {"routes": {}, "project_routes": {}}}
            if fn:
                fn(d)
            captured_config.append(d)

        with patch("kiro_crew.claude_routes.cli.get_config_dir", return_value=str(tmp_path)):
            with patch(
                "kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)
            ):
                with patch(
                    "kiro_crew.claude_routes.cli.update_config_locked", side_effect=_mock_update
                ):
                    rc = _cmd_import(
                        argparse.Namespace(path=str(proj), route_id="custom-route-1", yes=True)
                    )
                    assert rc == 0

        # Vault holds key
        vault = SecretVault(tmp_path)
        secret = vault.get("claude-route-custom-route-1")
        assert secret is not None
        assert secret.reveal() == "sk-imported-test-token"

        # Config contains route and binding
        created_route = captured_config[-1]["claude"]["routes"]["custom-route-1"]
        assert created_route["api_key"] == "secret://claude-route-custom-route-1"
        assert created_route["base_url"] == "http://gw.local:20128/v1"
        assert created_route["models"]["opus"] == "code-high"
        assert created_route["models"]["sonnet"] == "code-medium"
        assert captured_config[-1]["claude"]["project_routes"][str(proj)] == "custom-route-1"

        # Preference set
        assert get_project_model_pref("custom-route-1") == "code-medium"
