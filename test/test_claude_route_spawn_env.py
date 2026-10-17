"""Tests for Claude per-project route spawn environment injection and isolation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kiro_crew.acp.harness.claude import ClaudeLaunch
from kiro_crew.acp.transport_errors import AcpError
from kiro_crew.claude_routes.config import (
    ClaudeConfig,
    ClaudeRouteConfig,
    prepare_claude_route_env,
)
from kiro_crew.claude_routes.settings_conflict import (
    check_claude_route_conflicts,
    read_settings_env,
)
from kiro_crew.secrets.vault import SecretVault


def _setup_vault(root: Path, key_name: str, secret_value: str) -> Path:
    """Store a test secret in a temporary vault.

    @param root - Root directory representing KIROCREW_HOME
    @param key_name - Secret key identifier
    @param secret_value - Plaintext secret value
    @returns Configuration directory path
    """
    v = SecretVault(root)
    v.set_sync(key_name, secret_value)
    return root


class TestPrepareClaudeRouteEnv:
    def test_unrouted_directory_returns_empty(self, tmp_path: Path) -> None:
        cfg = ClaudeConfig()
        env_out, to_pop = prepare_claude_route_env(
            None, str(tmp_path), config_dir=tmp_path, cfg=cfg
        )
        assert env_out == {}
        assert to_pop == set()

    def test_routed_directory_resolves_secret_and_aliases(self, tmp_path: Path) -> None:
        cfg_dir = _setup_vault(tmp_path, "route-key-1", "sk-proj-test-secret")
        proj_dir = tmp_path / "project-a"
        proj_dir.mkdir(parents=True, exist_ok=True)

        route = ClaudeRouteConfig(
            base_url="http://192.168.1.4:20128/v1",
            api_key="secret://route-key-1",
            auth_env="ANTHROPIC_API_KEY",
            models={"sonnet": "code-medium", "opus": "code-high", "haiku": "code-low"},
            subagent_model="code-medium",
        )
        claude_cfg = ClaudeConfig(
            routes={"route-a": route},
            project_routes={str(proj_dir): "route-a"},
        )

        class DummyConfig:
            claude = claude_cfg

        env_out, to_pop = prepare_claude_route_env(
            None, str(proj_dir), config_dir=cfg_dir, cfg=DummyConfig()
        )

        assert env_out["ANTHROPIC_BASE_URL"] == "http://192.168.1.4:20128/v1"
        assert env_out["ANTHROPIC_API_KEY"] == "sk-proj-test-secret"
        assert env_out["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "code-high"
        assert env_out["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "code-medium"
        assert env_out["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "code-low"
        assert env_out["CLAUDE_CODE_SUBAGENT_MODEL"] == "code-medium"
        assert "ANTHROPIC_BASE_URL" in to_pop
        assert "ANTHROPIC_API_KEY" in to_pop

    def test_missing_secret_raises_value_error(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project-a"
        proj_dir.mkdir(parents=True, exist_ok=True)
        route = ClaudeRouteConfig(
            base_url="http://host/v1",
            api_key="secret://non-existent-secret",
        )
        claude_cfg = ClaudeConfig(
            routes={"r1": route},
            project_routes={str(proj_dir): "r1"},
        )

        class DummyConfig:
            claude = claude_cfg

        with pytest.raises(ValueError) as excinfo:
            prepare_claude_route_env(None, str(proj_dir), config_dir=tmp_path, cfg=DummyConfig())
        assert "secret://non-existent-secret" in str(excinfo.value) or "claude.routes.r1" in str(
            excinfo.value
        )


class TestSettingsConflicts:
    def test_detects_global_settings_env_conflict(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        (claude_dir / "settings.json").write_text(
            '{"env": {"ANTHROPIC_DEFAULT_SONNET_MODEL": "other-model"}}',
            encoding="utf-8",
        )
        route_env = {"ANTHROPIC_DEFAULT_SONNET_MODEL": "code-medium"}
        conflicts = check_claude_route_conflicts("r1", route_env, home_dir=tmp_path)
        assert len(conflicts) == 1
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in conflicts[0]

    def test_detects_project_local_settings_conflict(self, tmp_path: Path) -> None:
        proj_claude = tmp_path / "proj" / ".claude"
        proj_claude.mkdir(parents=True, exist_ok=True)
        (proj_claude / "settings.local.json").write_text(
            '{"env": {"ANTHROPIC_BASE_URL": "http://conflicting-url/v1"}}',
            encoding="utf-8",
        )
        route_env = {"ANTHROPIC_BASE_URL": "http://192.168.1.4:20128/v1"}
        conflicts = check_claude_route_conflicts(
            "r1", route_env, project_dir=str(tmp_path / "proj"), home_dir=tmp_path
        )
        assert len(conflicts) == 1
        assert "ANTHROPIC_BASE_URL" in conflicts[0]

    def test_read_settings_env_handles_missing_and_valid(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "missing.json"
        assert read_settings_env(non_existent) == {}

        valid = tmp_path / "valid.json"
        valid.write_text('{"env": {"FOO": "bar", "NUM": 123}}', encoding="utf-8")
        assert read_settings_env(valid) == {"FOO": "bar"}


class TestClaudeLaunchPrepareSpawnEnv:
    @pytest.mark.asyncio
    async def test_prepare_spawn_env_injects_route_and_pops_ambient(self, tmp_path: Path) -> None:
        launch = ClaudeLaunch()
        launch._work_dir = str(tmp_path)

        resolved_route_env = {
            "ANTHROPIC_BASE_URL": "http://custom-gw/v1",
            "ANTHROPIC_API_KEY": "sk-secret-key",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "code-medium",
        }
        keys_to_pop = {"ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_DEFAULT_SONNET_MODEL"}

        env = {
            "KIROCREW_CLAUDE_ROUTE": "route-1",
            "ANTHROPIC_BASE_URL": "http://old-global/v1",
            "ANTHROPIC_API_KEY": "old-global-key",
            "PATH": "/bin:/usr/bin",
        }

        async def _fake_to_thread(func, *args, **kwargs):
            return resolved_route_env, keys_to_pop

        with patch.object(launch, "apply_spawn_env"):
            await launch.prepare_spawn_env(env, to_thread=_fake_to_thread)

        assert env["ANTHROPIC_BASE_URL"] == "http://custom-gw/v1"
        assert env["ANTHROPIC_API_KEY"] == "sk-secret-key"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "code-medium"
        # Marker was popped
        assert "KIROCREW_CLAUDE_ROUTE" not in env

    @pytest.mark.asyncio
    async def test_prepare_spawn_env_converts_value_error_to_acp_error(
        self, tmp_path: Path
    ) -> None:
        launch = ClaudeLaunch()
        launch._work_dir = str(tmp_path)
        env = {"KIROCREW_CLAUDE_ROUTE": "route-missing-secret"}

        async def _failing_to_thread(func, *args, **kwargs):
            raise ValueError("vault secret not found")

        with patch.object(launch, "apply_spawn_env"):
            with pytest.raises(AcpError) as excinfo:
                await launch.prepare_spawn_env(env, to_thread=_failing_to_thread)
            assert "Claude route credential resolution failed" in str(excinfo.value)
