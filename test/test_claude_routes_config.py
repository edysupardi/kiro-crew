"""Tests for per-project Claude routes configuration and directory resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiro_crew.claude_routes.config import (
    CLAUDE_ROUTES_SECTION_KEY,
    ClaudeConfig,
    ClaudeRouteConfig,
    ResolvedClaudeRoute,
    claude_route_refusals,
    coerce_claude_routes,
    configured_claude_routes,
    refuse_claude_routes_on_write,
    resolve_claude_route,
)
from kiro_crew.config.loader import ConfigWriteRefused


def _project(root: Path, name: str) -> Path:
    """Create a project directory helper.

    @param root - Root directory path
    @param name - Project folder name
    @returns Path object for project directory
    """
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestClaudeRouteConfig:
    def test_default_values(self) -> None:
        cfg = ClaudeRouteConfig()
        assert cfg.base_url == ""
        assert cfg.api_key == ""
        assert cfg.auth_env == "ANTHROPIC_API_KEY"
        assert cfg.models == {}
        assert cfg.subagent_model == ""
        assert cfg.extra_models == []
        assert cfg.models_url == ""
        assert cfg.discover_models is True
        assert cfg.effective_models_url == ""

    def test_effective_models_url_resolution(self) -> None:
        cfg_custom = ClaudeRouteConfig(
            base_url="http://host:1234/v1", models_url="http://host:1234/custom/models"
        )
        assert cfg_custom.effective_models_url == "http://host:1234/custom/models"

        cfg_v1 = ClaudeRouteConfig(base_url="http://host:1234/v1")
        assert cfg_v1.effective_models_url == "http://host:1234/v1/models"

        cfg_root = ClaudeRouteConfig(base_url="http://host:1234")
        assert cfg_root.effective_models_url == "http://host:1234/v1/models"


class TestClaudeRouteRefusals:
    def test_valid_configuration_returns_no_refusals(self) -> None:
        raw = {
            "routes": {
                "proj-a": {
                    "base_url": "http://192.168.1.4:20128/v1",
                    "api_key": "secret://claude-proj-a",
                    "auth_env": "ANTHROPIC_API_KEY",
                    "models": {"sonnet": "code-medium", "opus": "code-high"},
                    "extra_models": ["opusplan"],
                    "discover_models": True,
                }
            },
            "project_routes": {
                "/home/dev/src/proj-a": "proj-a",
            },
        }
        assert claude_route_refusals(raw) == []

    def test_non_dict_section_refused(self) -> None:
        assert len(claude_route_refusals("invalid")) == 1

    def test_invalid_route_id_refused(self) -> None:
        raw = {"routes": {"INVALID_UPPER": {"base_url": "http://host/v1"}}}
        refusals = claude_route_refusals(raw)
        assert any("pattern" in r for r in refusals)

    def test_plaintext_api_key_refused(self) -> None:
        raw = {
            "routes": {
                "proj-a": {
                    "base_url": "http://host/v1",
                    "api_key": "sk-ant-1234567890",
                }
            }
        }
        refusals = claude_route_refusals(raw)
        assert any("secret:// URI reference, not a plaintext key" in r for r in refusals)

    def test_invalid_url_schemes_and_userinfo_refused(self) -> None:
        raw = {
            "routes": {
                "r1": {"base_url": "ftp://host/v1"},
                "r2": {"base_url": "http://user:pass@host/v1"},
                "r3": {"base_url": "http://host/v1?query=1"},
            }
        }
        refusals = claude_route_refusals(raw)
        assert any("scheme" in r for r in refusals)
        assert any("userinfo" in r for r in refusals)
        assert any("query parameters" in r for r in refusals)

    def test_relative_project_directory_refused(self) -> None:
        raw = {
            "routes": {"r1": {"base_url": "http://host/v1"}},
            "project_routes": {"relative/path": "r1"},
        }
        refusals = claude_route_refusals(raw)
        assert any("absolute directory path" in r for r in refusals)

    def test_undefined_route_target_refused(self) -> None:
        raw = {
            "routes": {"r1": {"base_url": "http://host/v1"}},
            "project_routes": {"/abs/path": "undefined-route"},
        }
        refusals = claude_route_refusals(raw)
        assert any("targets undefined route" in r for r in refusals)


class TestCoerceClaudeRoutes:
    def test_empty_or_none_returns_empty_config(self) -> None:
        assert coerce_claude_routes(None) == ClaudeConfig()
        assert coerce_claude_routes({}) == ClaudeConfig()

    def test_coerces_valid_and_drops_invalid(self) -> None:
        raw = {
            "routes": {
                "valid-route": {
                    "base_url": "http://host/v1",
                    "api_key": "secret://key1",
                    "models": {"sonnet": "model-1"},
                },
                "bad-route": {
                    "base_url": "not-a-valid-url",
                },
            },
            "project_routes": {
                "/abs/path/a/": "valid-route",
                "relative/path": "valid-route",
            },
        }
        cfg = coerce_claude_routes(raw)
        assert "valid-route" in cfg.routes
        assert "bad-route" not in cfg.routes
        # Directory trailing slash is normalized
        assert os.path.normpath("/abs/path/a") in cfg.project_routes
        assert "relative/path" not in cfg.project_routes


class TestResolveClaudeRoute:
    def test_exact_and_nested_directory_bound(self, tmp_path: Path) -> None:
        proj = _project(tmp_path, "proj-a")
        nested = _project(proj, "src/components")
        cfg = ClaudeConfig(
            routes={"r1": ClaudeRouteConfig(base_url="http://host/v1")},
            project_routes={str(proj): "r1"},
        )
        resolved = resolve_claude_route(cfg, str(proj))
        assert isinstance(resolved, ResolvedClaudeRoute)
        assert resolved.route_id == "r1"
        assert resolved.bound_dir == str(proj)

        nested_resolved = resolve_claude_route(cfg, str(nested))
        assert nested_resolved is not None
        assert nested_resolved.route_id == "r1"

    def test_unmapped_and_sibling_prefix_unbound(self, tmp_path: Path) -> None:
        proj = _project(tmp_path, "proj")
        sibling = _project(tmp_path, "project2")
        cfg = ClaudeConfig(
            routes={"r1": ClaudeRouteConfig(base_url="http://host/v1")},
            project_routes={str(proj): "r1"},
        )
        assert resolve_claude_route(cfg, str(sibling)) is None

    def test_deepest_directory_wins(self, tmp_path: Path) -> None:
        outer = _project(tmp_path, "workspace")
        inner = _project(outer, "subproject")
        cfg = ClaudeConfig(
            routes={
                "r-outer": ClaudeRouteConfig(base_url="http://outer/v1"),
                "r-inner": ClaudeRouteConfig(base_url="http://inner/v1"),
            },
            project_routes={
                str(outer): "r-outer",
                str(inner): "r-inner",
            },
        )
        assert resolve_claude_route(cfg, str(inner / "subfile")).route_id == "r-inner"
        assert resolve_claude_route(cfg, str(outer / "rootfile")).route_id == "r-outer"

    def test_configured_claude_routes_helper(self) -> None:
        assert CLAUDE_ROUTES_SECTION_KEY == "claude"
        cfg = ClaudeConfig(routes={"r1": ClaudeRouteConfig()})
        assert configured_claude_routes(cfg) == cfg

        class DummyHostConfig:
            claude = cfg

        assert configured_claude_routes(DummyHostConfig()) == cfg
        assert configured_claude_routes(None) == ClaudeConfig()


class TestWriteFloorRefusal:
    def test_write_refused_on_new_invalid_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        bad_data = {"claude": {"routes": {"bad": {"api_key": "plaintext-not-secret"}}}}
        with pytest.raises(ConfigWriteRefused) as excinfo:
            refuse_claude_routes_on_write(bad_data, config_path)
        assert "secret:// URI reference" in str(excinfo.value)

    def test_write_allowed_when_clean(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        clean_data = {
            "claude": {
                "routes": {
                    "clean": {
                        "base_url": "http://host/v1",
                        "api_key": "secret://vault-key",
                    }
                }
            }
        }
        refuse_claude_routes_on_write(clean_data, config_path)
