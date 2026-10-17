"""Tests for Claude project route dashboard endpoints (GET, PUT, POST /import)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from kiro_crew.claude_routes.config import ClaudeConfig, ClaudeRouteConfig
from kiro_crew.claude_routes.handlers import (
    api_claude_project_route_get,
    api_claude_project_route_import,
    api_claude_project_route_put,
)
from kiro_crew.claude_routes.preferences import get_project_model_pref
from kiro_crew.secrets.vault import SecretVault


def _make_mock_request(
    query: dict | None = None,
    body: dict | None = None,
) -> web.Request:
    req = MagicMock(spec=web.Request)
    req.query = query or {}
    if body is not None:
        req.json = AsyncMock(return_value=body)
    return req


class TestClaudeProjectRouteEndpoints:
    @pytest.mark.asyncio
    async def test_get_unbound_and_bound_directory(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project-one"
        proj_dir.mkdir(parents=True, exist_ok=True)

        claude_cfg = ClaudeConfig(
            routes={
                "r1": ClaudeRouteConfig(
                    base_url="http://host:20128/v1",
                    api_key="secret://k1",
                    models={"sonnet": "code-medium"},
                )
            },
            project_routes={str(proj_dir): "r1"},
        )

        class DummyConfig:
            claude = claude_cfg

        with patch(
            "kiro_crew.dashboard.handlers._shared.require_owner_dashboard_request",
            return_value=None,
        ):
            with patch("kiro_crew.config.loader.KiroCrewConfig.load", return_value=DummyConfig()):
                with patch(
                    "kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)
                ):
                    # 1. Bound project query
                    req_bound = _make_mock_request(query={"path": str(proj_dir)})
                    resp_bound = await api_claude_project_route_get(req_bound)
                    assert resp_bound.status == 200
                    data_bound = json.loads(resp_bound.body.decode("utf-8"))
                    assert data_bound["routeId"] == "r1"
                    assert data_bound["boundDir"] == str(proj_dir)
                    assert data_bound["hasKey"] is True
                    assert data_bound["baseUrl"] == "http://host:20128/v1"

                    # 2. Unbound directory query
                    req_unbound = _make_mock_request(query={"path": str(tmp_path / "unmapped")})
                    resp_unbound = await api_claude_project_route_get(req_unbound)
                    assert resp_unbound.status == 200
                    data_unbound = json.loads(resp_unbound.body.decode("utf-8"))
                    assert data_unbound["routeId"] is None
                    assert data_unbound["boundDir"] is None

    @pytest.mark.asyncio
    async def test_put_binds_and_unbinds_route(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "project-b"
        target_dir.mkdir(parents=True, exist_ok=True)

        captured_updates = []

        async def _mock_run_config_write(func, path, *args, **kwargs):
            fn = kwargs.get("mutate") or (args[0] if args else None)
            dummy_data: dict = {"claude": {"routes": {"route-b": {}}, "project_routes": {}}}
            if fn:
                fn(dummy_data)
            captured_updates.append(dummy_data)

        with patch(
            "kiro_crew.dashboard.handlers._shared.require_owner_dashboard_request",
            return_value=None,
        ):
            with patch(
                "kiro_crew.claude_routes.handlers.run_config_write",
                side_effect=_mock_run_config_write,
            ):
                # Bind
                req_bind = _make_mock_request(body={"path": str(target_dir), "routeId": "route-b"})
                resp_bind = await api_claude_project_route_put(req_bind)
                assert resp_bind.status == 200
                assert (
                    captured_updates[-1]["claude"]["project_routes"][str(target_dir)] == "route-b"
                )

                # Unbind
                req_unbind = _make_mock_request(body={"path": str(target_dir), "routeId": ""})
                resp_unbind = await api_claude_project_route_put(req_unbind)
                assert resp_unbind.status == 200
                assert str(target_dir) not in captured_updates[-1]["claude"]["project_routes"]

    @pytest.mark.asyncio
    async def test_import_preview_and_confirm(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "imported-proj"
        claude_dir = proj_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)

        # Write local settings with secret and alias
        (claude_dir / "settings.local.json").write_text(
            json.dumps(
                {
                    "model": "code-medium",
                    "env": {
                        "ANTHROPIC_BASE_URL": "http://192.168.1.4:20128/v1",
                        "ANTHROPIC_API_KEY": "sk-super-secret-key-12345",
                        "ANTHROPIC_DEFAULT_OPUS_MODEL": "code-high",
                        "ANTHROPIC_DEFAULT_SONNET_MODEL": "code-medium",
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch(
            "kiro_crew.dashboard.handlers._shared.require_owner_dashboard_request",
            return_value=None,
        ):
            with patch(
                "kiro_crew.claude_routes.handlers.get_config_dir", return_value=str(tmp_path)
            ):
                with patch(
                    "kiro_crew.claude_routes.preferences.get_config_dir", return_value=str(tmp_path)
                ):
                    # 1. Preview
                    req_prev = _make_mock_request(
                        body={"path": str(proj_dir), "routeId": "imp-r1", "confirm": False}
                    )
                    resp_prev = await api_claude_project_route_import(req_prev)
                    assert resp_prev.status == 200
                    prev_data = json.loads(resp_prev.body.decode("utf-8"))
                    assert prev_data["preview"] is True
                    assert prev_data["baseUrl"] == "http://192.168.1.4:20128/v1"
                    assert prev_data["hasKey"] is True
                    assert prev_data["models"]["opus"] == "code-high"
                    assert prev_data["preferredModel"] == "code-medium"
                    # Plaintext key is NOT returned in preview response!
                    assert "sk-super-secret" not in str(prev_data)

                    # 2. Confirm import
                    captured_config = []

                    async def _mock_run_config_write(func, path, *args, **kwargs):
                        fn = kwargs.get("mutate") or (args[0] if args else None)
                        cfg_dict: dict = {"claude": {"routes": {}, "project_routes": {}}}
                        if fn:
                            fn(cfg_dict)
                        captured_config.append(cfg_dict)

                    with patch(
                        "kiro_crew.claude_routes.handlers.run_config_write",
                        side_effect=_mock_run_config_write,
                    ):
                        req_conf = _make_mock_request(
                            body={"path": str(proj_dir), "routeId": "imp-r1", "confirm": True}
                        )
                        resp_conf = await api_claude_project_route_import(req_conf)
                        assert resp_conf.status == 200
                        conf_data = json.loads(resp_conf.body.decode("utf-8"))
                        assert conf_data["ok"] is True
                        assert conf_data["routeId"] == "imp-r1"

                        # Key stored in vault
                        vault = SecretVault(tmp_path)
                        stored_secret = vault.get("claude-route-imp-r1")
                        assert stored_secret is not None
                        assert stored_secret.reveal() == "sk-super-secret-key-12345"

                        # Route created with secret:// reference (never plaintext)
                        saved_route = captured_config[-1]["claude"]["routes"]["imp-r1"]
                        assert saved_route["api_key"] == "secret://claude-route-imp-r1"
                        assert saved_route["base_url"] == "http://192.168.1.4:20128/v1"
                        assert (
                            captured_config[-1]["claude"]["project_routes"][str(proj_dir)]
                            == "imp-r1"
                        )

                        # Model preference set
                        assert get_project_model_pref("imp-r1") == "code-medium"
