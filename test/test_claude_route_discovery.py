"""Tests for Claude route model cache, live discovery, and API endpoint integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from kiro_crew.claude_routes.config import ClaudeConfig, ClaudeRouteConfig
from kiro_crew.claude_routes.discovery import discover_route_models
from kiro_crew.claude_routes.discovery_handler import routed_cc_models
from kiro_crew.claude_routes.model_cache import (
    _ROUTE_ADVERTISED,
    get_route_models,
    set_route_models,
)
from kiro_crew.secrets.vault import SecretVault


def _setup_vault(root: Path, key_name: str, secret_value: str) -> Path:
    v = SecretVault(root)
    v.set_sync(key_name, secret_value)
    return root


class TestModelCache:
    def test_cache_set_and_get_isolated_by_route(self, tmp_path: Path) -> None:
        with patch(
            "kiro_crew.claude_routes.model_cache.get_config_dir", return_value=str(tmp_path)
        ):
            _ROUTE_ADVERTISED.clear()
            set_route_models("route-1", ["m1", "m2", "m1"])
            set_route_models("route-2", ["m3", "m4"])

            assert get_route_models("route-1") == ["m1", "m2"]
            assert get_route_models("route-2") == ["m3", "m4"]
            assert get_route_models("unknown") == []

            # Check persistent file on disk
            cache_file = tmp_path / "claude_route_models.json"
            assert cache_file.is_file()
            disk_data = json.loads(cache_file.read_text(encoding="utf-8"))
            assert disk_data["route-1"] == ["m1", "m2"]
            assert disk_data["route-2"] == ["m3", "m4"]


class TestLiveDiscovery:
    @pytest.mark.asyncio
    async def test_discovers_openai_style_models_with_bearer_and_x_api_key(
        self, tmp_path: Path
    ) -> None:
        received_headers: dict[str, str] = {}

        async def handler(request: web.Request) -> web.Response:
            received_headers["authorization"] = request.headers.get("Authorization", "")
            received_headers["x-api-key"] = request.headers.get("x-api-key", "")
            data = {
                "object": "list",
                "data": [
                    {"id": "code-medium", "object": "model"},
                    {"id": "code-high", "object": "model"},
                ],
            }
            return web.json_response(data)

        app = web.Application()
        app.router.add_get("/v1/models", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}/v1"

        try:
            cfg_dir = _setup_vault(tmp_path, "route-key", "secret-token-123")
            route = ClaudeRouteConfig(
                base_url=base_url,
                api_key="secret://route-key",
                auth_env="ANTHROPIC_API_KEY",
                discover_models=True,
            )

            with patch(
                "kiro_crew.claude_routes.model_cache.get_config_dir", return_value=str(tmp_path)
            ):
                _ROUTE_ADVERTISED.clear()
                models = await discover_route_models(route, "r1", config_dir=cfg_dir, force=True)

            assert "code-medium" in models
            assert "code-high" in models
            assert received_headers["authorization"] == "Bearer secret-token-123"
            assert received_headers["x-api-key"] == "secret-token-123"
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_fails_closed_on_error_and_returns_cached(self, tmp_path: Path) -> None:
        async def err_handler(request: web.Request) -> web.Response:
            return web.Response(status=500, text="Internal Server Error")

        app = web.Application()
        app.router.add_get("/v1/models", err_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}/v1"

        try:
            cfg_dir = _setup_vault(tmp_path, "route-key", "secret-token-123")
            route = ClaudeRouteConfig(
                base_url=base_url,
                api_key="secret://route-key",
                discover_models=True,
            )

            with patch(
                "kiro_crew.claude_routes.model_cache.get_config_dir", return_value=str(tmp_path)
            ):
                _ROUTE_ADVERTISED["r1"] = ["cached-prior-model"]
                models = await discover_route_models(route, "r1", config_dir=cfg_dir, force=True)

            # Returns cached fallback on HTTP failure
            assert models == ["cached-prior-model"]
        finally:
            await runner.cleanup()


class TestDiscoveryHandler:
    @pytest.mark.asyncio
    async def test_routed_cc_models_formats_dropdown_entries(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project-alpha"
        proj_dir.mkdir(parents=True, exist_ok=True)

        route = ClaudeRouteConfig(
            base_url="http://mock-gw/v1",
            api_key="secret://key",
            extra_models=["opusplan"],
            models={"sonnet": "code-medium"},
            discover_models=False,
        )
        claude_cfg = ClaudeConfig(
            routes={"route-alpha": route},
            project_routes={str(proj_dir): "route-alpha"},
        )

        class DummyConfig:
            claude = claude_cfg

        mock_request = MagicMock()

        with patch(
            "kiro_crew.claude_routes.model_cache.get_config_dir", return_value=str(tmp_path)
        ):
            _ROUTE_ADVERTISED["route-alpha"] = ["gw-model-1", "gw-model-2"]
            result = await routed_cc_models(mock_request, DummyConfig(), str(proj_dir))

        assert result is not None
        names = [entry["model_name"] for entry in result]
        assert names[0] == "auto"
        assert "opusplan" in names
        assert "gw-model-1" in names
        assert "gw-model-2" in names
        assert "code-medium" in names

    @pytest.mark.asyncio
    async def test_unrouted_project_returns_none(self, tmp_path: Path) -> None:
        cfg = ClaudeConfig()

        class DummyConfig:
            claude = cfg

        mock_request = MagicMock()
        result = await routed_cc_models(mock_request, DummyConfig(), str(tmp_path))
        assert result is None
