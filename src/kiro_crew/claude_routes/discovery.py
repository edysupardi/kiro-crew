"""Live model discovery for Claude gateway routes via /v1/models endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

import aiohttp

from kiro_crew.claude_routes.config import ClaudeRouteConfig
from kiro_crew.claude_routes.model_cache import get_route_models, set_route_models
from kiro_crew.config.loader import config_dir as get_config_dir
from kiro_crew.mcp_gateway.secret_uri import resolve_secret_uris

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

DISCOVERY_TIMEOUT_SECS = 5.0
DISCOVERY_MAX_BODY_BYTES = 1024 * 1024  # 1 MiB
DISCOVERY_CACHE_TTL_SECS = 300.0  # 5 minutes

_DISCOVERY_LOCKS: dict[str, asyncio.Lock] = {}
_LAST_DISCOVERED_AT: dict[str, float] = {}


def _get_lock(route_id: str) -> asyncio.Lock:
    """Return an asyncio lock for single-flight discovery per route."""
    if route_id not in _DISCOVERY_LOCKS:
        _DISCOVERY_LOCKS[route_id] = asyncio.Lock()
    return _DISCOVERY_LOCKS[route_id]


async def discover_route_models(
    route: ClaudeRouteConfig,
    route_id: str,
    *,
    config_dir: Path | None = None,
    force: bool = False,
    now_fn: object = None,
) -> list[str]:
    """Query gateway /v1/models endpoint to discover available models for a route.

    @param route - Configuration of the Claude route
    @param route_id - Unique identifier for the route
    @param config_dir - Optional path to the configuration directory
    @param force - If True, bypass the in-memory TTL check
    @param now_fn - Optional clock function for deterministic testing
    @returns Discovered model IDs, or cached/fallback IDs on network error
    """
    if not route.discover_models or not route.base_url:
        return get_route_models(route_id)

    clock = now_fn if callable(now_fn) else time.monotonic
    now = clock()
    last_run = _LAST_DISCOVERED_AT.get(route_id, 0.0)
    if not force and (now - last_run) < DISCOVERY_CACHE_TTL_SECS:
        cached = get_route_models(route_id)
        if cached:
            return cached

    lock = _get_lock(route_id)
    async with lock:
        # Re-check under lock in case another coroutine just finished discovery
        last_run = _LAST_DISCOVERED_AT.get(route_id, 0.0)
        if not force and (clock() - last_run) < DISCOVERY_CACHE_TTL_SECS:
            cached = get_route_models(route_id)
            if cached:
                return cached

        target_url = route.effective_models_url
        if not target_url:
            return get_route_models(route_id)

        target_config_dir = config_dir or get_config_dir()
        api_key = ""
        if route.api_key:
            try:
                resolved, _ = resolve_secret_uris(
                    {"KEY": route.api_key},
                    target_config_dir,
                    subject=f"claude.routes.{route_id}",
                )
                api_key = resolved.get("KEY", "")
            except Exception:
                logger.warning(
                    "Claude route '%s': secret resolution failed for discovery", route_id
                )
                return get_route_models(route_id)

        headers = {
            "Accept": "application/json",
            "User-Agent": "KiroCrew-ClaudeRoutes/1.0",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            if route.auth_env == "ANTHROPIC_API_KEY":
                headers["x-api-key"] = api_key

        timeout = aiohttp.ClientTimeout(total=DISCOVERY_TIMEOUT_SECS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(target_url, headers=headers, allow_redirects=False) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Claude route '%s': model discovery returned HTTP %d",
                            route_id,
                            resp.status,
                        )
                        return get_route_models(route_id)

                    content_bytes = bytearray()
                    while True:
                        chunk = await resp.content.read(65536)
                        if not chunk:
                            break
                        content_bytes.extend(chunk)
                        if len(content_bytes) > DISCOVERY_MAX_BODY_BYTES:
                            logger.warning(
                                "Claude route '%s': model discovery response exceeded 1 MiB limit",
                                route_id,
                            )
                            return get_route_models(route_id)

                    text = content_bytes.decode("utf-8", errors="replace")
                    parsed = json.loads(text)
        except Exception:
            logger.debug(
                "Claude route '%s': discovery network request failed", route_id, exc_info=True
            )
            return get_route_models(route_id)

        discovered: list[str] = []
        if isinstance(parsed, dict):
            items = parsed.get("data") or parsed.get("models")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "id" in item:
                        m_id = str(item["id"]).strip()
                        if m_id and len(m_id) <= 256:
                            discovered.append(m_id)
                    elif isinstance(item, str):
                        m_id = item.strip()
                        if m_id and len(m_id) <= 256:
                            discovered.append(m_id)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    m_id = item.strip()
                    if m_id and len(m_id) <= 256:
                        discovered.append(m_id)

        if discovered:
            set_route_models(route_id, discovered)
            _LAST_DISCOVERED_AT[route_id] = clock()
            return discovered

        return get_route_models(route_id)
