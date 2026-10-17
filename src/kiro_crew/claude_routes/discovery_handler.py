"""Dashboard HTTP handler support for per-project Claude route model discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kiro_crew import model_registry
from kiro_crew.claude_routes.config import ClaudeConfig, resolve_claude_route
from kiro_crew.claude_routes.discovery import discover_route_models
from kiro_crew.claude_routes.model_cache import get_route_models

if TYPE_CHECKING:
    from aiohttp import web

    from kiro_crew.config import KiroCrewConfig

logger = logging.getLogger(__name__)


async def routed_cc_models(
    request: web.Request,
    cfg: KiroCrewConfig,
    project_path: str,
) -> list[dict[str, Any]] | None:
    """Return the available models list for a routed Claude project.

    @param request - The incoming HTTP web request
    @param cfg - The active KiroCrewConfig
    @param project_path - The absolute project directory path
    @returns List of model dicts for UI dropdown, or None if unrouted
    """
    claude_cfg = getattr(cfg, "claude", None)
    if not isinstance(claude_cfg, ClaudeConfig):
        return None
    if not claude_cfg.project_routes or not claude_cfg.routes:
        return None

    resolved = resolve_claude_route(claude_cfg, project_path)
    if not resolved:
        return None

    route_id = resolved.route_id
    route = resolved.route

    # Discover live from gateway or read route-scoped cache
    discovered = await discover_route_models(route, route_id)
    if not discovered:
        discovered = get_route_models(route_id)

    # Gather all candidate model IDs for this route
    seen: set[str] = set()
    result: list[dict[str, Any]] = [
        {
            "model_name": "auto",
            "display_name": "auto",
            "description": "Default configuration for this route",
        }
    ]
    seen.add("auto")

    # Add extra models (e.g. opusplan)
    for extra in route.extra_models:
        if extra and extra not in seen:
            seen.add(extra)
            result.append(
                {
                    "model_name": extra,
                    "display_name": extra,
                    "description": f"Route option ({route_id})",
                }
            )

    # Add discovered models
    for m_id in discovered:
        if m_id and m_id not in seen:
            seen.add(m_id)
            win = model_registry.model_window(m_id)
            entry: dict[str, Any] = {
                "model_name": m_id,
                "display_name": m_id,
                "description": f"Gateway model ({route_id})",
            }
            if win:
                entry["context_window"] = win
            result.append(entry)

    # Add alias models if configured
    for alias_name, alias_id in route.models.items():
        if alias_id and alias_id not in seen:
            seen.add(alias_id)
            result.append(
                {
                    "model_name": alias_id,
                    "display_name": alias_id,
                    "description": f"Alias for {alias_name} ({route_id})",
                }
            )

    return result
