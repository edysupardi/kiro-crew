"""Dashboard owner handlers for per-project Claude routes: GET, PUT, POST /import."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from aiohttp import web

from kiro_crew.claude_routes.config import (
    CLAUDE_ROUTES_SECTION_KEY,
    configured_claude_routes,
    resolve_claude_route,
)
from kiro_crew.claude_routes.preferences import (
    get_project_model_pref,
    set_project_model_pref,
)
from kiro_crew.claude_routes.settings_conflict import _safe_read_json_file
from kiro_crew.config.loader import (
    KiroCrewConfig,
)
from kiro_crew.config.loader import config_dir as get_config_dir
from kiro_crew.config.loader import (
    config_path,
    update_config_locked,
)
from kiro_crew.dashboard.chat_utils import run_config_write
from kiro_crew.secrets.vault import SecretVault

logger = logging.getLogger(__name__)

PROJECT_DIR_MAX_LEN = 4096


def _project_path(raw: object) -> str | None:
    """Normalize and validate an absolute project path.

    @param raw - Input value representing path
    @returns Clean normalized absolute path or None
    """
    if not isinstance(raw, str):
        return None
    path = raw.strip()
    if not path or len(path) > PROJECT_DIR_MAX_LEN or "\x00" in path:
        return None
    if not os.path.isabs(path):
        return None
    return os.path.normpath(path)


async def api_claude_project_route_get(request: web.Request) -> web.Response:
    """GET /api/claude/project-route?path=<dir> — View route binding for a directory.

    @param request - HTTP request object
    @returns JSON response with route details or 403 if unauthorized
    """
    from kiro_crew.dashboard.handlers._shared import require_owner_dashboard_request

    denied = await require_owner_dashboard_request(request, "claude_project_route_get")
    if denied is not None:
        return denied

    raw_path = request.query.get("path")
    norm_path = _project_path(raw_path)
    if not norm_path:
        return web.json_response({"error": "invalid path parameter"}, status=400)

    cfg = await asyncio.to_thread(KiroCrewConfig.load)
    claude_cfg = configured_claude_routes(cfg)
    resolved = resolve_claude_route(claude_cfg, norm_path)

    available_routes = sorted(claude_cfg.routes.keys())
    if not resolved:
        return web.json_response(
            {
                "path": norm_path,
                "boundDir": None,
                "routeId": None,
                "baseUrl": None,
                "hasKey": False,
                "models": {},
                "extraModels": [],
                "preferredModel": None,
                "routes": available_routes,
            }
        )

    route = resolved.route
    pref = get_project_model_pref(resolved.route_id)
    return web.json_response(
        {
            "path": norm_path,
            "boundDir": resolved.bound_dir,
            "routeId": resolved.route_id,
            "baseUrl": route.base_url,
            "hasKey": bool(route.api_key),
            "models": route.models,
            "extraModels": route.extra_models,
            "preferredModel": pref,
            "routes": available_routes,
        }
    )


async def api_claude_project_route_put(request: web.Request) -> web.Response:
    """PUT /api/claude/project-route — Bind or unbind a project directory to a route.

    @param request - HTTP request with JSON body {path, routeId}
    @returns JSON confirmation response
    """
    from kiro_crew.dashboard.handlers._shared import require_owner_dashboard_request

    denied = await require_owner_dashboard_request(request, "claude_project_route_put")
    if denied is not None:
        return denied

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    norm_path = _project_path(body.get("path"))
    if not norm_path:
        return web.json_response({"error": "invalid path parameter"}, status=400)

    route_id = body.get("routeId")
    if route_id is not None and not isinstance(route_id, str):
        return web.json_response({"error": "invalid routeId"}, status=400)

    clean_route_id = (route_id or "").strip()

    def _update(data: dict[str, Any]) -> None:
        claude_sect = data.setdefault(CLAUDE_ROUTES_SECTION_KEY, {})
        if not isinstance(claude_sect, dict):
            claude_sect = {}
            data[CLAUDE_ROUTES_SECTION_KEY] = claude_sect
        proj_routes = claude_sect.setdefault("project_routes", {})
        if not isinstance(proj_routes, dict):
            proj_routes = {}
            claude_sect["project_routes"] = proj_routes

        if not clean_route_id:
            proj_routes.pop(norm_path, None)
        else:
            proj_routes[norm_path] = clean_route_id

    try:
        await run_config_write(update_config_locked, config_path(), mutate=_update)
    except Exception as exc:
        logger.warning("Failed to save claude project route binding: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)

    return web.json_response({"ok": True, "path": norm_path, "routeId": clean_route_id or None})


async def api_claude_project_route_import(request: web.Request) -> web.Response:
    """POST /api/claude/project-route/import — Import route settings from project local settings.

    @param request - HTTP request with JSON body {path, routeId, confirm}
    @returns Preview dict if confirm is false, or imported route info if true
    """
    from kiro_crew.dashboard.handlers._shared import require_owner_dashboard_request

    denied = await require_owner_dashboard_request(request, "claude_project_route_import")
    if denied is not None:
        return denied

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    norm_path = _project_path(body.get("path"))
    if not norm_path:
        return web.json_response({"error": "invalid path parameter"}, status=400)

    route_id = str(body.get("routeId") or Path(norm_path).name or "imported").strip().lower()
    route_id = "".join(c if c.isalnum() or c in "._-" else "-" for c in route_id)
    confirm = bool(body.get("confirm", False))

    # Read project .claude/settings.local.json and .claude/settings.json safely
    proj_dir = Path(norm_path)
    extracted_env: dict[str, str] = {}
    extracted_model: str = ""

    for rel in (".claude/settings.local.json", ".claude/settings.json"):
        f_path = proj_dir / rel
        data = _safe_read_json_file(f_path)
        if isinstance(data, dict):
            if not extracted_model and isinstance(data.get("model"), str):
                extracted_model = str(data["model"]).strip()
            env_map = data.get("env")
            if isinstance(env_map, dict):
                for k, v in env_map.items():
                    if isinstance(k, str) and isinstance(v, str) and k not in extracted_env:
                        extracted_env[k] = v

    base_url = extracted_env.get("ANTHROPIC_BASE_URL", "")
    api_key = extracted_env.get("ANTHROPIC_API_KEY", "") or extracted_env.get(
        "ANTHROPIC_AUTH_TOKEN", ""
    )
    auth_env = (
        "ANTHROPIC_AUTH_TOKEN"
        if "ANTHROPIC_AUTH_TOKEN" in extracted_env and "ANTHROPIC_API_KEY" not in extracted_env
        else "ANTHROPIC_API_KEY"
    )

    models_map: dict[str, str] = {}
    if extracted_env.get("ANTHROPIC_DEFAULT_OPUS_MODEL"):
        models_map["opus"] = extracted_env["ANTHROPIC_DEFAULT_OPUS_MODEL"]
    if extracted_env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"):
        models_map["sonnet"] = extracted_env["ANTHROPIC_DEFAULT_SONNET_MODEL"]
    if extracted_env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        models_map["haiku"] = extracted_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"]

    subagent_model = extracted_env.get("CLAUDE_CODE_SUBAGENT_MODEL", "")

    if not confirm:
        return web.json_response(
            {
                "preview": True,
                "path": norm_path,
                "routeId": route_id,
                "baseUrl": base_url,
                "hasKey": bool(api_key),
                "authEnv": auth_env,
                "models": models_map,
                "subagentModel": subagent_model,
                "preferredModel": extracted_model,
            }
        )

    # When confirm is True, store key in Vault (if key exists) and update config.json
    secret_ref = ""
    if api_key:
        secret_name = f"claude-route-{route_id}"
        vault = SecretVault(get_config_dir())
        await vault.set(secret_name, api_key)
        secret_ref = f"secret://{secret_name}"

    def _commit_import(data: dict[str, Any]) -> None:
        claude_sect = data.setdefault(CLAUDE_ROUTES_SECTION_KEY, {})
        if not isinstance(claude_sect, dict):
            claude_sect = {}
            data[CLAUDE_ROUTES_SECTION_KEY] = claude_sect

        routes = claude_sect.setdefault("routes", {})
        if not isinstance(routes, dict):
            routes = {}
            claude_sect["routes"] = routes

        route_entry: dict[str, Any] = {
            "base_url": base_url,
            "auth_env": auth_env,
            "models": models_map,
            "subagent_model": subagent_model,
            "extra_models": ["opusplan"],
            "discover_models": True,
        }
        if secret_ref:
            route_entry["api_key"] = secret_ref
        routes[route_id] = route_entry

        proj_routes = claude_sect.setdefault("project_routes", {})
        if not isinstance(proj_routes, dict):
            proj_routes = {}
            claude_sect["project_routes"] = proj_routes
        proj_routes[norm_path] = route_id

    try:
        await run_config_write(update_config_locked, config_path(), mutate=_commit_import)
    except Exception as exc:
        logger.warning("Failed to commit imported claude route: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)

    if extracted_model:
        set_project_model_pref(route_id, extracted_model)

    return web.json_response(
        {
            "ok": True,
            "path": norm_path,
            "routeId": route_id,
            "baseUrl": base_url,
            "hasKey": bool(secret_ref),
            "preferredModel": extracted_model or None,
        }
    )
