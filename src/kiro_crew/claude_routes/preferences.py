"""Persistent per-project model preferences for Claude routes."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.loader import config_dir as get_config_dir

logger = logging.getLogger(__name__)

_PREFERENCES: dict[str, dict[str, str]] = {}
_INITIALIZED: bool = False


def _pref_file_path() -> Path:
    """Return the filesystem path for persisted project model preferences."""
    return Path(get_config_dir()) / "claude_project_models.json"


def _ensure_loaded() -> None:
    """Load persisted preferences from disk if not yet in memory."""
    global _INITIALIZED  # noqa: PLW0603
    if _INITIALIZED:
        return
    _INITIALIZED = True
    path = _pref_file_path()
    try:
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            routes = data.get("routes")
            if isinstance(routes, dict):
                for route_id, info in routes.items():
                    if isinstance(route_id, str) and isinstance(info, dict):
                        m = info.get("model")
                        if isinstance(m, str):
                            _PREFERENCES[route_id] = {
                                "model": m,
                                "updated_at": str(info.get("updated_at", "")),
                            }
    except Exception:
        logger.debug("Could not load claude_project_models.json", exc_info=True)


def get_project_model_pref(route_id: str) -> str | None:
    """Get the preferred model ID for a Claude route.

    @param route_id - The route identifier
    @returns Preferred model name/ID or None
    """
    _ensure_loaded()
    entry = _PREFERENCES.get(route_id)
    return entry.get("model") if entry else None


def set_project_model_pref(route_id: str, model: str) -> None:
    """Save the preferred model ID for a Claude route.

    @param route_id - The route identifier
    @param model - Model name/ID to associate with the route
    """
    _ensure_loaded()
    clean_model = model.strip()
    if clean_model in ("", "auto"):
        clear_project_model_pref(route_id)
        return

    _PREFERENCES[route_id] = {
        "model": clean_model,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _persist()


def clear_project_model_pref(route_id: str) -> None:
    """Clear the preferred model ID for a Claude route.

    @param route_id - The route identifier
    """
    _ensure_loaded()
    if route_id in _PREFERENCES:
        _PREFERENCES.pop(route_id, None)
        _persist()


def _persist() -> None:
    """Persist memory preferences to disk atomically."""
    try:
        path = _pref_file_path()
        payload = {
            "version": 1,
            "routes": _PREFERENCES,
        }
        atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        logger.debug("Could not persist claude_project_models.json", exc_info=True)


def resolve_slot_project_model(cfg: object, project_path: str | None) -> str | None:
    """Resolve project-preferred model if project is bound to a Claude route.

    @param cfg - Configuration object
    @param project_path - Working directory of the slot
    @returns Preferred model string or None
    """
    if not project_path:
        return None
    from kiro_crew.claude_routes.config import configured_claude_routes, resolve_claude_route

    claude_cfg = configured_claude_routes(cfg)
    resolved = resolve_claude_route(claude_cfg, project_path)
    if resolved:
        return get_project_model_pref(resolved.route_id)
    return None


def handle_project_model_pick(
    cfg: object,
    project_path: str | None,
    model_name: str,
) -> bool:
    """Commit a project-scoped model pick if project is bound to a Claude route.

    @param cfg - Configuration object
    @param project_path - Working directory of the slot
    @param model_name - Model to save as preference
    @returns True if committed to a project route, False if unrouted
    """
    if not project_path:
        return False
    from kiro_crew.claude_routes.config import configured_claude_routes, resolve_claude_route

    claude_cfg = configured_claude_routes(cfg)
    resolved = resolve_claude_route(claude_cfg, project_path)
    if not resolved:
        return False

    set_project_model_pref(resolved.route_id, model_name)
    return True
