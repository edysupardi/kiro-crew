"""Route-scoped advertised model cache for Claude gateway routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.loader import config_dir as get_config_dir

logger = logging.getLogger(__name__)

_ROUTE_ADVERTISED: dict[str, list[str]] = {}
_INITIALIZED: bool = False


def _route_cache_path() -> Path:
    """Return the filesystem path for persisted route-scoped model discoveries."""
    return Path(get_config_dir()) / "claude_route_models.json"


def _ensure_loaded() -> None:
    """Load persisted route discoveries into memory if not already initialized."""
    global _INITIALIZED  # noqa: PLW0603
    if _INITIALIZED:
        return
    _INITIALIZED = True
    path = _route_cache_path()
    try:
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for route_id, models in data.items():
                if isinstance(route_id, str) and isinstance(models, list):
                    _ROUTE_ADVERTISED[route_id] = [str(m) for m in models if isinstance(m, str)]
    except Exception:
        logger.debug("Could not load claude_route_models.json", exc_info=True)


def get_route_models(route_id: str) -> list[str]:
    """Retrieve the cached list of model IDs for a specific route.

    @param route_id - The identifier of the route
    @returns List of model IDs known for this route
    """
    _ensure_loaded()
    return list(_ROUTE_ADVERTISED.get(route_id, []))


def set_route_models(route_id: str, models: list[str]) -> None:
    """Update and persist the model cache for a specific route.

    @param route_id - The identifier of the route
    @param models - List of discovered or advertised model IDs
    """
    _ensure_loaded()
    deduped: list[str] = []
    seen: set[str] = set()
    for m in models:
        clean = str(m).strip()
        if clean and clean not in seen and len(clean) <= 256:
            seen.add(clean)
            deduped.append(clean)

    _ROUTE_ADVERTISED[route_id] = deduped
    try:
        path = _route_cache_path()
        atomic_write(path, json.dumps(_ROUTE_ADVERTISED, indent=2, ensure_ascii=False))
    except Exception:
        logger.debug("Could not persist claude_route_models.json", exc_info=True)
