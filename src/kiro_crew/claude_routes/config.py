"""Configuration DTOs, validation, and resolution for per-project Claude routes."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

from kiro_crew.aws_project_profiles import resolve_project_binding

logger = logging.getLogger(__name__)

CLAUDE_ROUTES_SECTION_KEY = "claude"
ROUTE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ALLOWED_AUTH_ENVS = frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"})
PROJECT_DIR_MAX_LEN = 4096
BASE_URL_MAX_LEN = 2048
MODEL_ID_MAX_LEN = 256


@dataclass(frozen=True)
class ClaudeRouteConfig:
    """Configuration for a single Claude gateway route."""

    base_url: str = ""
    api_key: str = ""
    auth_env: str = "ANTHROPIC_API_KEY"
    models: dict[str, str] = field(default_factory=dict)
    subagent_model: str = ""
    extra_models: list[str] = field(default_factory=list)
    models_url: str = ""
    discover_models: bool = True

    @property
    def effective_models_url(self) -> str:
        """Endpoint to query for model discovery."""
        if self.models_url:
            return self.models_url
        if not self.base_url:
            return ""
        trimmed = self.base_url.rstrip("/")
        if trimmed.endswith("/v1"):
            return f"{trimmed}/models"
        return f"{trimmed}/v1/models"


@dataclass
class ClaudeConfig:
    """Container for Claude routes and project-to-route directory mappings."""

    routes: dict[str, ClaudeRouteConfig] = field(default_factory=dict)
    project_routes: dict[str, str] = field(default_factory=dict)


class ResolvedClaudeRoute(NamedTuple):
    """The resolved route binding for a session working directory."""

    route_id: str
    bound_dir: str
    route: ClaudeRouteConfig


def _validate_url(url: str, *, label: str) -> str | None:
    """Validate a gateway URL for scheme, length, host, and absence of userinfo.

    @param url - The URL string to validate
    @param label - The field name label for error reporting
    @returns Error message if invalid, None if valid
    """
    if not url:
        return f"{label} must not be empty"
    if len(url) > BASE_URL_MAX_LEN:
        return f"{label} exceeds maximum length of {BASE_URL_MAX_LEN} characters"
    try:
        parts = urlsplit(url)
    except Exception:
        return f"{label} is not a valid URL"
    if parts.scheme not in ("http", "https"):
        return f"{label} must use http or https scheme"
    if not parts.netloc or not parts.hostname:
        return f"{label} must include a host"
    if parts.username or parts.password:
        return f"{label} must not contain userinfo credentials"
    if parts.query or parts.fragment:
        return f"{label} must not contain query parameters or fragments"
    return None


def _validate_route(route_id: str, raw: object) -> list[str]:
    """Validate a single ClaudeRouteConfig raw mapping.

    @param route_id - The route identifier string
    @param raw - The raw dictionary value representing route settings
    @returns List of refusal reasons, empty if valid
    """
    if not ROUTE_ID_RE.match(route_id):
        return [f"route '{route_id}': route ID must match pattern [a-z0-9][a-z0-9._-]{{0,63}}"]
    if not isinstance(raw, dict):
        return [f"route '{route_id}': value must be a dictionary"]

    refusals: list[str] = []
    base_url = raw.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str):
            refusals.append(f"route '{route_id}': base_url must be a string")
        else:
            err = _validate_url(base_url, label="base_url")
            if err:
                refusals.append(f"route '{route_id}': {err}")

    api_key = raw.get("api_key")
    if api_key is not None:
        if not isinstance(api_key, str):
            refusals.append(f"route '{route_id}': api_key must be a secret:// URI string")
        elif not api_key.startswith("secret://") or len(api_key) <= 9:
            refusals.append(
                f"route '{route_id}': api_key must be a secret:// URI reference, not a plaintext key"
            )

    auth_env = raw.get("auth_env", "ANTHROPIC_API_KEY")
    if auth_env not in ALLOWED_AUTH_ENVS:
        refusals.append(f"route '{route_id}': auth_env must be one of {sorted(ALLOWED_AUTH_ENVS)}")

    models = raw.get("models")
    if models is not None and not isinstance(models, dict):
        refusals.append(f"route '{route_id}': models must be an object of alias -> model ID")
    elif isinstance(models, dict):
        for alias, m_id in models.items():
            if not isinstance(alias, str) or not isinstance(m_id, str):
                refusals.append(f"route '{route_id}': models keys and values must be strings")
                break
            if len(m_id) > MODEL_ID_MAX_LEN:
                refusals.append(f"route '{route_id}': model ID '{alias}' exceeds maximum length")

    extra_models = raw.get("extra_models")
    if extra_models is not None and not isinstance(extra_models, list):
        refusals.append(f"route '{route_id}': extra_models must be a list of model ID strings")
    elif isinstance(extra_models, list):
        for m_id in extra_models:
            if not isinstance(m_id, str):
                refusals.append(f"route '{route_id}': extra_models entries must be strings")
                break
            if len(m_id) > MODEL_ID_MAX_LEN:
                refusals.append(f"route '{route_id}': extra_models entry exceeds maximum length")

    models_url = raw.get("models_url")
    if models_url:
        if not isinstance(models_url, str):
            refusals.append(f"route '{route_id}': models_url must be a string")
        else:
            err = _validate_url(models_url, label="models_url")
            if err:
                refusals.append(f"route '{route_id}': {err}")

    return refusals


def claude_route_refusals(raw: object) -> list[str]:
    """Inspect a raw claude section and list any configuration validation errors.

    @param raw - The raw value under config key 'claude'
    @returns List of refusal strings; empty if completely valid
    """
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{CLAUDE_ROUTES_SECTION_KEY} must be a dictionary"]

    refusals: list[str] = []
    routes = raw.get("routes")
    defined_routes: set[str] = set()
    if routes is not None:
        if not isinstance(routes, dict):
            refusals.append(
                f"{CLAUDE_ROUTES_SECTION_KEY}.routes must be an object of route_id -> config"
            )
        else:
            for route_id, route_raw in routes.items():
                if isinstance(route_id, str):
                    defined_routes.add(route_id)
                refusals.extend(_validate_route(str(route_id), route_raw))

    project_routes = raw.get("project_routes")
    if project_routes is not None:
        if not isinstance(project_routes, dict):
            refusals.append(
                f"{CLAUDE_ROUTES_SECTION_KEY}.project_routes must be an object of directory -> route_id"
            )
        else:
            for dir_key, target_route in project_routes.items():
                if (
                    not isinstance(dir_key, str)
                    or not dir_key
                    or len(dir_key) > PROJECT_DIR_MAX_LEN
                    or "\x00" in dir_key
                    or not os.path.isabs(dir_key)
                ):
                    refusals.append(
                        f"project_routes key '{dir_key}': must be an absolute directory path"
                    )
                if not isinstance(target_route, str) or not ROUTE_ID_RE.match(target_route):
                    refusals.append(
                        f"project_routes entry for '{dir_key}': route target '{target_route}' is invalid"
                    )
                elif (
                    routes is not None
                    and isinstance(routes, dict)
                    and target_route not in defined_routes
                ):
                    refusals.append(
                        f"project_routes entry for '{dir_key}': targets undefined route '{target_route}'"
                    )

    return refusals


def coerce_claude_routes(raw: object) -> ClaudeConfig:
    """Normalize and validate raw dictionary into a typed ClaudeConfig.

    Invalid entries are dropped with logging warnings so that an unparseable entry
    fails closed for that route without crashing the entire configuration load.

    @param raw - The raw configuration section
    @returns Normalized ClaudeConfig object
    """
    if raw is None or not isinstance(raw, dict):
        return ClaudeConfig()

    routes_out: dict[str, ClaudeRouteConfig] = {}
    routes_raw = raw.get("routes")
    if isinstance(routes_raw, dict):
        for route_id, r_data in routes_raw.items():
            if not isinstance(route_id, str) or not isinstance(r_data, dict):
                continue
            errors = _validate_route(route_id, r_data)
            if errors:
                logger.warning(
                    "claude.routes: dropping invalid route '%s': %s", route_id, "; ".join(errors)
                )
                continue
            routes_out[route_id] = ClaudeRouteConfig(
                base_url=str(r_data.get("base_url") or ""),
                api_key=str(r_data.get("api_key") or ""),
                auth_env=str(r_data.get("auth_env") or "ANTHROPIC_API_KEY"),
                models=dict(r_data.get("models") or {}),
                subagent_model=str(r_data.get("subagent_model") or ""),
                extra_models=list(r_data.get("extra_models") or []),
                models_url=str(r_data.get("models_url") or ""),
                discover_models=bool(r_data.get("discover_models", True)),
            )

    proj_routes_out: dict[str, str] = {}
    proj_raw = raw.get("project_routes")
    if isinstance(proj_raw, dict):
        for dir_key, target_route in proj_raw.items():
            if (
                not isinstance(dir_key, str)
                or not dir_key
                or len(dir_key) > PROJECT_DIR_MAX_LEN
                or "\x00" in dir_key
                or not os.path.isabs(dir_key)
            ):
                logger.warning("claude.project_routes: ignoring invalid directory '%s'", dir_key)
                continue
            if not isinstance(target_route, str) or not ROUTE_ID_RE.match(target_route):
                logger.warning(
                    "claude.project_routes: ignoring invalid route target '%s'", target_route
                )
                continue
            normalized_dir = os.path.normpath(dir_key)
            proj_routes_out[normalized_dir] = target_route

    return ClaudeConfig(routes=routes_out, project_routes=proj_routes_out)


def resolve_claude_route(cfg: object, cwd: str | None) -> ResolvedClaudeRoute | None:
    """Resolve the active Claude route for a given working directory.

    @param cfg - The gateway configuration object (or ClaudeConfig)
    @param cwd - The working directory of the session
    @returns ResolvedClaudeRoute with binding info, or None if unbound
    """
    if not cwd:
        return None
    claude_cfg = getattr(cfg, "claude", cfg)
    if not isinstance(claude_cfg, ClaudeConfig):
        return None
    if not claude_cfg.project_routes or not claude_cfg.routes:
        return None

    binding = resolve_project_binding(claude_cfg.project_routes, cwd)
    if not binding:
        return None
    bound_dir, route_id = binding
    route = claude_cfg.routes.get(route_id)
    if not route:
        return None
    return ResolvedClaudeRoute(route_id=route_id, bound_dir=bound_dir, route=route)


def configured_claude_routes(cfg: object) -> ClaudeConfig:
    """Extract the ClaudeConfig from a general configuration object.

    @param cfg - The host configuration object
    @returns Active ClaudeConfig or an empty default
    """
    if isinstance(cfg, ClaudeConfig):
        return cfg
    claude = getattr(cfg, "claude", None)
    return claude if isinstance(claude, ClaudeConfig) else ClaudeConfig()


def prepare_claude_route_env(
    route_id: str | None,
    cwd: str | None,
    config_dir: Path | None = None,
    cfg: object = None,
) -> tuple[dict[str, str], set[str]]:
    """Resolve and build process environment overrides for a Claude route.

    @param route_id - Optional explicit route identifier
    @param cwd - Current working directory of the session
    @param config_dir - Optional path to the configuration directory
    @param cfg - Optional loaded configuration object
    @returns Tuple of (resolved_env_dict, keys_to_pop_from_ambient)
    @throws ValueError if secret URI resolution fails
    """
    from kiro_crew.claude_routes.settings_conflict import check_claude_route_conflicts
    from kiro_crew.config.loader import KiroCrewConfig
    from kiro_crew.config.loader import config_dir as get_config_dir
    from kiro_crew.mcp_gateway.secret_uri import resolve_secret_uris

    host_cfg = cfg if cfg is not None else KiroCrewConfig.load()
    claude_cfg = configured_claude_routes(host_cfg)
    target_route: ClaudeRouteConfig | None = None
    target_id: str = ""

    if route_id and route_id in claude_cfg.routes:
        target_id = route_id
        target_route = claude_cfg.routes[route_id]
    elif cwd:
        resolved = resolve_claude_route(claude_cfg, cwd)
        if resolved:
            target_id = resolved.route_id
            target_route = resolved.route

    if not target_route:
        return {}, set()

    keys_to_pop = {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    }

    target_config_dir = config_dir or get_config_dir()
    resolved_mapping, _ = resolve_secret_uris(
        {"KEY": target_route.api_key},
        target_config_dir,
        subject=f"claude.routes.{target_id}",
    )
    secret_value = resolved_mapping.get("KEY", "")

    new_env = {
        "ANTHROPIC_BASE_URL": target_route.base_url,
        target_route.auth_env: secret_value,
    }
    if target_route.models.get("opus"):
        new_env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = target_route.models["opus"]
    if target_route.models.get("sonnet"):
        new_env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = target_route.models["sonnet"]
    if target_route.models.get("haiku"):
        new_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = target_route.models["haiku"]
    if target_route.subagent_model:
        new_env["CLAUDE_CODE_SUBAGENT_MODEL"] = target_route.subagent_model

    conflicts = check_claude_route_conflicts(target_id, new_env, project_dir=cwd)
    for warning in conflicts:
        logger.warning("Claude route '%s': %s", target_id, warning)

    return new_env, keys_to_pop


def refuse_claude_routes_on_write(data: dict, path: object) -> None:
    """Refuse a ``claude`` configuration entry that fails validation on write.

    Scoped to what THIS write does: a bad entry already on disk in a mapping
    this write leaves unchanged publishes with a warning.

    @param data - The configuration dictionary being written
    @param path - The filesystem path of the configuration file
    @throws ConfigWriteRefused if the write introduces invalid claude routing config
    """
    raw_claude = data.get(CLAUDE_ROUTES_SECTION_KEY) if isinstance(data, dict) else None
    refusals = claude_route_refusals(raw_claude)
    if not refusals:
        return
    detail = "; ".join(refusals)
    if _claude_routes_on_disk(path) == raw_claude:
        logger.warning(
            "config.json already holds %s entries the load ignores (%s); this write "
            "left that mapping as it found it.",
            CLAUDE_ROUTES_SECTION_KEY,
            detail,
        )
        return
    from kiro_crew.config.loader import ConfigWriteRefused

    raise ConfigWriteRefused(
        f"{CLAUDE_ROUTES_SECTION_KEY} refuses {detail}. Nothing was written: fix or remove the entry."
    )


def _claude_routes_on_disk(path: object) -> object:
    """Extract the ``claude`` section currently on disk at path.

    @param path - The filesystem path of the configuration file
    @returns The unparsed/raw claude object on disk, or None
    """
    try:
        from pathlib import Path

        from kiro_crew.config.loader import read_config_text

        target = path if isinstance(path, Path) else Path(str(path))
        document = json.loads(read_config_text(target))
    except (OSError, ValueError):
        return None
    return document.get(CLAUDE_ROUTES_SECTION_KEY) if isinstance(document, dict) else None
