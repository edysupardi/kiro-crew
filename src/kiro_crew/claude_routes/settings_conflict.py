"""Conflict detection for Claude route environment settings."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_SETTINGS_FILE_BYTES = 1024 * 1024  # 1 MiB
ROUTING_ENV_VARS = frozenset(
    {
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
)


def _safe_read_json_file(path: Path) -> dict | None:
    """Read a JSON file using non-blocking, no-follow file descriptor semantics.

    @param path - Path to file to read
    @returns Parsed dict or None if unreadable/invalid
    """
    try:
        if path.is_symlink():
            return None
    except OSError:
        return None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return {}
    except OSError:
        return None

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_SETTINGS_FILE_BYTES:
            return None
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_SETTINGS_FILE_BYTES:
            chunk = os.read(fd, MAX_SETTINGS_FILE_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > MAX_SETTINGS_FILE_BYTES:
            return None
        raw = b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)

    try:
        decoded = raw.decode("utf-8").lstrip("﻿")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except (ValueError, RecursionError):
        return None


def read_settings_env(path: Path) -> dict[str, str] | None:
    """Extract string-to-string env mapping from a settings.json file.

    @param path - Path to settings file
    @returns Extracted env dictionary, empty dict if absent, or None if unreadable
    """
    data = _safe_read_json_file(path)
    if data is None:
        return None
    env = data.get("env")
    if env is None:
        return {}
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items() if isinstance(k, str) and isinstance(v, str)}


def check_claude_route_conflicts(
    route_id: str,
    route_env: dict[str, str],
    project_dir: str | None = None,
    home_dir: str | Path | None = None,
) -> list[str]:
    """Check for conflicting routing environment variables in global and project settings.

    @param route_id - The identifier of the active route
    @param route_env - The environment variables prepared for the route
    @param project_dir - The active project working directory
    @param home_dir - The home directory (defaults to Path.home())
    @returns List of conflict warning/error descriptions
    """
    conflicts: list[str] = []
    base_home = Path(home_dir) if home_dir else Path.home()
    global_settings = base_home / ".claude" / "settings.json"

    # 1. Check user global settings.json (Phase 0 showed global settings env outranks process env)
    global_env = read_settings_env(global_settings)
    if global_env:
        for var_name, global_val in global_env.items():
            if var_name in ROUTING_ENV_VARS and var_name in route_env:
                route_val = route_env[var_name]
                h_global = hashlib.sha256(global_val.encode("utf-8")).digest()
                h_route = hashlib.sha256(route_val.encode("utf-8")).digest()
                if not hmac.compare_digest(h_global, h_route):
                    conflicts.append(
                        f"~/.claude/settings.json sets '{var_name}' which conflicts with "
                        f"route '{route_id}' and shadows process environment. Remove or update it."
                    )

    # 2. Check project-local settings files
    if project_dir:
        p_dir = Path(project_dir)
        for rel in (".claude/settings.json", ".claude/settings.local.json"):
            p_file = p_dir / rel
            p_env = read_settings_env(p_file)
            if p_env:
                for var_name, p_val in p_env.items():
                    if var_name in ROUTING_ENV_VARS and var_name in route_env:
                        route_val = route_env[var_name]
                        h_p = hashlib.sha256(p_val.encode("utf-8")).digest()
                        h_route = hashlib.sha256(route_val.encode("utf-8")).digest()
                        if not hmac.compare_digest(h_p, h_route):
                            conflicts.append(
                                f"Project file '{rel}' sets '{var_name}' which differs from "
                                f"route '{route_id}' configuration."
                            )

    return conflicts
