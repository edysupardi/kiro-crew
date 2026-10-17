"""CLI subcommands for managing per-project Claude routes: kirocrew claude-route."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

from kiro_crew import cli_help
from kiro_crew.claude_routes.config import (
    CLAUDE_ROUTES_SECTION_KEY,
    configured_claude_routes,
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
from kiro_crew.secrets.vault import SecretVault


def register_claude_route_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``kirocrew claude-route`` and its subcommands.

    @param sub - Subparsers action from top-level CLI parser
    """
    parser = cli_help.add_command(sub, "claude-route")
    route_sub = parser.add_subparsers(dest="route_cmd", required=True)

    # list
    list_cmd = route_sub.add_parser(
        "list", help="List configured Claude routes and directory bindings"
    )
    list_cmd.add_argument("--json", action="store_true", help="Output as JSON")

    # bind
    bind_cmd = route_sub.add_parser("bind", help="Bind a project directory to a Claude route")
    bind_cmd.add_argument("path", help="Absolute or relative path to project directory")
    bind_cmd.add_argument("route_id", help="Route identifier to bind")

    # unbind
    unbind_cmd = route_sub.add_parser(
        "unbind", help="Unbind a project directory from its Claude route"
    )
    unbind_cmd.add_argument("path", help="Absolute or relative path to project directory")

    # set-key
    key_cmd = route_sub.add_parser(
        "set-key", help="Securely set API key for a Claude route in vault"
    )
    key_cmd.add_argument("route_id", help="Route identifier")

    # import
    import_cmd = route_sub.add_parser(
        "import", help="Import route configuration from a project's .claude settings"
    )
    import_cmd.add_argument("path", help="Path to project directory containing .claude settings")
    import_cmd.add_argument("--route-id", help="Optional route identifier override")
    import_cmd.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")


def claude_route_cmd(args: argparse.Namespace) -> int:
    """Dispatcher for ``kirocrew claude-route`` subcommands.

    @param args - Parsed command-line arguments
    @returns Exit code (0 for success, non-zero for error)
    """
    cmd = getattr(args, "route_cmd", "")
    if cmd == "list":
        return _cmd_list(args)
    if cmd == "bind":
        return _cmd_bind(args)
    if cmd == "unbind":
        return _cmd_unbind(args)
    if cmd == "set-key":
        return _cmd_set_key(args)
    if cmd == "import":
        return _cmd_import(args)
    print(f"Unknown claude-route command: {cmd}", file=sys.stderr)
    return 1


def _cmd_list(args: argparse.Namespace) -> int:
    cfg = KiroCrewConfig.load()
    claude_cfg = configured_claude_routes(cfg)

    if getattr(args, "json", False):
        data = {
            "routes": {
                r_id: {
                    "base_url": r.base_url,
                    "has_key": bool(r.api_key),
                    "auth_env": r.auth_env,
                    "models": r.models,
                    "extra_models": r.extra_models,
                    "preferred_model": get_project_model_pref(r_id),
                }
                for r_id, r in claude_cfg.routes.items()
            },
            "project_routes": claude_cfg.project_routes,
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    if not claude_cfg.routes:
        print("No Claude routes configured.")
        return 0

    print("Configured Claude Routes:")
    for r_id, r in claude_cfg.routes.items():
        pref = get_project_model_pref(r_id) or "auto"
        print(f"  • {r_id}:")
        print(f"      base_url: {r.base_url or '(not set)'}")
        print(f"      api_key:  {'present' if r.api_key else 'missing'}")
        print(f"      preferred model: {pref}")
        if r.models:
            print(f"      model aliases: {r.models}")

    if claude_cfg.project_routes:
        print("\nProject Directory Bindings:")
        for p_dir, r_id in claude_cfg.project_routes.items():
            print(f"  • {p_dir} -> {r_id}")
    else:
        print("\nNo project directory bindings.")

    return 0


def _cmd_bind(args: argparse.Namespace) -> int:
    norm_path = os.path.normpath(os.path.abspath(args.path))
    route_id = args.route_id.strip()

    cfg = KiroCrewConfig.load()
    claude_cfg = configured_claude_routes(cfg)
    if route_id not in claude_cfg.routes:
        print(f"Error: route '{route_id}' does not exist in claude.routes", file=sys.stderr)
        return 1

    def _update(data: dict[str, Any]) -> None:
        claude_sect = data.setdefault(CLAUDE_ROUTES_SECTION_KEY, {})
        proj_routes = claude_sect.setdefault("project_routes", {})
        proj_routes[norm_path] = route_id

    update_config_locked(config_path(), mutate=_update)
    print(f"Bound {norm_path} -> {route_id}")
    return 0


def _cmd_unbind(args: argparse.Namespace) -> int:
    norm_path = os.path.normpath(os.path.abspath(args.path))

    def _update(data: dict[str, Any]) -> None:
        claude_sect = data.get(CLAUDE_ROUTES_SECTION_KEY) or {}
        proj_routes = claude_sect.get("project_routes") or {}
        proj_routes.pop(norm_path, None)

    update_config_locked(config_path(), mutate=_update)
    print(f"Unbound {norm_path}")
    return 0


def _cmd_set_key(args: argparse.Namespace) -> int:
    route_id = args.route_id.strip()
    key_val = getpass.getpass(f"Enter API key for Claude route '{route_id}': ").strip()
    if not key_val:
        print("Error: empty key provided", file=sys.stderr)
        return 1

    secret_name = f"claude-route-{route_id}"
    vault = SecretVault(get_config_dir())
    vault.set_sync(secret_name, key_val)

    # Ensure route config uses the secret:// reference
    secret_ref = f"secret://{secret_name}"

    def _update(data: dict[str, Any]) -> None:
        claude_sect = data.setdefault(CLAUDE_ROUTES_SECTION_KEY, {})
        routes = claude_sect.setdefault("routes", {})
        route_entry = routes.setdefault(route_id, {})
        route_entry["api_key"] = secret_ref

    update_config_locked(config_path(), mutate=_update)
    print(f"Stored API key in vault as {secret_ref}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    norm_path = os.path.normpath(os.path.abspath(args.path))
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
    route_id = args.route_id or proj_dir.name or "imported"
    route_id = "".join(c if c.isalnum() or c in "._-" else "-" for c in route_id.lower()).strip("-")

    print(f"Discovered settings for project '{norm_path}':")
    print(f"  Route ID:        {route_id}")
    print(f"  Base URL:        {base_url or '(not found)'}")
    print(f"  API Key:         {'found in settings' if api_key else 'not found'}")
    print(f"  Auth env:        {auth_env}")
    print(f"  Model aliases:   {models_map}")
    if subagent_model:
        print(f"  Subagent model:  {subagent_model}")
    if extracted_model:
        print(f"  Preferred model: {extracted_model}")

    if not args.yes:
        ans = (
            input(f"\nImport as Claude route '{route_id}' and bind {norm_path}? [y/N]: ")
            .strip()
            .lower()
        )
        if ans not in ("y", "yes"):
            print("Import cancelled.")
            return 0

    secret_ref = ""
    if api_key:
        secret_name = f"claude-route-{route_id}"
        vault = SecretVault(get_config_dir())
        vault.set_sync(secret_name, api_key)
        secret_ref = f"secret://{secret_name}"

    def _commit(data: dict[str, Any]) -> None:
        claude_sect = data.setdefault(CLAUDE_ROUTES_SECTION_KEY, {})
        routes = claude_sect.setdefault("routes", {})
        routes[route_id] = {
            "base_url": base_url,
            "auth_env": auth_env,
            "models": models_map,
            "subagent_model": subagent_model,
            "extra_models": ["opusplan"],
            "discover_models": True,
            **({"api_key": secret_ref} if secret_ref else {}),
        }
        proj_routes = claude_sect.setdefault("project_routes", {})
        proj_routes[norm_path] = route_id

    update_config_locked(config_path(), mutate=_commit)
    if extracted_model:
        set_project_model_pref(route_id, extracted_model)

    print(f"Successfully imported route '{route_id}' bound to {norm_path}")
    return 0
