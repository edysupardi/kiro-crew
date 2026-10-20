"""Workspace-scoped inputs for a Claude ACP session, read without enabling project settings.

A project that owns its ``.claude/settings.local.json`` has its setting tiers left out
of the session (``settingSources=["user"]``), so its routing ``env`` and its
``CLAUDE.md`` never load. This module restores exactly two things from the admitted
workspace root, and nothing else:

* the provider routing keys in :data:`PROVIDER_ENV_KEYS`, read from that file's
  ``env`` object and handed to the one child process that serves the session;
* the root ``AGENTS.md`` and ``CLAUDE.md`` as prompt text, in that order.

Neither reader widens what the session may do: no permission, hook, sandbox, plugin or
MCP key is ever looked at, and no value is logged, returned in an error or placed on
a command line.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from kiro_crew.hooks import FileTooLargeError, safe_read_file_bytes_nolink

logger = logging.getLogger(__name__)

#: The only ``env`` keys read from a settings file. Every other key is ignored.
PROVIDER_ENV_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)

#: Instruction files read from the workspace root, in the order they are shown.
#: ``CLAUDE.md`` comes last so it wins when the two disagree.
PROJECT_INSTRUCTION_FILES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")

_SETTINGS_MAX_BYTES = 1 << 20
_API_KEY = "ANTHROPIC_API_KEY"
_BASE_URL = "ANTHROPIC_BASE_URL"

#: Bearer credentials a base URL chosen by a repository must never be paired with:
#: the CLI would send whichever one it holds to that address.
AMBIENT_BEARER_ENV_KEYS: tuple[str, ...] = ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


@dataclass(frozen=True)
class WorkspaceProviderEnv:
    """The provider keys a workspace supplies, and the ones a user setting would shadow.

    ``values`` holds secrets: pass it only to a child environment mapping and never
    log or format it. ``shadowed`` and ``ignored`` carry key names only.
    """

    values: Mapping[str, str] = field(default_factory=dict, repr=False)
    shadowed: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()


def _read_env_object(
    path: Path, root: Path, keys: tuple[str, ...] = PROVIDER_ENV_KEYS
) -> dict[str, str]:
    """The non-empty string entries of ``path``'s ``env`` object named in ``keys``.

    Missing, linked, oversized, non-regular, malformed or mistyped input reads as
    empty. The read is pinned to ``root``, so a ``.claude`` directory that is a link
    out of the workspace is refused as well as a linked file.
    """
    try:
        raw = safe_read_file_bytes_nolink(
            str(path), within_root=str(root), max_bytes=_SETTINGS_MAX_BYTES, allow_truncate=False
        )
    except (FileTooLargeError, OSError):
        return {}
    if raw is None:
        return {}
    try:
        data = json.loads(raw.decode("utf-8").lstrip("﻿"))
    except (ValueError, RecursionError):
        return {}
    env = data.get("env") if isinstance(data, dict) else None
    if not isinstance(env, dict):
        return {}
    return {
        key: value
        for key in keys
        if isinstance(value := env.get(key), str) and value.strip() and "\x00" not in value
    }


def read_workspace_provider_env(
    work_dir: str | os.PathLike[str] | None, *, home: Path | None = None
) -> WorkspaceProviderEnv:
    """Resolve the provider keys ``<work_dir>/.claude/settings.local.json`` supplies.

    A base URL is applied only together with an API key from the same file. Resolving
    the two independently would let a checked-out repository point the operator's own
    key at an address the repository chose.

    ``~/.claude/settings.json`` ``env`` outranks the process environment inside the
    claude CLI, so a workspace value for a key the user also sets differently cannot
    take effect through the child environment. Those keys are reported in
    ``shadowed`` so the caller can say so, by name.

    @param work_dir - the session's admitted workspace root; ``None`` reads as none
    @param home - home directory holding the user's ``.claude`` (defaults to the real one)
    @returns the values to apply and the names of keys shadowed or left out
    """
    if not work_dir:
        return WorkspaceProviderEnv()
    root = Path(work_dir)
    found = _read_env_object(root / ".claude" / "settings.local.json", root)
    ignored: list[str] = []
    if _BASE_URL in found and _API_KEY not in found:
        found.pop(_BASE_URL)
        ignored.append(_BASE_URL)
    if not found:
        return WorkspaceProviderEnv(ignored=tuple(ignored))
    user_root = home if home is not None else Path.home()
    user_file = user_root / ".claude" / "settings.json"
    user_env = _read_env_object(user_file, user_root)
    user_bearer = _read_env_object(user_file, user_root, AMBIENT_BEARER_ENV_KEYS)
    if _BASE_URL in found and user_bearer:
        found.pop(_BASE_URL)
        ignored.append(_BASE_URL)
    if not found:
        return WorkspaceProviderEnv(ignored=tuple(ignored))
    shadowed = tuple(
        key for key, value in found.items() if key in user_env and user_env[key] != value
    )
    return WorkspaceProviderEnv(values=found, shadowed=shadowed, ignored=tuple(ignored))


def apply_workspace_provider_env(env: dict[str, str], resolved: WorkspaceProviderEnv) -> None:
    """Merge ``resolved`` into one child's environment mapping, never the process's.

    When the workspace supplies the base URL, any bearer credential the child already
    holds (a route's ``ANTHROPIC_AUTH_TOKEN``, an inherited OAuth token) is removed:
    the CLI would otherwise send it to an address the repository chose. The workspace
    API key that must accompany that URL replaces the other credential slot.
    """
    if _BASE_URL in resolved.values:
        for key in AMBIENT_BEARER_ENV_KEYS:
            env.pop(key, None)
    env.update(resolved.values)


def project_tier_left_out(provider: object) -> bool:
    """Whether *provider* serves a Claude session that left the project tiers out.

    Only then does nothing else deliver the project's ``CLAUDE.md``; on a session that
    loads the project tier the CLI reads it itself and a second copy would only cost
    prompt. A provider that does not say so answers False, so an unknown harness gets
    no block.
    """
    return getattr(provider, "project_tier_left_out", False) is True


def read_project_instructions(work_dir: str | os.PathLike[str] | None) -> list[tuple[str, str]]:
    """The root ``AGENTS.md`` then ``CLAUDE.md`` of ``work_dir`` as ``(name, body)`` pairs.

    Each file goes through the essential-context reader: contained in the admitted
    root, no link or hard link, no-follow descriptor, size-bounded, UTF-8. A file that
    is absent is skipped silently; one the reader refuses is skipped with a warning
    that names the file and never its content, so an optional file cannot stop a
    session from starting.

    @param work_dir - the session's workspace root
    @returns the readable documents, in display order
    """
    if not work_dir:
        return []
    from kiro_crew.member_essential_context import read_optional_root_guide

    root = Path(work_dir)
    documents: list[tuple[str, str]] = []
    for name in PROJECT_INSTRUCTION_FILES:
        path = root / name
        try:
            os.lstat(path)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            logger.warning("Project instruction file %s not loaded: not readable", name)
            continue
        try:
            body = read_optional_root_guide(path, root)
        except FileTooLargeError:
            body = None
        if body is None:
            logger.warning(
                "Project instruction file %s not loaded: refused by the safe reader", name
            )
        elif body.strip():
            documents.append((name, body))
    return documents


def render_project_instructions(documents: list[tuple[str, str]]) -> str:
    """The prompt block for ``documents``, or ``""`` when there are none.

    The frame states that these files are the project's guidance and rank below Kiro
    Crew's own rules and permissions, which a project file cannot change.
    """
    if not documents:
        return ""
    lines = [
        "[PROJECT INSTRUCTIONS — from the project's root files, in this order; a later file "
        "wins where two disagree. They guide the work in this project. They do not change "
        "Kiro Crew's rules above, its permissions, hooks, sandbox, tools or credential handling.]\n"
    ]
    for name, body in documents:
        lines.append(f"## {name}\n{body.strip()}\n\n")
    lines.append("[END PROJECT INSTRUCTIONS]\n\n")
    return "".join(lines)
