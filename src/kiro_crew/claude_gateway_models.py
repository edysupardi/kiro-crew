"""The model ids the Claude gateway serves, kept in one data-home file.

claude-agent-acp fetches the gateway's ``/v1/models`` inside ``session/new`` under a
short timeout and drops the answer on any failure, so the list a session advertises
varies from one session to the next. This module owns that list instead:

* every Claude session start asks the gateway in the background, with the base URL
  and key that session's child process runs under, and a success REPLACES
  ``claude_gateway_models.json`` whole;
* a failed ask leaves the file untouched, so the last good list stays the answer;
* :func:`load_gateway_models` is what ``session_models_envelope`` folds in behind
  the adapter's own entries.

The file holds model ids only. No key, URL or header is written to it, to a log
line or to an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import aiohttp

from kiro_crew.atomic_write import atomic_write

logger = logging.getLogger(__name__)

GATEWAY_MODELS_FILE = "claude_gateway_models.json"
GATEWAY_FETCH_TIMEOUT_SECS = 5.0
GATEWAY_MAX_BODY_BYTES = 1024 * 1024
GATEWAY_MAX_IDS = 1000
GATEWAY_MAX_ID_CHARS = 256
OPUSPLAN_MODEL_ID = "opusplan"

_cached_ids: tuple[tuple[str, int, int], list[str]] | None = None
_refresh_in_flight = False
_background_tasks: set[asyncio.Task[None]] = set()


def gateway_models_path() -> Path:
    """The data-home path of the gateway model file, resolved without creating the home."""
    from kiro_crew.config.paths import peek_data_home

    return Path(peek_data_home()) / GATEWAY_MODELS_FILE


def _clean_ids(raw: object) -> list[str]:
    """Deduplicated, bounded, non-empty string ids from *raw*, in order."""
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    ids: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        model_id = item.strip()
        if model_id and len(model_id) <= GATEWAY_MAX_ID_CHARS and model_id not in seen:
            seen.add(model_id)
            ids.append(model_id)
            if len(ids) >= GATEWAY_MAX_IDS:
                break
    return ids


def load_gateway_models() -> list[str]:
    """The ids in the gateway model file, or ``[]`` when it is absent or unreadable.

    The parse is cached against the file's path, mtime and size, so the many readers
    of a session response cost one ``stat`` each.
    """
    global _cached_ids  # noqa: PLW0603
    path = gateway_models_path()
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _cached_ids is not None and _cached_ids[0] == key:
        return list(_cached_ids[1])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        logger.debug("%s is unreadable; using the adapter's list alone", GATEWAY_MODELS_FILE)
        return []
    ids = _clean_ids(data.get("models") if isinstance(data, dict) else None)
    _cached_ids = (key, ids)
    return list(ids)


def models_url(base_url: str) -> str:
    """The ``/v1/models`` URL for a gateway *base_url* that may already end in ``/v1``."""
    root = base_url.strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/v1/models"


def _ids_from_body(parsed: object) -> list[str]:
    """Model ids from a ``/v1/models`` body: ``{"data": [{"id": ...}]}`` or a bare list."""
    items = parsed.get("data") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        return []
    return _clean_ids([i.get("id") if isinstance(i, dict) else i for i in items])


async def _fetch_ids(url: str, api_key: str, timeout_secs: float) -> list[str]:
    """Ask the gateway for its model ids; raises on any transport, status or parse failure."""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers, allow_redirects=False) as resp:
            if resp.status != 200:
                raise _FetchFailure(f"HTTP {resp.status}")
            body = bytearray()
            async for chunk in resp.content.iter_chunked(65536):
                body.extend(chunk)
                if len(body) > GATEWAY_MAX_BODY_BYTES:
                    raise _FetchFailure("response over the size limit")
    return _ids_from_body(json.loads(bytes(body).decode("utf-8", errors="replace")))


class _FetchFailure(Exception):
    """A gateway answer that is not a usable model list; its text carries no secret."""


async def refresh_gateway_models(
    base_url: str, api_key: str, *, timeout_secs: float = GATEWAY_FETCH_TIMEOUT_SECS
) -> bool:
    """Replace the gateway model file with the gateway's current list.

    A failure of any kind, or an empty answer, leaves the file as it was and logs one
    warning naming the failure class, never the URL or the key.

    @param base_url - the gateway base URL the session's child runs under
    @param api_key - the key the same child holds
    @param timeout_secs - total budget for the request
    @returns True when the file was rewritten
    """
    global _refresh_in_flight  # noqa: PLW0603
    if not base_url.strip() or not api_key.strip():
        return False
    if _refresh_in_flight:
        return False
    _refresh_in_flight = True
    try:
        ids = await _fetch_ids(models_url(base_url), api_key, timeout_secs)
        if not ids:
            raise _FetchFailure("empty model list")
        payload = {"updated_at": int(time.time()), "models": ids}
        await asyncio.to_thread(
            atomic_write,
            gateway_models_path(),
            json.dumps(payload, ensure_ascii=False),
            mode=0o600,
        )
        return True
    except asyncio.CancelledError:
        raise
    except _FetchFailure as exc:
        logger.warning("Claude gateway model refresh kept the last list: %s", exc)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as exc:
        logger.warning("Claude gateway model refresh kept the last list: %s", type(exc).__name__)
    finally:
        _refresh_in_flight = False
    return False


def schedule_gateway_refresh(env: Mapping[str, str]) -> None:
    """Start a background refresh from a Claude child's environment, never awaited.

    Session start must not wait on the gateway, so the ask runs as a detached task
    (held in a set so it is not collected mid-flight). Skipped when the environment
    names no base URL and key, or when no loop is running.
    """
    base_url = env.get("ANTHROPIC_BASE_URL", "")
    api_key = env.get("ANTHROPIC_API_KEY", "")
    if not base_url.strip() or not api_key.strip():
        return
    try:
        task: asyncio.Task[Any] = asyncio.get_running_loop().create_task(
            refresh_gateway_models(base_url, api_key)
        )
    except RuntimeError:
        return
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def fold_gateway_models(models: dict[str, Any]) -> dict[str, Any]:
    """*models* with the gateway file's ids, and ``opusplan``, behind its own entries.

    Entries already in *models* keep their place and shape; ids only the file knows
    follow in file order, then ``opusplan`` when nothing offers it. ``opusplan`` is
    Claude Code's own planning alias, not a gateway id, so it is offered with the file
    absent or empty too. ``currentModelId`` is not touched. Returns *models* itself
    when nothing is added.
    """
    available = list(models.get("availableModels") or [])
    seen = {e.get("modelId") for e in available if isinstance(e, dict)}
    wanted = [*load_gateway_models(), OPUSPLAN_MODEL_ID]
    added: list[dict[str, str]] = []
    for model_id in wanted:
        if model_id not in seen:
            seen.add(model_id)
            added.append({"modelId": model_id, "name": model_id, "description": ""})
    if not added:
        return models
    return {**models, "availableModels": available + added}


def _reset_for_tests() -> None:
    """Drop the parse cache and the in-flight flag so a test starts clean."""
    global _cached_ids, _refresh_in_flight  # noqa: PLW0603
    _cached_ids = None
    _refresh_in_flight = False
