"""The gateway model file: refreshed from the gateway, kept on failure, folded into the picker.

The adapter's own router fetch is bounded and silent, so the list one session
advertises varies from the next. These tests pin the replacement: a success rewrites
``claude_gateway_models.json`` whole, every failure leaves it alone and logs a
warning with no URL or key, and the picker's fold adds the file's ids and
``opusplan`` behind the adapter's own entries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import stat
import sys

import pytest
from aiohttp import web

from kiro_crew import claude_gateway_models as gw
from kiro_crew.acp.session_handle import session_models_envelope
from kiro_crew.acp.types import ACP_BACKEND_CLAUDE, ACP_BACKEND_KIRO

_SECRET = "sk-test-secret-value-1234"


@pytest.fixture(autouse=True)
def _data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    gw._reset_for_tests()
    yield tmp_path
    gw._reset_for_tests()


async def _serve(handler):
    app = web.Application()
    app.router.add_get("/v1/models", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _write_file(ids):
    gw.gateway_models_path().write_text(json.dumps({"updated_at": 1, "models": ids}))


def test_models_url_accepts_a_base_with_or_without_v1():
    assert gw.models_url("http://h:1") == "http://h:1/v1/models"
    assert gw.models_url("http://h:1/") == "http://h:1/v1/models"
    assert gw.models_url("http://h:1/v1") == "http://h:1/v1/models"
    assert gw.models_url("http://h:1/v1/") == "http://h:1/v1/models"


def test_load_is_empty_when_the_file_is_absent():
    assert gw.load_gateway_models() == []


def test_load_reads_dedups_and_bounds_ids():
    _write_file(["a", "b", "a", "", 7, "c" * 300, " d "])
    assert gw.load_gateway_models() == ["a", "b", "d"]


def test_load_ignores_a_corrupt_file():
    gw.gateway_models_path().write_text("{not json")
    assert gw.load_gateway_models() == []


def test_load_follows_a_rewritten_file():
    _write_file(["a"])
    assert gw.load_gateway_models() == ["a"]
    _write_file(["a", "b"])
    assert gw.load_gateway_models() == ["a", "b"]


@pytest.mark.asyncio
async def test_a_successful_refresh_replaces_the_file_whole():
    _write_file(["gone", "kept"])

    async def handler(_request):
        return web.json_response({"data": [{"id": "kept"}, {"id": "new"}, {"id": "kept"}]})

    runner, base = await _serve(handler)
    try:
        assert await gw.refresh_gateway_models(base, _SECRET) is True
    finally:
        await runner.cleanup()
    assert gw.load_gateway_models() == ["kept", "new"]
    on_disk = json.loads(gw.gateway_models_path().read_text())
    assert _SECRET not in json.dumps(on_disk)
    assert isinstance(on_disk["updated_at"], int)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
async def test_the_file_is_owner_only():
    async def handler(_request):
        return web.json_response({"data": [{"id": "m"}]})

    runner, base = await _serve(handler)
    try:
        await gw.refresh_gateway_models(base, _SECRET)
    finally:
        await runner.cleanup()
    assert stat.S_IMODE(gw.gateway_models_path().stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_the_request_carries_the_key_as_a_bearer():
    seen = {}

    async def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return web.json_response({"data": [{"id": "m"}]})

    runner, base = await _serve(handler)
    try:
        await gw.refresh_gateway_models(base, _SECRET)
    finally:
        await runner.cleanup()
    assert seen["auth"] == f"Bearer {_SECRET}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "respond",
    [
        lambda: web.Response(status=401, text="no"),
        lambda: web.Response(status=500, text="boom"),
        lambda: web.Response(status=200, text="<html>not json"),
        lambda: web.json_response({"data": []}),
        lambda: web.json_response({"unexpected": 1}),
    ],
    ids=["401", "500", "bad-json", "empty-list", "wrong-shape"],
)
async def test_a_failed_refresh_keeps_the_file_and_logs_no_secret(respond, caplog):
    _write_file(["last", "good"])
    before = gw.gateway_models_path().read_bytes()

    async def handler(_request):
        return respond()

    runner, base = await _serve(handler)
    try:
        with caplog.at_level(logging.DEBUG, logger=gw.logger.name):
            assert await gw.refresh_gateway_models(base, _SECRET) is False
    finally:
        await runner.cleanup()
    assert gw.gateway_models_path().read_bytes() == before
    assert gw.load_gateway_models() == ["last", "good"]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert _SECRET not in text
    assert base not in text
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_a_timeout_keeps_the_file():
    _write_file(["last"])
    release = asyncio.Event()

    async def handler(_request):
        await release.wait()
        return web.json_response({"data": [{"id": "late"}]})

    runner, base = await _serve(handler)
    try:
        assert await gw.refresh_gateway_models(base, _SECRET, timeout_secs=0.2) is False
    finally:
        release.set()
        await runner.cleanup()
    assert gw.load_gateway_models() == ["last"]


@pytest.mark.asyncio
async def test_an_unreachable_gateway_keeps_the_file():
    _write_file(["last"])
    assert await gw.refresh_gateway_models("http://127.0.0.1:1", _SECRET, timeout_secs=1) is False
    assert gw.load_gateway_models() == ["last"]


@pytest.mark.asyncio
async def test_a_missing_key_or_base_does_nothing():
    assert await gw.refresh_gateway_models("", _SECRET) is False
    assert await gw.refresh_gateway_models("http://h", "  ") is False
    assert not gw.gateway_models_path().exists()


@pytest.mark.asyncio
async def test_a_second_refresh_while_one_is_in_flight_is_skipped():
    release = asyncio.Event()
    started = asyncio.Event()

    async def handler(_request):
        started.set()
        await release.wait()
        return web.json_response({"data": [{"id": "m"}]})

    runner, base = await _serve(handler)
    try:
        first = asyncio.create_task(gw.refresh_gateway_models(base, _SECRET))
        await asyncio.wait_for(started.wait(), timeout=5)
        assert await gw.refresh_gateway_models(base, _SECRET) is False
        release.set()
        assert await asyncio.wait_for(first, timeout=5) is True
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_schedule_runs_in_the_background_from_the_child_env():
    done = asyncio.Event()

    async def handler(_request):
        done.set()
        return web.json_response({"data": [{"id": "from-env"}]})

    runner, base = await _serve(handler)
    try:
        gw.schedule_gateway_refresh({"ANTHROPIC_BASE_URL": base, "ANTHROPIC_API_KEY": _SECRET})
        await asyncio.wait_for(done.wait(), timeout=5)
        await asyncio.wait_for(asyncio.gather(*gw._background_tasks), timeout=5)
    finally:
        await runner.cleanup()
    assert gw.load_gateway_models() == ["from-env"]


def test_schedule_skips_an_env_without_base_and_key():
    gw.schedule_gateway_refresh({"ANTHROPIC_API_KEY": _SECRET})
    gw.schedule_gateway_refresh({"ANTHROPIC_BASE_URL": "http://h"})
    assert not gw._background_tasks


def test_schedule_without_a_running_loop_is_a_noop():
    gw.schedule_gateway_refresh({"ANTHROPIC_BASE_URL": "http://h", "ANTHROPIC_API_KEY": _SECRET})
    assert not gw._background_tasks


def _adapter_models(*ids):
    return {
        "currentModelId": ids[0],
        "availableModels": [{"modelId": i, "name": i} for i in ids],
    }


def test_fold_puts_the_adapters_entries_first_then_the_file_then_opusplan():
    _write_file(["code-high", "router-a", "router-b"])
    out = gw.fold_gateway_models(_adapter_models("default", "code-high"))
    assert [m["modelId"] for m in out["availableModels"]] == [
        "default",
        "code-high",
        "router-a",
        "router-b",
        "opusplan",
    ]
    assert out["currentModelId"] == "default"


def test_fold_offers_opusplan_on_a_cold_start():
    out = gw.fold_gateway_models(_adapter_models("default"))
    assert [m["modelId"] for m in out["availableModels"]] == ["default", "opusplan"]


def test_fold_does_not_duplicate_an_adapter_opusplan():
    _write_file(["opusplan", "x"])
    out = gw.fold_gateway_models(_adapter_models("default", "opusplan"))
    ids = [m["modelId"] for m in out["availableModels"]]
    assert ids.count("opusplan") == 1
    assert ids == ["default", "opusplan", "x"]


def test_a_seven_entry_session_still_shows_the_whole_file():
    ids = [f"router-{i}" for i in range(172)]
    _write_file(ids)
    seven = _adapter_models(
        "default",
        "code-high",
        "code-medium",
        "code-low",
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-haiku-5-5",
    )
    out = gw.fold_gateway_models(seven)
    assert len(out["availableModels"]) >= 172
    assert set(ids) <= {m["modelId"] for m in out["availableModels"]}


def test_envelope_folds_for_claude_only():
    _write_file(["router-a"])
    resp = {"models": _adapter_models("default")}
    claude = session_models_envelope(resp, ACP_BACKEND_CLAUDE)
    kiro = session_models_envelope(resp, ACP_BACKEND_KIRO)
    assert "router-a" in {m["modelId"] for m in claude["availableModels"]}
    assert [m["modelId"] for m in kiro["availableModels"]] == ["default"]


def test_envelope_folds_when_the_response_carries_only_a_select():
    _write_file(["router-a"])
    resp = {
        "configOptions": [
            {
                "id": "model",
                "type": "select",
                "currentValue": "default",
                "options": [{"value": "default", "name": "default"}],
            }
        ]
    }
    out = session_models_envelope(resp, ACP_BACKEND_CLAUDE)
    assert [m["modelId"] for m in out["availableModels"]] == ["default", "router-a", "opusplan"]
