"""A session carrying BOTH a ``models`` payload and a ``model`` select is unioned.

claude-agent-acp behind a custom gateway is that shape: the ``models`` payload
lists the slot-resolved ids (the ``ANTHROPIC_DEFAULT_*_MODEL`` targets plus the
real Claude ids) while the ``configOptions`` ``model`` select lists every id the
gateway serves, because ``session/set_config_option`` accepts the whole select.
Stopping at the payload offers a fraction of what a pick can name — the picker
showed eight models against a gateway serving over a hundred. These tests pin
the union: payload entries first, select-only ids appended, claude-only — the
other advertised-selection hosts keep payload-wins, pinned where that contract
was authored (test_acp_codex_harness).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import kiro_crew.model_registry as model_registry
from kiro_crew.acp.client import AcpClient
from kiro_crew.acp.session_handle import (
    AcpSessionHandle,
    WatchdogSettings,
    advertised_models_from_session,
    merge_model_select_options,
    session_models_envelope,
)
from kiro_crew.acp.types import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_KIRO,
)

_PAYLOAD_MODELS = [
    {"modelId": "default", "name": "Default (recommended)"},
    {"modelId": "code-high", "name": "code-high"},
    {"modelId": "code-medium", "name": "code-medium"},
    {"modelId": "claude-sonnet-5-5", "name": "Sonnet 5.5"},
]

_SELECT_ONLY = [
    {"value": "ssb-light-model", "name": "ssb-light-model", "description": "combo"},
    {"value": "glm/glm-5.3", "name": "glm/glm-5.3", "description": "combo"},
]


def _both_shapes(resp_models: dict | None = None, current: str = "code-medium") -> dict:
    """A claude ``session/new`` result carrying the payload AND the select."""
    return {
        "sessionId": "sess-1",
        "models": (
            resp_models
            if resp_models is not None
            else {"currentModelId": current, "availableModels": list(_PAYLOAD_MODELS)}
        ),
        "configOptions": [
            {
                "id": "model",
                "type": "select",
                "currentValue": current,
                "options": [
                    *[{"value": m["modelId"], "name": m["name"]} for m in _PAYLOAD_MODELS],
                    *_SELECT_ONLY,
                ],
            }
        ],
    }


class TestMergeModelSelectOptions:
    def test_payload_entries_first_then_select_only(self) -> None:
        merged = merge_model_select_options(
            {"currentModelId": "code-medium", "availableModels": list(_PAYLOAD_MODELS)},
            {"availableModels": [dict(e) for e in _SELECT_ONLY]},
        )
        assert [m["modelId"] for m in merged["availableModels"]] == [
            "default",
            "code-high",
            "code-medium",
            "claude-sonnet-5-5",
            "ssb-light-model",
            "glm/glm-5.3",
        ]
        assert merged["currentModelId"] == "code-medium"

    def test_duplicate_ids_resolve_to_the_payload_shape(self) -> None:
        merged = merge_model_select_options(
            {"availableModels": [{"modelId": "m1", "name": "Payload name"}]},
            {"availableModels": [{"value": "m1", "name": "Select name"}]},
        )
        assert merged["availableModels"] == [
            {"modelId": "m1", "name": "Payload name", "description": ""}
        ]

    def test_current_model_id_falls_back_to_the_select(self) -> None:
        merged = merge_model_select_options(
            {"availableModels": []},
            {"currentModelId": "code-low", "availableModels": []},
        )
        assert merged["currentModelId"] == "code-low"
        assert merged["availableModels"] == []


class TestSessionModelsEnvelope:
    def test_unions_when_an_opted_in_host_carries_both(self) -> None:
        env = session_models_envelope(_both_shapes(), ACP_BACKEND_CLAUDE)
        assert isinstance(env, dict)
        ids = [m["modelId"] for m in env["availableModels"]]
        assert ids[:4] == ["default", "code-high", "code-medium", "claude-sonnet-5-5"]
        assert "ssb-light-model" in ids and "glm/glm-5.3" in ids

    def test_a_host_outside_the_table_is_left_on_its_payload(self) -> None:
        for backend in (ACP_BACKEND_KIRO, ACP_BACKEND_CODEX):
            env = session_models_envelope(_both_shapes(), backend)
            assert isinstance(env, dict)
            assert [m["modelId"] for m in env["availableModels"]] == [
                m["modelId"] for m in _PAYLOAD_MODELS
            ], backend

    def test_no_select_leaves_the_payload_envelope_untouched(self) -> None:
        resp = {"models": {"currentModelId": "m1", "availableModels": [{"modelId": "m1"}]}}
        env = session_models_envelope(resp, ACP_BACKEND_CLAUDE)
        assert env == {"currentModelId": "m1", "availableModels": [{"modelId": "m1"}]}

    def test_a_malformed_select_does_not_touch_the_payload(self) -> None:
        resp = _both_shapes()
        resp["configOptions"][0]["options"] = "not-a-list"
        env = session_models_envelope(resp, ACP_BACKEND_CLAUDE)
        assert [m["modelId"] for m in env["availableModels"]] == [
            m["modelId"] for m in _PAYLOAD_MODELS
        ]


class TestClientCapture:
    def _client(self) -> AcpClient:
        return AcpClient(acp_backend=ACP_BACKEND_CLAUDE)

    def test_capture_unions_the_select_ids(self) -> None:
        client = self._client()
        client._capture_available_models(_both_shapes())
        ids = [m["modelId"] for m in client.available_models()]
        assert ids[:4] == ["default", "code-high", "code-medium", "claude-sonnet-5-5"]
        assert "ssb-light-model" in ids and "glm/glm-5.3" in ids
        assert client._resolved_model_id == "code-medium"

    def test_capture_feeds_the_union_to_the_provider_cache(self, monkeypatch) -> None:
        """The cross-session snapshot the cold dashboard reads must carry the
        union too, or the picker falls back to the slot ids after a restart."""
        seen: dict[str, list[str]] = {}

        def _fake_refresh(namespace: str, ids: list[str]) -> bool:
            seen[namespace] = ids
            return False

        monkeypatch.setattr(model_registry, "refresh_advertised_models", _fake_refresh)
        client = self._client()
        client._capture_available_models(_both_shapes())
        assert "ssb-light-model" in seen["claude_code"]
        assert "code-high" in seen["claude_code"]

    def test_capture_without_select_unchanged(self) -> None:
        client = self._client()
        client._capture_available_models({"models": {"availableModels": [{"modelId": "m1"}]}})
        assert [m["modelId"] for m in client.available_models()] == ["m1"]


class TestPooledAndProbePaths:
    def _handle(self) -> AcpSessionHandle:
        runtime = MagicMock()
        runtime.acp_backend = ACP_BACKEND_CLAUDE
        runtime.is_alive.return_value = True
        runtime.send_request = AsyncMock(return_value=1)
        runtime.send_notification = AsyncMock()
        return AcpSessionHandle(
            session_id="sess-1",
            queue=asyncio.Queue(),
            runtime=runtime,
            watchdog=WatchdogSettings(),
        )

    def test_store_session_config_unions_the_select_ids(self) -> None:
        handle = self._handle()
        handle.store_session_config(_both_shapes())
        assert "ssb-light-model" in handle._advertised_model_ids()
        assert handle._advertised_model_ids()[:2] == ["default", "code-high"]

    def test_probe_answer_carries_the_union(self) -> None:
        ids = [
            m["modelId"] for m in advertised_models_from_session(_both_shapes(), ACP_BACKEND_CLAUDE)
        ]
        assert ids[:4] == ["default", "code-high", "code-medium", "claude-sonnet-5-5"]
        assert "ssb-light-model" in ids and "glm/glm-5.3" in ids

    @pytest.mark.parametrize(
        "backend", sorted({ACP_BACKEND_CLAUDE, ACP_BACKEND_KIRO, ACP_BACKEND_CODEX})
    )
    def test_probe_and_pooled_agree_for_every_host(self, backend: str) -> None:
        handle = self._handle()
        handle._runtime.acp_backend = backend
        handle.store_session_config(_both_shapes())
        assert handle._advertised_model_ids() == [
            m["modelId"] for m in advertised_models_from_session(_both_shapes(), backend)
        ]
