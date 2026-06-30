"""Integration-style tests for checkpoint recommendation and download endpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from api_types import ModelComponentPaths, ModelProfilePayload
from _routes._errors import HTTPError
import handlers.models_handler as models_handler_module
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    LTXLocalModelDeprecated,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    get_model_cp_spec,
    resolve_downloading_dir,
    resolve_downloading_target_path,
    resolve_model_path,
)
from state.app_state_types import DownloadSessionComplete, DownloadSessionError, DownloadingSession, FileDownloadRunning
from tests.conftest import TEST_ADMIN_TOKEN, _test_model_path
from tests.http_error_assertions import assert_http_error

_ADMIN_HEADERS = {"X-Admin-Token": TEST_ADMIN_TOKEN}


def _current_ltx_spec():
    return get_ltx_model_spec(get_latest_ltx_model_id())


def _cp_path(test_state, cp_id: str) -> Path:
    return resolve_model_path(test_state.config.default_models_dir, cp_id)


class TestRecommendations:
    def test_ltx_recommendation_requires_primary_local_bundle(self, client):
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
                spec.text_encoder_cp,
            ],
        }

    def test_ltx_recommendation_skips_text_encoder_when_api_key_exists(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "test-key"
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
            ],
        }

    def test_ltx_recommendation_ok_when_required_bundle_is_downloaded(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ltx_recommendation_reports_missing_text_encoder_for_current_model(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [_current_ltx_spec().text_encoder_cp],
        }

    def test_img_gen_recommendation(self, client, create_fake_model_files):
        response = client.get("/api/models/img-gen-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] == IMG_GEN_MODEL_CP_ID

        create_fake_model_files(include_zit=True)
        response = client.get("/api/models/img-gen-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] is None

    def test_adapter_status_uses_user_path_then_models_dir(self, client, test_state, tmp_path):
        override_path = tmp_path / "union.safetensors"
        override_path.write_bytes(b"fake")
        test_state.state.app_settings.adapter_paths["union_control"] = str(override_path)

        # Adapter at canonical adapters/ subfolder (not root).
        models_dir_path = test_state.config.default_models_dir / "adapters" / "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
        models_dir_path.parent.mkdir(parents=True, exist_ok=True)
        models_dir_path.write_bytes(b"fake")

        response = client.get("/api/models/adapters/status", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        adapters = {item["id"]: item for item in response.json()["adapters"]}
        assert adapters["union_control"]["status"] == "available"
        assert adapters["union_control"]["path"] == str(override_path)
        assert adapters["hdr"]["status"] == "available"
        assert adapters["hdr"]["path"] == str(models_dir_path)
        assert adapters["lipdub"]["status"] == "missing"
        assert adapters["lipdub"]["path"] is None

    def test_adapter_status_uses_active_profile_paths(self, client, test_state, tmp_path):
        adapter_path = tmp_path / "ingredients.safetensors"
        adapter_path.write_bytes(b"fake")
        profile = ModelProfilePayload(
            id="profile-with-adapters",
            name="Profile With Adapters",
            source="official",
            components=ModelComponentPaths(official_adapters={"ingredients": str(adapter_path)}),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = profile.id

        response = client.get("/api/models/adapters/status", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        adapters = {item["id"]: item for item in response.json()["adapters"]}
        assert adapters["ingredients"]["status"] == "available"
        assert adapters["ingredients"]["path"] == str(adapter_path)

    def test_adapter_recommendation_reports_pipeline_missing_items(self, client):
        response = client.get(
            "/api/models/adapters/recommendation",
            params={"pipeline": "hdr"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["pipeline"] == "hdr"
        assert payload["missing"] == ["hdr", "hdr_scene_embeddings"]
        assert [item["adapter_id"] for item in payload["required"]] == ["hdr", "hdr_scene_embeddings"]
        assert payload["cps_to_download"] == ["ltx-2.3-22b-ic-lora-hdr-0.9", "ltx-2.3-22b-ic-lora-hdr-scene-emb"]

    def test_adapter_recommendation_returns_cps_for_missing_adapters(self, client):
        response = client.get(
            "/api/models/adapters/recommendation",
            params={"pipeline": "ingredients"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["missing"] == ["ingredients"]
        _CP_ID = "ltx-2.3-22b-ic-lora-ingredients-0.9"
        assert payload["cps_to_download"] == [_CP_ID]

    def test_adapter_recommendation_cps_empty_when_all_available(self, client, create_fake_model_files, test_state):
        create_fake_model_files()
        _CP_ID = "ltx-2.3-22b-ic-lora-ingredients-0.9"
        cp_path = _test_model_path(test_state, _CP_ID)
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        cp_path.write_bytes(b"\x00" * 1024)
        response = client.get(
            "/api/models/adapters/recommendation",
            params={"pipeline": "ingredients"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["missing"] == []
        assert payload["cps_to_download"] == []

    def test_adapter_status_requires_admin(self, client):
        response = client.get("/api/models/adapters/status")
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_legacy_ic_lora_recommendation_uses_union_control(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-ic-lora-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "cps_to_download": ["ltx-2.3-22b-ic-lora-union-control-ref0.5"]
        }

    def test_text_encoder_recommendation(self, client, create_fake_model_files, test_state):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/text-encoder-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] == _current_ltx_spec().text_encoder_cp
        assert response.json()["expected_size_bytes"] > 0

    def test_ic_lora_recommendation(self, client, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-ic-lora-recommendation")
        assert response.status_code == 200
        assert response.json()["cps_to_download"] == [
            *get_ic_loras_cp_ids(_current_ltx_spec().ic_loras_spec),
        ]

        create_fake_ic_lora_files()
        response = client.get("/api/models/ltx-ic-lora-recommendation")
        assert response.status_code == 200
        assert response.json()["cps_to_download"] == []


class TestDownloadProgress:
    def test_unknown_session_returns_404(self, client):
        response = client.get("/api/models/download/progress", params={"sessionId": "nonexistent"})
        assert_http_error(response, status_code=404, code="UNKNOWN_DOWNLOAD_SESSION")

    def test_active_progress(self, client, test_state):
        test_state.state.downloading_session = DownloadingSession(
            id="test-session",
            current_running_file=FileDownloadRunning(
                file_type="ltx-2.3-22b-distilled",
                target_path="ltx-2.3-22b-distilled.safetensors",
                downloaded_bytes=5_000_000_000,
                speed_bytes_per_sec=50_000_000.0,
            ),
            files_to_download={"ltx-2.3-22b-distilled"},
            completed_files=set(),
            completed_bytes=0,
        )
        response = client.get("/api/models/download/progress", params={"sessionId": "test-session"})
        assert response.status_code == 200
        assert response.json()["status"] == "downloading"
        assert response.json()["current_downloading_file"] == "ltx-2.3-22b-distilled"

    def test_completed_and_error_sessions(self, client, test_state):
        test_state.state.completed_download_sessions["done-session"] = DownloadSessionComplete()
        test_state.state.completed_download_sessions["err-session"] = DownloadSessionError(error_message="network error")

        complete = client.get("/api/models/download/progress", params={"sessionId": "done-session"})
        assert complete.status_code == 200
        assert complete.json()["status"] == "complete"

        failed = client.get("/api/models/download/progress", params={"sessionId": "err-session"})
        assert failed.status_code == 200
        assert failed.json()["status"] == "error"
        assert failed.json()["error"] == "network error"
        # Structured error code defaults to UNKNOWN_ERROR when not specified.
        assert failed.json()["error_code"] == "UNKNOWN_ERROR"


class TestModelDownloads:
    def test_download_start_success(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "started"
        assert _cp_path(test_state, IMG_GEN_MODEL_CP_ID).exists()

    def test_download_requires_admin(self, client):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_download_conflicts_when_another_session_is_running(self, client, test_state):
        test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
            headers=_ADMIN_HEADERS,
        )
        assert_http_error(response, status_code=409, code="DOWNLOAD_ALREADY_RUNNING")

    def test_upgrade_without_downloaded_model_is_rejected(self, client):
        response = client.post(
            "/api/models/download",
            json={"type": "upgrade", "cp_ids": [_current_ltx_spec().model_cp]},
            headers=_ADMIN_HEADERS,
        )
        assert_http_error(response, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_upgrade_raises_500_for_internal_ltx_mapping_inconsistency(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: None)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_upgrade_raises_500_when_latest_ltx_model_is_not_relevant(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_latest_ltx_model_id", lambda: "ltx-2.3-22b-distilled")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: "ltx-2.3-22b-distilled")

        original_get_ltx_model_spec = models_handler_module.get_ltx_model_spec

        def _get_ltx_model_spec(model_id):
            spec = original_get_ltx_model_spec(model_id)
            if model_id == "ltx-2.3-22b-distilled":
                return replace(spec, relevance=LTXLocalModelDeprecated())
            return spec

        monkeypatch.setattr(models_handler_module, "get_ltx_model_spec", _get_ltx_model_spec)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_download_error_is_reported(self, client, test_state):
        test_state.model_downloader.fail_next = RuntimeError("Connection refused")

        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        session_id = response.json()["sessionId"]

        progress = client.get("/api/models/download/progress", params={"sessionId": session_id})
        assert progress.status_code == 200
        assert progress.json()["status"] == "error"

    def test_download_uses_progress_callback(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        assert test_state.model_downloader.calls
        assert all(call["on_progress"] is not None for call in test_state.model_downloader.calls)

    def test_failed_download_cleans_session_staging(self, test_state):
        """Session-owned staging is cleaned; .downloading/ dir may persist."""
        test_state.model_downloader.fail_next = RuntimeError("network error")
        test_state.downloads.start_model_download(download_type="download", cp_ids={IMG_GEN_MODEL_CP_ID})
        # Worker finalizes errors internally (Phase 3B); no task-runner error.
        assert len(test_state.task_runner.errors) == 0
        # Session's staged file should be cleaned
        staging_path = resolve_downloading_target_path(
            test_state.config.default_models_dir, IMG_GEN_MODEL_CP_ID
        )
        assert not staging_path.exists()


class TestDownloadCancel:
    def test_cancel_requires_admin(self, client):
        response = client.post("/api/models/download/cancel")
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_cancel_no_active_returns_no_active_download(self, client, test_state):
        response = client.post("/api/models/download/cancel", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assert response.json() == {"status": "no_active_download"}

    def test_cancel_active_returns_cancelling_and_progress_cancelled(self, client, test_state):
        session_id = test_state.downloads.start_download({IMG_GEN_MODEL_CP_ID})
        response = client.post("/api/models/download/cancel", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assert response.json()["status"] == "cancelling"
        assert response.json()["sessionId"] == str(session_id)

        # Progress for that session reports cancelled while cleanup is pending.
        progress = client.get(
            "/api/models/download/progress", params={"sessionId": str(session_id)}
        )
        assert progress.status_code == 200
        assert progress.json()["status"] == "cancelled"

    def test_repeated_cancel_returns_cancelling_with_same_session_id(self, client, test_state):
        session_id = test_state.downloads.start_download({IMG_GEN_MODEL_CP_ID})
        first = client.post("/api/models/download/cancel", headers=_ADMIN_HEADERS)
        second = client.post("/api/models/download/cancel", headers=_ADMIN_HEADERS)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["status"] == "cancelling"
        assert second.json()["status"] == "cancelling"
        assert first.json()["sessionId"] == second.json()["sessionId"] == str(session_id)


class TestCheckpointDeletion:
    def test_delete_requires_admin(self, client):
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_delete_missing_checkpoint_is_noop(self, client):
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_delete_rejects_current_ltx_bundle(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [_current_ltx_spec().model_cp]},
            headers=_ADMIN_HEADERS,
        )
        assert_http_error(response, status_code=409, code="DELETE_PROTECTED_CHECKPOINT")

    def test_delete_removes_non_protected_checkpoint(self, client, test_state):
        img_gen_path = _cp_path(test_state, IMG_GEN_MODEL_CP_ID)
        img_gen_path.mkdir(parents=True, exist_ok=True)
        (img_gen_path / "model.safetensors").write_bytes(b"\x00" * 1024)

        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert not img_gen_path.exists()


# ---- Live model selection (Step 4) ----
# Full registry-derived option ids in catalog order (Fast family first, then
# Full family). Driven by the unified base-video registry, not downloadable CP
# ids — includes scanner-only Kijai/QuantStack distilled entries and the
# official dev safetensors.
_EXPECTED_MODEL_SELECTION_IDS: list[str] = [
    # Fast family (pipeline_family="fast")
    "ltx-2.3-22b-distilled",
    "ltx-2.3-22b-distilled-fp8-kijai-v3",
    "ltx-2.3-22b-distilled-gguf-quantstack-q2-k",
    "ltx-2.3-22b-distilled-gguf-quantstack-q3-k-s",
    "ltx-2.3-22b-distilled-gguf-quantstack-q3-k-m",
    "ltx-2.3-22b-distilled-gguf-quantstack-q4-k-s",
    "ltx-2.3-22b-distilled-gguf-quantstack-q4-k-m",
    "ltx-2.3-22b-distilled-gguf-quantstack-q5-k-s",
    "ltx-2.3-22b-distilled-gguf-quantstack-q5-k-m",
    # Full family (pipeline_family="full")
    "ltx-2.3-22b-dev",
    "ltx-2.3-22b-dev-gguf-q4-k-m",
    "ltx-2.3-22b-dev-gguf-ud-q4-k-m",
    "ltx-2.3-22b-dev-gguf-q6-k",
    "ltx-2.3-22b-dev-gguf-ud-q5-k-m",
]
_FAST_FAMILY_IDS: frozenset[str] = frozenset(_EXPECTED_MODEL_SELECTION_IDS[:9])
_FULL_FAMILY_IDS: frozenset[str] = frozenset(_EXPECTED_MODEL_SELECTION_IDS[9:])
_UNSUPPORTED_WORKFLOW_REASON = (
    "Live model selection is currently available for text-to-video, "
    "image-to-video, and HDR IC-LoRA only"
)
_NOT_INSTALLED_REASON = "Model checkpoint is not installed"


class TestModelSelectionOptions:
    def test_endpoint_requires_admin_token(self, client):
        # Valid workflow so param validation passes; guard rejects without token.
        response = client.get(
            "/api/models/model-options", params={"workflow": "text-to-video"}
        )
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_t2v_enumerates_all_candidates_disabled_when_missing(self, client, test_state):
        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["workflow"] == "text-to-video"
        assert data["models_dir"] == str(test_state.config.default_models_dir)

        options = data["options"]
        assert [o["id"] for o in options] == _EXPECTED_MODEL_SELECTION_IDS

        for opt in options:
            # Supported workflow + nothing installed → all missing-disabled.
            assert opt["installed"] is False
            assert opt["disabled_reason"] == _NOT_INSTALLED_REASON
            # Stable catalog-like metadata (universal across registry entries).
            assert opt["group"] == "Base video model"
            assert opt["source_url"] == f"https://huggingface.co/{opt['repo_id']}"
            assert opt["pipeline_family"] in ("fast", "full")

        # Spot-check the distilled candidate (section=full, empty variant group).
        distilled = options[0]
        assert distilled["id"] == "ltx-2.3-22b-distilled"
        assert distilled["section"] == "full"
        assert distilled["variant_group"] == "ltx-2.3-distilled"
        assert distilled["pipeline_family"] == "fast"
        assert distilled["repo_id"] == "Lightricks/LTX-2.3"
        assert distilled["label"].startswith("LTX-2.3 22B distilled")
        assert distilled["source_url"] == "https://huggingface.co/Lightricks/LTX-2.3"

        # Spot-check the Kijai FP8 candidate (Fast family, scanner-only).
        kijai = next(o for o in options if o["id"] == "ltx-2.3-22b-distilled-fp8-kijai-v3")
        assert kijai["section"] == "kijai"
        assert kijai["variant_group"] == "ltx-2.3-distilled-fp8"
        assert kijai["pipeline_family"] == "fast"
        assert kijai["repo_id"] == "Kijai/LTX2.3_comfy"
        assert kijai["downloadable"] is False

        # Spot-check a Full-family dev GGUF candidate (section=gguf).
        gguf = next(o for o in options if o["id"] == "ltx-2.3-22b-dev-gguf-q4-k-m")
        assert gguf["section"] == "gguf"
        assert gguf["variant_group"] == "ltx-2.3-dev-gguf"
        assert gguf["pipeline_family"] == "full"
        assert gguf["repo_id"] == "unsloth/LTX-2.3-GGUF"
        assert gguf["source_url"] == "https://huggingface.co/unsloth/LTX-2.3-GGUF"

    def test_installed_distilled_is_enabled_missing_gguf_remain_disabled(
        self, client, test_state, create_fake_model_files
    ):
        # create_fake_model_files installs the distilled transformer at its
        # canonical path (among other bundle files), but no GGUF candidates.
        create_fake_model_files()

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        by_id = {o["id"]: o for o in response.json()["options"]}

        distilled = by_id["ltx-2.3-22b-distilled"]
        assert distilled["installed"] is True
        assert distilled["disabled_reason"] is None

        # Every other registry entry is absent on disk → missing-disabled.
        for other_id in _EXPECTED_MODEL_SELECTION_IDS:
            if other_id == "ltx-2.3-22b-distilled":
                continue
            other = by_id[other_id]
            assert other["installed"] is False
            assert other["disabled_reason"] == _NOT_INSTALLED_REASON

    def test_options_tag_pipeline_family_distilled_fast_dev_full(
        self, client, test_state, create_fake_model_files
    ):
        # Each option carries a machine-readable pipeline family so the
        # frontend can filter the popover by the current model dropdown family.
        # Fast family = distilled/Kijai/QuantStack; Full family = dev/unsloth.
        create_fake_model_files()
        gguf_cp = "ltx-2.3-22b-dev-gguf-q4-k-m"
        gguf_path = resolve_model_path(test_state.config.default_models_dir, gguf_cp)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"GGUF")

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        by_id = {o["id"]: o for o in response.json()["options"]}

        for fast_id in _FAST_FAMILY_IDS:
            assert by_id[fast_id]["pipeline_family"] == "fast", fast_id
        for full_id in _FULL_FAMILY_IDS:
            assert by_id[full_id]["pipeline_family"] == "full", full_id

    def test_unsupported_workflow_disables_even_installed_options(self, client, create_fake_model_files):
        create_fake_model_files()

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "retake"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["workflow"] == "retake"
        assert [o["id"] for o in data["options"]] == _EXPECTED_MODEL_SELECTION_IDS

        for opt in data["options"]:
            assert opt["disabled_reason"] == _UNSUPPORTED_WORKFLOW_REASON

        # Distilled is installed on disk, yet disabled because of the workflow.
        distilled = next(o for o in data["options"] if o["id"] == "ltx-2.3-22b-distilled")
        assert distilled["installed"] is True
        assert distilled["disabled_reason"] == _UNSUPPORTED_WORKFLOW_REASON

    def test_installed_gguf_disabled_without_active_profile(
        self, client, test_state, create_fake_model_files
    ):
        # GGUF installed + canonical distilled LoRA + canonical upscaler, but NO
        # active profile → the dev/GGUF selection cannot run (no split sidecars).
        create_fake_model_files()
        gguf_cp = "ltx-2.3-22b-dev-gguf-q4-k-m"
        gguf_path = resolve_model_path(test_state.config.default_models_dir, gguf_cp)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"GGUF")
        lora = (
            test_state.config.default_models_dir
            / "adapters"
            / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        lora.parent.mkdir(parents=True, exist_ok=True)
        lora.write_bytes(b"LORA")

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        by_id = {o["id"]: o for o in response.json()["options"]}
        gguf = by_id[gguf_cp]
        assert gguf["installed"] is True
        assert gguf["disabled_reason"] is not None
        assert "active model profile" in gguf["disabled_reason"].lower()

    def test_installed_gguf_enabled_with_suitable_active_profile(
        self, client, test_state, tmp_path, create_fake_model_files
    ):
        # Active split-component profile + installed GGUF + canonical distilled
        # LoRA → the dev/GGUF selection is runtime-ready and enabled.
        create_fake_model_files()  # canonical upscaler + text encoder
        sidecars: dict[str, str] = {}
        for name in ("tp", "ec", "vvae", "avae", "ups"):
            p = tmp_path / f"{name}.safetensors"
            p.write_bytes(b"x")
            sidecars[name] = str(p)
        profile = ModelProfilePayload(
            id="dev-split",
            name="Dev Split",
            source="kijai",
            components=ModelComponentPaths(
                transformer="/placeholder-dev.gguf",
                transformer_format="gguf",
                text_projection=sidecars["tp"],
                embeddings_connector=sidecars["ec"],
                video_vae=sidecars["vvae"],
                audio_vae=sidecars["avae"],
                upsampler=sidecars["ups"],
                text_encoder_root="/placeholder-gemma",
                text_encoder_format="gguf",
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "dev-split"

        gguf_cp = "ltx-2.3-22b-dev-gguf-q4-k-m"
        gguf_path = resolve_model_path(test_state.config.default_models_dir, gguf_cp)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"GGUF")
        lora = (
            test_state.config.default_models_dir
            / "adapters"
            / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        lora.parent.mkdir(parents=True, exist_ok=True)
        lora.write_bytes(b"LORA")

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        by_id = {o["id"]: o for o in response.json()["options"]}
        gguf = by_id[gguf_cp]
        assert gguf["installed"] is True
        assert gguf["disabled_reason"] is None

    def test_installed_gguf_enabled_when_embeddings_connector_optional(
        self, client, test_state, tmp_path, create_fake_model_files
    ):
        # QuantStack-like profile: embeddings_connector is null but the runtime
        # treats it as optional, so a dev/GGUF selection must still be enabled
        # when the required sidecars (text projection, VAEs) + upscaler +
        # distilled LoRA are available.
        create_fake_model_files()  # canonical upscaler + text encoder
        sidecars: dict[str, str] = {}
        for name in ("tp", "vvae", "avae", "ups"):
            p = tmp_path / f"{name}.safetensors"
            p.write_bytes(b"x")
            sidecars[name] = str(p)
        profile = ModelProfilePayload(
            id="quantstack-like",
            name="QuantStack-like",
            source="quantstack",
            components=ModelComponentPaths(
                transformer="/placeholder-dev.gguf",
                transformer_format="gguf",
                text_projection=sidecars["tp"],
                embeddings_connector=None,  # optional — must not disable
                video_vae=sidecars["vvae"],
                audio_vae=sidecars["avae"],
                upsampler=sidecars["ups"],
                text_encoder_root="/placeholder-gemma",
                text_encoder_format="gguf",
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "quantstack-like"

        gguf_cp = "ltx-2.3-22b-dev-gguf-q4-k-m"
        gguf_path = resolve_model_path(test_state.config.default_models_dir, gguf_cp)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"GGUF")
        lora = (
            test_state.config.default_models_dir
            / "adapters"
            / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        lora.parent.mkdir(parents=True, exist_ok=True)
        lora.write_bytes(b"LORA")

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        by_id = {o["id"]: o for o in response.json()["options"]}
        gguf = by_id[gguf_cp]
        assert gguf["installed"] is True
        # embeddings_connector absent must NOT disable the option.
        assert gguf["disabled_reason"] is None

    def test_installed_gguf_disabled_when_profile_missing_sidecars(
        self, client, test_state, tmp_path, create_fake_model_files
    ):
        # Active profile exists but lacks the split sidecars → dev/GGUF disabled
        # with a clear reason naming the missing components.
        create_fake_model_files()
        profile = ModelProfilePayload(
            id="incomplete",
            name="Incomplete",
            source="kijai",
            components=ModelComponentPaths(
                transformer="/placeholder-dev.gguf",
                transformer_format="gguf",
                # No text_projection / embeddings_connector / VAEs.
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "incomplete"

        gguf_cp = "ltx-2.3-22b-dev-gguf-q4-k-m"
        gguf_path = resolve_model_path(test_state.config.default_models_dir, gguf_cp)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"GGUF")

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        by_id = {o["id"]: o for o in response.json()["options"]}
        gguf = by_id[gguf_cp]
        assert gguf["installed"] is True
        assert gguf["disabled_reason"] is not None
        assert "split components" in gguf["disabled_reason"].lower()

    def test_model_options_use_base_video_registry_fast_and_full_families(
        self, client, test_state, create_fake_model_files
    ):
        """The model-options endpoint is driven by the unified base-video
        registry (plan: source-of-truth fix). It must enumerate Fast-family
        Kijai/QuantStack distilled entries AND Full-family dev entries (not
        only downloadable CP ids), with installed/disabled status derived from
        fake filesystem evidence.

        Coverage:
        - Fast family includes the Kijai FP8 + a QuantStack distilled GGUF.
        - Full family includes the official dev safetensors + an unsloth dev GGUF.
        - Placing a scanner-only file at its canonical path flips it to
          ``installed=True`` (registry evidence semantics).
        - A supported-workflow installed distilled selection is enabled; a
          scanner-only installed entry that requires sidecars is disabled when
          no active profile provides them.
        """
        # Distilled bundle (upscaler/text encoder) + the official distilled
        # transformer — the canonical distilled selection is installed & enabled.
        create_fake_model_files()

        models_dir = test_state.config.default_models_dir

        # Place Fast-family scanner-only files at their canonical registry paths.
        kijai_rel = (
            "diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors"
        )
        kijai_path = models_dir / kijai_rel
        kijai_path.parent.mkdir(parents=True, exist_ok=True)
        kijai_path.write_bytes(b"FP8")

        quantstack_rel = (
            "gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/"
            "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
        )
        quantstack_path = models_dir / quantstack_rel
        quantstack_path.parent.mkdir(parents=True, exist_ok=True)
        quantstack_path.write_bytes(b"GGUF")

        # Place a Full-family scanner-only file (official dev safetensors).
        dev_rel = "diffusion_models/ltx-2.3-22b-dev.safetensors"
        dev_path = models_dir / dev_rel
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        dev_path.write_bytes(b"DEV")

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "text-to-video"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        by_id = {o["id"]: o for o in response.json()["options"]}

        # Fast-family Kijai/QuantStack entries are present and now installed.
        kijai = by_id["ltx-2.3-22b-distilled-fp8-kijai-v3"]
        assert kijai["pipeline_family"] == "fast"
        assert kijai["installed"] is True
        assert kijai["repo_id"] == "Kijai/LTX2.3_comfy"
        assert kijai["canonical_relative_path"] == kijai_rel
        assert kijai["expected_absolute_path"] == str(kijai_path)
        # Scanner-only (not downloadable); requires sidecars → disabled here
        # because no active profile provides them.
        assert kijai["downloadable"] is False
        assert kijai["disabled_reason"] is not None

        quantstack = by_id["ltx-2.3-22b-distilled-gguf-quantstack-q4-k-m"]
        assert quantstack["pipeline_family"] == "fast"
        assert quantstack["installed"] is True
        assert quantstack["repo_id"] == "QuantStack/LTX-2.3-GGUF"
        assert quantstack["canonical_relative_path"] == quantstack_rel
        assert quantstack["downloadable"] is False
        assert quantstack["disabled_reason"] is not None

        # Full-family dev entries are present.
        dev = by_id["ltx-2.3-22b-dev"]
        assert dev["pipeline_family"] == "full"
        assert dev["installed"] is True
        assert dev["repo_id"] == "Lightricks/LTX-2.3"
        assert dev["canonical_relative_path"] == dev_rel
        assert dev["expected_absolute_path"] == str(dev_path)
        assert dev["disabled_reason"] is not None

        # The official distilled (runtime_readiness=none) is installed & enabled.
        distilled = by_id["ltx-2.3-22b-distilled"]
        assert distilled["installed"] is True
        assert distilled["disabled_reason"] is None

        # Family partitioning is exhaustive and disjoint over the registry ids.
        assert _FAST_FAMILY_IDS | _FULL_FAMILY_IDS == set(_EXPECTED_MODEL_SELECTION_IDS)
        assert _FAST_FAMILY_IDS.isdisjoint(_FULL_FAMILY_IDS)

    def test_hdr_ic_lora_model_options_are_supported(
        self, client, test_state, create_fake_model_files
    ):
        """``GET /api/models/model-options?workflow=hdr-ic-lora`` is supported.

        HDR IC-LoRA joins text-to-video and image-to-video as a supported
        workflow for live model selection (dedicated V2V pipeline). The
        official distilled monolith (runtime_readiness="none") must be enabled
        when installed; dev/GGUF candidates remain disabled when no active
        profile provides sidecars (same gating as T2V/I2V).
        """
        create_fake_model_files()

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "hdr-ic-lora"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["workflow"] == "hdr-ic-lora"
        by_id = {o["id"]: o for o in data["options"]}

        # Same candidate catalog as T2V/I2V.
        assert [o["id"] for o in data["options"]] == _EXPECTED_MODEL_SELECTION_IDS

        # Supported workflow: no option carries the unsupported-workflow reason.
        for opt in data["options"]:
            assert opt["disabled_reason"] != _UNSUPPORTED_WORKFLOW_REASON, (
                f"{opt['id']} must not be unsupported-reason under hdr-ic-lora"
            )

        # The official distilled (runtime_readiness="none") is installed and
        # enabled (no sidecars required).
        distilled = by_id["ltx-2.3-22b-distilled"]
        assert distilled["installed"] is True
        assert distilled["disabled_reason"] is None

        # Dev/GGUF candidates are installed-on-disk-absent here, so they stay
        # missing-disabled (NOT unsupported-disabled).
        gguf = by_id["ltx-2.3-22b-dev-gguf-q4-k-m"]
        assert gguf["installed"] is False
        assert gguf["disabled_reason"] == _NOT_INSTALLED_REASON

    def test_generic_ic_lora_model_options_remain_unsupported(
        self, client, create_fake_model_files
    ):
        """``workflow=ic-lora`` stays unsupported for live model selection.

        Generic IC-LoRA has no live base-model selection path. Every candidate
        enumerates but is uniformly disabled with the unsupported-workflow
        reason (now mentioning text-to-video, image-to-video, and HDR IC-LoRA).
        """
        create_fake_model_files()

        response = client.get(
            "/api/models/model-options",
            params={"workflow": "ic-lora"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["workflow"] == "ic-lora"

        # Same candidate catalog, but every option is unsupported-disabled.
        assert [o["id"] for o in data["options"]] == _EXPECTED_MODEL_SELECTION_IDS
        for opt in data["options"]:
            assert opt["disabled_reason"] == _UNSUPPORTED_WORKFLOW_REASON, (
                f"{opt['id']} must be unsupported under generic ic-lora"
            )

        # Reason text mentions HDR IC-LoRA (updated gating copy).
        assert "HDR IC-LoRA" in _UNSUPPORTED_WORKFLOW_REASON
