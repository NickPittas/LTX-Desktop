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

        models_dir_path = test_state.config.default_models_dir / "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
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
