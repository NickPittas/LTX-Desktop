"""Integration-style tests for model profile endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from api_types import CURRENT_MODEL_PROFILE_SCHEMA_VERSION
from runtime_config.model_download_specs import resolve_model_path
from tests.conftest import TEST_ADMIN_TOKEN
from tests.http_error_assertions import assert_http_error

_ADMIN_HEADERS = {"X-Admin-Token": TEST_ADMIN_TOKEN}


def _make_official_payload(profile_id: str = "test-official", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": profile_id,
        "name": "Test Official",
        "family": "ltx-2.3",
        "source": "official",
        "components": {
            "transformer": "/tmp/test_model.safetensors",
            "upsampler": "/tmp/test_upsampler.safetensors",
            "text_encoder_format": "api",
        },
        "capabilities": ["t2v"],
        "notes": "",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    payload.update(overrides)
    return payload


def _scanner_known_components(test_state) -> dict[str, str]:
    """Write scanner-recognized base diffusion + upscaler at canonical paths
    under the effective models root and return components pointing at them."""
    models_dir: Path = test_state.config.default_models_dir
    transformer = resolve_model_path(models_dir, "ltx-2.3-22b-distilled")
    upsampler = resolve_model_path(models_dir, "ltx-2.3-spatial-upscaler-x2-1.0")
    transformer.parent.mkdir(parents=True, exist_ok=True)
    transformer.write_bytes(b"model")
    upsampler.parent.mkdir(parents=True, exist_ok=True)
    upsampler.write_bytes(b"upsampler")
    return {
        "transformer": str(transformer),
        "upsampler": str(upsampler),
        "text_encoder_format": "api",
    }


class TestModelProfiles:
    def test_list_profiles_empty(self, client):
        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["active_model_profile_id"] is None
        assert data["profiles"] == []

    def test_list_requires_admin(self, client):
        response = client.get("/api/model-profiles")
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_create_official_profile(self, client, tmp_path):
        profile = _make_official_payload(
            profile_id="official-1",
            components={
                "transformer": str(tmp_path / "model.safetensors"),
                "upsampler": str(tmp_path / "upsampler.safetensors"),
                "text_encoder_format": "api",
            },
        )

        response = client.post("/api/model-profiles", json=profile, headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "official-1"
        assert data["source"] == "official"

        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assert len(response.json()["profiles"]) == 1

    def test_create_requires_admin(self, client):
        response = client.post("/api/model-profiles", json=_make_official_payload())
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_duplicate_id_rejected(self, client):
        profile = _make_official_payload(profile_id="dup")
        client.post("/api/model-profiles", json=profile, headers=_ADMIN_HEADERS)
        response = client.post("/api/model-profiles", json=profile, headers=_ADMIN_HEADERS)
        assert response.status_code == 409

    def test_validate_missing_file_reports_issues(self, client):
        response = client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                components={
                    "transformer": "/nonexistent/model.safetensors",
                    "upsampler": "/nonexistent/upsampler.safetensors",
                    "text_encoder_format": "api",
                },
            ),
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200

        profile_id = response.json()["id"]
        response = client.post(f"/api/model-profiles/{profile_id}/validate", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["issues"]) >= 2

    def test_validate_bad_extension_reported(self, client):
        response = client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="bad-ext",
                components={
                    "transformer": "/tmp/model.txt",
                    "upsampler": "/tmp/upsampler.txt",
                    "text_encoder_format": "api",
                },
            ),
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200

        response = client.post("/api/model-profiles/bad-ext/validate", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        fields = [issue["field"] for issue in response.json()["issues"]]
        assert "components.transformer" in fields

    def test_activate_valid_profile(self, client, test_state):
        components = _scanner_known_components(test_state)

        response = client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="valid-profile",
                components=components,
            ),
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200

        response = client.post("/api/model-profiles/valid-profile/validate", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assert response.json()["valid"] is True

        response = client.post("/api/model-profiles/valid-profile/activate", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["active_model_profile_id"] == "valid-profile"

        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        assert response.json()["active_model_profile_id"] == "valid-profile"

    def test_activate_invalid_profile_returns_409(self, client):
        response = client.post(
            "/api/model-profiles",
            json=_make_official_payload(profile_id="invalid-profile"),
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200

        response = client.post("/api/model-profiles/invalid-profile/activate", headers=_ADMIN_HEADERS)
        assert response.status_code == 409

    def test_activate_requires_admin(self, client, tmp_path):
        model_file = tmp_path / "model.safetensors"
        upscaler_file = tmp_path / "upsampler.safetensors"
        model_file.write_bytes(b"model")
        upscaler_file.write_bytes(b"upscaler")

        client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="no-admin-profile",
                components={
                    "transformer": str(model_file),
                    "upsampler": str(upscaler_file),
                    "text_encoder_format": "api",
                },
            ),
            headers=_ADMIN_HEADERS,
        )

        response = client.post("/api/model-profiles/no-admin-profile/activate")
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_validate_nonexistent_profile_returns_404(self, client):
        response = client.post("/api/model-profiles/nonexistent/validate", headers=_ADMIN_HEADERS)
        assert response.status_code == 404

    def test_patch_profile(self, client):
        client.post("/api/model-profiles", json=_make_official_payload(profile_id="patchable"), headers=_ADMIN_HEADERS)

        response = client.request(
            "PATCH",
            "/api/model-profiles/patchable",
            json={"name": "Patched Name", "notes": "Updated"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Patched Name"
        assert data["notes"] == "Updated"

    def test_patch_profile_validates_nested_components(self, client):
        client.post("/api/model-profiles", json=_make_official_payload(profile_id="patchable-nested"), headers=_ADMIN_HEADERS)

        response = client.request(
            "PATCH",
            "/api/model-profiles/patchable-nested",
            json={"components": {"official_adapters": {"ingredients": "/tmp/ingredients.safetensors"}}},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200

        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        profile = next(p for p in response.json()["profiles"] if p["id"] == "patchable-nested")
        assert profile["components"]["official_adapters"]["ingredients"] == "/tmp/ingredients.safetensors"

    def test_delete_profile(self, client):
        client.post("/api/model-profiles", json=_make_official_payload(profile_id="deletable"), headers=_ADMIN_HEADERS)

        response = client.delete("/api/model-profiles/deletable", headers=_ADMIN_HEADERS)
        assert response.status_code == 200

        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        assert len(response.json()["profiles"]) == 0

    def test_delete_deactivates_if_active(self, client, test_state):
        components = _scanner_known_components(test_state)

        profile = _make_official_payload(
            profile_id="active-delete",
            components=components,
        )
        client.post("/api/model-profiles", json=profile, headers=_ADMIN_HEADERS)
        client.post("/api/model-profiles/active-delete/activate", headers=_ADMIN_HEADERS)

        response = client.delete("/api/model-profiles/active-delete", headers=_ADMIN_HEADERS)
        assert response.status_code == 200

        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        assert response.json()["active_model_profile_id"] is None

    def test_delete_nonexistent_returns_404(self, client):
        response = client.delete("/api/model-profiles/nope", headers=_ADMIN_HEADERS)
        assert response.status_code == 404


class TestCreateEmptyId:
    """Regression: creating a profile with empty/blank ID gets an assigned ID."""

    def test_empty_id_gets_assigned_and_usable(self, client, test_state):
        components = _scanner_known_components(test_state)

        payload = _make_official_payload(
            profile_id="",
            components=components,
        )
        response = client.post("/api/model-profiles", json=payload, headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assigned_id = response.json()["id"]
        assert assigned_id and assigned_id.strip()

        # validate works by assigned id
        validate_r = client.post(f"/api/model-profiles/{assigned_id}/validate", headers=_ADMIN_HEADERS)
        assert validate_r.status_code == 200

        # activate works by assigned id (scanner-recognized transformer → 200)
        activate_r = client.post(f"/api/model-profiles/{assigned_id}/activate", headers=_ADMIN_HEADERS)
        assert activate_r.status_code == 200

        # delete works by assigned id
        delete_r = client.delete(f"/api/model-profiles/{assigned_id}", headers=_ADMIN_HEADERS)
        assert delete_r.status_code == 200

    def test_blank_id_gets_assigned(self, client):
        payload = _make_official_payload(profile_id="  ")
        response = client.post("/api/model-profiles", json=payload, headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assert response.json()["id"] and response.json()["id"].strip()

    def test_explicit_id_still_works(self, client):
        payload = _make_official_payload(profile_id="my-profile")
        response = client.post("/api/model-profiles", json=payload, headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assert response.json()["id"] == "my-profile"


class TestLoadRepair:
    """Regression: load_profiles repairs bad persisted data."""

    def _write_profiles(self, handler, data: dict) -> Path:
        path: Path = handler._profiles_path
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_blank_ids_repaired_on_load(self, test_state):
        self._write_profiles(
            test_state.model_profiles,
            {
                "active_model_profile_id": "",
                "profiles": [
                    {
                        "id": "",
                        "name": "Bad Profile",
                        "family": "ltx-2.3",
                        "source": "official",
                        "components": {"text_encoder_format": "api"},
                        "capabilities": ["t2v"],
                        "notes": "",
                        "created_at": "",
                        "updated_at": "",
                    },
                    {
                        "id": "good-profile",
                        "name": "Good Profile",
                        "family": "ltx-2.3",
                        "source": "official",
                        "components": {"text_encoder_format": "api"},
                        "capabilities": ["t2v"],
                        "notes": "",
                        "created_at": "",
                        "updated_at": "",
                    },
                ],
            },
        )
        test_state.model_profiles.load_profiles()

        assert len(test_state.state.model_profiles) == 2
        bad = [p for p in test_state.state.model_profiles if p.name == "Bad Profile"][0]
        assert bad.id and bad.id.strip()
        good = [p for p in test_state.state.model_profiles if p.name == "Good Profile"][0]
        assert good.id == "good-profile"

        # Repaired file was saved back
        raw = json.loads(test_state.model_profiles._profiles_path.read_text())
        assert raw["active_model_profile_id"] is None

    def test_blank_active_id_cleared_on_load(self, test_state):
        self._write_profiles(
            test_state.model_profiles,
            {
                "active_model_profile_id": "",
                "profiles": [],
            },
        )
        test_state.model_profiles.load_profiles()
        assert test_state.state.active_model_profile_id is None

    def test_missing_active_profile_cleared_on_load(self, test_state):
        self._write_profiles(
            test_state.model_profiles,
            {
                "active_model_profile_id": "ghost",
                "profiles": [],
            },
        )
        test_state.model_profiles.load_profiles()
        assert test_state.state.active_model_profile_id is None


class TestRecommendationWithProfile:
    def test_ltx_recommendation_ok_when_valid_official_profile_active(self, client, test_state):
        components = _scanner_known_components(test_state)

        client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="official-ready",
                components=components,
            ),
            headers=_ADMIN_HEADERS,
        )
        client.post("/api/model-profiles/official-ready/activate", headers=_ADMIN_HEADERS)

        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestProfileSchemaMigration:
    """Phase 1: backward-compatible model_profiles.json schema migration."""

    def _write_profiles(self, handler, data: dict) -> Path:
        path: Path = handler._profiles_path
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _legacy_profile_dict(self, profile_id: str = "legacy-1") -> dict:
        return {
            "id": profile_id,
            "name": "Legacy Profile",
            "family": "ltx-2.3",
            "source": "official",
            "components": {"text_encoder_format": "api"},
            "capabilities": ["t2v"],
            "notes": "",
            "created_at": "",
            "updated_at": "",
        }

    def test_legacy_profile_gets_schema_defaults(self, test_state):
        """Absent schema fields get defaults in the loaded (in-memory) profile."""
        self._write_profiles(
            test_state.model_profiles,
            {
                "active_model_profile_id": "legacy-1",
                "profiles": [self._legacy_profile_dict()],
            },
        )
        test_state.model_profiles.load_profiles()

        assert len(test_state.state.model_profiles) == 1
        profile = test_state.state.model_profiles[0]
        assert profile.schema_version == CURRENT_MODEL_PROFILE_SCHEMA_VERSION
        assert profile.created_by == "user"
        assert profile.validation_status == "candidate"
        assert profile.last_scanned_at is None
        assert profile.problems == []

    def test_legacy_profile_defaults_in_api_response(self, client, test_state):
        """Defaults appear in the API response without any explicit save."""
        path: Path = test_state.model_profiles._profiles_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "active_model_profile_id": None,
                "profiles": [self._legacy_profile_dict("api-legacy")],
            }),
            encoding="utf-8",
        )
        # Reload so the handler picks up the legacy file
        test_state.model_profiles.load_profiles()

        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        profiles = response.json()["profiles"]
        assert len(profiles) == 1
        p = profiles[0]
        assert p["schema_version"] == CURRENT_MODEL_PROFILE_SCHEMA_VERSION
        assert p["created_by"] == "user"
        assert p["validation_status"] == "candidate"
        assert p["last_scanned_at"] is None
        assert p["problems"] == []

    def test_no_autosave_on_migration_without_repair(self, test_state):
        """Load does NOT rewrite the file when there are no blank IDs to repair."""
        path = self._write_profiles(
            test_state.model_profiles,
            {
                "active_model_profile_id": None,
                "profiles": [self._legacy_profile_dict()],
            },
        )
        original_text = path.read_text(encoding="utf-8")
        test_state.model_profiles.load_profiles()

        # File must be unchanged — no schema_version written
        assert path.read_text(encoding="utf-8") == original_text
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" not in raw["profiles"][0]

    def test_blank_id_repair_still_saves_with_new_fields(self, test_state):
        """Existing blank-ID repair path saves and includes new fields."""
        path = self._write_profiles(
            test_state.model_profiles,
            {
                "active_model_profile_id": "",
                "profiles": [self._legacy_profile_dict(profile_id="")],
            },
        )
        test_state.model_profiles.load_profiles()

        # Repair triggered a save → new fields are persisted
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["profiles"][0]["schema_version"] == CURRENT_MODEL_PROFILE_SCHEMA_VERSION

    def test_round_trip_on_explicit_save(self, test_state):
        """Explicit save after load persists the new schema fields."""
        path = self._write_profiles(
            test_state.model_profiles,
            {
                "active_model_profile_id": None,
                "profiles": [self._legacy_profile_dict("round-trip")],
            },
        )
        test_state.model_profiles.load_profiles()

        # Before save — legacy file
        raw_before = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" not in raw_before["profiles"][0]

        # Explicit save
        test_state.model_profiles.save_profiles()

        # After save — new fields persisted
        raw_after = json.loads(path.read_text(encoding="utf-8"))
        assert raw_after["profiles"][0]["schema_version"] == CURRENT_MODEL_PROFILE_SCHEMA_VERSION
        assert raw_after["profiles"][0]["created_by"] == "user"
        assert raw_after["profiles"][0]["validation_status"] == "candidate"
        assert raw_after["profiles"][0]["problems"] == []

    def test_extra_forbid_preserved_on_create(self, client):
        """Creating a profile with an unknown field must still be rejected (422)."""
        payload = _make_official_payload(unknown_field="should_be_rejected")
        response = client.post("/api/model-profiles", json=payload, headers=_ADMIN_HEADERS)
        assert response.status_code == 422

    def test_extra_forbid_preserved_on_patch(self, client):
        """Patching with server-owned fields must be rejected (422)."""
        client.post(
            "/api/model-profiles",
            json=_make_official_payload(profile_id="patch-forbid"),
            headers=_ADMIN_HEADERS,
        )
        response = client.request(
            "PATCH",
            "/api/model-profiles/patch-forbid",
            json={"schema_version": 99},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 422

    def test_create_with_schema_fields_accepted(self, client):
        """ModelProfilePayload can carry the new fields on create."""
        payload = _make_official_payload(
            profile_id="with-schema",
            schema_version=CURRENT_MODEL_PROFILE_SCHEMA_VERSION,
            created_by="wizard",
            validation_status="candidate",
        )
        response = client.post("/api/model-profiles", json=payload, headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["schema_version"] == CURRENT_MODEL_PROFILE_SCHEMA_VERSION
        assert data["created_by"] == "wizard"

    def test_patch_does_not_touch_server_owned_fields(self, client):
        """Patching name/notes must not reset server-owned schema fields."""
        client.post(
            "/api/model-profiles",
            json=_make_official_payload(profile_id="patch-preserve", created_by="wizard"),
            headers=_ADMIN_HEADERS,
        )

        response = client.request(
            "PATCH",
            "/api/model-profiles/patch-preserve",
            json={"name": "New Name"},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"
        # Server-owned fields preserved
        assert data["created_by"] == "wizard"
        assert data["schema_version"] == CURRENT_MODEL_PROFILE_SCHEMA_VERSION


class TestProfileActivationSafety:
    """Phase 5: activation hardening (generation gate + resolver gating)."""

    def test_activate_profile_rejects_while_generation_running(self, client, test_state):
        from state.app_state_types import (
            ApiGeneration,
            GenerationProgress,
            GenerationRunning,
        )

        components = _scanner_known_components(test_state)
        client.post(
            "/api/model-profiles",
            json=_make_official_payload(profile_id="gen-blocked", components=components),
            headers=_ADMIN_HEADERS,
        )

        # Simulate an active running generation.
        test_state.state.active_generation = ApiGeneration(
            state=GenerationRunning(
                id="gen-1",
                progress=GenerationProgress(phase="encoding", progress=0, current_step=None, total_steps=None),
            )
        )

        response = client.post("/api/model-profiles/gen-blocked/activate", headers=_ADMIN_HEADERS)
        assert response.status_code == 409
        assert response.json()["code"] == "MODEL_PROFILE_ACTIVATION_GENERATION_RUNNING"
        # Profile was not activated.
        assert test_state.state.active_model_profile_id is None

    def test_activate_profile_rejects_missing_required_transformer(self, client, test_state):
        # Profile points at a path that is not scanner-recognized.
        client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="missing-base",
                components={
                    "transformer": str(test_state.config.default_models_dir / "nonexistent.safetensors"),
                    "upsampler": str(test_state.config.default_models_dir / "also-missing.safetensors"),
                    "text_encoder_format": "api",
                },
            ),
            headers=_ADMIN_HEADERS,
        )

        response = client.post("/api/model-profiles/missing-base/activate", headers=_ADMIN_HEADERS)
        assert response.status_code == 409
        assert response.json()["code"] == "MODEL_PROFILE_REQUIRED_ARTIFACTS_MISSING"
        assert test_state.state.active_model_profile_id is None

    def test_activate_profile_allows_scanned_wrong_folder_transformer_when_profile_points_to_it(
        self, client, test_state
    ):
        # Place the base diffusion model in a non-canonical subfolder so the
        # scanner reports it as wrong_folder_usable.
        models_dir: Path = test_state.config.default_models_dir
        wrong_folder = models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
        wrong_folder.parent.mkdir(parents=True, exist_ok=True)
        wrong_folder.write_bytes(b"model")

        client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="wrong-folder",
                components={
                    "transformer": str(wrong_folder),
                    "text_encoder_format": "api",
                },
            ),
            headers=_ADMIN_HEADERS,
        )

        response = client.post("/api/model-profiles/wrong-folder/activate", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        assert response.json()["active_model_profile_id"] == "wrong-folder"

    def test_activate_profile_does_not_promote_candidate_to_validated(self, client, test_state):
        components = _scanner_known_components(test_state)
        client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="candidate-activate",
                components=components,
                validation_status="candidate",
            ),
            headers=_ADMIN_HEADERS,
        )

        response = client.post("/api/model-profiles/candidate-activate/activate", headers=_ADMIN_HEADERS)
        assert response.status_code == 200

        # Activation must not change validation_status to validated.
        profiles = client.get("/api/model-profiles", headers=_ADMIN_HEADERS).json()["profiles"]
        profile = next(p for p in profiles if p["id"] == "candidate-activate")
        assert profile["validation_status"] == "candidate"

    def test_activate_profile_rejects_catalog_fallback_without_explicit_transformer(self, client, test_state):
        """Canonical transformer exists in models root, but the profile omits
        components.transformer — activation must reject (no catalog fallback)."""
        # Write a scanner-known transformer at the canonical path, but do NOT
        # point the profile at it.
        models_dir: Path = test_state.config.default_models_dir
        transformer = resolve_model_path(models_dir, "ltx-2.3-22b-distilled")
        transformer.parent.mkdir(parents=True, exist_ok=True)
        transformer.write_bytes(b"model")

        client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="no-transformer",
                components={"text_encoder_format": "api"},
            ),
            headers=_ADMIN_HEADERS,
        )

        response = client.post("/api/model-profiles/no-transformer/activate", headers=_ADMIN_HEADERS)
        assert response.status_code == 409
        assert response.json()["code"] == "MODEL_PROFILE_REQUIRED_ARTIFACTS_MISSING"
        assert test_state.state.active_model_profile_id is None
