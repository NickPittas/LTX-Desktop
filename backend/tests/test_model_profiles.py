"""Integration-style tests for model profile endpoints."""

from __future__ import annotations

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

    def test_activate_valid_profile(self, client, tmp_path):
        model_file = tmp_path / "model.safetensors"
        upscaler_file = tmp_path / "upsampler.safetensors"
        model_file.write_bytes(b"model")
        upscaler_file.write_bytes(b"upscaler")

        response = client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="valid-profile",
                components={
                    "transformer": str(model_file),
                    "upsampler": str(upscaler_file),
                    "text_encoder_format": "api",
                },
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

    def test_delete_profile(self, client):
        client.post("/api/model-profiles", json=_make_official_payload(profile_id="deletable"), headers=_ADMIN_HEADERS)

        response = client.delete("/api/model-profiles/deletable", headers=_ADMIN_HEADERS)
        assert response.status_code == 200

        response = client.get("/api/model-profiles", headers=_ADMIN_HEADERS)
        assert len(response.json()["profiles"]) == 0

    def test_delete_deactivates_if_active(self, client, tmp_path):
        model_file = tmp_path / "model.safetensors"
        upscaler_file = tmp_path / "upsampler.safetensors"
        model_file.write_bytes(b"model")
        upscaler_file.write_bytes(b"upscaler")

        profile = _make_official_payload(
            profile_id="active-delete",
            components={
                "transformer": str(model_file),
                "upsampler": str(upscaler_file),
                "text_encoder_format": "api",
            },
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


class TestRecommendationWithProfile:
    def test_ltx_recommendation_ok_when_valid_official_profile_active(self, client, tmp_path):
        model_file = tmp_path / "model.safetensors"
        upscaler_file = tmp_path / "upsampler.safetensors"
        model_file.write_bytes(b"model")
        upscaler_file.write_bytes(b"upscaler")

        client.post(
            "/api/model-profiles",
            json=_make_official_payload(
                profile_id="official-ready",
                components={
                    "transformer": str(model_file),
                    "upsampler": str(upscaler_file),
                    "text_encoder_format": "api",
                },
            ),
            headers=_ADMIN_HEADERS,
        )
        client.post("/api/model-profiles/official-ready/activate", headers=_ADMIN_HEADERS)

        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
