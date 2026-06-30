"""Integration-style tests for IC-LoRA endpoints."""

from __future__ import annotations

from pathlib import Path

from api_types import ModelComponentPaths, ModelProfilePayload
from runtime_config.model_download_specs import DEPTH_PROCESSOR_CP_ID, resolve_model_path
from tests.http_error_assertions import assert_http_error
from tests.fakes import FakeCapture


class TestIcLoraExtractConditioning:
    def test_canny_extraction(self, client, test_state):
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "canny", "frame_time": 0},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["conditioning_type"] == "canny"
        assert payload["conditioning"].startswith("data:image/jpeg;base64,")

    def test_depth_extraction(self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "depth", "frame_time": 0},
        )
        assert response.status_code == 200
        assert response.json()["conditioning_type"] == "depth"
        assert fake_services.depth_processor_pipeline.apply_calls == ["frame-a"]

    def test_depth_extraction_requires_downloaded_ltx_model(self, client, test_state):
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "depth", "frame_time": 0},
        )
        assert_http_error(response, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")


class TestIcLoraGenerate:
    def test_happy_path(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert Path(response.json()["video_path"]).exists()

    def test_generate_prores_output_format(self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
        """Phase 2a: IC-LoRA generate threads output_format/proxy_path/encoder."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "output_format": "prores_422_hq",
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["video_path"].endswith(".mov")
        assert data["proxy_path"] is not None
        assert data["proxy_path"].endswith("_proxy.mp4")

        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert str(call["output_format"]) == "OutputFormat.PRORES_422_HQ"
        assert call["proxy_path"] == data["proxy_path"]
        assert call["encoder"] is fake_services.media_encoder

    def test_adapter_id_from_active_profile(self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        # Create adapter file
        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        # Set up profile with official_adapters pointing to adapter file
        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"water_simulation": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        # Verify adapter path was routed to pipeline create()
        assert fake_services.ic_lora_pipeline.last_lora_path == str(adapter_file)

    def test_multiple_images_forwarded(
        self, client, test_state, create_fake_model_files, create_fake_ic_lora_files, fake_services
    ):
        """Prove N>1 images all forward to pipeline with correct order."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [
                    {"path": "/fake/img1.png", "frame": 0, "strength": 0.5},
                    {"path": "/fake/img2.png", "frame": 1, "strength": 1.0},
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        recorded_images = fake_services.ic_lora_pipeline.generate_calls[0]["images"]
        assert len(recorded_images) == 2
        assert recorded_images[0].path == "/fake/img1.png"
        assert recorded_images[0].frame_idx == 0
        assert recorded_images[0].strength == 0.5
        assert recorded_images[1].path == "/fake/img2.png"
        assert recorded_images[1].frame_idx == 1
        assert recorded_images[1].strength == 1.0

    def test_canny_uses_active_profile_union_control(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True
        adapter_path = tmp_path / "union-control.safetensors"
        adapter_path.write_bytes(b"fake")
        profile = ModelProfilePayload(
            id="profile-with-union",
            name="Profile With Union",
            source="official",
            components=ModelComponentPaths(official_adapters={"union_control": str(adapter_path)}),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = profile.id

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_path == str(adapter_path)

    def test_canny_with_adapter_no_depth_processor(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """Canny generate with adapter_id works without depth processor on disk."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        # Create adapter files — NO depth processor
        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        union_file = adapter_dir / "union_control.safetensors"
        union_file.write_bytes(b"\x00" * 1024)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={
                    "water_simulation": str(adapter_file),
                    "union_control": str(union_file),
                },
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_path == str(adapter_file)

    def test_mask_path_forwards_to_pipeline(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files, fake_services):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "mask_path": str(mask_path),
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.generate_calls[0]["mask_path"] == str(mask_path)

    def test_conditioning_strength_forwards_to_pipeline(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files, fake_services):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "mask_path": str(mask_path),
                "conditioning_strength": 0.35,
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.generate_calls[0]["conditioning_strength"] == 0.35

    def test_mask_path_missing_returns_400(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "mask_path": "/nonexistent/mask.mp4",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400", message="Mask not found: /nonexistent/mask.mp4")

    def test_lora_strength_default_1_0(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files, fake_services):
        """Default lora_strength=1.0 reaches pipeline create."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
            },
        )
        assert response.status_code == 200
        assert fake_services.ic_lora_pipeline.last_lora_strength == 1.0

    def test_lora_strength_explicit_forwards(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files, fake_services):
        """Explicit lora_strength=0.5 reaches pipeline create."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "lora_strength": 0.5,
            },
        )
        assert response.status_code == 200
        assert fake_services.ic_lora_pipeline.last_lora_strength == 0.5

    def test_lora_strength_change_causes_reload(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files, fake_services):
        """Changing lora_strength triggers pipeline rebuild (cache miss)."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        # First call with default lora_strength (1.0)
        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
            },
        )
        assert response.status_code == 200
        assert fake_services.ic_lora_pipeline.last_lora_strength == 1.0

        # Second call with different lora_strength should force reload
        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "lora_strength": 0.75,
            },
        )
        assert response.status_code == 200
        assert fake_services.ic_lora_pipeline.last_lora_strength == 0.75

    def test_adapter_id_non_ic_lora_returns_error(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "distilled_lora_384",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400", message="Adapter distilled_lora_384 is not an IC-LoRA adapter (kind=distilled_lora)")

    def test_adapter_id_unknown_returns_error(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "nonexistent_adapter",
            },
        )
        assert response.status_code == 422
        assert "adapter_id" in response.text

    def test_active_profile_transformer_without_official_model(self, client, test_state, fake_services, create_fake_ic_lora_files):
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        # Active profile with transformer path — no official model files created
        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="gguf-profile",
            name="GGUF Profile",
            source="official",
            components=ModelComponentPaths(
                transformer="/fake/path/model.safetensors",
                text_encoder_root="/fake/text/encoder",
                text_encoder_format="hf_folder",
                official_adapters={"water_simulation": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "gguf-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_path == str(adapter_file)

    def test_adapter_id_no_legacy_loras_succeeds(self, client, test_state, fake_services, create_fake_model_files):
        """adapter_id from active profile should not require legacy IC-LoRA checkpoint under models_dir."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        # Create depth processor folder (still required) but NOT legacy IC-LoRA checkpoints
        depth_path = resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.mkdir(parents=True, exist_ok=True)
        (depth_path / "config.json").write_text("{}", encoding="utf-8")

        # Create adapter files
        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        union_file = adapter_dir / "union_control.safetensors"
        union_file.write_bytes(b"\x00" * 1024)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        # Profile with official_adapters — no legacy IC-LoRA checkpoints under models_dir
        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={
                    "union_control": str(union_file),
                    "water_simulation": str(adapter_file),
                },
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_path == str(adapter_file)

    def test_no_conditioning_only_adapter_skips_preprocessing(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image
    ):
        """conditioning_type=None with adapter loads only adapter, no canny/depth preproc."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"water_simulation": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_paths == [str(adapter_file)]

    def test_canny_with_adapter_loads_union_then_water_simulation(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image
    ):
        """Canny + adapter: loads union first then adapter, passes a control video."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        union_file = adapter_dir / "union_control.safetensors"
        union_file.write_bytes(b"\x00" * 1024)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={
                    "union_control": str(union_file),
                    "water_simulation": str(adapter_file),
                },
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_paths == [str(union_file), str(adapter_file)]

    def test_canny_no_adapter_loads_only_union(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image
    ):
        """Canny with no adapter: loads only union_control and passes a control video."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        union_file = adapter_dir / "union_control.safetensors"
        union_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"union_control": str(union_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_paths == [str(union_file)]


class TestIcLoraWorkflowGating:
    """Workflow-aware validation for IC-LoRA adapter dispatch."""

    def test_in_outpainting_requires_mask_path(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        """Inpaint requires mask_path (outpaint remains gated via mask_path check)."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "in_outpainting",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="In/outpainting requires a mask_path")

    def test_ingredients_requires_images(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        # video_path provided but will be ignored — ingredients no longer requires it
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "ingredients",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="Ingredients adapter requires at least one image in images[]")

    def test_ingredients_without_video_path_ok(self, client, test_state,
                                                create_fake_model_files,
                                                fake_services):
        """Ingredients T2V works without video_path; no video opened, uses request dims."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="ingredients-profile",
            name="Ingredients Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"ingredients": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "ingredients-profile"

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "ingredients",
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        assert response.json()["status"] == "complete"

        # No video was opened
        assert test_state.video_processor.open_video_calls == [], (
            f"Expected no open_video calls, got: {test_state.video_processor.open_video_calls}"
        )

        # Generate called with video_conditioning=[] and images
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs.get("video_conditioning") == [], (
            f"Expected video_conditioning=[], got {kwargs.get('video_conditioning')!r}"
        )
        assert len(kwargs.get("images", [])) == 1
        assert kwargs["images"][0].path == "/fake/img.png"

        # Default T2V dims
        assert kwargs["width"] == 704
        assert kwargs["height"] == 1280
        assert kwargs["num_frames"] == 121
        assert kwargs["frame_rate"] == 24.0

    def test_non_ingredients_without_video_path_returns_400(self, client, test_state,
                                                             create_fake_model_files,
                                                             create_fake_ic_lora_files):
        """Standard adapter without video_path returns 400, not 500."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="ws-profile",
            name="WaterSim Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"water_simulation": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "ws-profile"

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="video_path is required for this adapter")

    def test_ingredients_with_conditioning_type_returns_400(self, client, test_state,
                                                             create_fake_model_files,
                                                             create_fake_ic_lora_files):
        """Ingredients rejects conditioning_type."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="ingredients-profile",
            name="Ingredients Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"ingredients": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "ingredients-profile"

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "ingredients",
                "conditioning_type": "canny",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="Ingredients adapter is image-only; omit conditioning_type")

    def test_ingredients_num_frames_snaps_valid(self, client, test_state,
                                                 create_fake_model_files,
                                                 fake_services):
        """Invalid num_frames (not 1+8k) is snapped to valid LTX frame count."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="ingredients-profile",
            name="Ingredients Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"ingredients": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "ingredients-profile"

        # 90 is not 1+8k; expected snap: 1 + 8 * ((90-1)//8) = 1 + 8*11 = 89
        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "ingredients",
                "num_frames": 90,
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs["num_frames"] == 89, (
            f"Expected num_frames snapped to 89, got {kwargs['num_frames']}"
        )

    def test_lipdub_returns_400_unavailable(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "lipdub",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="LipDub requires audio conditioning and lip-sync pipeline which is not wired yet")

    def _setup_ingredients_api_encoding(self, test_state, fake_services, create_fake_model_files):
        """Wire API text encoding for an ingredients (image-only) workflow."""
        from types import SimpleNamespace

        create_fake_model_files()
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = False
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)
        # prepare_text_encoding → encode_via_api must return a truthy result.
        fake_services.text_encoder.encode_responses.append(
            SimpleNamespace(video_context="fake_tensor", audio_context=None)
        )

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="ingredients-profile",
            name="Ingredients Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"ingredients": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "ingredients-profile"

    def test_ingredients_image_workflow_uses_i2v_enhancer_setting(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """Ingredients (image-only) workflow consults prompt_enhancer_enabled_i2v, not t2v.

        With i2v ON and t2v OFF, enhancement must still be requested (i2v wins)
        — proving image workflows no longer silently use the T2V setting.
        """
        self._setup_ingredients_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "ingredients",
            },
        )
        assert response.status_code == 200, response.text

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_ingredients_image_workflow_ignores_t2v_enhancer_setting(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """Image workflow must not consult t2v: with i2v OFF and t2v ON, no enhancement."""
        self._setup_ingredients_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = False
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "ingredients",
            },
        )
        assert response.status_code == 200, response.text

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def _setup_hdr_artifacts(self, test_state, *, include_scene_embeddings: bool = True):
        """Install HDR LoRA (+ optional scene embeddings) at canonical paths.

        Returns ``(models_dir, adapters_dir, hdr_lora_path, scene_emb_path)``.
        """
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        from safetensors.torch import save_file
        import torch as _torch

        models_dir = test_state.config.default_models_dir
        adapters_dir = models_dir / "adapters"
        adapters_dir.mkdir(parents=True, exist_ok=True)
        hdr_lora = adapters_dir / OFFICIAL_LTX23_ADAPTERS["hdr"].filename
        hdr_lora.write_bytes(b"\x00" * 1024)
        scene_emb = adapters_dir / OFFICIAL_LTX23_ADAPTERS["hdr_scene_embeddings"].filename
        if include_scene_embeddings:
            save_file(
                {"video_context": _torch.zeros(1, 768, dtype=_torch.float32)},
                str(scene_emb),
            )
        return models_dir, adapters_dir, str(hdr_lora), str(scene_emb)

    def test_hdr_requires_video_path(self, client, test_state, create_fake_model_files):
        """HDR is a V2V workflow: missing video_path returns 400 (no early T2V dispatch)."""
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "HDR test",
                "images": [],
                "adapter_id": "hdr",
            },
        )
        assert_http_error(
            response, status_code=400, code="HTTP_400",
            message="video_path is required for this adapter",
        )

    def test_hdr_ignores_conditioning_type(self, client, test_state, fake_services, create_fake_model_files):
        """HDR ignores conditioning_type (the source video is the guide) — request succeeds."""
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a"] * 9, width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR test",
                "images": [],
                "adapter_id": "hdr",
                "conditioning_type": "canny",
            },
        )
        assert response.status_code == 200, response.text
        # conditioning_type is ignored (not forwarded): HDR is handled by the
        # dedicated HDR pipeline, not the generic IC-LoRA pipeline.
        assert fake_services.ic_lora_pipeline.generate_calls == []
        assert len(fake_services.hdr_ic_lora_pipeline.generate_calls) == 1

    def test_hdr_ignores_images(self, client, test_state, fake_services, create_fake_model_files):
        """HDR ignores images (the source video is the guide) — request succeeds."""
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a"] * 9, width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR test",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "hdr",
            },
        )
        assert response.status_code == 200, response.text
        # images are ignored (not forwarded): HDR is handled by the dedicated
        # HDR pipeline, not the generic IC-LoRA pipeline.
        assert fake_services.ic_lora_pipeline.generate_calls == []
        assert len(fake_services.hdr_ic_lora_pipeline.generate_calls) == 1

    def test_hdr_rejects_model_selection_for_non_hdr(
        self, client, test_state, create_fake_model_files, create_fake_ic_lora_files
    ):
        """Non-HDR IC-LoRA must reject model_selection with HTTP 400."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a", "b"])
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "model_selection": "ltx-2.3-22b-distilled",
            },
        )
        assert_http_error(
            response, status_code=400, code="HTTP_400",
            message="model_selection is supported only for HDR IC-LoRA",
        )

    def test_hdr_rejects_sequence_input(self, client, test_state, create_fake_model_files, tmp_path):
        """HDR sequence inputs are gated: official load_video_conditioning_hdr decodes
        via PyAV which cannot open a single EXR/PNG-seq frame. Reject with HTTP 400
        until the sequence → HDR conditioning adapter lands."""
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        # A sequence file (single .exr frame) — is_sequence_file() returns True
        # for supported sequence extensions even though the file exists.
        seq_dir = test_state.config.outputs_dir / "seq"
        seq_dir.mkdir(parents=True, exist_ok=True)
        seq_file = seq_dir / "shot_0001.exr"
        seq_file.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(seq_file),
                "prompt": "HDR sequence test",
                "images": [],
                "adapter_id": "hdr",
            },
        )
        assert response.status_code == 400
        msg = response.json()["message"].lower()
        assert ".mp4" in msg and ".mov" in msg, (
            f"Sequence gate message must name .mp4/.mov, got: {msg!r}"
        )

    def test_hdr_uses_source_video_metadata_and_conditioning(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """HDR success contract: source-video-driven metadata + dedicated HDR pipeline.

        Verifies:
        - Source video dims/fps drive the pipeline (request width/height/fps/num_frames ignored).
        - num_frames snapped DOWN to nearest 1+8k ≤ source frame count.
        - output_format forced to EXR_ZIP_HALF; proxy_path non-null.
        - source_video_path threaded as the IC-LoRA guide.
        - No text encoder call (scene embeddings replace prompt encoding).
        """
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        # Source video: 1920x1080, 30fps, 25 frames. num_frames must snap to
        # 1 + 8*((25-1)//8) = 1 + 8*3 = 25. Height 1080 → aligned to 1088.
        video_path = test_state.config.outputs_dir / "hdr_src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path),
            FakeCapture(frames=[f"f{i}" for i in range(25)], width=1920, height=1080, fps=30.0),
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR test",
                "images": [],
                "adapter_id": "hdr",
                # Request dims/fps/num_frames must be IGNORED for HDR.
                "num_frames": 9,
                "height": 512,
                "width": 512,
                "frame_rate": 24,
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "complete"
        assert "_exr" in data["video_path"]  # EXR forced
        assert data["proxy_path"] is not None
        assert data["proxy_path"].endswith("_exr_proxy.mp4")

        # Dedicated HDR pipeline received the call (NOT the generic IC-LoRA one).
        assert fake_services.ic_lora_pipeline.generate_calls == [], (
            "HDR must not dispatch through the generic IC-LoRA pipeline"
        )
        assert len(fake_services.hdr_ic_lora_pipeline.generate_calls) == 1
        call = fake_services.hdr_ic_lora_pipeline.generate_calls[0]
        # Source video threaded as the IC-LoRA guide.
        assert call["source_video_path"] == str(video_path)
        # Source-derived ORIGINAL dims (request 512x512 ignored). The handler
        # passes the original source dims (1920x1080); the pipeline internally
        # aligns to 64 (1088) for generation and crops back to 1080.
        assert call["width"] == 1920
        assert call["height"] == 1080
        assert call["frame_rate"] == 30.0
        # num_frames snapped from source frame count (25), not request (9).
        assert call["num_frames"] == 25
        # EXR primary + non-null proxy.
        assert str(call["output_format"]) == "OutputFormat.EXR_ZIP_HALF"
        assert call["proxy_path"] == data["proxy_path"]
        # No text encoder call (scene embeddings replace prompt encoding).
        assert fake_services.text_encoder.encode_calls == []

    def test_hdr_does_not_reject_short_source_video(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """HDR never rejects by frame count — a 2-frame source succeeds (wrapper pads in memory)."""
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "short.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a", "b"], width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR test",
                "images": [],
                "adapter_id": "hdr",
            },
        )
        assert response.status_code == 200, response.text
        # No handler-side frame-count rejection. num_frames is passed only as
        # an advisory legacy arg (== source frame count); in-memory padding to
        # 8n+1 is wrapper-owned (the fake does not pad, so it is not asserted).
        assert len(fake_services.hdr_ic_lora_pipeline.generate_calls) == 1
        assert fake_services.hdr_ic_lora_pipeline.generate_calls[0]["num_frames"] == 2

    def test_hdr_rejects_dev_base(
        self, client, test_state, create_fake_model_files, tmp_path
    ):
        """HDR initial support is distilled-only — a dev base is rejected."""
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        # Dev profile with NO distilled LoRA installed.
        dev_transformer = tmp_path / "dev.safetensors"
        dev_transformer.write_bytes(b"\x00" * 1024)
        upsampler = tmp_path / "ups.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)
        profile = ModelProfilePayload(
            id="dev-no-lora",
            name="Dev No LoRA",
            source="official",
            components=ModelComponentPaths(
                transformer=str(dev_transformer),
                transformer_format="official_safetensors",
                upsampler=str(upsampler),
                text_encoder_format="api",
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = profile.id

        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a"] * 17, width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR dev test",
                "images": [],
                "adapter_id": "hdr",
            },
        )
        # Dev base is not supported for HDR initial support — explicit reject
        # with an actionable code (no silent fallback).
        assert response.status_code == 409
        payload = response.json()
        assert payload["code"] == "UNSUPPORTED_MODEL_BASE_FAMILY"

    def test_hdr_distilled_selected_base_does_not_require_distilled_lora(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """Distilled base family runs HDR WITHOUT any distilled LoRA."""
        create_fake_model_files()
        _models, _adapters, _hdr_lora, scene_emb_path = self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        # Default active profile is None → downloaded distilled bundle. The
        # official distilled monolith is a distilled base; no distilled LoRA
        # is installed and none is required.
        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a"] * 17, width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR distilled test",
                "images": [],
                "adapter_id": "hdr",
            },
        )
        assert response.status_code == 200, response.text
        # Dedicated HDR pipeline loaded with base_family=distilled, no distilled LoRA.
        assert fake_services.hdr_ic_lora_pipeline.last_base_family == "distilled"
        assert fake_services.hdr_ic_lora_pipeline.last_distilled_lora_path is None
        # scene_embeddings_path is forwarded into the HDR pipeline create().
        assert fake_services.hdr_ic_lora_pipeline.last_scene_embeddings_path == scene_emb_path

    def test_hdr_model_selection_threads_to_component_resolver(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        """HDR threads model_selection through the component resolver.

        With an active profile + a selected base, ``_resolve_active_components``
        returns a ``ResolvedLtxComponents`` carrying the selection (selected_cp_id
        appears in the cache key) so the HDR pipeline ``create()`` receives it.
        """
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        # Active profile so _resolve_active_components(model_selection) returns
        # a non-None ResolvedLtxComponents carrying the selection metadata.
        transformer = tmp_path / "distilled.safetensors"
        transformer.write_bytes(b"\x00" * 1024)
        profile = ModelProfilePayload(
            id="hdr-sel-profile",
            name="HDR Selection Profile",
            source="official",
            components=ModelComponentPaths(
                transformer=str(transformer),
                transformer_format="official_safetensors",
                text_encoder_format="api",
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "hdr-sel-profile"

        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a"] * 17, width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR selection test",
                "images": [],
                "adapter_id": "hdr",
                "model_selection": "ltx-2.3-22b-distilled",
            },
        )
        assert response.status_code == 200, response.text
        # The HDR pipeline create() received components carrying the selection.
        components = fake_services.hdr_ic_lora_pipeline.last_components
        assert components is not None
        assert "model_selection" in components.cache_key
        assert "ltx-2.3-22b-distilled" in components.cache_key

    def test_hdr_rejects_kijai_or_split_selection(
        self, client, test_state, create_fake_model_files, tmp_path
    ):
        """HDR initial support rejects Kijai/split-safetensors selections (distilled-only)."""
        create_fake_model_files()
        self._setup_hdr_artifacts(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        models_dir = test_state.config.default_models_dir
        # Kijai-style split distilled transformer at its canonical registry path.
        kijai_rel = (
            "diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors"
        )
        kijai_path = models_dir / kijai_rel
        kijai_path.parent.mkdir(parents=True, exist_ok=True)
        kijai_path.write_bytes(b"FP8")

        # Active split-component profile providing the sidecars the Kijai
        # build needs.
        sidecars: dict[str, str] = {}
        for name in ("tp", "vvae", "avae", "ups"):
            p = tmp_path / f"{name}.safetensors"
            p.write_bytes(b"x")
            sidecars[name] = str(p)
        profile = ModelProfilePayload(
            id="kijai-hdr",
            name="Kijai HDR",
            source="kijai",
            components=ModelComponentPaths(
                transformer="/placeholder.safetensors",
                transformer_format="split_safetensors",
                text_projection=sidecars["tp"],
                video_vae=sidecars["vvae"],
                audio_vae=sidecars["avae"],
                upsampler=sidecars["ups"],
                text_encoder_format="api",
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "kijai-hdr"

        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a"] * 17, width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR Kijai test",
                "images": [],
                "adapter_id": "hdr",
                "model_selection": "ltx-2.3-22b-distilled-fp8-kijai-v3",
            },
        )
        # Kijai/split-safetensors is not supported for HDR initial support —
        # explicit reject with an actionable gating code (no silent fallback).
        assert response.status_code == 409, response.text
        payload = response.json()
        assert payload["code"] in ("UNSUPPORTED_MODEL_BASE_FAMILY", "UNSUPPORTED_MODEL_FORMAT"), (
            f"Expected HDR model gating rejection, got {payload!r}"
        )

    def test_hdr_missing_scene_embeddings_returns_400(
        self, client, test_state, create_fake_model_files
    ):
        """HDR without scene embeddings returns a clear 400 error (after video_path validates)."""
        create_fake_model_files()
        # Install HDR LoRA but NOT scene embeddings.
        self._setup_hdr_artifacts(test_state, include_scene_embeddings=False)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "src.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a"] * 9, width=512, height=512)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "HDR test",
                "images": [],
                "adapter_id": "hdr",
            },
        )
        assert_http_error(
            response, status_code=400, code="HTTP_400",
            message="HDR scene embeddings not found. Configure hdr_scene_embeddings adapter path or install the file.",
        )

    def test_hdr_scene_embeddings_returns_400_unavailable(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        """hdr_scene_embeddings is a support asset, not a standalone adapter."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "hdr_scene_embeddings",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="HDR scene embeddings is a support asset for the HDR workflow and cannot be used as a standalone adapter")

    def test_motion_track_control_returns_400_unavailable(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "motion_track_control",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="Motion Track Control requires trajectory/reference video processing which is not wired yet")

class TestIcLoraEmptyPromptWorkflow:
    """in_outpainting allows empty prompt; other adapters reject it."""

    def test_in_outpainting_accepts_empty_prompt(self, client, test_state,
                                                    create_fake_model_files, create_fake_ic_lora_files):
        """in_outpainting allows empty prompt (inpaint may work with just the mask)."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        # Also create the in_outpainting adapter file (not included in default IC-LoRA set)
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
            },
        )
        # Should not return 400 — inpaint now accepts empty prompt
        assert response.status_code != 400, f"Unexpected 400: {response.json()}"

    def test_in_outpainting_blank_prompt_stays_empty(self, client, test_state,
                                                       create_fake_model_files, create_fake_ic_lora_files,
                                                       fake_services):
        """Blank in_outpainting prompt passes empty string to pipeline, not a default."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "  \t  ",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        prompt = fake_services.ic_lora_pipeline.generate_calls[0]["prompt"]
        # Whitespace-only prompt must not get default replacement
        assert prompt == "  \t  ", f"Expected whitespace prompt preserved, got: {prompt!r}"
        assert "empty background, no person" not in prompt, f"Default prompt leaked through: {prompt!r}"

    def test_in_outpainting_explicit_prompt_preserved(self, client, test_state,
                                                       create_fake_model_files, create_fake_ic_lora_files,
                                                       fake_services):
        """Explicit in_outpainting prompt passed unchanged (not replaced by default)."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a red car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        prompt = fake_services.ic_lora_pipeline.generate_calls[0]["prompt"]
        assert prompt == "add a red car", f"Expected explicit prompt preserved, got: {prompt!r}"

    def test_in_outpainting_mask_grow_px_defaults_to_30(self, client, test_state,
                                                          create_fake_model_files, create_fake_ic_lora_files,
                                                          fake_services):
        """Default mask_grow_px is 30 and forwarded to pipeline."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
                # No mask_grow_px — use default
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs.get("mask_grow_px") == 30, f"Expected default mask_grow_px=30, got: {kwargs.get('mask_grow_px')!r}"

    def test_in_outpainting_mask_grow_px_custom(self, client, test_state,
                                                  create_fake_model_files, create_fake_ic_lora_files,
                                                  fake_services):
        """Custom mask_grow_px forwarded to pipeline."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
                "mask_grow_px": 10,
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs.get("mask_grow_px") == 10, f"Expected mask_grow_px=10, got: {kwargs.get('mask_grow_px')!r}"

    def test_standard_adapter_empty_prompt_returns_400(self, client, test_state,
                                                        create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"water_simulation": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "",
                "images": [],
                "adapter_id": "water_simulation",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="Prompt is required for this adapter")


    def test_union_control_requires_conditioning_type(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "union_control",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="Union Control requires conditioning_type (canny or depth)")

    def test_in_outpainting_calls_generate_inpaint(self, client, test_state,
                                                    create_fake_model_files, create_fake_ic_lora_files):
        """in_outpainting now calls generate_inpaint instead of the generic generate."""
        create_fake_model_files()
        create_fake_ic_lora_files()
        # Also create the in_outpainting adapter file
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
            },
        )
        # Should route to generate_inpaint (not return 400)
        assert response.status_code != 400, f"Unexpected 400: {response.json()}"

    def test_standard_video_adapter_without_conditioning_loads_only_adapter(
        self, client, test_state, fake_services, create_fake_model_files,
    ):
        """standard_video adapter (water_simulation) without conditioning loads only the adapter, no union."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"water_simulation": str(adapter_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "test-profile"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_paths == [str(adapter_file)]


class TestIcLoraLaplacianBlendGrow:
    """laplacian_blend_grow separate field forwarded to pipeline for inpaint final blend."""

    def test_default_forwards_12(self, client, test_state,
                                 create_fake_model_files, create_fake_ic_lora_files,
                                 fake_services):
        """Default laplacian_blend_grow=12 and final_mask_blur_px=6 forwarded."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path),
                                                   FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs.get("laplacian_blend_grow") == 12, (
            f"Expected default laplacian_blend_grow=12, got {kwargs.get('laplacian_blend_grow')!r}"
        )
        assert kwargs.get("final_mask_blur_px") == 6, (
            f"Expected default final_mask_blur_px=6, got {kwargs.get('final_mask_blur_px')!r}"
        )

    def test_custom_laplacian_blend_grow(self, client, test_state,
                                         create_fake_model_files, fake_services):
        """Custom laplacian_blend_grow forwarded to pipeline."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path),
                                                   FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
                "laplacian_blend_grow": 3,
                "mask_grow_px": 10,
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs.get("laplacian_blend_grow") == 3, (
            f"Expected laplacian_blend_grow=3, got {kwargs.get('laplacian_blend_grow')!r}"
        )

    def test_independent_of_mask_grow_px(self, client, test_state,
                                         create_fake_model_files, fake_services):
        """laplacian_blend_grow does not affect mask_grow_px."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path),
                                                   FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
                "laplacian_blend_grow": 10,
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs.get("laplacian_blend_grow") == 10, (
            f"Expected laplacian_blend_grow=10, got {kwargs.get('laplacian_blend_grow')!r}"
        )
        assert kwargs.get("mask_grow_px") == 30, (
            f"mask_grow_px should still default to 30, got {kwargs.get('mask_grow_px')!r}"
        )

    def test_custom_final_mask_blur_px(self, client, test_state,
                                        create_fake_model_files, fake_services):
        """Custom final_mask_blur_px=14 forwarded independently of laplacian_blend_grow."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path),
                                                   FakeCapture(frames=["frame-a", "frame-b"]))

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
                "laplacian_blend_grow": 20,
                "final_mask_blur_px": 14,
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs.get("laplacian_blend_grow") == 20, (
            f"Expected laplacian_blend_grow=20, got {kwargs.get('laplacian_blend_grow')!r}"
        )
        assert kwargs.get("final_mask_blur_px") == 14, (
            f"Expected final_mask_blur_px=14, got {kwargs.get('final_mask_blur_px')!r}"
        )


class TestIcLoraResolution:
    """All IC-LoRA workflows preserve input resolution (aligned to 64)."""

    def test_in_outpainting_1920x1080_preserves_resolution(self, client, test_state,
                                                            create_fake_model_files,
                                                            fake_services):
        """1920x1080 input in_outpainting calls generate_inpaint with 1920x1088."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / "adapters" / adapter.filename
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path),
            FakeCapture(frames=["frame-a", "frame-b"], width=1920, height=1080),
        )

        mask_path = test_state.config.outputs_dir / "test_mask.mp4"
        mask_path.write_bytes(b"\x00" * 100)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "add a car",
                "images": [],
                "adapter_id": "in_outpainting",
                "mask_path": str(mask_path),
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs["width"] == 1920, f"Expected width=1920, got {kwargs['width']}"
        assert kwargs["height"] == 1088, f"Expected height=1088 (1080 aligned to 64), got {kwargs['height']}"

    def test_plain_canny_1920x1080_aligns_to_128(self, client, test_state,
                                                  create_fake_model_files,
                                                  fake_services):
        """Plain canny (no adapter_id) also aligns to 128 because union control is loaded."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        union_file = adapter_dir / "union_control.safetensors"
        union_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="canny-only",
            name="Canny Only",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"union_control": str(union_file)},
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "canny-only"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path),
            FakeCapture(frames=["frame-a", "frame-b"], width=1920, height=1080),
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs["width"] == 1920, f"Expected width=1920, got {kwargs['width']}"
        assert kwargs["height"] == 1152, f"Expected height=1152 (1080 aligned to 128), got {kwargs['height']}"

    def test_canny_with_adapter_1920x1080_aligns_to_128(self, client, test_state,
                                                        create_fake_model_files,
                                                        fake_services):
        """Canny + another adapter (ingredients) aligns to 128 because union control is loaded."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        union_file = adapter_dir / "union_control.safetensors"
        union_file.write_bytes(b"\x00" * 1024)
        adapter_file = adapter_dir / "water_simulation.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="canny-plus-adapter",
            name="Canny Plus Adapter",
            source="official",
            components=ModelComponentPaths(
                official_adapters={
                    "union_control": str(union_file),
                    "water_simulation": str(adapter_file),
                },
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "canny-plus-adapter"

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path),
            FakeCapture(frames=["frame-a", "frame-b"], width=1920, height=1080),
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [{"path": "/fake/img.png", "frame": 0, "strength": 1.0}],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        assert kwargs["width"] == 1920, f"Expected width=1920, got {kwargs['width']}"
        assert kwargs["height"] == 1152, f"Expected height=1152 (1080 aligned to 128), got {kwargs['height']}"


class TestHdrPromptEncoderAudioFallback:
    """HDR must never feed an audio modality with context=None into upstream.

    The pinned ``ICLoraPipeline.__call__`` UNCONDITIONALLY builds an audio
    modality from the prompt-encoder's ``audio_encoding``. HDR is video-only,
    so when no explicit ``audio_context`` is supplied the wrapper must borrow a
    valid ``audio_encoding`` from the real prompt encoder — otherwise the
    transformer's audio args preprocessor crashes with::

        AttributeError: 'NoneType' object has no attribute 'view'

    at ``audio_args_preprocessor.prepare(audio, video)``. These tests pin that
    contract directly on the wrapper (the handler-level fake pipeline cannot
    exercise this code path).
    """

    def test_none_audio_falls_back_to_real_encoder(self):
        """audio_context=None borrows a valid audio_encoding from the real encoder."""
        import torch

        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
            _HDRPromptEncoderWrapper,
        )

        video = torch.zeros(1, 1024, 4096, dtype=torch.float32)
        real_audio = torch.ones(1, 64, 4096, dtype=torch.float32)
        calls: list[dict] = []

        class _FakeRealCtx:
            def __init__(self, audio_encoding: torch.Tensor) -> None:
                # video_encoding is intentionally ignored by the wrapper.
                self.video_encoding = torch.full((1, 1024, 4096), -7.0)
                self.audio_encoding = audio_encoding

        class _FakeRealEncoder:
            def __call__(self, prompts: list[str], **kwargs: object):
                calls.append({"prompts": prompts, "kwargs": kwargs})
                return (_FakeRealCtx(real_audio),)

        wrapper = _HDRPromptEncoderWrapper(
            video_context=video,
            audio_context=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
            original_encoder=_FakeRealEncoder(),
        )

        (ctx,) = wrapper(["prompt"], enhance_first_prompt=True)

        # Real encoder invoked exactly once for the audio fallback.
        assert len(calls) == 1, f"expected one fallback call, got {len(calls)}"
        # HDR must disable enhancement on the fallback call (we only need a
        # validly-shaped audio tensor; HDR video comes from scene embeddings).
        assert calls[0]["kwargs"].get("enhance_first_prompt") is False, (
            "HDR audio fallback must force enhance_first_prompt=False"
        )

        # HDR video_encoding always comes from scene embeddings, never the encoder.
        assert torch.equal(ctx.video_encoding, video)
        # audio_encoding must be non-None — this is the crash root cause.
        assert ctx.audio_encoding is not None
        # Borrowed from the real encoder, cast to the wrapper's dtype.
        assert torch.equal(ctx.audio_encoding, real_audio)

    def test_explicit_audio_context_skips_real_encoder(self):
        """When audio_context is supplied, the real encoder must not run."""
        import torch

        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
            _HDRPromptEncoderWrapper,
        )

        video = torch.zeros(1, 1024, 4096, dtype=torch.float32)
        audio = torch.ones(1, 64, 4096, dtype=torch.float32)

        class _ExplodingEncoder:
            def __call__(self, *args, **kwargs):
                raise AssertionError(
                    "real encoder must not run when an explicit audio_context is supplied"
                )

        wrapper = _HDRPromptEncoderWrapper(
            video_context=video,
            audio_context=audio,
            device=torch.device("cpu"),
            dtype=torch.float32,
            original_encoder=_ExplodingEncoder(),
        )

        (ctx,) = wrapper(["prompt"], enhance_first_prompt=False)

        assert ctx.audio_encoding is not None
        assert torch.equal(ctx.audio_encoding, audio)
        assert torch.equal(ctx.video_encoding, video)

    def test_swap_restores_original_encoder_on_exception(self):
        """The original prompt_encoder is restored even if HDR inference raises."""
        import contextlib

        import torch

        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
            _swap_prompt_encoder_for_hdr,
        )

        class _FakePipeline:
            device = torch.device("cpu")  # type: ignore[assignment]
            dtype = torch.float32  # type: ignore[assignment]

        class _FakeEncoder:
            pass

        pipe = _FakePipeline()
        original = _FakeEncoder()
        pipe.prompt_encoder = original

        with contextlib.suppress(RuntimeError):
            with _swap_prompt_encoder_for_hdr(pipe, torch.zeros(1, 1), None):
                assert pipe.prompt_encoder is not original, "wrapper not installed"
                raise RuntimeError("simulated HDR inference failure")

        assert pipe.prompt_encoder is original, "original encoder not restored"

    def test_swap_yields_wrapper_for_hdr_video_only(self):
        """Inside the swap, a None audio_context yields a non-None audio_encoding."""
        import torch

        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
            _swap_prompt_encoder_for_hdr,
        )

        real_audio = torch.ones(1, 8, 8, dtype=torch.float32)

        class _FakeRealCtx:
            def __init__(self) -> None:
                self.video_encoding = torch.zeros(1, 1, 1)
                self.audio_encoding = real_audio

        class _FakePipeline:
            device = torch.device("cpu")  # type: ignore[assignment]
            dtype = torch.float32  # type: ignore[assignment]

        class _FakeEncoder:
            def __call__(self, prompts, **kwargs):
                return (_FakeRealCtx(),)

        pipe = _FakePipeline()
        pipe.prompt_encoder = _FakeEncoder()

        hdr_video = torch.zeros(1, 1024, 4096, dtype=torch.float32)
        with _swap_prompt_encoder_for_hdr(pipe, hdr_video, None):
            (ctx,) = pipe.prompt_encoder(["p"], enhance_first_prompt=False)

        # HDR video from scene embeddings, audio borrowed from real encoder.
        assert torch.equal(ctx.video_encoding, hdr_video)
        assert ctx.audio_encoding is not None
        assert torch.equal(ctx.audio_encoding, real_audio)
