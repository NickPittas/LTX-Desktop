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

    def test_hdr_generates_with_proper_artifacts(self, client, test_state, create_fake_model_files):
        """HDR adapter dispatches to the HDR path and succeeds when artifacts are present."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        # Create HDR LoRA at canonical adapters/ path.
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        models_dir = test_state.config.default_models_dir
        adapters_dir = models_dir / "adapters"
        adapters_dir.mkdir(parents=True, exist_ok=True)
        hdr_lora = adapters_dir / OFFICIAL_LTX23_ADAPTERS["hdr"].filename
        hdr_lora.write_bytes(b"\x00" * 1024)

        # Create synthetic scene embeddings (valid safetensors with both
        # video_context AND audio_context). HDR is video-only, so even when an
        # audio_context is present it must NOT reach the pipeline.
        from safetensors.torch import save_file
        import torch as _torch
        scene_emb = adapters_dir / OFFICIAL_LTX23_ADAPTERS["hdr_scene_embeddings"].filename
        save_file(
            {
                "video_context": _torch.zeros(1, 768, dtype=_torch.float32),
                "audio_context": _torch.zeros(1, 768, dtype=_torch.float32),
            },
            str(scene_emb),
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "HDR test",
                "images": [],
                "adapter_id": "hdr",
                "num_frames": 9,
                "height": 512,
                "width": 512,
                "frame_rate": 24,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert "_exr" in data["video_path"]  # EXR output forced
        # Lane A contract: proxy_path must be non-null for EXR primaries
        # (forwarded to the encoder; Lane B owns the tonemap policy inside
        # MediaEncoder proxy generation).
        assert data["proxy_path"] is not None
        assert data["proxy_path"].endswith("_exr_proxy.mp4")

        # Verify HDR-specific parameters reached the pipeline.
        from tests.fakes.services import FakeIcLoraPipeline
        singleton = FakeIcLoraPipeline._singleton
        assert singleton is not None
        assert len(singleton.generate_calls) == 1
        call = singleton.generate_calls[0]
        # Non-null proxy_path forwarded to the pipeline/encoder.
        assert call["proxy_path"] == data["proxy_path"]
        # Scene embeddings threaded into inference path.
        assert "hdr_video_context" in call
        assert call["hdr_video_context"] is not None
        # HDR is video-only: audio_context must be dropped even though the
        # synthetic embeddings file above contains one.
        assert call.get("hdr_audio_context") is None
        # LogC3 → linear postprocess applied before encode.
        assert "output_postprocess" in call
        assert call["output_postprocess"] is not None
        # Linear EXR passthrough (no EOTF).
        assert call.get("input_colorspace") is not None
        assert call["input_colorspace"].transfer == "linear"

    def test_hdr_missing_scene_embeddings_returns_400(
        self, client, test_state, create_fake_model_files
    ):
        """HDR without scene embeddings returns a clear 400 error."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        # Create HDR LoRA but NOT scene embeddings.
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        models_dir = test_state.config.default_models_dir
        adapters_dir = models_dir / "adapters"
        adapters_dir.mkdir(parents=True, exist_ok=True)
        hdr_lora = adapters_dir / OFFICIAL_LTX23_ADAPTERS["hdr"].filename
        hdr_lora.write_bytes(b"\x00" * 1024)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "HDR test",
                "images": [],
                "adapter_id": "hdr",
                "num_frames": 9,
                "height": 512,
                "width": 512,
                "frame_rate": 24,
            },
        )
        assert_http_error(
            response, status_code=400, code="HTTP_400",
            message="HDR scene embeddings not found. Configure hdr_scene_embeddings adapter path or install the file.",
        )

    def test_hdr_requires_prompt(self, client, test_state, create_fake_model_files):
        """HDR adapter requires a non-blank prompt."""
        create_fake_model_files()

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "",
                "images": [],
                "adapter_id": "hdr",
                "num_frames": 9,
                "height": 512,
                "width": 512,
                "frame_rate": 24,
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="Prompt is required for HDR adapter")

    def test_hdr_works_with_kijai_gguf_profile(self, client, test_state, create_fake_model_files):
        """HDR must work with Kijai/GGUF profiles, not only official checkpoints.

        Verifies structurally that HDR reuses the same profile-aware pipeline
        loading (load_ic_lora → ResolvedLtxComponents) as non-HDR IC-LoRA,
        not a separate official-only path.
        """
        create_fake_model_files()

        # Create a profile with a custom (Kijai-style) transformer path.
        from api_types import ModelComponentPaths, ModelProfilePayload
        from pathlib import Path as _Path
        models_dir = test_state.config.default_models_dir
        kijai_transformer = models_dir / "diffusion_models" / "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors"
        kijai_transformer.parent.mkdir(parents=True, exist_ok=True)
        kijai_transformer.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="kijai-hdr-profile",
            name="Kijai HDR Profile",
            source="kijai",
            components=ModelComponentPaths(
                transformer=str(kijai_transformer),
                transformer_format="official_safetensors",
                transformer_quantization="fp8_input_scaled",
                upsampler=str(models_dir / "latent_upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                text_encoder_format="api",
            ),
            capabilities=["t2v"],
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = profile.id

        # Create HDR LoRA + scene embeddings at adapters/.
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        from safetensors.torch import save_file
        import torch as _torch
        adapters_dir = models_dir / "adapters"
        adapters_dir.mkdir(parents=True, exist_ok=True)
        (adapters_dir / OFFICIAL_LTX23_ADAPTERS["hdr"].filename).write_bytes(b"\x00" * 1024)
        save_file(
            {"video_context": _torch.zeros(1, 768, dtype=_torch.float32)},
            str(adapters_dir / OFFICIAL_LTX23_ADAPTERS["hdr_scene_embeddings"].filename),
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "prompt": "Kijai HDR test",
                "images": [],
                "adapter_id": "hdr",
                "num_frames": 9,
                "height": 512,
                "width": 512,
                "frame_rate": 24,
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"

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

        (ctx,) = wrapper(["prompt"], enhance_first_prompt=True, streaming_prefetch_count=None)

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

    def test_none_audio_propagates_streaming_prefetch(self):
        """Fallback call forwards caller kwargs (e.g. streaming_prefetch_count)."""
        import torch

        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
            _HDRPromptEncoderWrapper,
        )

        class _FakeRealCtx:
            def __init__(self) -> None:
                self.video_encoding = torch.zeros(1, 1, 1)
                self.audio_encoding = torch.zeros(1, 8, 8)

        class _FakeRealEncoder:
            def __call__(self, prompts, **kwargs):
                self.last_kwargs = kwargs
                return (_FakeRealCtx(),)

        enc = _FakeRealEncoder()
        wrapper = _HDRPromptEncoderWrapper(
            video_context=torch.zeros(1, 1024, 4096),
            audio_context=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
            original_encoder=enc,
        )
        wrapper(["p"], enhance_first_prompt=False, streaming_prefetch_count=3)
        assert enc.last_kwargs.get("streaming_prefetch_count") == 3

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
