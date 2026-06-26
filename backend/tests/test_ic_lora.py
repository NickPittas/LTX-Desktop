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
                official_adapters={"ingredients": str(adapter_file)},
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
                "adapter_id": "ingredients",
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
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={
                    "ingredients": str(adapter_file),
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
                "adapter_id": "ingredients",
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
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="gguf-profile",
            name="GGUF Profile",
            source="official",
            components=ModelComponentPaths(
                transformer="/fake/path/model.safetensors",
                text_encoder_root="/fake/text/encoder",
                text_encoder_format="hf_folder",
                official_adapters={"ingredients": str(adapter_file)},
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
                "adapter_id": "ingredients",
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
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        # Profile with official_adapters — no legacy IC-LoRA checkpoints under models_dir
        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={
                    "union_control": str(union_file),
                    "ingredients": str(adapter_file),
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
                "adapter_id": "ingredients",
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
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={"ingredients": str(adapter_file)},
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
                "adapter_id": "ingredients",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert fake_services.ic_lora_pipeline.last_lora_paths == [str(adapter_file)]

    def test_canny_with_adapter_loads_union_then_ingredients(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image
    ):
        """Canny + ingredients: loads union first then ingredients, passes a control video."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        adapter_dir = test_state.config.default_models_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        union_file = adapter_dir / "union_control.safetensors"
        union_file.write_bytes(b"\x00" * 1024)
        adapter_file = adapter_dir / "ingredients.safetensors"
        adapter_file.write_bytes(b"\x00" * 1024)

        profile = ModelProfilePayload(
            id="test-profile",
            name="Test Profile",
            source="official",
            components=ModelComponentPaths(
                official_adapters={
                    "union_control": str(union_file),
                    "ingredients": str(adapter_file),
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
                "adapter_id": "ingredients",
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

    def test_hdr_returns_400_unavailable(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
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
                "adapter_id": "hdr",
            },
        )
        assert_http_error(response, status_code=400, code="HTTP_400",
                          message="HDR workflow requires HDR scene embeddings and tone-mapping pipeline which is not wired yet")

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
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        adapter_path = test_state.config.default_models_dir / adapter.filename
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

    def test_default_forwards_6(self, client, test_state,
                                create_fake_model_files, create_fake_ic_lora_files,
                                fake_services):
        """Default laplacian_blend_grow=6 forwarded to pipeline generate_inpaint."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        assert kwargs.get("laplacian_blend_grow") == 6, (
            f"Expected default laplacian_blend_grow=6, got {kwargs.get('laplacian_blend_grow')!r}"
        )

    def test_custom_laplacian_blend_grow(self, client, test_state,
                                         create_fake_model_files, fake_services):
        """Custom laplacian_blend_grow forwarded to pipeline."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        adapter_path = test_state.config.default_models_dir / adapter.filename
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


class TestIcLoraResolution:
    """All IC-LoRA workflows preserve input resolution (aligned to 64)."""

    def test_in_outpainting_1920x1080_preserves_resolution(self, client, test_state,
                                                            create_fake_model_files,
                                                            fake_services):
        """1920x1080 input in_outpainting calls generate_inpaint with 1920x1088."""
        create_fake_model_files()
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS
        adapter = OFFICIAL_LTX23_ADAPTERS["in_outpainting"]
        adapter_path = test_state.config.default_models_dir / adapter.filename
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
        assert kwargs["height"] == 1088, f"Expected height=1088, got {kwargs['height']}"

    def test_non_inpaint_preserves_aligned_input_dimensions(self, client, test_state,
                                                              create_fake_model_files,
                                                              fake_services):
        """Non-in_outpainting adapter also preserves aligned input dimensions."""
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.ic_lora_pipeline.bind_singleton(fake_services.ic_lora_pipeline)

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
        test_state.video_processor.register_video(
            str(video_path),
            FakeCapture(frames=["frame-a", "frame-b"], width=1920, height=1080),
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "prompt": "test prompt",
                "images": [],
                "adapter_id": "water_simulation",
            },
        )
        assert response.status_code == 200, f"Unexpected status: {response.json()}"
        kwargs = fake_services.ic_lora_pipeline.generate_calls[0]
        # Non-inpaint now also preserves aligned input resolution
        assert kwargs["width"] == 1920, f"Expected width=1920, got {kwargs['width']}"
        assert kwargs["height"] == 1088, f"Expected height=1088, got {kwargs['height']}"
