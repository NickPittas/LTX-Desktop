"""Integration-style tests for generation and image endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from api_types import ModelComponentPaths, ModelProfilePayload
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from runtime_config.model_download_specs import resolve_model_path
from state.app_state_types import GpuSlot, VideoPipelineState
from tests.http_error_assertions import assert_http_error
from tests.fakes.services import FakeFastVideoPipeline


@dataclass
class _FakeEncodingResult:
    """Minimal stand-in for TextEncodingResult in tests."""

    video_context: object = "fake_tensor"
    audio_context: object = None

_T2V_JSON = {
    "prompt": "test",
    "resolution": "540p",
    "model": "fast",
    "duration": 5,
    "fps": 24,
}


def _write_test_wav(path: Path, *, duration_seconds: float = 0.1, sample_rate: int = 8000) -> None:
    import wave

    frame_count = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def _enable_local_text_encoding(test_state) -> None:
    test_state.state.app_settings.use_local_text_encoder = True


def _fake_running_generation_state(test_state) -> None:
    pipeline = FakeFastVideoPipeline()
    test_state.state.gpu_slot = GpuSlot(
        active_pipeline=VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
        ),
    )
    test_state.generation.start_generation("running")


class TestGenerate:
    def test_t2v_requires_downloaded_ltx_model(self, client):
        r = client.post("/api/generate", json=_T2V_JSON)
        assert_http_error(r, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_t2v_happy_path(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A beautiful sunset",
                "resolution": "1080p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "cameraMotion": "none",
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None
        assert Path(data["video_path"]).exists()

        pipeline = fake_services.fast_video_pipeline
        assert len(pipeline.generate_calls) == 1

    def test_already_running(self, client, test_state):
        _fake_running_generation_state(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 409

    def test_i2v_nonexistent_image(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": "/no/such/file.png"},
        )
        assert r.status_code == 400

    def test_i2v_rejects_invalid_image_content_400(self, client, test_state, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        bad_image = tmp_path / "bad.png"
        bad_image.write_bytes(b"not-a-real-png")

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": str(bad_image)},
        )
        data = assert_http_error(
            r,
            status_code=400,
            code="HTTP_400",
            message=f"Invalid image file: {bad_image}",
        )
        assert "Invalid image file" in data["message"]

    def test_resolution_mapping_540p(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 960
        assert call["height"] == 512

    def test_resolution_mapping_720p(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json={**_T2V_JSON, "resolution": "720p"})
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 1280
        assert call["height"] == 704

    def test_locked_seed(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 123

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        assert pipeline.generate_calls[0]["seed"] == 123

    def test_error_sets_generation_error(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        fake_services.fast_video_pipeline.raise_on_generate = RuntimeError("GPU OOM")

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 500

        progress = test_state.generation.get_generation_progress()
        assert progress.status == "error"

    def test_cancelled_response(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        fake_services.fast_video_pipeline.raise_on_generate = RuntimeError("cancelled")

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_t2v_active_profile_local_text_encoder_passes_gate(
        self, client, test_state, fake_services, tmp_path
    ):
        """Active profile with local text encoder passes gate without use_local_text_encoder or API key."""
        d = tmp_path / "profile"
        d.mkdir()
        files = {
            "ltx-2.3-22b-distilled.gguf": b"GGUF",
            "tp.safetensors": b"x",
            "ec.safetensors": b"x",
            "vvae.safetensors": b"x",
            "avae.safetensors": b"x",
            "gemma.gguf": b"GEMMA",
            "upsampler.safetensors": b"UPSAMPLER",
        }
        paths = {}
        for name, content in files.items():
            p = d / name
            p.write_bytes(content)
            key = "transformer" if name.startswith("ltx-") else name.rsplit(".", 1)[0]
            paths[key] = str(p)

        profile = ModelProfilePayload(
            id="gguf-profile",
            name="GGUF Profile",
            source="kijai",
            components=ModelComponentPaths(
                transformer=paths["transformer"],
                transformer_format="gguf",
                upsampler=paths["upsampler"],
                text_projection=paths["tp"],
                embeddings_connector=paths["ec"],
                video_vae=paths["vvae"],
                audio_vae=paths["avae"],
                text_encoder_root=paths["gemma"],
                text_encoder_format="gguf",
            ),
        )
        test_state.state.model_profiles = [profile]
        test_state.state.active_model_profile_id = "gguf-profile"
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True

        r = client.post("/api/generate", json=_T2V_JSON)
        # The gate passes — response should be 200, not TEXT_ENCODING_NOT_CONFIGURED
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert fake_services.fast_video_pipeline.last_gemma_root == paths["gemma"]
        assert fake_services.fast_video_pipeline.generate_calls[-1]["enhance_prompt"] is True


class TestA2VGenerate:
    def test_a2v_generation_happy_path(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None
        assert Path(data["video_path"]).exists()

        pipeline = fake_services.a2v_pipeline
        assert len(pipeline.generate_calls) == 1
        call = pipeline.generate_calls[0]
        assert call["audio_path"] == str(audio_file)
        assert call["audio_start_time"] == 0.0
        assert call["audio_max_duration"] is None

    def test_a2v_rejects_missing_audio_file(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": "/no/such/audio.wav",
            },
        )
        assert r.status_code == 400

    def test_a2v_rejects_invalid_audio_content_400(self, client, test_state, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "bad.wav"
        audio_file.write_bytes(b"not-a-real-wav")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )
        data = assert_http_error(
            r,
            status_code=400,
            code="HTTP_400",
            message=f"Invalid audio file: {audio_file}",
        )
        assert "Invalid audio file" in data["message"]

    def test_a2v_forced_api_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(audio_file)
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["audio_uri"] == "storage://uploaded/test_audio.wav"
        assert call["image_uri"] is None
        assert call["model"] == "ltx-2-3-pro"
        assert call["resolution"] == "1920x1080"

    def test_a2v_prefers_api_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(audio_file)
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        assert len(fake_services.a2v_pipeline.generate_calls) == 0

    def test_a2v_prefers_api_without_key_falls_back_to_local(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = ""
        _enable_local_text_encoding(test_state)
        create_fake_model_files()
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 0
        assert len(fake_services.a2v_pipeline.generate_calls) == 1

    def test_a2v_forced_api_routes_to_ltx_api_with_audio_and_image(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video with a still frame",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 2
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(audio_file)
        assert fake_services.ltx_api_client.upload_file_calls[1]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["audio_uri"] == "storage://uploaded/test_audio.wav"
        assert call["image_uri"] == "storage://uploaded/input.png"
        assert call["model"] == "ltx-2-3-pro"
        assert call["resolution"] == "1920x1080"

    def test_a2v_uses_resolution_map(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        for resolution, expected_w, expected_h in [
            ("540p", 960, 576),
            ("720p", 1280, 704),
            ("1080p", 1920, 1088),
        ]:
            fake_services.a2v_pipeline.generate_calls.clear()
            r = client.post(
                "/api/generate",
                json={
                    "prompt": "A music video",
                    "resolution": resolution,
                    "model": "fast",
                    "duration": 5,
                    "fps": 24,
                    "audioPath": str(audio_file),
                },
            )

            assert r.status_code == 200
            call = fake_services.a2v_pipeline.generate_calls[0]
            assert call["width"] == expected_w, f"{resolution}: expected width {expected_w}, got {call['width']}"
            assert call["height"] == expected_h, f"{resolution}: expected height {expected_h}, got {call['height']}"

    def test_a2v_forced_api_rejects_missing_audio_file(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": "/no/such/audio.wav",
            },
        )

        data = assert_http_error(
            r,
            status_code=400,
            code="HTTP_400",
            message="Audio file not found: /no/such/audio.wav",
        )
        assert "Audio file not found" in data["message"]

    def test_a2v_forced_api_missing_key_returns_integrity_error(self, client, test_state, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = ""
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )

        assert_http_error(r, status_code=400, code="PRO_API_KEY_REQUIRED")

    def test_a2v_forced_api_cancelled_response(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_audio_to_video = RuntimeError("cancelled")
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


class TestOutputFormat:
    """Phase 2a: output_format / proxy_path plumbing through video generation."""

    def test_t2v_prores_422_hq_output(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "output_format": "prores_422_hq"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["video_path"].endswith(".mov")
        assert data["proxy_path"] is not None
        assert data["proxy_path"].endswith("_proxy.mp4")

        call = fake_services.fast_video_pipeline.generate_calls[-1]
        assert str(call["output_format"]) == "OutputFormat.PRORES_422_HQ"
        assert call["proxy_path"] == data["proxy_path"]
        # Handler passes the (fake) encoder explicitly — exercises the DI wiring.
        assert call["encoder"] is fake_services.media_encoder

    def test_t2v_exr_half_output(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "output_format": "exr_zip_half"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["video_path"].endswith("_exr")
        assert data["proxy_path"] is not None
        assert data["proxy_path"].endswith("_proxy.mp4")

    def test_t2v_mp4_default_no_proxy(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200, r.text
        data = r.json()
        # MP4 default: no proxy, primary is .mp4 (byte-identical path/behavior).
        assert data["video_path"].endswith(".mp4")
        assert data["proxy_path"] is None

    def test_api_gate_rejects_prores(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 50,
                "cameraMotion": "dolly_in",
                "output_format": "prores_422_hq",
            },
        )
        assert r.status_code == 400
        assert "API mode cannot produce primary ProRes/EXR" in r.json()["message"]

    def test_handler_passes_on_progress(self, client, test_state, fake_services, create_fake_model_files):
        """Phase 4a: handler passes an on_progress callback to the pipeline."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json={**_T2V_JSON, "output_format": "prores_422_hq"})
        assert r.status_code == 200
        call = fake_services.fast_video_pipeline.generate_calls[-1]
        assert call["on_progress"] is not None, "handler must pass on_progress for non-MP4"
        # Invoking the callback should not raise (it calls update_progress with int pct).
        call["on_progress"](0.3)  # encode phase
        call["on_progress"](0.8)  # proxy phase

    def test_handler_forwards_negative_prompt_to_fast_pipeline(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """Phase 3D: handler forwards req.negativePrompt to the fast pipeline.

        Negative prompt is required by the dev route (TI2VidTwoStagesPipeline
        CFG) and harmless for the distilled route. The handler must not drop it.
        """
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "negativePrompt": "blurry, low quality"},
        )
        assert r.status_code == 200, r.text

        call = fake_services.fast_video_pipeline.generate_calls[-1]
        assert call["negative_prompt"] == "blurry, low quality"

    def test_handler_defaults_negative_prompt_to_empty_string(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """When the request omits negativePrompt, the fast pipeline receives ``""``.

        Distilled route ignores it; dev route treats it as no negative
        conditioning. The handler default must be an empty string (not None).
        """
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200, r.text

        call = fake_services.fast_video_pipeline.generate_calls[-1]
        assert call["negative_prompt"] == ""


class TestForcedApiGenerate:
    def test_prefers_api_video_routes_to_ltx_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 50,
                "audio": True,
                "cameraMotion": "dolly_in",
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 1
        assert len(fake_services.fast_video_pipeline.generate_calls) == 0

    def test_prefers_api_video_without_key_falls_back_to_local(self, client, test_state, fake_services, create_fake_model_files):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = ""
        _enable_local_text_encoding(test_state)
        create_fake_model_files()

        r = client.post("/api/generate", json=_T2V_JSON)

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 0
        assert len(fake_services.fast_video_pipeline.generate_calls) == 1

    def test_t2v_routes_to_ltx_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 50,
                "audio": True,
                "cameraMotion": "dolly_in",
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 1
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["model"] == "ltx-2-3-fast"
        assert call["resolution"] == "1920x1080"
        assert call["duration"] == 6.0
        assert call["fps"] == 50.0
        assert call["generate_audio"] is True
        assert call["camera_motion"] == "dolly_in"

    def test_i2v_routes_to_ltx_api(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame",
                "resolution": "2160p",
                "model": "pro",
                "duration": 8,
                "fps": 25,
                "audio": False,
                "cameraMotion": "jib_up",
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.image_to_video_calls) == 1
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/input.png"
        assert call["model"] == "ltx-2-3-pro"
        assert call["resolution"] == "3840x2160"
        assert call["duration"] == 8.0
        assert call["fps"] == 25.0
        assert call["camera_motion"] == "jib_up"

    def test_camera_motion_none_maps_to_none_for_t2v(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 50,
                "audio": True,
                "cameraMotion": "none",
            },
        )

        assert r.status_code == 200
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 1
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["camera_motion"] == "none"

    def test_camera_motion_none_maps_to_none_for_i2v(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input-none.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame",
                "resolution": "2160p",
                "model": "pro",
                "duration": 8,
                "fps": 25,
                "audio": False,
                "cameraMotion": "none",
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.image_to_video_calls) == 1
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/input-none.png"
        assert call["camera_motion"] == "none"

    def test_i2v_fast_routes_to_fast_model(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input-fast.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame quickly",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "audio": False,
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.image_to_video_calls) == 1
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/input-fast.png"
        assert call["model"] == "ltx-2-3-fast"
        assert call["resolution"] == "1920x1080"
        assert call["duration"] == 6.0
        assert call["fps"] == 25.0

    def test_invalid_forced_model_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "ultra",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert r.status_code == 422

    def test_missing_api_key_returns_integrity_error(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = ""

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(r, status_code=400, code="PRO_API_KEY_REQUIRED")

    def test_invalid_forced_resolution_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "720p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video resolution '720p' for pipeline 'pro'",
        )

    def test_invalid_forced_duration_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 5,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video duration '5' for pipeline 'pro' at resolution '1080p' and fps '25'",
        )

    def test_invalid_forced_fps_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 30,
                "audio": False,
            },
        )

        assert r.status_code == 422

    def test_forced_api_surfaces_insufficient_funds_as_custom_402(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_text_to_video = LTXAPIClientError(
            402,
            'LTX API generation failed (402): {"type":"error","error":{"type":"insufficient_funds_error","message":"Insufficient funds. Required: 36 cents"}}',
            stage="generation",
            provider_error_type="insufficient_funds_error",
            provider_message="Insufficient funds. Required: 36 cents",
            request_id="req-123",
        )

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=402,
            code="LTX_INSUFFICIENT_FUNDS",
            message="Your LTX API credits are insufficient for this generation. Buy more credits and try again.",
        )

        progress = test_state.generation.get_generation_progress()
        assert progress.status == "error"
        assert progress.phase == "error"

    def test_invalid_camera_motion_rejected_with_422(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
                "cameraMotion": "orbit",
            },
        )

        assert r.status_code == 422

    def test_forced_api_cancelled_response(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_text_to_video = RuntimeError("cancelled")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_portrait_resolution_1080p(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "1080x1920"

    def test_portrait_resolution_1440p(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait video",
                "resolution": "1440p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "1440x2560"

    def test_portrait_resolution_4k(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait video",
                "resolution": "2160p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "2160x3840"

    def test_default_landscape_when_aspect_ratio_omitted(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A landscape video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "1920x1080"

    def test_invalid_aspect_ratio_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "4:3",
            },
        )

        assert r.status_code == 422

    def test_extended_durations_for_fast_1080p_24fps(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A long video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 20,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["duration"] == 20.0

    def test_extended_duration_rejected_for_pro_1080p_24fps(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A long video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 20,
                "fps": 24,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video duration '20' for pipeline 'pro' at resolution '1080p' and fps '24'",
        )

    def test_extended_duration_rejected_for_fast_1440p_24fps(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A long video",
                "resolution": "1440p",
                "model": "fast",
                "duration": 20,
                "fps": 24,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video duration '20' for pipeline 'fast' at resolution '1440p' and fps '24'",
        )

    def test_fps_24_accepted(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["fps"] == 24.0

    def test_fps_48_accepted(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 48,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["fps"] == 48.0

    def test_a2v_portrait_resolution(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audioPath": str(audio_file),
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["resolution"] == "1080x1920"

    def test_a2v_forced_api_rejects_non_1080p(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "test_key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A big video",
                "resolution": "2160p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "audioPath": str(audio_file),
                "aspectRatio": "9:16",
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api audio-to-video resolution '2160p' for pipeline 'fast'",
        )

    def test_a2v_forced_api_passes_through_model_and_aspect(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "test_key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait music video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "audioPath": str(audio_file),
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["resolution"] == "1080x1920"
        assert call["model"] == "ltx-2-3-fast"


class TestGenerateCancel:
    def test_cancel_active(self, client, test_state):
        _fake_running_generation_state(test_state)

        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "cancelling"

    def test_cancel_no_active(self, client):
        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "no_active_generation"


class TestGenerateModelSpecs:
    def test_models_specs_endpoint_returns_ordered_backend_specs(self, client):
        r = client.get("/api/generate/models-specs")

        assert r.status_code == 200
        data = r.json()
        assert [item["pipeline"] for item in data["local_models"]] == ["fast"]
        assert data["local_models"][0]["spec"]["display_name"] == "LTX 2.3 Fast"
        assert list(data["local_models"][0]["spec"]["supported_resolutions_durations"]["540p"]["fps_to_durations"].keys()) == ["24"]
        assert [item["pipeline"] for item in data["api_models"]] == ["fast", "pro"]
        assert list(data["api_models"][0]["spec"]["a2v_supported_resolutions_durations"].keys()) == ["1080p"]
        assert data["api_models"][0]["spec"]["supported_resolutions_durations"]["1080p"]["fps_to_durations"]["24"] == [
            6, 8, 10, 12, 14, 16, 18, 20,
        ]


class TestGenerationProgress:
    def test_idle(self, client):
        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"

    def test_running(self, client, test_state):
        _fake_running_generation_state(test_state)
        test_state.generation.update_progress("inference", 50, 4, 8)

        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["phase"] == "inference"
        assert data["progress"] == 50
        assert data["currentStep"] == 4
        assert data["totalSteps"] == 8

    def test_running_from_api_generation_state(self, client, test_state):
        test_state.generation.start_api_generation("api-running")
        test_state.generation.update_progress("inference", 35)

        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["phase"] == "inference"
        assert data["progress"] == 35
        assert data["currentStep"] is None
        assert data["totalSteps"] is None


class TestGenerateImage:
    def test_happy_path(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "width": 1024, "height": 1024, "numSteps": 4},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 1
        assert Path(data["image_paths"][0]).exists()

    def test_dimension_clamping(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "width": 1023, "height": 1023},
        )
        assert r.status_code == 200

        call = fake_services.image_generation_pipeline.generate_calls[0]
        assert call["width"] == 1008
        assert call["height"] == 1008

    def test_num_images_clamped(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "numImages": 20},
        )
        assert r.status_code == 200

        assert len(fake_services.image_generation_pipeline.generate_calls) == 12

    def test_error(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("GPU OOM")

        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 500

    def test_cancelled(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("cancelled")

        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


class TestForcedApiGenerateImage:
    def test_generate_image_routes_to_zit_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = "fal-key"

        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "width": 1024, "height": 1024, "numSteps": 4, "numImages": 2},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 2
        assert len(fake_services.zit_api_client.text_to_image_calls) == 2
        assert len(fake_services.image_generation_pipeline.generate_calls) == 0

    def test_generate_image_missing_fal_key(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = ""

        r = client.post("/api/generate-image", json={"prompt": "A cat"})

        assert_http_error(r, status_code=500, code="FAL_API_KEY_NOT_CONFIGURED")

    def test_generate_image_cancelled(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = "fal-key"
        fake_services.zit_api_client.raise_on_text_to_image = RuntimeError("cancelled")

        r = client.post("/api/generate-image", json={"prompt": "A cat"})

        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


class TestEmptyPromptRejected:
    def test_empty_prompt_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": ""})
        assert r.status_code == 422

    def test_whitespace_prompt_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": "   "})
        assert r.status_code == 422

    def test_missing_prompt_rejected(self, client):
        r = client.post("/api/generate", json={})
        assert r.status_code == 422

    def test_empty_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={"prompt": ""})
        assert r.status_code == 422

    def test_whitespace_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={"prompt": "   "})
        assert r.status_code == 422

    def test_missing_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={})
        assert r.status_code == 422


class TestModelSelectionDto:
    """Step 4 / Phase 1: ``model_selection`` DTO contract only.

    Phase 1 establishes acceptance/rejection of the optional field. It is NOT
    routed yet — when present and valid it simply runs like current behavior.
    Runtime routing (and any bad/unsupported handling) arrives in Phase 2.
    """

    def test_omitted_still_works(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "complete"

    def test_null_still_works(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json={**_T2V_JSON, "model_selection": None})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "complete"

    def test_valid_candidate_literal_accepted_by_request_validation(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # Distilled is the transformer installed by create_fake_model_files, so a
        # valid candidate runs like current behavior (no Phase 1 routing change).
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": "ltx-2.3-22b-distilled"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "complete"

    def test_invalid_string_value_rejected_with_422(self, client):
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": "not-a-real-model"},
        )
        assert r.status_code == 422

    def test_wrong_type_rejected_with_422(self, client):
        # Strict mode: an int is neither a matching literal nor None → 422.
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": 123},
        )
        assert r.status_code == 422

    def test_camelcase_model_selection_field_rejected(self, client):
        # extra="forbid": a stale/misspelled camelCase ``modelSelection`` must
        # 422 instead of being silently ignored and falling back to no-selection.
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "modelSelection": "ltx-2.3-22b-distilled"},
        )
        assert r.status_code == 422


class TestModelSelectionRouting:
    """Step 4 / Phase 2: runtime model_selection resolver + cache hardening.

    Only absent/None falls back to current behavior; present-but-bad/unsupported
    selections reject clearly. T2V/I2V only (A2V rejects).
    """

    def test_selected_distilled_routes_and_records_selected_checkpoint_path(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": "ltx-2.3-22b-distilled"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "complete"

        expected = str(
            resolve_model_path(test_state.config.default_models_dir, "ltx-2.3-22b-distilled")
        )
        assert fake_services.fast_video_pipeline.last_checkpoint_path == expected

    def test_selected_missing_gguf_rejects_with_not_installed(
        self, client, test_state, create_fake_model_files
    ):
        # Distilled bundle installed, but the GGUF candidate is not.
        create_fake_model_files()

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": "ltx-2.3-22b-dev-gguf-q4-k-m"},
        )
        canonical = str(
            resolve_model_path(
                test_state.config.default_models_dir, "ltx-2.3-22b-dev-gguf-q4-k-m"
            )
        )
        assert r.status_code == 409
        payload = r.json()
        assert payload["code"] == "MODEL_SELECTION_NOT_INSTALLED"
        assert canonical in payload["message"]

    def test_present_model_selection_with_audio_rejects_as_unsupported_for_a2v(
        self, client, test_state, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "audioPath": str(audio_file),
                "model_selection": "ltx-2.3-22b-distilled",
            },
        )
        assert r.status_code == 409
        payload = r.json()
        assert payload["code"] == "MODEL_SELECTION_UNSUPPORTED_FOR_A2V"

    def test_non_base_cp_literal_rejects_with_unsupported_selection(self, client):
        # Valid ModelCheckpointID but not a selectable base video transformer.
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": "ltx-2.3-spatial-upscaler-x2-1.0"},
        )
        assert r.status_code == 422
        payload = r.json()
        assert payload["code"] == "UNSUPPORTED_MODEL_SELECTION"

    def test_switching_selection_changes_pipeline_cache(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        """Create/load once for distilled, then a different dev GGUF selection
        causes a new pipeline/cache key (cache miss → new ``create()``).

        Uses an active profile with split components so the dev GGUF selection
        can resolve its sidecars; the distilled LoRA is provided canonically so
        the dev route can build.
        """
        # Active profile with split sidecar components (reuses the GGUF helper).
        paths = _make_gguf_profile(test_state, tmp_path)
        # Distilled monolith + upscaler + text encoder installed canonically.
        create_fake_model_files()

        # Install the selected dev GGUF candidate at its canonical placement path.
        gguf_cp = "ltx-2.3-22b-dev-gguf-q4-k-m"
        gguf_path = resolve_model_path(test_state.config.default_models_dir, gguf_cp)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"GGUF-Q4")

        # Canonical distilled LoRA required by the dev route.
        distilled_lora = (
            test_state.config.default_models_dir
            / "adapters"
            / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        distilled_lora.parent.mkdir(parents=True, exist_ok=True)
        distilled_lora.write_bytes(b"LORA")

        # Gen 1: select distilled → override produces a distilled monolith build.
        r1 = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": "ltx-2.3-22b-distilled"},
        )
        assert r1.status_code == 200, r1.text
        distilled_expected = str(
            resolve_model_path(test_state.config.default_models_dir, "ltx-2.3-22b-distilled")
        )
        assert fake_services.fast_video_pipeline.last_checkpoint_path == distilled_expected
        assert fake_services.fast_video_pipeline.last_base_family == "distilled"

        # Gen 2: select a different installed dev GGUF → cache key differs → the
        # fast pipeline is recreated with the GGUF split build (new path tuple),
        # proving the selection switch invalidated the cache.
        r2 = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": gguf_cp},
        )
        assert r2.status_code == 200, r2.text
        assert fake_services.fast_video_pipeline.last_base_family == "dev"
        # Split build tuple: selected GGUF + profile sidecars.
        assert fake_services.fast_video_pipeline.last_checkpoint_path == (
            str(gguf_path),
            paths["tp"],
            paths["ec"],
            paths["vvae"],
            paths["avae"],
        )
        # The distilled str recorded in Gen 1 was overwritten by the new create.
        assert fake_services.fast_video_pipeline.last_checkpoint_path != distilled_expected

    def test_dev_cache_key_uses_effective_distilled_lora_with_stale_explicit(
        self, test_state, tmp_path
    ):
        """The dev cache key must reflect the ACTUAL effective distilled LoRA
        path even when an explicit (stale) path is present and the handler
        falls back to canonical. Removing the canonical fallback must change
        the key."""
        _make_gguf_profile(test_state, tmp_path)
        # Point the explicit distilled LoRA at a non-existent (stale) file.
        test_state.state.model_profiles[0].components.official_adapters[
            "distilled_lora_384_1_1"
        ] = str(tmp_path / "does-not-exist.safetensors")

        gguf_cp = "ltx-2.3-22b-dev-gguf-q4-k-m"
        gguf_path = resolve_model_path(test_state.config.default_models_dir, gguf_cp)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"GGUF")

        canonical_lora = (
            test_state.config.default_models_dir
            / "adapters"
            / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        canonical_lora.parent.mkdir(parents=True, exist_ok=True)
        canonical_lora.write_bytes(b"LORA")

        # Effective path = canonical fallback (explicit is stale) → key reflects it.
        key = test_state.pipelines._current_cache_key(model_selection=gguf_cp)
        assert str(canonical_lora) in key

        # Remove the canonical fallback → effective path becomes None → key changes
        # and no longer carries the canonical path.
        canonical_lora.unlink()
        key_after = test_state.pipelines._current_cache_key(model_selection=gguf_cp)
        assert str(canonical_lora) not in key_after
        assert key != key_after

    def test_present_model_selection_rejects_in_forced_api_mode(self, client, test_state, fake_services):
        # Forced API mode (local generations unsupported) + a present selection
        # must reject clearly instead of silently routing to the API. The guard
        # runs BEFORE generic spec validation, so _T2V_JSON's 540p (which would
        # otherwise fail api spec validation) surfaces the selection-specific
        # error, not a masked spec error.
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model_selection": "ltx-2.3-22b-distilled"},
        )
        assert r.status_code == 409
        payload = r.json()
        assert payload["code"] == "MODEL_SELECTION_UNSUPPORTED_FOR_API_GENERATIONS"
        # The API client must never have been called.
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 0


class TestEnhancePromptFlag:
    """Verify enhance_prompt is passed correctly to the text encoder API."""

    def _setup_api_encoding(self, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = False
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())

    def test_t2v_enhance_enabled(self, client, test_state, fake_services, create_fake_model_files):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_t2v_enhance_disabled(self, client, test_state, fake_services, create_fake_model_files):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def test_i2v_enhance_enabled(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_i2v_enhance_disabled(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = False
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def test_a2v_without_image_uses_t2v_setting(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post("/api/generate", json={**_T2V_JSON, "model": "fast", "audioPath": str(audio_file)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_a2v_with_image_uses_i2v_setting(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model": "fast", "audioPath": str(audio_file), "imagePath": str(image_path)},
        )
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_local_encoding_skips_api(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 0
        assert fake_services.fast_video_pipeline.generate_calls[-1]["enhance_prompt"] is True


def _make_gguf_profile(test_state, tmp_path: Path) -> dict[str, str]:
    """Create an active GGUF local profile (text encoder + transformer) and return file paths.

    Mirrors the profile shape used by ``test_t2v_active_profile_local_text_encoder_passes_gate``:
    a GGUF transformer + GGUF Gemma text encoder (vision tower stripped at runtime).
    Includes an upsampler so generation can proceed past the fast-pipeline upscaler gate.
    """
    d = tmp_path / "profile"
    d.mkdir()
    files = {
        "ltx-2.3-22b-distilled.gguf": b"GGUF",
        "tp.safetensors": b"x",
        "ec.safetensors": b"x",
        "vvae.safetensors": b"x",
        "avae.safetensors": b"x",
        "gemma.gguf": b"GEMMA",
        "upsampler.safetensors": b"UPSAMPLER",
    }
    paths: dict[str, str] = {}
    for name, content in files.items():
        p = d / name
        p.write_bytes(content)
        key = "transformer" if name.startswith("ltx-") else name.rsplit(".", 1)[0]
        paths[key] = str(p)

    profile = ModelProfilePayload(
        id="gguf-profile",
        name="GGUF Profile",
        source="kijai",
        components=ModelComponentPaths(
            transformer=paths["transformer"],
            transformer_format="gguf",
            upsampler=paths["upsampler"],
            text_projection=paths["tp"],
            embeddings_connector=paths["ec"],
            video_vae=paths["vvae"],
            audio_vae=paths["avae"],
            text_encoder_root=paths["gemma"],
            text_encoder_format="gguf",
        ),
    )
    test_state.state.model_profiles = [profile]
    test_state.state.active_model_profile_id = "gguf-profile"
    return paths


class TestGgufI2vEnhancerGate:
    """Phase 3B: the stale GGUF I2V 409 gate is removed.

    Local GGUF Gemma now handles image-conditioned prompt enhancement via
    mmproj multimodal enhancement (when configured) or an observable text-only
    llama.cpp degrade (image dropped from prompt enhancement only). Generation
    proceeds through the normal ``images=`` image-conditioning path either way.
    """

    def test_i2v_enhancer_on_with_gguf_profile_generates(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        """I2V + enhancer on + GGUF profile: no longer gated, returns 200.

        The fast pipeline ``generate`` is called with ``enhance_prompt=True``
        (local encoding path). The runtime PromptEncoder patch handles the
        GGUF+image enhancement internally (mmproj multimodal or text-only
        degrade); the handler no longer 409s.
        """
        _make_gguf_profile(test_state, tmp_path)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})

        assert r.status_code == 200, r.text
        assert r.json()["status"] == "complete"
        # Local encoding → enhance_prompt forwarded as True to the fast pipeline.
        assert fake_services.fast_video_pipeline.generate_calls[-1]["enhance_prompt"] is True

    def test_i2v_enhancer_off_with_gguf_profile_still_generates(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        _make_gguf_profile(test_state, tmp_path)
        # I2V enhancer OFF → no image-conditioned enhancement requested → generation proceeds.
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = False
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})

        assert r.status_code == 200, r.text
        assert r.json()["status"] == "complete"
        # No image-conditioned enhancement: enhance_prompt forwarded as False.
        assert fake_services.fast_video_pipeline.generate_calls[-1]["enhance_prompt"] is False

    def test_t2v_enhancer_on_with_gguf_profile_not_gated(
        self, client, test_state, fake_services, tmp_path
    ):
        """T2V (no image) + GGUF + t2v enhancer on: text-only llama.cpp enhancement, not gated."""
        _make_gguf_profile(test_state, tmp_path)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True

        r = client.post("/api/generate", json=_T2V_JSON)

        assert r.status_code == 200, r.text
        assert r.json()["status"] == "complete"
        assert fake_services.fast_video_pipeline.generate_calls[-1]["enhance_prompt"] is True

    def test_i2v_enhancer_on_with_non_gguf_profile_not_gated(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        """I2V + enhancer on + non-GGUF (API) encoding: API Gemma is multimodal-capable, not gated."""
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})

        assert r.status_code == 200, r.text
        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    @staticmethod
    def _setup_api_encoding(test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = False
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())
