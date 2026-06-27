"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

from api_types import (
    ConditioningType,
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateCompleteResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
    ImageConditioningInput,
    ModelComponentPaths,
    OutputFormat,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    OFFICIAL_LTX23_ADAPTERS,
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
)
from runtime_config.runtime_config import RuntimeConfig
from state.conditioning_cache import ConditioningCacheEntry, ConditioningCacheKey
from services.interfaces import VideoProcessor
from services.exr_input import is_exr_input, resolve_video_input_path
from services.color_management import detect_colorspace
from services.ltx_pipeline_common import make_encode_progress_callback, make_primary_output_path, make_proxy_output_path
from services.media_encoder.media_encoder import MediaEncoder
from services.services_utils import FrameArray
from state.app_state_types import AppState, ICLoraState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# ponytail: workflow types for IC-LoRA adapter dispatch; add to AdapterComponent when migrating to schema
WorkflowType = Literal[
    "standard_video",
    "ingredients",
    "in_outpainting",
    "union_control",
    "motion_track_control",
    "hdr",
    "hdr_scene_embeddings",
    "lipdub",
]

_ADAPTER_WORKFLOW: dict[str, WorkflowType] = {
    "water_simulation": "standard_video",
    "decompression": "standard_video",
    "deblur": "standard_video",
    "colorization": "standard_video",
    "day_to_night": "standard_video",
    "instant_shave": "standard_video",
    "cross_eyed": "standard_video",
    "ingredients": "ingredients",
    "in_outpainting": "in_outpainting",
    "union_control": "union_control",
    "motion_track_control": "motion_track_control",
    "hdr": "hdr",
    "lipdub": "lipdub",
    "hdr_scene_embeddings": "hdr_scene_embeddings",
}

_UNAVAILABLE_WORKFLOWS: frozenset[str] = frozenset({
    "motion_track_control",
    "hdr",
    "hdr_scene_embeddings",
    "lipdub",
})

_UNAVAILABLE_MESSAGES: dict[str, str] = {
    "motion_track_control": "Motion Track Control requires trajectory/reference video processing which is not wired yet",
    "hdr": "HDR workflow requires HDR scene embeddings and tone-mapping pipeline which is not wired yet",
    "hdr_scene_embeddings": "HDR scene embeddings is a support asset for the HDR workflow and cannot be used as a standalone adapter",
    "lipdub": "LipDub requires audio conditioning and lip-sync pipeline which is not wired yet",
}

# ponytail: 7 typed IC-LoRA adapter fields on ModelComponentPaths; future adapters use official_adapters dict
_TYPED_ADAPTER_FIELD: dict[str, str] = {
    "union_control": "ic_lora_union",
    "motion_track_control": "ic_lora_motion_track",
    "ingredients": "ic_lora_ingredients",
    "hdr": "ic_lora_hdr",
    "hdr_scene_embeddings": "ic_lora_hdr_scene_embeddings",
    "lipdub": "ic_lora_lipdub",
    "in_outpainting": "ic_lora_in_outpainting",
}


# ponytail: simple align helper; replace with math.ceil division if Python adds one
def _align_up(value: int, multiple: int) -> int:
    """Round up to the next multiple of `multiple`. Minimum `multiple`."""
    return max((value + multiple - 1) // multiple * multiple, multiple)


# ponytail: simple frame-count snap for LTX 1+8k constraint; move to LTX utils if reused
_SNAP_FRAME_MIN = 9


def _snap_frame_count(n: int) -> int:
    """Snap to valid LTX frame count: min {_SNAP_FRAME_MIN}, format 1+8k."""
    if n < _SNAP_FRAME_MIN:
        return _SNAP_FRAME_MIN
    k = max(0, (n - 1) // 8)
    return 1 + 8 * k


class IcLoraHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        video_processor: VideoProcessor,
        media_encoder: MediaEncoder,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._video_processor = video_processor
        self.media_encoder = media_encoder

    def _build_conditioning_frame(
        self,
        frame: FrameArray,
        conditioning_type: ConditioningType,
        ic_state: ICLoraState | None = None,
    ) -> FrameArray:
        match conditioning_type:
            case "canny":
                return self._video_processor.apply_canny(frame)
            case "depth":
                if ic_state is None or ic_state.depth_pipeline is None:
                    raise HTTPError(500, "Depth conditioning requires depth processor resources")
                return self._video_processor.apply_depth(frame, ic_state.depth_pipeline)
            case _:
                raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")

    def _active_profile_components(self) -> ModelComponentPaths | None:
        profile_id = self.state.active_model_profile_id
        if profile_id is None:
            return None
        profile = next((p for p in self.state.model_profiles if p.id == profile_id), None)
        return None if profile is None else profile.components

    def _profile_adapter_path(self, adapter_id: str) -> str | None:
        comps = self._active_profile_components()
        if comps is None:
            return None
        adapter_path = comps.official_adapters.get(adapter_id)
        if adapter_path:
            path = Path(adapter_path)
            if path.is_file():
                return str(path)

        field_name = _TYPED_ADAPTER_FIELD.get(adapter_id)
        typed_path = getattr(comps, field_name, None) if field_name is not None else None
        if typed_path:
            path = Path(typed_path)
            if path.is_file():
                return str(path)
        return None

    def _resolve_ic_lora_adapter_path(self, adapter_id: str) -> str:
        if adapter_id not in OFFICIAL_LTX23_ADAPTERS:
            raise HTTPError(400, f"Unknown adapter: {adapter_id}")

        adapter = OFFICIAL_LTX23_ADAPTERS[adapter_id]
        if adapter.kind != "ic_lora":
            raise HTTPError(400, f"Adapter {adapter_id} is not an IC-LoRA adapter (kind={adapter.kind})")

        profile_path = self._profile_adapter_path(adapter_id)
        if profile_path is not None:
            return profile_path

        installed = self.models_dir / adapter.filename
        if installed.is_file():
            return str(installed)

        raise HTTPError(400, f"Adapter not found: {adapter_id}")

    def _require_ic_lora_model_paths(
        self,
        conditioning_type: ConditioningType | None,
        require_lora: bool = True,
    ) -> tuple[Path | None, Path | None]:
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            # Active profile with transformer path satisfies model presence
            # (GGUF/local profile without official download)
            profile_id = self.state.active_model_profile_id
            profile = (
                next((p for p in self.state.model_profiles if p.id == profile_id), None)
                if profile_id is not None
                else None
            )
            if profile is not None and profile.components.transformer is not None:
                model_id = get_latest_ltx_model_id()
            else:
                raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")

        lora_path: Path | None = None
        if require_lora:
            ic_loras_spec = get_ltx_model_spec(model_id).ic_loras_spec
            match conditioning_type:
                case "canny":
                    lora_cp_id = ic_loras_spec.canny_cp
                case "depth":
                    lora_cp_id = ic_loras_spec.depth_cp
                case _:
                    raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")
            profile_union = self._profile_adapter_path("union_control")
            lora_path = Path(profile_union) if profile_union else get_existing_cp_path(self.models_dir, lora_cp_id)
        # ponytail: require_lora=False skips legacy LoRA check; adapter_path in generate() covers it
        depth_model_path: Path | None = None
        if conditioning_type == "depth":
            profile_components = self._active_profile_components()
            profile_depth = profile_components.depth_processor if profile_components is not None else None
            if profile_depth:
                depth_path = Path(profile_depth)
                depth_model_path = depth_path if depth_path.exists() else None
            else:
                depth_model_path = get_existing_cp_path(self.models_dir, DEPTH_PROCESSOR_CP_ID)
            if depth_model_path is None:
                raise HTTPError(400, "Depth conditioning requires dpt-hybrid-midas or profile depth_processor")
        return lora_path, depth_model_path

    def extract_conditioning(self, req: IcLoraExtractRequest) -> IcLoraExtractResponse:
        video_file = Path(req.video_path)
        if not video_file.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")

        cap = self._video_processor.open_video(str(video_file))
        info = self._video_processor.get_video_info(cap)
        target_frame = int(req.frame_time * float(info["fps"]))
        frame = self._video_processor.read_frame(cap, frame_idx=target_frame)
        self._video_processor.release(cap)

        if frame is None:
            raise HTTPError(400, "Could not read frame from video")

        ic_state: ICLoraState | None = None
        if req.conditioning_type == "depth":
            lora_path, depth_model_path = self._require_ic_lora_model_paths(req.conditioning_type)
            ic_state = self._pipelines.load_ic_lora(
                [str(lora_path)],
                str(depth_model_path),
            )

        result = self._build_conditioning_frame(frame, req.conditioning_type, ic_state)

        conditioning = self._video_processor.encode_frame_jpeg(result, quality=85)
        original = self._video_processor.encode_frame_jpeg(frame, quality=85)

        return IcLoraExtractResponse(
            conditioning="data:image/jpeg;base64," + base64.b64encode(conditioning).decode("utf-8"),
            original="data:image/jpeg;base64," + base64.b64encode(original).decode("utf-8"),
            conditioning_type=req.conditioning_type,
            frame_time=req.frame_time,
        )

    def _resolve_seed(self) -> int:
        settings = self.state.app_settings
        if settings.seed_locked:
            return settings.locked_seed
        if self.config.dev_mode:
            return 1000
        return int(time.time()) % 2147483647

    def _resolve_base_lora_path(
        self,
        conditioning_type: ConditioningType,
    ) -> tuple[str, Path | None]:
        """Get base LoRA path for conditioning: union_control from profile/adapter, else legacy checkpoint.
        Returns (base_path, depth_model_path)."""
        try:
            base = self._resolve_ic_lora_adapter_path("union_control")
            _, depth_model_path = self._require_ic_lora_model_paths(
                conditioning_type, require_lora=False
            )
            return base, depth_model_path
        except HTTPError:
            legacy_path, depth_model_path = self._require_ic_lora_model_paths(
                conditioning_type, require_lora=True
            )
            return str(legacy_path), depth_model_path

    def _generate_ingredients(
        self, req: IcLoraGenerateRequest, workflow: str
    ) -> IcLoraGenerateResponse:
        """Ingredients T2V path: no video, no conditioning, use request dims."""
        generation_id = uuid.uuid4().hex[:8]
        t_total_start = time.perf_counter()
        logger.info("[ic-lora] Ingredients generation started")

        try:
            t_load_start = time.perf_counter()
            # ponytail: no conditioning → load just the adapter, no union control
            assert req.adapter_id is not None  # guarded by workflow == "ingredients" above
            adapter_path = self._resolve_ic_lora_adapter_path(req.adapter_id)
            model_id = get_downloaded_ltx_model_id(self.models_dir)
            if model_id is None:
                profile_id = self.state.active_model_profile_id
                profile = (
                    next((p for p in self.state.model_profiles if p.id == profile_id), None)
                    if profile_id is not None else None
                )
                if profile is None or profile.components.transformer is None:
                    raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")

            ic_state = self._pipelines.load_ic_lora(
                [adapter_path],
                None,  # no depth
                adapter_path=adapter_path,
                lora_strength=req.lora_strength,
            )
            t_load_end = time.perf_counter()
            logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            s = self.state.app_settings
            use_api = not self._text.should_use_local_encoding()
            encoding_method = "api" if use_api else "local"
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(
                req.prompt, enhance_prompt=use_api and s.prompt_enhancer_enabled_t2v
            )
            t_text_end = time.perf_counter()
            logger.info("[ic-lora] Text encoding (%s): %.2fs", encoding_method, t_text_end - t_text_start)

            height = _align_up(req.height, 64)
            width = _align_up(req.width, 64)
            num_frames = _snap_frame_count(req.num_frames)
            frame_rate = req.frame_rate

            images: list[ImageConditioningInput] = [
                ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                for img in req.images
            ]

            self._generation.update_progress("inference", 15, 0, 1)

            output_format = req.output_format or OutputFormat.MP4
            output_path = make_primary_output_path(
                str(self.config.outputs_dir), "ic_lora", output_format, uuid.uuid4().hex[:8]
            )
            proxy_path = make_proxy_output_path(output_path, output_format)

            t_inference_start = time.perf_counter()
            ic_state.pipeline.generate(
                prompt=req.prompt,
                seed=self._resolve_seed(),
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                video_conditioning=[],
                output_path=output_path,
                mask_path=None,
                conditioning_strength=req.conditioning_strength,
                original_video_path=None,
                output_format=output_format,
                encoder=self.media_encoder,
                proxy_path=proxy_path,
                on_progress=make_encode_progress_callback(self._generation.update_progress),
            )
            t_inference_end = time.perf_counter()
            logger.info("[ic-lora] Inference: %.2fs", t_inference_end - t_inference_start)

            t_total_end = time.perf_counter()
            logger.info(
                "[ic-lora] Total ingredients generation: %.2fs (load=%.2fs, text=%.2fs, inference=%.2fs)",
                t_total_end - t_total_start,
                t_load_end - t_load_start,
                t_text_end - t_text_start,
                t_inference_end - t_inference_start,
            )

            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(output_path)
            return IcLoraGenerateCompleteResponse(
                status="complete", video_path=output_path, proxy_path=proxy_path
            )

        except HTTPError:
            self._generation.fail_generation("IC-LoRA generation failed")
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return IcLoraGenerateCancelledResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()

    def generate(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        # ponytail: resolve workflow before any Path(req.video_path) so ingredients can skip
        workflow: str | None = _ADAPTER_WORKFLOW.get(req.adapter_id) if req.adapter_id else None
        if workflow in _UNAVAILABLE_WORKFLOWS:
            raise HTTPError(400, _UNAVAILABLE_MESSAGES[workflow])

        # ponytail: dispatch ingredients before any video path validation
        if workflow == "ingredients":
            if not (req.prompt or "").strip():
                raise HTTPError(400, "Prompt is required for this adapter")
            if not req.images:
                raise HTTPError(400, "Ingredients adapter requires at least one image in images[]")
            if req.conditioning_type is not None:
                raise HTTPError(400, "Ingredients adapter is image-only; omit conditioning_type")
            return self._generate_ingredients(req, workflow)

        if not req.video_path:
            raise HTTPError(400, "video_path is required for this adapter")
        video_path = Path(req.video_path)
        if not video_path.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")
        if req.mask_path is not None:
            mask_path = Path(req.mask_path)
            if not mask_path.exists():
                raise HTTPError(400, f"Mask not found: {req.mask_path}")

        # P0-3: EXR source resolution — the handler's VideoProcessor.open_video
        # at :516 can't open an EXR dir/file. Resolve to a temp MP4 BEFORE that
        # call so metadata reads succeed. Non-EXR: resolve_video_input_path
        # returns the path UNCHANGED (pure-suffix gate, zero I/O).
        resolved_video_path = resolve_video_input_path(str(video_path))

        if workflow == "union_control" and req.conditioning_type is None:
            raise HTTPError(400, "Union Control requires conditioning_type (canny or depth)")
        if workflow == "in_outpainting" and req.mask_path is None:
            raise HTTPError(400, "In/outpainting requires a mask_path")
        # ponytail: in_outpainting allows empty prompt; other adapters require non-blank
        if workflow != "in_outpainting" and not (req.prompt or "").strip():
            raise HTTPError(400, "Prompt is required for this adapter")

        resolved_prompt: str = req.prompt or ""

        resolved_adapter_path: str | None = None
        if req.adapter_id is not None:
            resolved_adapter_path = self._resolve_ic_lora_adapter_path(req.adapter_id)

        # Build lora_paths list and resolve depth model
        lora_paths: list[str] = []
        depth_model_path: Path | None = None

        if req.conditioning_type is not None:
            base_path, depth_model_path = self._resolve_base_lora_path(req.conditioning_type)
            lora_paths.append(base_path)
            # Stack adapter on top if it differs from base union path
            if resolved_adapter_path is not None and resolved_adapter_path != base_path:
                lora_paths.append(resolved_adapter_path)
        elif resolved_adapter_path is not None:
            # No conditioning: load just the adapter
            lora_paths.append(resolved_adapter_path)
            # Verify LTX model exists for non-conditioning adapter use
            model_id = get_downloaded_ltx_model_id(self.models_dir)
            if model_id is None:
                profile_id = self.state.active_model_profile_id
                profile = (
                    next((p for p in self.state.model_profiles if p.id == profile_id), None)
                    if profile_id is not None else None
                )
                if profile is None or profile.components.transformer is None:
                    raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        else:
            raise HTTPError(400, "Either conditioning_type or adapter_id must be provided")

        generation_id = uuid.uuid4().hex[:8]
        t_total_start = time.perf_counter()
        logger.info("[ic-lora] Generation started (conditioning=%s)", req.conditioning_type)

        try:
            t_load_start = time.perf_counter()
            ic_state = self._pipelines.load_ic_lora(
                lora_paths,
                str(depth_model_path) if depth_model_path else None,
                adapter_path=resolved_adapter_path,
                lora_strength=req.lora_strength,
            )
            t_load_end = time.perf_counter()
            logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            s = self.state.app_settings
            use_api = not self._text.should_use_local_encoding()
            encoding_method = "api" if use_api else "local"
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(req.prompt, enhance_prompt=use_api and s.prompt_enhancer_enabled_t2v)
            t_text_end = time.perf_counter()
            logger.info("[ic-lora] Text encoding (%s): %.2fs", encoding_method, t_text_end - t_text_start)

            cap = self._video_processor.open_video(resolved_video_path)
            if not cap.isOpened():
                raise HTTPError(400, f"Cannot open video: {video_path}")
            info = self._video_processor.get_video_info(cap)
            input_width = int(info["width"])
            input_height = int(info["height"])
            frame_count = int(info["frame_count"])
            fps = float(info["fps"])

            t_preprocess_start: float = 0.0
            t_preprocess_end: float = 0.0

            if req.conditioning_type is not None:
                cache_key = ConditioningCacheKey(str(video_path), req.conditioning_type)
                cached = ic_state.conditioning_cache.get(cache_key)

                if cached is not None:
                    self._video_processor.release(cap)
                    control_video_path = cached.control_video_path
                    frame_count = cached.frame_count
                    fps = cached.fps
                    logger.info(
                        "[ic-lora] Conditioning cache hit for %s/%s",
                        video_path.name, req.conditioning_type,
                    )
                else:
                    t_preprocess_start = time.perf_counter()

                    control_video_path = str(
                        self.config.outputs_dir
                        / f"_control_{req.conditioning_type}_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    writer = self._video_processor.create_writer(
                        control_video_path,
                        fourcc="mp4v",
                        fps=fps,
                        size=(int(info["width"]), int(info["height"])),
                    )

                    assert req.conditioning_type is not None
                    frame_idx = 0
                    while frame_idx < frame_count:
                        frame = self._video_processor.read_frame(cap)
                        if frame is None:
                            break
                        control_frame = self._build_conditioning_frame(
                            frame, req.conditioning_type, ic_state
                        )
                        writer.write(control_frame)
                        frame_idx += 1

                    self._video_processor.release(cap)
                    self._video_processor.release(writer)
                    t_preprocess_end = time.perf_counter()
                    logger.info(
                        "[ic-lora] Preprocessing (%s, %d frames): %.2fs",
                        req.conditioning_type, frame_idx, t_preprocess_end - t_preprocess_start,
                    )

                    ic_state.conditioning_cache.put(
                        cache_key, ConditioningCacheEntry(control_video_path, frame_count, fps)
                    )
            else:
                # No conditioning: use original video as the control signal
                self._video_processor.release(cap)
                control_video_path = str(video_path)

            images: list[ImageConditioningInput] = [
                ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                for img in req.images
            ]

            self._generation.update_progress("inference", 15, 0, 1)

            uses_union_control = req.conditioning_type is not None
            align_to = 128 if uses_union_control else 64
            # ponytail: canny/depth loads Union Control ref_downscale=2 LoRA; half-res ref
            # must still be VAE 32x compatible; 64*2 alignment avoids VAE SpaceToDepth odd latent dims
            width = _align_up(input_width, align_to)
            height = _align_up(input_height, align_to)

            output_format = req.output_format or OutputFormat.MP4
            # CM-2: detect source CS for EXR inputs (output-CS preservation).
            input_colorspace = detect_colorspace(req.video_path) if (req.video_path and is_exr_input(req.video_path)) else None
            output_path = make_primary_output_path(
                str(self.config.outputs_dir), "ic_lora", output_format, uuid.uuid4().hex[:8]
            )
            proxy_path = make_proxy_output_path(output_path, output_format)

            t_inference_start = time.perf_counter()
            if workflow == "in_outpainting":
                ic_state.pipeline.generate_inpaint(
                    prompt=resolved_prompt,
                    seed=self._resolve_seed(),
                    height=height,
                    width=width,
                    num_frames=frame_count,
                    frame_rate=fps,
                    images=images,
                    video_path=resolved_video_path,
                    mask_path=str(req.mask_path),
                    output_path=output_path,
                    conditioning_strength=req.conditioning_strength,
                    mask_grow_px=req.mask_grow_px,
                    laplacian_blend_grow=req.laplacian_blend_grow,
                    final_mask_blur_px=req.final_mask_blur_px,
                    output_format=output_format,
                    encoder=self.media_encoder,
                    proxy_path=proxy_path,
                    on_progress=make_encode_progress_callback(self._generation.update_progress),
                    input_colorspace=input_colorspace,
                )
            else:
                ic_state.pipeline.generate(
                    prompt=req.prompt,
                    seed=self._resolve_seed(),
                    height=height,
                    width=width,
                    num_frames=frame_count,
                    frame_rate=fps,
                    images=images,
                    video_conditioning=[(control_video_path, req.conditioning_strength)],
                    output_path=output_path,
                    mask_path=req.mask_path,
                    conditioning_strength=req.conditioning_strength,
                    original_video_path=None,
                    output_format=output_format,
                    encoder=self.media_encoder,
                    proxy_path=proxy_path,
                    on_progress=make_encode_progress_callback(self._generation.update_progress),
                    input_colorspace=input_colorspace,
                )
            t_inference_end = time.perf_counter()
            logger.info("[ic-lora] Inference: %.2fs", t_inference_end - t_inference_start)

            t_total_end = time.perf_counter()
            preprocess_time = (t_preprocess_end - t_preprocess_start) if req.conditioning_type is not None else 0.0
            logger.info(
                "[ic-lora] Total generation: %.2fs (load=%.2fs, text=%.2fs, preprocess=%.2fs, inference=%.2fs)",
                t_total_end - t_total_start,
                t_load_end - t_load_start,
                t_text_end - t_text_start,
                preprocess_time,
                t_inference_end - t_inference_start,
            )

            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(output_path)
            return IcLoraGenerateCompleteResponse(
                status="complete", video_path=output_path, proxy_path=proxy_path
            )

        except HTTPError:
            self._generation.fail_generation("IC-LoRA generation failed")
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return IcLoraGenerateCancelledResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()
