"""Pipeline lifecycle handler."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import LTXLocalModelId, ModelCheckpointID
from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    OFFICIAL_LTX23_ADAPTERS,
    SELECTABLE_BASE_VIDEO_CP_IDS,
    UPSAMPLER_CP_ID,
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_ltx_model_spec,
    is_cp_downloaded,
    resolve_model_path,
)
from runtime_config.runtime_policy import streaming_prefetch_count_for_mode
from services.ltx_components import (
    CheckpointPath,
    ResolvedLtxComponents,
    checkpoint_path_arg,
    resolve_components,
)
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    ImageGenerationPipeline,
    GpuCleaner,
    IcLoraPipeline,
    PoseProcessorPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from services.services_utils import device_supports_fp8, get_device_type
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GpuGeneration,
    GenerationRunning,
    GpuSlot,
    ICLoraState,
    RetakePipelineState,
    VideoPipelineState,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class PipelinesHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner
        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._runtime_device = get_device_type(self.config.device)

    def _ensure_no_running_generation(self) -> None:
        match self.state.active_generation:
            case GpuGeneration(state=GenerationRunning()) if self.state.gpu_slot is not None:
                raise RuntimeError("Generation already running; cannot swap pipelines")
            case _:
                return

    def _resolve_selection(self, model_selection: ModelCheckpointID) -> str:
        """Validate a present ``model_selection`` and return its canonical path.

        Raises clear, actionable HTTP errors (never silent fallback) when the
        selection is unsupported or not installed:

        - ``UNSUPPORTED_MODEL_SELECTION`` (422): the CP id is a valid
          ``ModelCheckpointID`` but not a live-selectable base video transformer
          (e.g. an upscaler or adapter id).
        - ``MODEL_SELECTION_NOT_INSTALLED`` (409): the candidate is not present
          under the effective models dir; the message names the exact canonical
          placement path.

        Called only when ``model_selection`` is present; absent/None selection
        always falls back to active/current behavior.
        """
        if model_selection not in SELECTABLE_BASE_VIDEO_CP_IDS:
            raise HTTPError(
                422,
                (
                    f"Model selection '{model_selection}' is not a selectable base video "
                    "transformer. Live model selection supports the LTX-2.3 distilled and "
                    "dev GGUF base video models only."
                ),
                code="UNSUPPORTED_MODEL_SELECTION",
            )
        if not is_cp_downloaded(self.models_dir, model_selection):
            canonical = resolve_model_path(self.models_dir, model_selection)
            raise HTTPError(
                409,
                (
                    f"Selected model '{model_selection}' is not installed. Install it at the "
                    f"canonical placement path: {canonical}"
                ),
                code="MODEL_SELECTION_NOT_INSTALLED",
            )
        return str(resolve_model_path(self.models_dir, model_selection))

    def _pipeline_matches_model_type(
        self, model_type: VideoPipelineModelType, model_selection: ModelCheckpointID | None = None
    ) -> bool:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline, cache_key=cached_key)):
                if pipeline.pipeline_kind != model_type:
                    return False
                # ponytail: cache_key comparison only; richer invalidation lands with split/GGUF
                expected_key = self._current_cache_key(model_selection)
                return cached_key == expected_key
            case _:
                return False

    def _video_cache_key_for_components(
        self,
        components: ResolvedLtxComponents | None,
        model_selection: ModelCheckpointID | None,
    ) -> tuple[str, ...]:
        """Effective fast-video cache key for resolved components.

        For dev base families the ACTUAL effective distilled LoRA path is
        included in the key (explicit-existing preferred, else canonical
        fallback). The effective path is appended whenever it differs from the
        explicit path already baked into ``components.cache_key`` — e.g. when
        the explicit path is stale/missing and the handler falls back to
        canonical — so the key always reflects the real runtime path and a
        second ``load_gpu_pipeline`` with the same selection/profile cache-hits.
        """
        if components is None:
            model_id = get_downloaded_ltx_model_id(self.models_dir)
            if model_id is None:
                return ()
            if model_selection is not None:
                return (model_id, "model_selection", model_selection)
            return (model_id,)
        cache_key = components.cache_key
        if components.base_family == "dev":
            effective_lora = self._resolve_distilled_lora_path(components)
            explicit_lora = components.distilled_lora_path
            if effective_lora is not None and effective_lora != explicit_lora:
                cache_key = (*cache_key, effective_lora)
        return cache_key

    def _current_cache_key(self, model_selection: ModelCheckpointID | None = None) -> tuple[str, ...]:
        components = self._resolve_active_components(model_selection)
        return self._video_cache_key_for_components(components, model_selection)

    def _assert_invariants(self) -> None:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=active_pipeline):
                gpu_has_image_generation_pipeline = isinstance(active_pipeline, ImageGenerationPipeline)
            case _:
                gpu_has_image_generation_pipeline = False

        if gpu_has_image_generation_pipeline and self.state.cpu_slot is not None:
            raise RuntimeError("Invariant violation: image generation pipeline cannot be in both GPU and CPU slots")

    def _install_text_patches_if_needed(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return
        te.service.install_patches(lambda: self.state)

    def _resolve_active_components(
        self, model_selection: ModelCheckpointID | None = None
    ) -> ResolvedLtxComponents | None:
        profile_id = self.state.active_model_profile_id
        profile = (
            next((p for p in self.state.model_profiles if p.id == profile_id), None)
            if profile_id is not None
            else None
        )
        if profile is not None:
            if model_selection is None:
                return resolve_components(profile)
            selected_path = self._resolve_selection(model_selection)
            return resolve_components(
                profile,
                selected_transformer_path=selected_path,
                selected_cp_id=model_selection,
            )

        # No active profile (legacy downloaded-model path).
        if model_selection is not None:
            cp_id = model_selection
            # Validate (unsupported / not installed) before the profile check.
            self._resolve_selection(cp_id)
            # Only the distilled monolith can run without an active profile —
            # it reuses the legacy downloaded bundle (upsampler + text encoder).
            # Dev/GGUF selections need an active profile with split sidecar
            # components (text projection, VAEs; embeddings connector optional);
            # reject clearly rather than falling through to a deep pipeline failure.
            if cp_id != "ltx-2.3-22b-distilled":
                raise HTTPError(
                    409,
                    (
                        f"Live model selection for '{cp_id}' requires an active model profile "
                        "with split components (text projection, VAEs). "
                        "Activate a profile that provides these components and retry."
                    ),
                    code="MODEL_SELECTION_REQUIRES_PROFILE",
                )
        return None

    def _require_downloaded_ltx_model_id(self) -> LTXLocalModelId:
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return model_id

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if not self.state.app_settings.use_torch_compile:
            return state
        if state.is_compiled:
            return state
        if self._runtime_device == "mps":
            logger.info("Skipping torch.compile() for %s - not supported on MPS", state.pipeline.pipeline_kind)
            return state
        # GGUF transformers use lazy per-forward dequant that torch.compile
        # cannot trace. Skip silently (info, no traceback) instead of calling
        # compile_transformer() and relying on its RuntimeError guard.
        if not state.pipeline.supports_torch_compile():
            logger.info(
                "Skipping torch.compile() for %s - unsupported transformer format",
                state.pipeline.pipeline_kind,
            )
            return state

        try:
            state.pipeline.compile_transformer()
            state.is_compiled = True
        except Exception as exc:
            logger.warning("Failed to compile transformer: %s", exc, exc_info=True)
        return state

    def _resolve_profile_upsampler_path(self) -> str:
        """Resolve a usable upscaler path for an active profile.

        Prefers the profile's explicit ``components.upsampler`` path when it
        exists on disk. If that explicit path is stale/missing AND the
        canonical upscaler (``latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors``
        under the effective models root) exists, returns the canonical path.
        Otherwise returns an empty string so callers can decide how to surface
        the missing artifact (e.g. fast video fails fast with HTTP 409).
        """
        components = self._resolve_active_components()
        if components is None:
            return ""
        explicit = components.upsampler_path or ""
        if explicit and Path(explicit).exists():
            return explicit
        canonical = resolve_model_path(self.models_dir, UPSAMPLER_CP_ID)
        if canonical.exists():
            return str(canonical)
        return ""

    def _canonical_distilled_lora_candidates(self) -> list[tuple[str, Path]]:
        """Canonical models-dir distilled LoRA paths in preference order.

        Returns ``(adapter_id, path)`` tuples for the newest-then-older
        distilled LoRA filenames declared in ``OFFICIAL_LTX23_ADAPTERS``.
        Adapter canonical placement is ``<models_dir>/adapters/<filename>``
        (matches the scanner's canonical subfolder).
        """
        candidates: list[tuple[str, Path]] = []
        for role in ("distilled_lora_384_1_1", "distilled_lora_384"):
            adapter = OFFICIAL_LTX23_ADAPTERS.get(role)  # type: ignore[arg-type]
            if adapter is None:
                continue
            candidates.append((role, self.models_dir / "adapters" / adapter.filename))
        return candidates

    def _resolve_distilled_lora_path(
        self,
        components: ResolvedLtxComponents | None,
    ) -> str | None:
        """Resolve the effective distilled LoRA path for a dev base profile.

        Preference order:
        1. explicit profile path (``components.distilled_lora_path``)
        2. canonical models-dir fallback using ``OFFICIAL_LTX23_ADAPTERS``
           filenames for ``distilled_lora_384_1_1`` then ``distilled_lora_384``.

        Returns ``None`` when neither exists on disk.
        """
        explicit = components.distilled_lora_path if components is not None else None
        if explicit and Path(explicit).exists():
            return explicit

        for _role, path in self._canonical_distilled_lora_candidates():
            if path.exists():
                return str(path)
        return None

    def _resolve_checkpoint_paths(
        self, model_selection: ModelCheckpointID | None = None
    ) -> tuple[CheckpointPath, str | None, str, tuple[str, ...]]:
        """Return (checkpoint_path, gemma_root, upsampler_path, cache_key)."""
        components = self._resolve_active_components(model_selection)
        gemma_root = self._text_handler.resolve_gemma_root(model_selection)
        if components is not None:
            return (
                checkpoint_path_arg(components),
                components.gemma_root or gemma_root,
                self._resolve_profile_upsampler_path(),
                components.cache_key,
            )
        model_id = self._require_downloaded_ltx_model_id()
        spec = get_ltx_model_spec(model_id)
        if model_selection is not None:
            # Distilled selected without a profile: use the selected checkpoint
            # path explicitly (it IS the distilled monolith) and carry the
            # selection marker in the cache key.
            selected_path = str(resolve_model_path(self.models_dir, model_selection))
            cache_key: tuple[str, ...] = (model_id, "model_selection", model_selection)
            return (
                selected_path,
                gemma_root,
                str(get_existing_cp_path(self.models_dir, spec.upscale_cp)),
                cache_key,
            )
        return (
            str(get_existing_cp_path(self.models_dir, spec.model_cp)),
            gemma_root,
            str(get_existing_cp_path(self.models_dir, spec.upscale_cp)),
            (model_id,),
        )

    def _create_video_pipeline(
        self,
        model_type: VideoPipelineModelType,
        model_selection: ModelCheckpointID | None = None,
    ) -> VideoPipelineState:
        checkpoint_path, gemma_root, upsampler_path, _resolved_cache_key = self._resolve_checkpoint_paths(model_selection)
        # Fast video pipeline always invokes the spatial upscaler during
        # inference. Fail early with an actionable error instead of letting a
        # FileNotFoundError surface deep inside the diffusers pipeline.
        if not upsampler_path:
            canonical = resolve_model_path(self.models_dir, UPSAMPLER_CP_ID)
            raise HTTPError(
                409,
                (
                    "Spatial upscaler is required for fast video generation but was not found. "
                    "The active profile's upsampler path is missing or stale, and no canonical "
                    f"upscaler is installed at {canonical}. "
                    "Install 'ltx-2.3-spatial-upscaler-x2-1.0' or update the profile's upsampler path."
                ),
                code="UPSCALER_REQUIRED",
            )
        components = self._resolve_active_components(model_selection)
        transformer_format = components.transformer_format if components is not None else "safetensors"

        # Phase 3D (plan §12): route dev/distilled pipeline selection via
        # base_family. Unknown base family fails fast with an actionable error
        # before any heavy GPU work — never silently guess.
        base_family = components.base_family if components is not None else "distilled"
        if base_family == "unknown":
            raise HTTPError(
                409,
                (
                    "Active model profile has an unrecognized base family. The fast video "
                    "pipeline supports 'dev' and 'distilled' LTX-2.3 base models only; "
                    "the transformer path/filename did not contain a 'dev' or 'distilled' "
                    "signal. Choose an official LTX-2.3 dev or distilled transformer "
                    "(the filename must contain 'dev' or 'distilled'; note that "
                    "'distilled-lora' / 'distilled_lora' is an adapter name and does not "
                    "imply a distilled base)."
                ),
                code="UNSUPPORTED_MODEL_BASE_FAMILY",
            )

        # Dev route requires a distilled LoRA. Resolve explicit → canonical
        # fallback; if neither exists, fail before pipeline creation with the
        # exact canonical placement path(s) the user needs.
        distilled_lora_path: str | None = None
        if base_family == "dev":
            distilled_lora_path = self._resolve_distilled_lora_path(components)
            if not distilled_lora_path:
                canonical_paths = ", ".join(
                    str(p) for _role, p in self._canonical_distilled_lora_candidates()
                )
                raise HTTPError(
                    409,
                    (
                        "Dev base model requires a distilled LoRA for the fast video "
                        "pipeline, but none was found. Install one of the official "
                        f"distilled LoRAs at: {canonical_paths}."
                    ),
                    code="DISTILLED_LORA_REQUIRED",
                )

        pipeline = self._fast_video_pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            components=components,
            transformer_format=transformer_format,
            distilled_lora_path=distilled_lora_path,
        )

        # Cache key must reflect the effective distilled LoRA path so a dev
        # profile that toggles between explicit and canonical fallback (or
        # whose fallback appears/disappears on disk) invalidates correctly.
        # Computed via the same helper as ``_current_cache_key`` so a second
        # ``load_gpu_pipeline`` (e.g. inside ``generate_video``) cache-hits.
        effective_cache_key = self._video_cache_key_for_components(components, model_selection)

        state = VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
            cache_key=effective_cache_key,
        )
        return self._compile_if_enabled(state)

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def park_image_generation_pipeline_on_cpu(self) -> None:
        image_generation_pipeline: ImageGenerationPipeline | None = None

        with self._lock:
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if not isinstance(active, ImageGenerationPipeline):
                return

            if isinstance(self.state.active_generation, GpuGeneration) and isinstance(
                self.state.active_generation.state, GenerationRunning
            ):
                raise RuntimeError("Cannot park image generation pipeline while generation is running")

            image_generation_pipeline = active
            self.state.gpu_slot = None

        assert image_generation_pipeline is not None
        image_generation_pipeline.to("cpu")
        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.cpu_slot = CpuSlot(active_pipeline=image_generation_pipeline)
            self._assert_invariants()

    def load_image_generation_pipeline_to_gpu(self) -> ImageGenerationPipeline:
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if isinstance(active, ImageGenerationPipeline):
                    return active
                self._ensure_no_running_generation()

        image_generation_pipeline: ImageGenerationPipeline | None = None

        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=stored):
                    image_generation_pipeline = stored
                    self.state.cpu_slot = None
                case _:
                    image_generation_pipeline = None

        if image_generation_pipeline is None:
            zit_path = get_existing_cp_path(self.models_dir, IMG_GEN_MODEL_CP_ID)
            image_generation_pipeline = self._image_generation_pipeline_class.create(str(zit_path), self._runtime_device)
        else:
            image_generation_pipeline.to(self._runtime_device)

        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=image_generation_pipeline)
            self._assert_invariants()

        return image_generation_pipeline

    def _evict_gpu_pipeline_for_swap(self) -> None:
        should_park_image_generation_pipeline = False
        should_cleanup = False

        with self._lock:
            self._ensure_no_running_generation()
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, ImageGenerationPipeline):
                should_park_image_generation_pipeline = True
            else:
                self.state.gpu_slot = None
                self._assert_invariants()
                should_cleanup = True

        if should_park_image_generation_pipeline:
            self.park_image_generation_pipeline_on_cpu()
        elif should_cleanup:
            self._gpu_cleaner.cleanup()

    def load_gpu_pipeline(
        self,
        model_type: VideoPipelineModelType,
        model_selection: ModelCheckpointID | None = None,
    ) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_model_type(model_type, model_selection):
                match self.state.gpu_slot:
                    case GpuSlot(active_pipeline=VideoPipelineState() as existing_state):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            self._evict_gpu_pipeline_for_swap()
            state = self._create_video_pipeline(model_type, model_selection)
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state)
                self._assert_invariants()

        return state

    def load_ic_lora(
        self,
        lora_paths: list[str],
        depth_model_path: str | None,
        adapter_path: str | None = None,
        lora_strength: float = 1.0,
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=ICLoraState(
                        lora_paths=current_lora_paths,
                        depth_model_path=current_depth_model_path,
                        adapter_path=current_adapter_path,
                        lora_strength=current_lora_strength,
                    ) as state
                ) if (
                    current_lora_paths == lora_paths
                    and current_depth_model_path == depth_model_path
                    and current_adapter_path == adapter_path
                    and abs(current_lora_strength - lora_strength) < 0.001
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        checkpoint_path, gemma_root, upsampler_path, _cache_key = self._resolve_checkpoint_paths()
        components = self._resolve_active_components()

        pipeline = self._ic_lora_pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            lora_paths,
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            components=components,
            lora_strength=lora_strength,
        )
        depth_pipeline: DepthProcessorPipeline | None = None
        if depth_model_path is not None:
            depth_pipeline = self._depth_processor_pipeline_class.create(depth_model_path, self.config.device)
        state = ICLoraState(
            pipeline=pipeline,
            lora_paths=lora_paths,
            lora_strength=lora_strength,
            depth_pipeline=depth_pipeline,
            depth_model_path=depth_model_path,
            adapter_path=adapter_path,
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_a2v_pipeline(self) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState() as state):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        checkpoint_path, gemma_root, upsampler_path, _cache_key = self._resolve_checkpoint_paths()
        components = self._resolve_active_components()

        pipeline = self._a2v_pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            components=components,
        )
        state = A2VPipelineState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_retake_pipeline(self, *, distilled: bool = True) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self.config.device)

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=RetakePipelineState(distilled=current_distilled, quantized=current_quantized) as state
                ) if current_distilled == distilled and current_quantized == quantized:
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        from ltx_core.quantization import QuantizationPolicy

        quantization = QuantizationPolicy.fp8_cast() if quantized else None
        checkpoint_path, gemma_root, _upsampler_path, _cache_key = self._resolve_checkpoint_paths()
        components = self._resolve_active_components()
        pipeline = self._retake_pipeline_class.create(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            device=self.config.device,
            streaming_prefetch_count=streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            components=components,
            loras=[],
            quantization=quantization,
        )
        state = RetakePipelineState(pipeline=pipeline, distilled=distilled, quantized=quantized)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state
