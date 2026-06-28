"""Pydantic request/response models and typed aliases for ltx2_server."""

from __future__ import annotations

from enum import Enum
from typing import Annotated
from typing import Literal, NamedTuple, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


class OutputFormat(str, Enum):
    """Primary output container/codec produced by the media encoder.

    Lives in ``api_types`` (not ``services.media_encoder``) per §0A.F — services
    import ``api_types`` so this keeps the DTO/service layering clean and avoids a
    circular import. ``str`` enum so pydantic/OpenAPI serialize it as a plain
    string.
    """

    MP4 = "mp4"
    PRORES_PROXY = "prores_proxy"
    PRORES_LT = "prores_lt"
    PRORES_422 = "prores_422"
    PRORES_422_HQ = "prores_422_hq"
    PRORES_4444 = "prores_4444"
    PRORES_4444_XQ = "prores_4444_xq"
    EXR_ZIP_HALF = "exr_zip_half"
    EXR_ZIP_FLOAT = "exr_zip_float"


NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelCheckpointID = Literal[
    "ltx-2.3-22b-distilled",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5",
    "ltx-2.3-22b-ic-lora-ingredients-0.9",
    "ltx-2.3-22b-ic-lora-water-simulation-0.9",
    "ltx-2.3-22b-ic-lora-decompression-0.9",
    "ltx-2.3-22b-ic-lora-deblur-0.9",
    "ltx-2.3-22b-ic-lora-colorization-0.9",
    "ltx-2.3-22b-ic-lora-day-to-night-0.9",
    "ltx-2.3-22b-ic-lora-in-outpainting-0.9",
    "ltx-2.3-22b-ic-lora-instant-shave-0.9",
    "ltx-2.3-22b-ic-lora-cross-eyed-0.9",
    "ltx-2.3-22b-ic-lora-hdr-0.9",
    "ltx-2.3-22b-ic-lora-hdr-scene-emb",
    "ltx-2.3-22b-ic-lora-lipdub-0.9",
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "z-image-turbo",
]
LTXLocalModelId = Literal["ltx-2.3-22b-distilled"]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


JsonObject: TypeAlias = dict[str, object]
VideoCameraMotion = Literal[
    "none",
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
]


# ============================================================
# Response Models
# ============================================================


class ModelStatusItem(BaseModel):
    id: str
    name: str
    loaded: bool
    downloaded: bool


class GpuTelemetry(BaseModel):
    name: str
    vram: int
    vramUsed: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    models_loaded: bool
    active_model: str | None
    gpu_info: GpuTelemetry
    sage_attention: bool
    models_status: list[ModelStatusItem]


class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry


class RuntimePolicyResponse(BaseModel):
    force_api_generations: bool


class GenerationProgressResponse(BaseModel):
    status: Literal["idle", "running", "complete", "cancelled", "error"]
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None
    # Live-only metrics (additive, backward-compatible; None when unavailable).
    elapsedSeconds: float | None = None
    phaseElapsedSeconds: float | None = None
    stepsPerSecond: float | None = None
    estimatedRemainingSeconds: float | None = None
    vramUsedMb: int | None = None
    vramTotalMb: int | None = None
    gpuUtilPct: float | None = None
    ramUsedMb: int | None = None
    ramTotalMb: int | None = None
    cpuUtilPct: float | None = None


class DownloadProgressRunningResponse(BaseModel):
    status: Literal["downloading"]
    current_downloading_file: ModelCheckpointID | None
    current_file_progress: float
    total_progress: float
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelCheckpointID]
    all_files: set[ModelCheckpointID]
    error: None = None
    speed_bytes_per_sec: float


class DownloadProgressCompleteResponse(BaseModel):
    status: Literal["complete"]


# Single source of truth for structured download error codes.
DownloadErrorCode: TypeAlias = Literal[
    "DOWNLOAD_LOCKED",
    "INSUFFICIENT_DISK_SPACE",
    "NETWORK_ERROR",
    "UNKNOWN_ERROR",
]


class DownloadProgressErrorResponse(BaseModel):
    status: Literal["error"]
    error: str
    error_code: DownloadErrorCode = "UNKNOWN_ERROR"


class DownloadProgressCancelledResponse(BaseModel):
    status: Literal["cancelled"]


DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse
    | DownloadProgressCompleteResponse
    | DownloadProgressErrorResponse
    | DownloadProgressCancelledResponse
)


class DownloadCancelCancellingResponse(BaseModel):
    status: Literal["cancelling"]
    sessionId: str


class DownloadCancelNoActiveResponse(BaseModel):
    status: Literal["no_active_download"]


DownloadCancelResponse: TypeAlias = DownloadCancelCancellingResponse | DownloadCancelNoActiveResponse


class SuggestGapPromptResponse(BaseModel):
    status: Literal["success"] = "success"
    suggested_prompt: str


class GenerateVideoCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str
    proxy_path: str | None = None


class GenerateVideoCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateVideoResponse: TypeAlias = GenerateVideoCompleteResponse | GenerateVideoCancelledResponse


class GenerateImageCompleteResponse(BaseModel):
    status: Literal["complete"]
    image_paths: list[str]


class GenerateImageCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateImageResponse: TypeAlias = GenerateImageCompleteResponse | GenerateImageCancelledResponse


class CancelCancellingResponse(BaseModel):
    status: Literal["cancelling"]
    id: str


class CancelNoActiveGenerationResponse(BaseModel):
    status: Literal["no_active_generation"]


CancelResponse: TypeAlias = CancelCancellingResponse | CancelNoActiveGenerationResponse


class RetakeVideoResponse(BaseModel):
    status: Literal["complete"]
    video_path: str
    proxy_path: str | None = None


class RetakePayloadResponse(BaseModel):
    status: Literal["complete"]
    result: JsonObject


class RetakeCancelledResponse(BaseModel):
    status: Literal["cancelled"]


RetakeResponse: TypeAlias = RetakeVideoResponse | RetakePayloadResponse | RetakeCancelledResponse


class IcLoraExtractResponse(BaseModel):
    conditioning: str
    original: str
    conditioning_type: ConditioningType
    frame_time: float


class IcLoraGenerateCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str
    proxy_path: str | None = None


class IcLoraGenerateCancelledResponse(BaseModel):
    status: Literal["cancelled"]


IcLoraGenerateResponse: TypeAlias = IcLoraGenerateCompleteResponse | IcLoraGenerateCancelledResponse


# ============================================================
# HuggingFace auth
# ============================================================


class HuggingFaceLoginResponse(BaseModel):
    client_id: str
    redirect_uri: str
    scope: str
    state: str
    code_challenge: str
    code_challenge_method: str


class HuggingFaceAuthStatusResponse(BaseModel):
    status: Literal["authenticated", "pending", "not_authenticated"]


class HuggingFaceLogoutResponse(BaseModel):
    status: Literal["logged_out"]


class ModelDownloadStartResponse(BaseModel):
    status: Literal["started"]
    message: str
    sessionId: str


class LtxDownloadRecommendationResponse(BaseModel):
    status: Literal["download"]
    cps_to_download: list[ModelCheckpointID]


class LtxUpgradeRecommendationResponse(BaseModel):
    status: Literal["upgrade"]
    ltx_model_id: LTXLocalModelId
    upgrade_message: str | None = None
    cps_to_download: list[ModelCheckpointID]
    cps_to_delete: list[ModelCheckpointID]


class LtxOkRecommendationResponse(BaseModel):
    status: Literal["ok"]


LtxRecommendationResponse: TypeAlias = (
    LtxDownloadRecommendationResponse | LtxUpgradeRecommendationResponse | LtxOkRecommendationResponse
)


class ImageGenRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None


class LtxIcLoraRecommendationResponse(BaseModel):
    cps_to_download: list[ModelCheckpointID]


class TextEncoderRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None
    expected_size_bytes: int
    expected_size_gb: float


class StatusResponse(BaseModel):
    status: str


class HTTPErrorResponse(BaseModel):
    code: str
    message: str


class LtxInsufficientFundsErrorResponse(BaseModel):
    code: Literal["LTX_INSUFFICIENT_FUNDS"]
    message: str


# ============================================================
# Model Profile Types
# ============================================================


ModelProfileId: TypeAlias = str
ModelProfileFamily: TypeAlias = Literal["ltx-2", "ltx-2.3", "ltxv2", "custom"]
ModelProfileSource: TypeAlias = Literal["official", "kijai", "quantstack", "custom"]
ModelProfileTransformerFormat: TypeAlias = Literal["official_safetensors", "split_safetensors", "gguf"]
ModelProfileTextEncoderFormat: TypeAlias = Literal["hf_folder", "safetensors", "gguf", "api"]
ModelProfileCapability: TypeAlias = Literal[
    "t2v", "i2v", "a2v", "retake", "ic_lora", "local_text", "gguf"
]

#: Current model profile schema version. Absent ``schema_version`` in a legacy
#: ``model_profiles.json`` entry is treated as legacy and normalized to this
#: value on load (in-memory only; no destructive auto-save).
CURRENT_MODEL_PROFILE_SCHEMA_VERSION: int = 1

ModelProfileCreatedBy: TypeAlias = Literal["user", "wizard", "official_template"]
ModelProfileValidationStatus: TypeAlias = Literal["candidate", "validated", "deprecated"]
ModelProfileProblemSeverity: TypeAlias = Literal["info", "warning", "error"]


class ModelProfileProblem(BaseModel):
    """Stable, typed problem object surfaced per profile/artifact.

    ``code`` is a machine-readable stable identifier (e.g. ``missing_path``,
    ``duplicate``, ``unknown_file``). ``severity`` drives UI badges.
    ``path``/``field`` are optional context anchors.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: ModelProfileProblemSeverity
    message: str
    path: str | None = None
    field: str | None = None


def _default_model_profile_problems() -> list[ModelProfileProblem]:
    return []


class ModelComponentPaths(BaseModel):
    transformer: str | None = None
    transformer_format: ModelProfileTransformerFormat = "official_safetensors"
    transformer_quantization: str | None = None
    upsampler: str | None = None
    text_encoder_root: str | None = None
    text_encoder_format: ModelProfileTextEncoderFormat = "api"
    text_projection: str | None = None
    embeddings_connector: str | None = None
    video_vae: str | None = None
    audio_vae: str | None = None
    vocoder: str | None = None
    ic_lora_union: str | None = None
    ic_lora_motion_track: str | None = None
    ic_lora_ingredients: str | None = None
    ic_lora_hdr: str | None = None
    ic_lora_hdr_scene_embeddings: str | None = None
    ic_lora_lipdub: str | None = None
    ic_lora_in_outpainting: str | None = None
    official_adapters: dict[str, str] = Field(default_factory=dict)
    depth_processor: str | None = None
    pose_processor: str | None = None
    person_detector: str | None = None


class ModelProfilePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    family: ModelProfileFamily = "ltx-2.3"
    source: ModelProfileSource = "official"
    components: ModelComponentPaths = Field(default_factory=ModelComponentPaths)
    capabilities: list[ModelProfileCapability] = Field(default_factory=lambda: ["t2v"])
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    # Phase 1 schema migration fields — backward-compatible defaults.
    # These are server-owned: persisted on explicit save/patch/create, but not
    # auto-saved on load (existing blank-ID repair behavior is unchanged).
    schema_version: int = CURRENT_MODEL_PROFILE_SCHEMA_VERSION
    created_by: ModelProfileCreatedBy = "user"
    validation_status: ModelProfileValidationStatus = "candidate"
    last_scanned_at: str | None = None
    problems: list[ModelProfileProblem] = Field(default_factory=_default_model_profile_problems)


class ModelProfilePatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    family: ModelProfileFamily | None = None
    source: ModelProfileSource | None = None
    components: ModelComponentPaths | None = None
    capabilities: list[ModelProfileCapability] | None = None
    notes: str | None = None


def _default_model_profiles() -> list[ModelProfilePayload]:
    return []


class ModelProfilesResponse(BaseModel):
    active_model_profile_id: str | None = None
    profiles: list[ModelProfilePayload] = Field(default_factory=_default_model_profiles)


class ModelProfileValidationIssuePayload(BaseModel):
    field: str
    issue: str


def _default_model_profile_issues() -> list[ModelProfileValidationIssuePayload]:
    return []


class ModelProfileValidationResponse(BaseModel):
    valid: bool = True
    issues: list[ModelProfileValidationIssuePayload] = Field(default_factory=_default_model_profile_issues)


class ModelProfileActivateResponse(BaseModel):
    status: str = "ok"
    active_model_profile_id: str | None = None


# ============================================================
# Adapter Registry Types
# ============================================================


AdapterID: TypeAlias = Literal[
    "distilled_lora_384",
    "distilled_lora_384_1_1",
    "union_control",
    "motion_track_control",
    "ingredients",
    "water_simulation",
    "decompression",
    "deblur",
    "colorization",
    "day_to_night",
    "in_outpainting",
    "instant_shave",
    "cross_eyed",
    "hdr",
    "hdr_scene_embeddings",
    "lipdub",
]
AdapterKind: TypeAlias = Literal["lora", "ic_lora", "distilled_lora", "embeddings"]
AdapterSource: TypeAlias = Literal["official", "kijai", "custom"]
AdapterPipeline: TypeAlias = Literal[
    "fast",
    "union_control",
    "motion_track_control",
    "ingredients",
    "water_simulation",
    "decompression",
    "deblur",
    "colorization",
    "day_to_night",
    "in_outpainting",
    "instant_shave",
    "cross_eyed",
    "hdr",
    "lipdub",
]
AdapterStatus: TypeAlias = Literal["available", "missing"]


def _default_adapter_pipelines() -> list[AdapterPipeline]:
    return []


class AdapterComponentPayload(BaseModel):
    id: AdapterID
    display_name: str
    kind: AdapterKind
    source: AdapterSource = "official"
    repo_id: str
    filename: str
    expected_size_bytes: int
    required_for: list[AdapterPipeline] = Field(default_factory=_default_adapter_pipelines)
    optional_for: list[AdapterPipeline] = Field(default_factory=_default_adapter_pipelines)


class AdapterStatusItem(AdapterComponentPayload):
    status: AdapterStatus
    path: str | None = None


class AdapterRequirementItem(BaseModel):
    adapter_id: AdapterID
    display_name: str
    satisfied: bool
    path: str | None = None
    downloadable: bool = True
    repo_id: str
    filename: str


class AdapterStatusResponse(BaseModel):
    adapters: list[AdapterStatusItem]


class AdapterRecommendationResponse(BaseModel):
    pipeline: AdapterPipeline
    required: list[AdapterRequirementItem]
    missing: list[AdapterID]
    cps_to_download: list[ModelCheckpointID]


# ============================================================
# Model Library Scanner / Catalog Types (Phase 1 — read-only)
# ============================================================

#: Broad physical kind of a discovered artifact. Not everything is a LoRA.
ArtifactKind: TypeAlias = Literal[
    "diffusion_model",
    "vae",
    "text_encoder",
    "gguf",
    "upscaler",
    "control_adapter",
    "lora",
    "scene_embeddings",
    "depth_processor",
    "pose_processor",
    "person_detector",
    "image_gen_model",
]

#: Semantic role within a profile/pipeline (e.g. adapter id, ``base_diffusion_model``).
#: ``str`` (not Literal) because roles grow with the adapter registry.
ComponentRole: TypeAlias = str

#: How precisely the scanner matched a discovered file to a catalog entry.
ScannerConfidence: TypeAlias = Literal[
    "exact_catalog_match", "filename_match", "heuristic_match", "unknown",
]

#: Known-artifact status computed by the scanner.
ScanArtifactStatus: TypeAlias = Literal[
    "installed", "missing", "wrong_folder_usable", "duplicate",
]

#: Workflow/pipeline support status — independent of file presence.
SupportStatus: TypeAlias = Literal[
    "supported", "gated", "unvalidated", "not_applicable",
]


class ModelLibraryArtifact(BaseModel):
    """A known catalog artifact with its scan status and provenance."""

    filename: str
    artifact_kind: ArtifactKind
    component_role: ComponentRole
    status: ScanArtifactStatus
    scanner_confidence: ScannerConfidence
    canonical_relative_path: str
    expected_size_bytes: int
    repo_id: str
    source_url: str
    is_folder: bool = False
    absolute_paths: list[str] = Field(default_factory=list)
    preferred_path: str | None = None
    size_bytes: int | None = None
    support_status: SupportStatus = "supported"
    gated: bool = False
    notes: str = ""
    cp_id: ModelCheckpointID | None = None
    adapter_id: AdapterID | None = None


class UnknownFile(BaseModel):
    """An unrecognized file in the models root (never deleted)."""

    absolute_path: str
    relative_path: str
    size_bytes: int


class PartialFile(BaseModel):
    """A partial download artifact (``*.part`` / ``*.tmp``) — never installed."""

    absolute_path: str
    relative_path: str
    size_bytes: int
    suffix: str


def _default_scan_artifacts() -> list[ModelLibraryArtifact]:
    return []


def _default_unknown_files() -> list[UnknownFile]:
    return []


def _default_partial_files() -> list[PartialFile]:
    return []


class ModelLibraryScanResponse(BaseModel):
    """Read-only scan result of the user-selected models root."""

    models_dir: str
    scanned_at: str
    artifacts: list[ModelLibraryArtifact] = Field(default_factory=_default_scan_artifacts)
    unknown_files: list[UnknownFile] = Field(default_factory=_default_unknown_files)
    partial_files: list[PartialFile] = Field(default_factory=_default_partial_files)


# ============================================================
# Request Models
# ============================================================


LTXVideoGenResolution: TypeAlias = Literal["540p", "720p", "1080p", "1440p", "2160p"]
LTXVideoGenDuration: TypeAlias = Literal[5, 6, 8, 10, 12, 14, 16, 18, 20]
LTXVideoGenFps: TypeAlias = Literal[24, 25, 48, 50]
LTXVideoGenPipeline: TypeAlias = Literal["fast", "pro"]


class LTXVideoGenerationResolutionSpec(BaseModel):
    fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]]


class LTXVideoGenerationSpec(BaseModel):
    display_name: str
    supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]
    a2v_supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None = None


class LTXVideoGenerationModelSpecItem(BaseModel):
    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class GenerateVideoModelsSpecsResponse(BaseModel):
    local_models: list[LTXVideoGenerationModelSpecItem]
    api_models: list[LTXVideoGenerationModelSpecItem]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    resolution: LTXVideoGenResolution = "1080p"
    model: LTXVideoGenPipeline = "fast"
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    duration: LTXVideoGenDuration = 5
    fps: LTXVideoGenFps = 24
    audio: bool = False
    imagePath: str | None = None
    audioPath: str | None = None
    aspectRatio: Literal["16:9", "9:16"] = "16:9"
    output_format: OutputFormat | None = Field(default=None, validate_default=True)

    @field_validator("output_format", mode="before")
    @classmethod
    def _parse_output_format(cls, v: object) -> OutputFormat | None:
        """Accept the enum value string or null under strict mode; normalize None→MP4.

        The field type is ``OutputFormat | None`` with a ``None`` default so the
        OpenAPI/TS schema marks it OPTIONAL (clients may omit it; ``?:`` in TS).
        ``validate_default=True`` routes the omitted-default None through this
        validator, normalizing it to MP4 — so downstream handlers always receive a
        concrete ``OutputFormat`` (never None). Invalid strings still 422.
        """
        if v is None:
            return OutputFormat.MP4
        if isinstance(v, str) and not isinstance(v, OutputFormat):
            return OutputFormat(v)
        return v  # type: ignore[return-value]


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    width: int = Field(default=1024, ge=16)
    height: int = Field(default=1024, ge=16)
    numSteps: int = Field(default=4, ge=1)
    numImages: int = Field(default=1, ge=1)


def _default_model_types() -> set[ModelCheckpointID]:
    return set()


class ModelDownloadRequest(BaseModel):
    type: Literal["download", "upgrade"] = "download"
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


ModelAccessStatus: TypeAlias = Literal["authorized", "not_authorized"]


class CheckModelAccessRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


class CheckModelAccessResponse(BaseModel):
    access: dict[str, ModelAccessStatus]


class ModelDeleteRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


GapPromptMode: TypeAlias = Literal["text-to-video", "image-to-video", "text-to-image"]


class SuggestGapPromptRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    beforePrompt: str = ""
    afterPrompt: str = ""
    beforeFrame: str | None = None
    afterFrame: str | None = None
    gapDuration: float = 5
    mode: GapPromptMode = "text-to-video"
    inputImage: str | None = None

    @model_validator(mode="after")
    def _validate_input_image_mode(self) -> "SuggestGapPromptRequest":
        if self.inputImage is not None and self.mode != "image-to-video":
            raise ValueError("inputImage is only valid for image-to-video mode")
        return self


RetakeMode: TypeAlias = Literal["replace_audio_and_video", "replace_video", "replace_audio"]


class RetakeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: RetakeMode = "replace_audio_and_video"
    output_format: OutputFormat | None = Field(default=None, validate_default=True)

    @field_validator("output_format", mode="before")
    @classmethod
    def _parse_output_format(cls, v: object) -> OutputFormat | None:
        if v is None:
            return OutputFormat.MP4
        if isinstance(v, str) and not isinstance(v, OutputFormat):
            return OutputFormat(v)
        return v  # type: ignore[return-value]


ConditioningType: TypeAlias = Literal["canny", "depth"]


class IcLoraExtractRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    conditioning_type: ConditioningType = "canny"
    frame_time: float = 0


class IcLoraImageInput(BaseModel):
    model_config = ConfigDict(strict=True)

    path: str
    frame: int = 0
    strength: float = 1.0


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class IcLoraGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    video_path: str | None = None
    conditioning_type: ConditioningType | None = None
    prompt: str = ""
    output_format: OutputFormat | None = Field(default=None, validate_default=True)

    @field_validator("output_format", mode="before")
    @classmethod
    def _parse_output_format(cls, v: object) -> OutputFormat | None:
        if v is None:
            return OutputFormat.MP4
        if isinstance(v, str) and not isinstance(v, OutputFormat):
            return OutputFormat(v)
        return v  # type: ignore[return-value]
    conditioning_strength: float = 1.0
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = ""
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)
    adapter_id: AdapterID | None = None
    mask_path: str | None = None
    mask_grow_px: int = Field(default=30, ge=0, le=128, description="Mask dilation radius in pixels. Controls LTXVDilateVideoMask radii. 0=no dilation, default=30 matches official full-res (stage2) radius")
    laplacian_blend_grow: int = Field(default=12, ge=0, le=64, description="Controls Laplacian pyramid blend mask_low_res_dilation for inpaint. Larger values expand blend region at low-res level. Separate from mask_grow_px (dilation radii) and final_mask_blur_px (raw mask feather).")
    final_mask_blur_px: int = Field(default=6, ge=0, le=64, description="Blur radius for final raw-mask guard feather. Smoothens inpaint edge. 0=no feather. Separate from laplacian_blend_grow (pyramid blend level) and mask_grow_px (model context dilation).")
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    width: int = Field(default=704, ge=64, description="T2V output width (ingredients/no-video only)")
    height: int = Field(default=1280, ge=64, description="T2V output height (ingredients/no-video only)")
    num_frames: int = Field(default=121, ge=9, description="T2V frame count (ingredients/no-video only)")
    frame_rate: float = Field(default=24.0, gt=0.0, description="T2V frame rate (ingredients/no-video only)")
