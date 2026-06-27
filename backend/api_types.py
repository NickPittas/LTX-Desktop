"""Pydantic request/response models and typed aliases for ltx2_server."""

from __future__ import annotations

from typing import Annotated
from typing import Literal, NamedTuple, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

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


class DownloadProgressErrorResponse(BaseModel):
    status: Literal["error"]
    error: str


DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse | DownloadProgressCompleteResponse | DownloadProgressErrorResponse
)


class SuggestGapPromptResponse(BaseModel):
    status: Literal["success"] = "success"
    suggested_prompt: str


class GenerateVideoCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


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
    video_path: str
    conditioning_type: ConditioningType | None = None
    prompt: str = ""
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
