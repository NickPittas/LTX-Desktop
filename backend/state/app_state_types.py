"""Canonical state model for backend runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType, Protocol

from api_types import DownloadErrorCode, ModelCheckpointID, ModelProfilePayload
from state.conditioning_cache import ConditioningCache

if TYPE_CHECKING:
    from state.app_settings import AppSettings
    from services.interfaces import (
        A2VPipeline,
        DepthProcessorPipeline,
        FastVideoPipeline,
        ImageGenerationPipeline,
        IcLoraPipeline,
        PoseProcessorPipeline,
        RetakePipeline,
        TextEncoder,
    )
    import torch


# Download session
# ============================================================


DownloadSessionId = NewType("DownloadSessionId", str)


@dataclass(frozen=True)
class DownloadSessionComplete:
    status: str = "complete"


@dataclass(frozen=True)
class DownloadSessionError:
    error_message: str
    error_code: DownloadErrorCode = "UNKNOWN_ERROR"
    status: str = "error"


@dataclass(frozen=True)
class DownloadSessionCancelled:
    status: str = "cancelled"


DownloadSessionResult = DownloadSessionComplete | DownloadSessionError | DownloadSessionCancelled


def _default_completed_download_sessions() -> dict[DownloadSessionId, DownloadSessionResult]:
    return {}


@dataclass
class FileDownloadRunning:
    file_type: ModelCheckpointID
    target_path: str
    downloaded_bytes: int
    speed_bytes_per_sec: float


@dataclass
class DownloadingSession:
    id: DownloadSessionId
    current_running_file: FileDownloadRunning | None
    files_to_download: set[ModelCheckpointID]
    completed_files: set[ModelCheckpointID]
    completed_bytes: int
    cancellation_requested: bool = False


# ============================================================
# Text encoding
# ============================================================


@dataclass
class TextEncodingResult:
    video_context: torch.Tensor
    audio_context: torch.Tensor | None


class CachedTextEncoder(Protocol):
    def to(self, device: torch.device) -> "CachedTextEncoder":
        ...


def _new_prompt_cache() -> dict[tuple[str, bool, str], TextEncodingResult]:
    return {}


@dataclass
class TextEncoderState:
    service: TextEncoder
    # Key: (prompt, enhance_prompt, model_identity). ``model_identity`` is the
    # effective base model identity (selected checkpoint path, active profile
    # transformer, or downloaded model id) so prompt/API embeddings never leak
    # across live model selections (Step 4 / Phase 2).
    prompt_cache: dict[tuple[str, bool, str], TextEncodingResult] = field(default_factory=_new_prompt_cache)
    api_embeddings: TextEncodingResult | None = None
    cached_encoder: CachedTextEncoder | None = None


# ============================================================
# Pipeline state
# ============================================================


@dataclass
class VideoPipelineState:
    pipeline: FastVideoPipeline
    is_compiled: bool
    cache_key: tuple[str, ...] = ()


@dataclass
class PoseResources:
    pipeline: PoseProcessorPipeline
    person_detector_model_path: str
    pose_model_path: str


@dataclass
class ICLoraState:
    pipeline: IcLoraPipeline
    lora_paths: list[str]
    lora_strength: float = 1.0
    depth_pipeline: DepthProcessorPipeline | None = None
    depth_model_path: str | None = None
    adapter_path: str | None = None
    pose_resources: PoseResources | None = None
    conditioning_cache: ConditioningCache = field(default_factory=ConditioningCache)


@dataclass
class A2VPipelineState:
    pipeline: A2VPipeline


@dataclass
class RetakePipelineState:
    pipeline: RetakePipeline
    distilled: bool
    quantized: bool


# ============================================================
# Generation state
# ============================================================


@dataclass
class GenerationMetrics:
    """Live-only resource telemetry snapshot (sampled by background sampler).

    Not persisted — ephemeral, updated ~1 Hz while generation is running.
    """

    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    gpu_util_pct: float | None = None
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    cpu_util_pct: float | None = None


@dataclass
class GenerationProgress:
    phase: str
    progress: int
    current_step: int | None
    total_steps: int | None
    # Monotonic timestamps (set by handler) for elapsed/ETA computation.
    started_at: float | None = None
    phase_started_at: float | None = None
    # Latest resource telemetry snapshot (updated by background sampler).
    metrics: GenerationMetrics | None = None


@dataclass
class GenerationRunning:
    id: str
    progress: GenerationProgress


@dataclass
class GenerationComplete:
    id: str
    result: str | list[str]


@dataclass
class GenerationError:
    id: str
    error: str


@dataclass
class GenerationCancelled:
    id: str


GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled


@dataclass
class GpuGeneration:
    state: GenerationState


@dataclass
class ApiGeneration:
    state: GenerationState


ActiveGeneration = GpuGeneration | ApiGeneration


# ============================================================
# Device slots
# ============================================================


@dataclass
class GpuSlot:
    active_pipeline: VideoPipelineState | ICLoraState | A2VPipelineState | RetakePipelineState | ImageGenerationPipeline


@dataclass
class CpuSlot:
    active_pipeline: ImageGenerationPipeline


# HuggingFace auth
# ============================================================


@dataclass(frozen=True)
class HfNotAuthenticated:
    pass


@dataclass(frozen=True)
class HfOAuthPending:
    state: str
    code_verifier: str
    created_at: float


@dataclass(frozen=True)
class HfAuthenticated:
    access_token: str
    expires_at: float


HfAuthState = HfNotAuthenticated | HfOAuthPending | HfAuthenticated


# ============================================================
# Top-level state
# ============================================================


def _default_model_profiles() -> list[ModelProfilePayload]:
    return []


@dataclass
class AppState:
    downloading_session: DownloadingSession | None
    gpu_slot: GpuSlot | None
    active_generation: ActiveGeneration | None
    cpu_slot: CpuSlot | None
    text_encoder: TextEncoderState | None
    app_settings: AppSettings
    completed_download_sessions: dict[DownloadSessionId, DownloadSessionResult] = field(
        default_factory=_default_completed_download_sessions
    )
    hf_auth_state: HfAuthState = field(default_factory=HfNotAuthenticated)
    model_profiles: list[ModelProfilePayload] = field(default_factory=_default_model_profiles)
    active_model_profile_id: str | None = None
