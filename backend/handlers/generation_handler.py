"""Generation lifecycle handler."""

from __future__ import annotations

import logging
import threading
import time
from threading import RLock
from typing import TYPE_CHECKING, Literal

from api_types import (
    CancelCancellingResponse,
    CancelNoActiveGenerationResponse,
    CancelResponse,
    GenerationProgressResponse,
)
from handlers.base import StateHandlerBase, with_state_lock
from services.system_info.system_info import SystemInfo
from state.app_state_types import (
    ApiGeneration,
    AppState,
    GenerationCancelled,
    GenerationComplete,
    GenerationError,
    GenerationMetrics,
    GenerationProgress,
    GenerationRunning,
    GenerationState,
    GpuGeneration,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)
GenerationSlot = Literal["gpu", "api"]

#: Background sampler interval (~1 Hz per oracle decision).
_METRICS_SAMPLE_INTERVAL_SECONDS = 1.0


class GenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        system_info: SystemInfo,
    ) -> None:
        super().__init__(state, lock, config)
        self._system_info = system_info

    @with_state_lock
    def start_generation(self, generation_id: str) -> None:
        if self.is_generation_running():
            raise RuntimeError("Generation already in progress")
        if self.state.gpu_slot is None:
            raise RuntimeError("No active GPU pipeline")

        now = time.monotonic()
        self.state.active_generation = GpuGeneration(
            state=GenerationRunning(
                id=generation_id,
                progress=GenerationProgress(
                    phase="", progress=0, current_step=0, total_steps=0,
                    started_at=now, phase_started_at=now,
                ),
            )
        )
        self._start_metrics_sampler(generation_id)

    @with_state_lock
    def start_api_generation(self, generation_id: str) -> None:
        if self.is_generation_running():
            raise RuntimeError("Generation already in progress")

        now = time.monotonic()
        self.state.active_generation = ApiGeneration(
            state=GenerationRunning(
                id=generation_id,
                progress=GenerationProgress(
                    phase="", progress=0, current_step=None, total_steps=None,
                    started_at=now, phase_started_at=now,
                ),
            )
        )
        self._start_metrics_sampler(generation_id)

    @with_state_lock
    def _gpu_generation(self) -> GenerationState | None:
        match self.state.active_generation:
            case GpuGeneration(state=generation) if self.state.gpu_slot is not None:
                return generation
            case _:
                return None

    @with_state_lock
    def _api_generation(self) -> GenerationState | None:
        match self.state.active_generation:
            case ApiGeneration(state=generation):
                return generation
            case _:
                return None

    @with_state_lock
    def _active_generation_state(self) -> tuple[GenerationSlot, GenerationState] | None:
        match self.state.active_generation:
            case GpuGeneration(state=generation) if self.state.gpu_slot is not None:
                return "gpu", generation
            case ApiGeneration(state=generation):
                return "api", generation
            case _:
                return None

    @with_state_lock
    def _running_slot(self) -> GenerationSlot | None:
        active = self._active_generation_state()
        if active is None:
            return None

        slot, generation = active
        match generation:
            case GenerationRunning():
                return slot
            case _:
                return None

    @with_state_lock
    def _running_generation(self) -> tuple[GenerationSlot, GenerationRunning] | None:
        active = self._active_generation_state()
        if active is None:
            return None

        slot, generation = active
        match generation:
            case GenerationRunning() as running:
                return slot, running
            case _:
                return None

    @with_state_lock
    def _cancelled_generation(self) -> tuple[GenerationSlot, GenerationCancelled] | None:
        active = self._active_generation_state()
        if active is None:
            return None

        slot, generation = active
        match generation:
            case GenerationCancelled() as cancelled:
                return slot, cancelled
            case _:
                return None

    @with_state_lock
    def _set_generation_state(self, slot: GenerationSlot, generation: GenerationState) -> None:
        if slot == "gpu":
            self.state.active_generation = GpuGeneration(state=generation)
            return
        self.state.active_generation = ApiGeneration(state=generation)

    @with_state_lock
    def _generation_for_polling(self) -> GenerationState | None:
        active = self._active_generation_state()
        return None if active is None else active[1]

    @with_state_lock
    def is_generation_cancelled(self) -> bool:
        match self._active_generation_state():
            case (_, GenerationCancelled()):
                return True
            case _:
                return False

    @with_state_lock
    def update_progress(
        self,
        phase: str,
        progress: int,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        running_generation = self._running_generation()
        if running_generation is None:
            return

        _, running = running_generation
        # Reset phase timer when phase changes (for phaseElapsedSeconds).
        if running.progress.phase != phase:
            running.progress.phase_started_at = time.monotonic()
        running.progress.phase = phase
        running.progress.progress = progress
        running.progress.current_step = current_step
        running.progress.total_steps = total_steps

    def _start_metrics_sampler(self, generation_id: str) -> None:
        """Start a ~1 Hz background telemetry sampler for *generation_id*.

        The daemon thread exits automatically when the generation is no longer
        running or has been replaced by a different generation. Sampling happens
        outside the state lock; only the snapshot write is locked.
        """
        def sampler() -> None:
            while True:
                # Sleep first so the initial state is set by the caller.
                time.sleep(_METRICS_SAMPLE_INTERVAL_SECONDS)

                # Sample outside the lock (IO may be slow).
                try:
                    telemetry = self._system_info.sample()
                except Exception:
                    logger.debug("Metrics sample failed", exc_info=True)
                    continue

                metrics = GenerationMetrics(
                    vram_used_mb=telemetry.get("vram_used_mb"),
                    vram_total_mb=telemetry.get("vram_total_mb"),
                    gpu_util_pct=telemetry.get("gpu_util_pct"),
                    ram_used_mb=telemetry.get("ram_used_mb"),
                    ram_total_mb=telemetry.get("ram_total_mb"),
                    cpu_util_pct=telemetry.get("cpu_util_pct"),
                )
                if not self._store_metrics_if_running(generation_id, metrics):
                    return

        thread = threading.Thread(target=sampler, daemon=True, name=f"metrics-{generation_id}")
        thread.start()

    @with_state_lock
    def _store_metrics_if_running(
        self, generation_id: str, metrics: GenerationMetrics
    ) -> bool:
        """Store *metrics* on the running generation if it is still *generation_id*.

        Returns False if the generation ended or was replaced (sampler exits).
        """
        running_generation = self._running_generation()
        if running_generation is None:
            return False
        _, running = running_generation
        if running.id != generation_id:
            return False
        running.progress.metrics = metrics
        return True

    @with_state_lock
    def cancel_generation(self) -> CancelResponse:
        running_generation = self._running_generation()
        if running_generation is not None:
            slot, running = running_generation
            self._set_generation_state(slot, GenerationCancelled(id=running.id))
            return CancelCancellingResponse(status="cancelling", id=running.id)

        cancelled_generation = self._cancelled_generation()
        match cancelled_generation:
            case (_, GenerationCancelled(id=generation_id)):
                return CancelCancellingResponse(status="cancelling", id=generation_id)
            case _:
                return CancelNoActiveGenerationResponse(status="no_active_generation")

    @with_state_lock
    def complete_generation(self, result: str | list[str]) -> None:
        running_generation = self._running_generation()
        if running_generation is None:
            return

        slot, running = running_generation
        self._set_generation_state(slot, GenerationComplete(id=running.id, result=result))

    @with_state_lock
    def fail_generation(self, error: str) -> None:
        running_generation = self._running_generation()
        if running_generation is not None:
            slot, running = running_generation
            logger.error("Generation %s failed: %s", running.id, error)
            self._set_generation_state(slot, GenerationError(id=running.id, error=error))
            return

        if self._cancelled_generation() is not None:
            return

        logger.error("Generation failed without active running job: %s", error)

    @with_state_lock
    def get_generation_progress(self) -> GenerationProgressResponse:
        gen = self._generation_for_polling()

        match gen:
            case GenerationRunning(progress=progress):
                now = time.monotonic()
                elapsed = (now - progress.started_at) if progress.started_at is not None else None
                phase_elapsed = (
                    (now - progress.phase_started_at)
                    if progress.phase_started_at is not None
                    else None
                )
                steps_per_second: float | None = None
                if (
                    progress.current_step is not None
                    and progress.current_step > 0
                    and elapsed is not None
                    and elapsed > 0
                ):
                    steps_per_second = progress.current_step / elapsed
                estimated_remaining: float | None = None
                if (
                    progress.total_steps is not None
                    and progress.current_step is not None
                    and steps_per_second is not None
                    and steps_per_second > 0
                ):
                    remaining_steps = progress.total_steps - progress.current_step
                    if remaining_steps > 0:
                        estimated_remaining = remaining_steps / steps_per_second

                m = progress.metrics
                return GenerationProgressResponse(
                    status="running",
                    phase=progress.phase,
                    progress=progress.progress,
                    currentStep=progress.current_step,
                    totalSteps=progress.total_steps,
                    elapsedSeconds=round(elapsed, 1) if elapsed is not None else None,
                    phaseElapsedSeconds=(
                        round(phase_elapsed, 1) if phase_elapsed is not None else None
                    ),
                    stepsPerSecond=(
                        round(steps_per_second, 2) if steps_per_second is not None else None
                    ),
                    estimatedRemainingSeconds=(
                        round(estimated_remaining, 1)
                        if estimated_remaining is not None
                        else None
                    ),
                    vramUsedMb=m.vram_used_mb if m is not None else None,
                    vramTotalMb=m.vram_total_mb if m is not None else None,
                    gpuUtilPct=m.gpu_util_pct if m is not None else None,
                    ramUsedMb=m.ram_used_mb if m is not None else None,
                    ramTotalMb=m.ram_total_mb if m is not None else None,
                    cpuUtilPct=m.cpu_util_pct if m is not None else None,
                )
            case GenerationComplete():
                return GenerationProgressResponse(
                    status="complete",
                    phase="complete",
                    progress=100,
                    currentStep=0,
                    totalSteps=0,
                )
            case GenerationCancelled():
                return GenerationProgressResponse(
                    status="cancelled",
                    phase="cancelled",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                )
            case GenerationError():
                return GenerationProgressResponse(
                    status="error",
                    phase="error",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                )
            case _:
                return GenerationProgressResponse(
                    status="idle",
                    phase="",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                )

    @with_state_lock
    def is_generation_running(self) -> bool:
        return self._running_slot() is not None
