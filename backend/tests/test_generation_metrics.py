"""Tests for generation metrics/progress infrastructure (live-only telemetry).

Verifies elapsed/phase-elapsed computation, resource snapshot fields via fakes,
camelCase response keys, steps-per-second/ETA derivation, and that the sampler
store method is generation-id-aware.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from state.app_state_types import (
    ApiGeneration,
    GenerationComplete,
    GenerationMetrics,
    GenerationProgress,
    GenerationRunning,
    GpuGeneration,
)


def _set_running_api(test_state, *, phase: str = "inference", progress: int = 50,
                     current_step: int | None = 5, total_steps: int | None = 10,
                     started_at: float | None = None, phase_started_at: float | None = None,
                     metrics: GenerationMetrics | None = None) -> None:
    """Directly set an active running API generation on state."""
    now = time.monotonic()
    test_state.state.active_generation = ApiGeneration(
        state=GenerationRunning(
            id="gen-test",
            progress=GenerationProgress(
                phase=phase,
                progress=progress,
                current_step=current_step,
                total_steps=total_steps,
                started_at=started_at if started_at is not None else now,
                phase_started_at=phase_started_at if phase_started_at is not None else now,
                metrics=metrics,
            ),
        )
    )


class TestElapsedAndMetrics:
    def test_running_response_has_camel_case_metric_fields(self, test_state):
        """All new metric fields are present and use camelCase in the JSON response."""
        _set_running_api(test_state, metrics=GenerationMetrics(
            vram_used_mb=4096, vram_total_mb=8192, gpu_util_pct=75.0,
            ram_used_mb=16384, ram_total_mb=32768, cpu_util_pct=25.0,
        ))

        resp = test_state.generation.get_generation_progress()

        # Core fields unchanged
        assert resp.status == "running"
        assert resp.phase == "inference"
        assert resp.progress == 50
        assert resp.currentStep == 5
        assert resp.totalSteps == 10

        # New additive fields are present
        assert resp.elapsedSeconds is not None
        assert resp.phaseElapsedSeconds is not None
        assert resp.vramUsedMb == 4096
        assert resp.vramTotalMb == 8192
        assert resp.gpuUtilPct == 75.0
        assert resp.ramUsedMb == 16384
        assert resp.ramTotalMb == 32768
        assert resp.cpuUtilPct == 25.0

    def test_elapsed_seconds_computed_from_started_at(self, test_state):
        """elapsedSeconds is computed from started_at monotonic timestamp."""
        started = time.monotonic() - 10.0  # 10 seconds ago
        _set_running_api(test_state, started_at=started, phase_started_at=started,
                         current_step=0, total_steps=0)

        resp = test_state.generation.get_generation_progress()
        assert resp.elapsedSeconds is not None
        assert resp.elapsedSeconds >= 9.0  # ~10s elapsed (allow timing slack)

    def test_phase_elapsed_resets_on_phase_change(self, test_state):
        """Phase change in update_progress resets phase_started_at."""
        started = time.monotonic() - 20.0
        _set_running_api(test_state, phase="loading_model", started_at=started,
                         phase_started_at=started)

        # Phase hasn't changed yet → phase_elapsed ~20s
        resp = test_state.generation.get_generation_progress()
        assert resp.phaseElapsedSeconds is not None
        assert resp.phaseElapsedSeconds >= 15.0

        # Change phase → phase_started_at resets
        time.sleep(0.05)
        test_state.generation.update_progress("inference", 50, current_step=5, total_steps=10)

        resp = test_state.generation.get_generation_progress()
        assert resp.phaseElapsedSeconds is not None
        assert resp.phaseElapsedSeconds < 5.0  # just reset

    def test_steps_per_second_and_eta(self, test_state):
        """stepsPerSecond and estimatedRemainingSeconds computed from step count and elapsed."""
        started = time.monotonic() - 10.0
        _set_running_api(test_state, started_at=started, phase_started_at=started,
                         current_step=5, total_steps=10)

        resp = test_state.generation.get_generation_progress()
        assert resp.stepsPerSecond is not None
        # 5 steps in ~10s → ~0.5 steps/s
        assert 0.3 < resp.stepsPerSecond < 1.0
        assert resp.estimatedRemainingSeconds is not None
        # 5 remaining steps at ~0.5 steps/s → ~10s
        assert resp.estimatedRemainingSeconds is not None
        assert resp.estimatedRemainingSeconds > 5.0

    def test_no_metrics_snapshot_means_none_resource_fields(self, test_state):
        """When no metrics snapshot exists (sampler hasn't run yet), resource fields are None."""
        _set_running_api(test_state, metrics=None)

        resp = test_state.generation.get_generation_progress()
        assert resp.vramUsedMb is None
        assert resp.ramUsedMb is None
        assert resp.cpuUtilPct is None
        # elapsed still computed
        assert resp.elapsedSeconds is not None


class TestNonRunningStatesHaveNoMetrics:
    """Idle/complete/cancelled/error states must not populate metric fields."""

    def test_idle_has_no_metrics(self, test_state):
        resp = test_state.generation.get_generation_progress()
        assert resp.status == "idle"
        assert resp.elapsedSeconds is None
        assert resp.vramUsedMb is None

    def test_complete_has_no_metrics(self, test_state):
        test_state.state.active_generation = ApiGeneration(
            state=GenerationComplete(id="gen-test", result="/output.mp4")
        )
        resp = test_state.generation.get_generation_progress()
        assert resp.status == "complete"
        assert resp.elapsedSeconds is None
        assert resp.vramUsedMb is None


class TestMetricsStoreIsGenerationAware:
    """_store_metrics_if_running only writes to the matching generation."""

    def test_store_metrics_for_matching_generation(self, test_state):
        _set_running_api(test_state, metrics=None)
        metrics = GenerationMetrics(vram_used_mb=2048, vram_total_mb=8192)

        stored = test_state.generation._store_metrics_if_running("gen-test", metrics)
        assert stored is True

        resp = test_state.generation.get_generation_progress()
        assert resp.vramUsedMb == 2048

    def test_store_metrics_rejects_wrong_generation_id(self, test_state):
        _set_running_api(test_state, metrics=None)
        metrics = GenerationMetrics(vram_used_mb=2048)

        stored = test_state.generation._store_metrics_if_running("different-gen", metrics)
        assert stored is False

        # Metrics not written
        resp = test_state.generation.get_generation_progress()
        assert resp.vramUsedMb is None

    def test_store_metrics_rejects_when_not_running(self, test_state):
        """No active generation → store returns False."""
        metrics = GenerationMetrics(vram_used_mb=2048)
        stored = test_state.generation._store_metrics_if_running("gen-test", metrics)
        assert stored is False
