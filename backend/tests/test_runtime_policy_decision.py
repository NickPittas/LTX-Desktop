"""Tests for runtime policy decision helpers."""

from __future__ import annotations

import pytest

from runtime_config.runtime_policy import (
    decide_local_generation_mode,
    offload_mode_value_for_mode,
)


def test_darwin_always_unsupported() -> None:
    assert decide_local_generation_mode(system="Darwin", cuda_available=True, vram_gb=64) == "unsupported"
    assert decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None) == "unsupported"


def test_windows_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=False, vram_gb=24) == "unsupported"


def test_windows_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=14) == "unsupported"


def test_windows_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=None) == "unsupported"


def test_windows_streaming_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=24) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_windows_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=31) == "full_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=96) == "full_models_loading"


def test_linux_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=False, vram_gb=24) == "unsupported"


def test_linux_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=14) == "unsupported"


def test_linux_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=None) == "unsupported"


def test_linux_streaming_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_linux_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=31) == "full_models_loading"


def test_other_systems_fail_closed() -> None:
    assert decide_local_generation_mode(system="FreeBSD", cuda_available=True, vram_gb=48) == "unsupported"


def test_offload_mode_value_for_full_loading_is_none() -> None:
    assert offload_mode_value_for_mode("full_models_loading") == "none"


def test_offload_mode_value_for_streaming_mode_is_cpu() -> None:
    assert offload_mode_value_for_mode("streaming_models_loading") == "cpu"


def test_offload_mode_value_for_unsupported_asserts() -> None:
    with pytest.raises(AssertionError):
        offload_mode_value_for_mode("unsupported")
