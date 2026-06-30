"""Runtime policy decisions for local generation mode."""

from __future__ import annotations

from typing import Literal

LocalGenerationMode = Literal[
    "full_models_loading",
    "streaming_models_loading",
    "unsupported",
]


def decide_local_generation_mode(
    system: str, cuda_available: bool, vram_gb: int | None
) -> LocalGenerationMode:
    """Pick the local-generation mode for this runtime.

    - "unsupported": local generation is not viable; caller must route to the API.
    - "streaming_models_loading": enough VRAM to run, but model weights must be
      streamed from pinned host RAM (15-30 GB range).
    - "full_models_loading": enough VRAM to hold the whole model resident (>=31 GB),
      so streaming is skipped to avoid unnecessary host-RAM pressure.
    """
    if system == "Darwin":
        return "unsupported"

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return "unsupported"
        if vram_gb is None:
            return "unsupported"
        if vram_gb < 15:
            return "unsupported"
        if vram_gb < 31:
            return "streaming_models_loading"
        return "full_models_loading"

    # Fail closed for non-target platforms unless explicitly relaxed.
    return "unsupported"


def offload_mode_value_for_mode(mode: LocalGenerationMode) -> Literal["none", "cpu"]:
    """Return the ``OffloadMode`` value string to pass to a local pipeline.

    Maps the local generation mode to the upstream offload strategy:
    - ``full_models_loading`` -> ``"none"`` (full residency, no streaming).
    - ``streaming_models_loading`` -> ``"cpu"`` (stream weights from pinned host RAM).
    The caller converts this to ``OffloadMode(value)`` (never ``OffloadMode.DISK``).

    Must not be called when local generation is unsupported — callers should
    route through the API instead.
    """
    if mode == "unsupported":
        raise AssertionError(
            "offload_mode_value_for_mode called with 'unsupported' mode; "
            "callers must route to the API instead of constructing a local pipeline."
        )
    if mode == "full_models_loading":
        return "none"
    if mode == "streaming_models_loading":
        return "cpu"
    raise AssertionError(f"Unexpected LocalGenerationMode: {mode!r}")
