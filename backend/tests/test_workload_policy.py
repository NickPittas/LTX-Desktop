"""VRAM-aware residency/prefetch for non-HDR large IC-LoRA workloads."""

from services.workload_policy import classify_lora_workload


def test_large_both_axes_high_vram_keeps_26_resident() -> None:
    plan = classify_lora_workload(
        workflow="standard_video",
        frame_count=193, width=1920, height=1080, vram_gib=31.0,
    )
    assert plan.resident_blocks == 26
    assert plan.blockswap_prefetch is None


def test_large_both_axes_low_vram_drops_residency_and_prefetch() -> None:
    plan = classify_lora_workload(
        workflow="standard_video",
        frame_count=193, width=1920, height=1080, vram_gib=24.0,
    )
    assert plan.resident_blocks == 20
    assert plan.blockswap_prefetch == 0
    assert "prefetch=0" in plan.cache_key_parts()


def test_large_single_axis_low_vram_reduces_and_disables_prefetch() -> None:
    plan = classify_lora_workload(
        workflow="standard_video",
        frame_count=193, width=1280, height=704, vram_gib=24.0,
    )
    assert plan.resident_blocks == 26
    assert plan.blockswap_prefetch == 0


def test_normal_workload_unchanged() -> None:
    plan = classify_lora_workload(
        workflow="standard_video",
        frame_count=97, width=768, height=432, vram_gib=24.0,
    )
    assert plan.label == "normal"
    assert plan.resident_blocks is None
    assert plan.blockswap_prefetch is None
