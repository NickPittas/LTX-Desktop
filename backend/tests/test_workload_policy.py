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


class TestInpaintWorkloadPolicy:
    def _large(self, tier: float | None):
        return classify_lora_workload(workflow="in_outpainting", frame_count=193, width=1920, height=1088, vram_gib=tier)

    def test_ordinary_has_no_context_override(self):
        plan = classify_lora_workload(workflow="in_outpainting", frame_count=121, width=960, height=576, vram_gib=31)
        assert plan.inpaint_context_window_px is None and plan.inpaint_context_overlap_px is None

    def _tuple(self, tier: float | None):
        plan = self._large(tier)
        return (plan.inpaint_context_window_px, plan.inpaint_context_overlap_px, plan.resident_blocks, plan.blockswap_prefetch)

    def test_large_policy_tier_31(self): assert self._tuple(31) == (65, 16, 26, None)
    def test_large_policy_tier_28(self): assert self._tuple(28) == (65, 16, 26, None)
    def test_large_policy_tier_24(self): assert self._tuple(24) == (49, 16, 20, 0)
    def test_24gb_short_1080p_inpaint_caps_residency(self):
        plan = classify_lora_workload(
            workflow="in_outpainting", frame_count=41, width=1920, height=1088, vram_gib=24,
        )
        assert (
            plan.inpaint_context_window_px,
            plan.inpaint_context_overlap_px,
            plan.resident_blocks,
            plan.blockswap_prefetch,
        ) == (49, 16, 20, 0)
        assert "resident=20" in plan.cache_key_parts()
        assert "prefetch=0" in plan.cache_key_parts()
    def test_24gb_duration_only_inpaint_keeps_26_resident(self):
        plan = classify_lora_workload(
            workflow="in_outpainting", frame_count=193, width=960, height=576, vram_gib=24,
        )
        assert (
            plan.inpaint_context_window_px,
            plan.inpaint_context_overlap_px,
            plan.resident_blocks,
            plan.blockswap_prefetch,
        ) == (49, 16, 26, 0)
    def test_large_policy_tier_16(self): assert self._tuple(16) == (33, 8, 20, 0)
    def test_large_policy_tier_15(self): assert self._tuple(15) == (33, 8, 20, 0)
    def test_large_policy_tier_12(self): assert self._tuple(12) == (33, 8, 0, 0)
    def test_large_policy_below_12(self): assert self._tuple(11) == (33, 8, 0, 0)
    def test_large_policy_unknown(self): assert self._tuple(None) == (33, 8, 0, 0)

    def test_both_large_axes_reduce_residency(self):
        one = classify_lora_workload(workflow="in_outpainting", frame_count=193, width=960, height=576, vram_gib=31)
        assert (one.inpaint_context_window_px, one.inpaint_context_overlap_px, one.resident_blocks, one.blockswap_prefetch) == (65, 16, 37, None)
        assert self._tuple(31) == (65, 16, 26, None)

    def test_context_window_and_overlap_enter_cache_key(self):
        parts = self._large(31).cache_key_parts()
        assert "inpaint_ctx=65" in parts and "inpaint_overlap=16" in parts
