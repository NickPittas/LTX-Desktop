"""Tests for HDR utilities (LogC3 compress/decompress, decode postprocess, tonemap)."""

from __future__ import annotations

import math

import pytest
import torch

from services.ic_lora_pipeline.hdr_utils import (
    apply_hdr_decode_postprocess,
    logc3_compress,
    logc3_decompress,
    tonemap_for_sdr,
)


class TestLogC3Roundtrip:
    def test_roundtrip_sdr_range(self):
        """compress → decompress is identity for typical SDR-range values."""
        x = torch.linspace(0.0, 1.0, 100)
        recovered = logc3_decompress(logc3_compress(x))
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_roundtrip_hdr_range(self):
        """Roundtrip works for HDR values > 1.0."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        recovered = logc3_decompress(logc3_compress(x))
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_roundtrip_below_cut(self):
        """Linear region below cut point roundtrips exactly."""
        x = torch.linspace(0.0, 0.005, 50)
        recovered = logc3_decompress(logc3_compress(x))
        assert torch.allclose(x, recovered, atol=1e-7)

    def test_zero_is_zero(self):
        """Zero maps to zero in both directions."""
        assert logc3_compress(torch.tensor([0.0])).item() == pytest.approx(0.0)
        assert logc3_decompress(torch.tensor([0.0])).item() == pytest.approx(0.0)


class TestLogC3Properties:
    def test_compress_is_monotonic(self):
        """LogC3 compression is monotonically increasing."""
        x = torch.linspace(0.0, 10.0, 500)
        y = logc3_compress(x)
        assert torch.all(y[1:] >= y[:-1] - 1e-7)

    def test_continuity_at_cut(self):
        """C0 continuity at the cut point (linear region meets log region)."""
        from services.ic_lora_pipeline.hdr_utils import _CUT, _Y_AT_CUT, _LINEAR_SLOPE

        eps = 1e-8
        y_below = logc3_compress(torch.tensor([_CUT - eps]))
        y_above = logc3_compress(torch.tensor([_CUT + eps]))
        assert abs(y_below.item() - y_above.item()) < 1e-5

    def test_compress_reduces_dynamic_range(self):
        """Compressed HDR values fit in a smaller range than linear."""
        x = torch.tensor([0.01, 1.0, 10.0])
        y = logc3_compress(x)
        # Compressed range should be smaller than linear range
        assert (y.max() - y.min()).item() < (x.max() - x.min()).item()


class TestApplyHdrDecodePostprocess:
    def test_decompresses_logc3(self):
        """apply_hdr_decode_postprocess decompresses LogC3 to linear."""
        logc3_output = logc3_compress(torch.tensor([0.5, 1.0, 2.0]))
        linear = apply_hdr_decode_postprocess(logc3_output)
        expected = logc3_decompress(logc3_output)
        assert torch.allclose(linear, expected, atol=1e-5)

    def test_clamps_negative_to_zero(self):
        """Negative values (invalid light) are clamped to zero."""
        logc3_with_negative = torch.tensor([-0.5, 0.0, 0.5])
        result = apply_hdr_decode_postprocess(logc3_with_negative)
        assert result.min().item() >= 0.0


class TestTonemapForSDR:
    def test_maps_to_unit_range(self):
        """Reinhard tonemap maps [0, ∞) → [0, 1)."""
        linear = torch.tensor([0.0, 0.5, 1.0, 5.0, 100.0])
        sdr = tonemap_for_sdr(linear)
        assert sdr.min().item() >= 0.0
        assert sdr.max().item() < 1.0

    def test_zero_is_zero(self):
        assert tonemap_for_sdr(torch.tensor([0.0])).item() == pytest.approx(0.0)

    def test_monotonic(self):
        """Tonemapping is monotonically increasing."""
        linear = torch.linspace(0.0, 10.0, 100)
        sdr = tonemap_for_sdr(linear)
        assert torch.all(sdr[1:] >= sdr[:-1] - 1e-7)
