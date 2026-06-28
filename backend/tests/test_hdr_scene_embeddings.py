"""Tests for HDR scene embeddings loader/validator."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from services.ic_lora_pipeline.hdr_scene_embeddings import (
    HDRSceneEmbeddings,
    load_hdr_scene_embeddings,
)


def _write_safetensors(path, tensors: dict[str, torch.Tensor]) -> None:
    """Write a small safetensors file for testing."""
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(path),
    )


class TestLoadHDRSceneEmbeddings:
    def test_loads_video_context(self, tmp_path):
        """Valid file with video_context loads successfully."""
        path = tmp_path / "embeddings.safetensors"
        _write_safetensors(path, {"video_context": torch.zeros(2, 768, dtype=torch.float32)})

        result = load_hdr_scene_embeddings(str(path))
        assert isinstance(result, HDRSceneEmbeddings)
        assert result.video_context.shape == (2, 768)
        assert result.audio_context is None

    def test_loads_video_and_audio_context(self, tmp_path):
        """File with both keys loads both tensors."""
        path = tmp_path / "embeddings.safetensors"
        _write_safetensors(
            path,
            {
                "video_context": torch.zeros(2, 768, dtype=torch.float32),
                "audio_context": torch.zeros(1, 256, dtype=torch.float32),
            },
        )

        result = load_hdr_scene_embeddings(str(path))
        assert result.video_context.shape == (2, 768)
        assert result.audio_context is not None
        assert result.audio_context.shape == (1, 256)

    def test_missing_video_context_raises(self, tmp_path):
        """Missing video_context key raises ValueError with clear message."""
        path = tmp_path / "bad.safetensors"
        _write_safetensors(path, {"other_key": torch.zeros(1, 1)})

        with pytest.raises(ValueError, match="missing required key 'video_context'"):
            load_hdr_scene_embeddings(str(path))

    def test_non_float_dtype_raises(self, tmp_path):
        """Integer dtype for video_context raises ValueError."""
        path = tmp_path / "bad_dtype.safetensors"
        _write_safetensors(path, {"video_context": torch.zeros(2, 768, dtype=torch.int32)})

        with pytest.raises(ValueError, match="must be a floating-point tensor"):
            load_hdr_scene_embeddings(str(path))

    def test_1d_tensor_raises(self, tmp_path):
        """1-D video_context (insufficient dimensions) raises ValueError."""
        path = tmp_path / "bad_shape.safetensors"
        _write_safetensors(path, {"video_context": torch.zeros(768, dtype=torch.float32)})

        with pytest.raises(ValueError, match="must have at least 2 dimensions"):
            load_hdr_scene_embeddings(str(path))
