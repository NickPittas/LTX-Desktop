"""Unit tests for _first_checkpoint_path helper.

Tests the pure-data path extraction helper without importing any
patched modules (no torch, no GPU dependencies).
"""

from __future__ import annotations

import pytest

from services.patches.safetensors_metadata_fix import _first_checkpoint_path


class TestFirstCheckpointPath:
    """_first_checkpoint_path extracts first path from str/tuple/list."""

    def test_str_passthrough(self) -> None:
        assert _first_checkpoint_path("/m/transformer.safetensors") == "/m/transformer.safetensors"

    def test_tuple_first_element(self) -> None:
        result = _first_checkpoint_path(("/m/a.safetensors", "/m/b.safetensors"))
        assert result == "/m/a.safetensors"

    def test_list_first_element(self) -> None:
        result = _first_checkpoint_path(["/m/a.safetensors", "/m/b.safetensors"])
        assert result == "/m/a.safetensors"

    def test_single_element_tuple(self) -> None:
        assert _first_checkpoint_path(("/m/only.safetensors",)) == "/m/only.safetensors"

    def test_empty_tuple_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _first_checkpoint_path(())

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _first_checkpoint_path([])
