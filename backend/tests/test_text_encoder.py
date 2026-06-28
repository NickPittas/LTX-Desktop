"""Tests for :class:`LTXTextEncoder` monkey-patch idempotency and silent
optional-module handling.

Covers the Phase 2 warning-cleanup fixes:

* ``PromptEncoder.__init__`` patch must be idempotent across repeated
  ``install_patches`` calls (no re-wrapping, no repeated "Installed" log).
* ``cleanup_memory`` patch must treat a missing optional module
  (``ModuleNotFoundError``) as a debug-level no-op instead of emitting a
  warning traceback.
"""

from __future__ import annotations

import logging

import pytest
import torch

from services.text_encoder.ltx_text_encoder import LTXTextEncoder


def _make_encoder() -> LTXTextEncoder:
    # http is only used by encode_via_api, never by the patch installers.
    return LTXTextEncoder(
        device=torch.device("cpu"),
        http=object(),  # type: ignore[arg-type]
        ltx_api_base_url="http://test.invalid",
    )


# ---------------------------------------------------------------------------
# PromptEncoder.__init__ patch idempotency
# ---------------------------------------------------------------------------


class TestPromptEncoderInitPatchIdempotent:
    def test_init_patch_sets_sentinel_and_does_not_rewrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ltx_pipelines.utils.blocks import PromptEncoder

        # Restore __init__ after the test so the patch does not leak.
        monkeypatch.setattr(PromptEncoder, "__init__", PromptEncoder.__init__)

        encoder = _make_encoder()
        encoder._install_prompt_encoder_init_patch()
        after_first = PromptEncoder.__init__
        assert getattr(after_first, "_ltx_desktop_api_init_patch", False)

        # A second install (and a fresh instance) must short-circuit on the
        # function-level sentinel rather than wrap the already-patched init.
        encoder2 = _make_encoder()
        encoder2._install_prompt_encoder_init_patch()
        assert PromptEncoder.__init__ is after_first

    def test_init_patch_does_not_relog_on_repeat(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from ltx_pipelines.utils.blocks import PromptEncoder

        monkeypatch.setattr(PromptEncoder, "__init__", PromptEncoder.__init__)

        encoder = _make_encoder()
        with caplog.at_level(logging.INFO):
            encoder._install_prompt_encoder_init_patch()
        first_count = sum(
            1 for r in caplog.records if "Installed PromptEncoder.__init__ patch" in r.message
        )
        assert first_count == 1

        caplog.clear()
        with caplog.at_level(logging.INFO):
            encoder._install_prompt_encoder_init_patch()

        repeat_count = sum(
            1 for r in caplog.records if "Installed PromptEncoder.__init__ patch" in r.message
        )
        assert repeat_count == 0


# ---------------------------------------------------------------------------
# PromptEncoder.__call__ patch idempotency (cross-instance)
# ---------------------------------------------------------------------------


class TestPromptEncoderCallPatchIdempotent:
    def test_call_patch_does_not_rewrap_across_instances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ltx_pipelines.utils.blocks import PromptEncoder

        monkeypatch.setattr(PromptEncoder, "__call__", PromptEncoder.__call__)

        def fake_state_getter() -> object:
            return object()

        encoder = _make_encoder()
        encoder._install_prompt_encoder_patch(fake_state_getter)
        after_first = PromptEncoder.__call__
        assert getattr(after_first, "_ltx_desktop_api_call_patch", False)

        # A fresh instance (new session) must not double-wrap __call__.
        encoder2 = _make_encoder()
        encoder2._install_prompt_encoder_patch(fake_state_getter)
        assert PromptEncoder.__call__ is after_first


# ---------------------------------------------------------------------------
# cleanup_memory patch: silent for missing optional modules + idempotent
# ---------------------------------------------------------------------------


class TestCleanupMemoryPatch:
    def test_missing_optional_module_is_debug_not_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing optional module logs at DEBUG, never a WARNING traceback."""
        import builtins

        from ltx_pipelines.utils import helpers as ltx_utils

        # Replace cleanup_memory with a bare stub (no sentinel) so the install
        # loop actually runs even if a prior run already patched it elsewhere.
        def stub_cleanup() -> None:
            return None

        monkeypatch.setattr(ltx_utils, "cleanup_memory", stub_cleanup)

        target = "ltx_pipelines.retake_pipeline"
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == target:
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)  # type: ignore[no-any-return]

        monkeypatch.setattr(builtins, "__import__", fake_import)

        encoder = _make_encoder()
        with caplog.at_level(logging.DEBUG):
            encoder._install_cleanup_memory_patch(lambda: object())

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        assert any(target in m and "absent" in m for m in debug_msgs), debug_msgs
        assert not any("Failed to patch cleanup_memory" in m for m in warning_msgs)
        assert encoder._cleanup_memory_patched is True

    def test_cleanup_patch_is_idempotent_across_instances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ltx_pipelines.utils import helpers as ltx_utils

        def stub_cleanup() -> None:
            return None

        monkeypatch.setattr(ltx_utils, "cleanup_memory", stub_cleanup)

        encoder = _make_encoder()
        encoder._install_cleanup_memory_patch(lambda: object())
        assert encoder._cleanup_memory_patched is True
        wrapped = ltx_utils.cleanup_memory
        assert getattr(wrapped, "_ltx_desktop_cleanup_patch", False)

        # Fresh instance must short-circuit on the function-level sentinel.
        encoder2 = _make_encoder()
        encoder2._install_cleanup_memory_patch(lambda: object())
        assert ltx_utils.cleanup_memory is wrapped
