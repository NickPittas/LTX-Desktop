"""Phase 3A tests: transactional download infrastructure.

Pure helper tests + handler-level tests for scanner-aware skip, disk preflight,
per-item locks, safe atomic promote, rollback, and admin guards.

No mocks — uses tmp_path filesystem and monkeypatch only.
"""

from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest
import requests as http_requests
from _routes._errors import HTTPError
from api_types import ModelLibraryArtifact
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    resolve_downloading_dir,
    resolve_downloading_target_path,
    resolve_model_path,
)
from services.model_downloader.download_transaction import (
    DownloadLockError,
    InsufficientDiskSpaceError,
    acquire_download_lock,
    assert_under_root,
    download_lock_path,
    preflight_disk_space,
    safe_atomic_promote,
    should_skip_download,
)
from state.app_state_types import DownloadSessionId
from tests.conftest import TEST_ADMIN_TOKEN
from tests.http_error_assertions import assert_http_error

_ADMIN_HEADERS = {"X-Admin-Token": TEST_ADMIN_TOKEN}

DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])


# ============================================================
# Helpers
# ============================================================


def _artifact(
    status: str,
    *,
    component_role: str = "base_diffusion_model",
    absolute_paths: list[str] | None = None,
    preferred_path: str | None = None,
    canonical_relative_path: str = "model.safetensors",
    cp_id: str | None = None,
) -> ModelLibraryArtifact:
    return ModelLibraryArtifact(
        filename="model.safetensors",
        artifact_kind="diffusion_model",
        component_role=component_role,
        status=status,  # type: ignore[arg-type]
        scanner_confidence="exact_catalog_match",
        canonical_relative_path=canonical_relative_path,
        expected_size_bytes=1000,
        repo_id="test/repo",
        source_url="https://huggingface.co/test/repo",
        is_folder=False,
        absolute_paths=absolute_paths or [],
        preferred_path=preferred_path,
        size_bytes=None,
        support_status="supported",
        gated=False,
        notes="",
        cp_id=cp_id,  # type: ignore[arg-type]
        adapter_id=None,
    )


# ============================================================
# Root path assertion tests
# ============================================================


class TestRootAssertion:
    def test_accepts_path_under_root(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        safe = root / "sub" / "file.safetensors"
        assert_under_root(root, safe)  # should not raise

    def test_rejects_path_outside_root(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        escape = tmp_path / "escape.safetensors"
        with pytest.raises(ValueError, match="escapes root"):
            assert_under_root(root, escape)

    def test_rejects_traversal(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        escape = root / ".." / ".." / "escape.safetensors"
        with pytest.raises(ValueError, match="escapes root"):
            assert_under_root(root, escape)


# ============================================================
# Scanner-aware skip rule tests (pure)
# ============================================================


class TestScannerSkipRule:
    def test_installed_skipped(self):
        art = _artifact("installed", absolute_paths=["/m/model.safetensors"])
        assert should_skip_download(art, Path("/m")) is True

    def test_duplicate_at_canonical_skipped(self):
        art = _artifact(
            "duplicate",
            absolute_paths=["/m/model.safetensors", "/m/sub/model.safetensors"],
            canonical_relative_path="model.safetensors",
        )
        assert should_skip_download(art, Path("/m")) is True

    def test_duplicate_wrong_folder_only_not_skipped(self):
        art = _artifact(
            "duplicate",
            absolute_paths=["/m/sub/model.safetensors", "/m/other/model.safetensors"],
            canonical_relative_path="model.safetensors",
        )
        assert should_skip_download(art, Path("/m")) is False

    def test_wrong_folder_usable_not_skipped(self):
        art = _artifact(
            "wrong_folder_usable",
            absolute_paths=["/m/sub/model.safetensors"],
            canonical_relative_path="model.safetensors",
        )
        assert should_skip_download(art, Path("/m")) is False

    def test_missing_not_skipped(self):
        art = _artifact("missing")
        assert should_skip_download(art, Path("/m")) is False

    def test_none_artifact_not_skipped(self):
        assert should_skip_download(None, Path("/m")) is False


# ============================================================
# Disk-space preflight tests
# ============================================================


class TestDiskPreflight:
    def test_preflight_passes_with_enough_space(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        preflight_disk_space(models_dir, 1)  # should not raise

    def test_preflight_fails_insufficient_space(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(
            "shutil.disk_usage",
            lambda p: DiskUsage(total=100, used=95, free=5),
        )
        with pytest.raises(InsufficientDiskSpaceError):
            preflight_disk_space(models_dir, 1000)

    def test_preflight_noop_for_zero_required(self, tmp_path):
        preflight_disk_space(tmp_path / "nonexistent", 0)  # should not raise

    def test_preflight_creates_models_dir_if_missing(self, tmp_path):
        models_dir = tmp_path / "models"
        preflight_disk_space(models_dir, 1)
        assert models_dir.exists()


# ============================================================
# Per-item lock tests
# ============================================================


class TestDownloadLocks:
    def test_acquire_and_release(self, tmp_path):
        models_dir = tmp_path / "models"
        cp_id = "z-image-turbo"
        lock = acquire_download_lock(models_dir, cp_id)
        assert lock.acquired
        assert download_lock_path(models_dir, cp_id).exists()

        lock.release()
        assert not download_lock_path(models_dir, cp_id).exists()

    def test_contention_blocks_second_acquire(self, tmp_path):
        models_dir = tmp_path / "models"
        cp_id = "z-image-turbo"
        lock1 = acquire_download_lock(models_dir, cp_id)
        assert lock1.acquired

        lock2 = acquire_download_lock(models_dir, cp_id)
        assert not lock2.acquired

        lock1.release()
        lock3 = acquire_download_lock(models_dir, cp_id)
        assert lock3.acquired
        lock3.release()

    def test_release_does_not_delete_unowned_lock(self, tmp_path):
        """Release only deletes if acquired=True (this session created it)."""
        models_dir = tmp_path / "models"
        cp_id = "z-image-turbo"
        # Pre-create lock (another session)
        lock_path = download_lock_path(models_dir, cp_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("other")

        lock = acquire_download_lock(models_dir, cp_id)
        assert not lock.acquired
        lock.release()  # should NOT delete
        assert lock_path.exists()

    def test_lock_path_under_downloading(self, tmp_path):
        models_dir = tmp_path / "models"
        path = download_lock_path(models_dir, "z-image-turbo")
        assert ".downloading" in str(path)
        assert "locks" in str(path)


# ============================================================
# Safe atomic promote tests
# ============================================================


class TestSafeAtomicPromote:
    def test_promote_when_dst_absent(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging.safetensors"
        dst = root / "final.safetensors"
        src.write_bytes(b"new")

        assert safe_atomic_promote(src, dst, root) is True
        assert dst.read_bytes() == b"new"
        assert not src.exists()

    def test_no_overwrite_when_dst_exists(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging.safetensors"
        dst = root / "final.safetensors"
        src.write_bytes(b"new-download")
        dst.write_bytes(b"pre-existing-user-file")

        assert safe_atomic_promote(src, dst, root) is False
        assert dst.read_bytes() == b"pre-existing-user-file"
        assert not src.exists()

    def test_promote_folder(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging_dir"
        dst = root / "final_dir"
        (src).mkdir()
        (src / "model.safetensors").write_bytes(b"data")

        assert safe_atomic_promote(src, dst, root) is True
        assert (dst / "model.safetensors").read_bytes() == b"data"
        assert not src.exists()

    def test_no_overwrite_folder(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging_dir"
        dst = root / "final_dir"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"new")
        dst.mkdir()
        (dst / "model.safetensors").write_bytes(b"old")

        assert safe_atomic_promote(src, dst, root) is False
        assert (dst / "model.safetensors").read_bytes() == b"old"
        assert not src.exists()

    def test_promote_rejects_src_outside_root(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        src = tmp_path / "escape.safetensors"
        dst = root / "final.safetensors"
        src.write_bytes(b"bad")

        with pytest.raises(ValueError):
            safe_atomic_promote(src, dst, root)

    def test_promote_rejects_dst_outside_root(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging.safetensors"
        dst = tmp_path / "escape.safetensors"
        src.write_bytes(b"bad")

        with pytest.raises(ValueError):
            safe_atomic_promote(src, dst, root)

    def test_no_overwrite_broken_symlink_dst(self, tmp_path):
        """A broken symlink at dst counts as existing (Path.exists misses it)."""
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging.safetensors"
        dst = root / "final.safetensors"
        src.write_bytes(b"new-download")
        os.symlink(root / "nonexistent", dst)  # broken symlink

        assert safe_atomic_promote(src, dst, root) is False
        # Broken symlink preserved (not followed/replaced)
        assert dst.is_symlink()
        assert not dst.exists()
        assert not src.exists()

    def test_no_overwrite_symlink_to_file_dst(self, tmp_path):
        """A symlink pointing at a real file at dst is treated as existing."""
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging.safetensors"
        dst = root / "final.safetensors"
        target = root / "real.safetensors"
        target.write_bytes(b"target-data")
        os.symlink(target, dst)
        src.write_bytes(b"new-download")

        assert safe_atomic_promote(src, dst, root) is False
        # Symlink + target preserved (not overwritten in-place)
        assert dst.is_symlink()
        assert target.read_bytes() == b"target-data"
        assert not src.exists()

    def test_no_overwrite_broken_symlink_dst_folder_src(self, tmp_path):
        """Broken symlink dst is no-clobber even when src is a directory."""
        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging_dir"
        dst = root / "final_dir"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"new")
        os.symlink(root / "nonexistent", dst)  # broken symlink

        assert safe_atomic_promote(src, dst, root) is False
        assert dst.is_symlink()
        assert not src.exists()

    def test_directory_promote_race_does_not_clobber_empty_dst(self, tmp_path, monkeypatch):
        """If an empty dst directory appears between the lexists check and the
        atomic rename, it must NOT be replaced/clobbered. Plain rename() would
        silently replace an empty destination directory on POSIX; only
        renameat2(RENAME_NOREPLACE) closes this race atomically."""
        from services.model_downloader import download_transaction as mod

        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging_dir"
        dst = root / "final_dir"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"new")
        # dst intentionally absent so the lexists fast-path passes.

        real_rename = mod._atomic_noreplace_rename

        def racing_rename(src_p: Path, dst_p: Path) -> None:
            # Simulate a concurrent creator making dst (empty dir) immediately
            # before the no-replace syscall runs.
            os.mkdir(dst_p)
            return real_rename(src_p, dst_p)

        monkeypatch.setattr(mod, "_atomic_noreplace_rename", racing_rename)

        assert safe_atomic_promote(src, dst, root) is False
        # The raced dst survives — renameat2(NOREPLACE) refused to replace it.
        assert dst.is_dir()
        assert dst.exists()
        # src was discarded (treated as a skipped promote).
        assert not src.exists()

    def test_atomic_noreplace_rename_rejects_existing_dst(self, tmp_path):
        """The no-replace syscall helper raises OSError(EEXIST) for an existing
        dst instead of clobbering it."""
        import errno as _errno

        from services.model_downloader import download_transaction as mod

        root = tmp_path / "models"
        root.mkdir()
        src = root / "staging_dir"
        dst = root / "final_dir"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"new")
        dst.mkdir()  # empty existing dst
        sentinel = dst / "preexisting.txt"
        sentinel.write_bytes(b"keep-me")

        with pytest.raises(OSError) as exc:
            mod._atomic_noreplace_rename(src, dst)
        assert exc.value.errno == _errno.EEXIST
        # dst untouched (not clobbered), src NOT moved.
        assert sentinel.read_bytes() == b"keep-me"
        assert src.is_dir()


# ============================================================
# Handler-level tests: scanner skip via start_model_download
# ============================================================


class TestScannerAwareDownloadSkip:
    def test_installed_cp_not_downloaded(self, test_state):
        """CP at canonical root path → scanner skip, no download."""
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID  # z-image-turbo (folder)
        final = resolve_model_path(models_dir, cp_id)
        final.mkdir(parents=True)
        (final / "model.safetensors").write_bytes(b"\x00")

        test_state.downloads.start_model_download(download_type="download", cp_ids={cp_id})

        assert len(test_state.model_downloader.calls) == 0

    def test_wrong_folder_cp_still_downloaded(self, test_state):
        """CP in wrong folder → NOT skipped, downloads to canonical."""
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID  # Z-Image-Turbo
        # Place in subfolder (wrong folder for current runtime)
        wrong = models_dir / "subfolder" / "Z-Image-Turbo"
        wrong.mkdir(parents=True)
        (wrong / "model.safetensors").write_bytes(b"\x00")

        test_state.downloads.start_model_download(download_type="download", cp_ids={cp_id})

        # Was NOT skipped — downloader was called
        assert len(test_state.model_downloader.calls) == 1
        # Final file at canonical root path
        assert resolve_model_path(models_dir, cp_id).exists()


# ============================================================
# Handler-level tests: disk preflight
# ============================================================


class TestDiskPreflightHandler:
    def test_preflight_fails_before_session(self, test_state, monkeypatch):
        monkeypatch.setattr(
            "shutil.disk_usage",
            lambda p: DiskUsage(total=100, used=95, free=5),
        )
        with pytest.raises(HTTPError) as exc:
            test_state.downloads.start_model_download(
                download_type="download",
                cp_ids={IMG_GEN_MODEL_CP_ID},
            )
        assert exc.value.status_code == 409
        assert exc.value.detail == "INSUFFICIENT_DISK_SPACE"
        # No session created
        assert test_state.state.downloading_session is None
        # No background task started
        assert len(test_state.task_runner.errors) == 0


# ============================================================
# Handler-level tests: locks
# ============================================================


class TestLockContentionHandler:
    def test_lock_contention_fails_download(self, test_state):
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID
        # Pre-create lock (another session)
        lock_path = download_lock_path(models_dir, cp_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("other-session")

        session_id = test_state.downloads.start_model_download(download_type="download", cp_ids={cp_id})

        # Worker finalized as error internally (lock contention handled by the
        # worker, not the generic task-runner error handler).
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "error"
        assert progress.error_code == "DOWNLOAD_LOCKED"
        assert len(test_state.task_runner.errors) == 0
        # Pre-existing lock NOT deleted
        assert lock_path.exists()

    def test_lock_cleaned_on_success(self, test_state):
        cp_id = IMG_GEN_MODEL_CP_ID
        test_state.downloads.start_model_download(download_type="download", cp_ids={cp_id})

        models_dir: Path = test_state.config.default_models_dir
        assert not download_lock_path(models_dir, cp_id).exists()

    def test_lock_cleaned_on_failure(self, test_state):
        test_state.model_downloader.fail_next = RuntimeError("boom")
        cp_id = IMG_GEN_MODEL_CP_ID
        session_id = test_state.downloads.start_model_download(download_type="download", cp_ids={cp_id})

        # Worker finalized as error internally (no task-runner propagation).
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "error"
        assert len(test_state.task_runner.errors) == 0
        models_dir: Path = test_state.config.default_models_dir
        assert not download_lock_path(models_dir, cp_id).exists()

    def test_lock_contention_preserves_other_session_staging(self, test_state):
        """A worker that cannot acquire the lock must NOT delete another
        session's pre-existing lock or staging file for the same CP."""
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID
        # Pre-create lock + staging (another session actively downloading)
        lock_path = download_lock_path(models_dir, cp_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("other-session")
        staging_path = resolve_downloading_target_path(models_dir, cp_id)
        staging_path.mkdir(parents=True, exist_ok=True)
        (staging_path / "model.safetensors").write_bytes(b"other-session-data")

        test_state.downloads.start_model_download(download_type="download", cp_ids={cp_id})

        # Worker finalized as error internally due to lock contention.
        assert len(test_state.task_runner.errors) == 0
        # Pre-existing lock NOT deleted by the losing session
        assert lock_path.exists()
        # Pre-existing staging NOT deleted (this worker owns nothing)
        assert staging_path.exists()
        assert (staging_path / "model.safetensors").read_bytes() == b"other-session-data"


# ============================================================
# Handler-level tests: safe promote / no overwrite
# ============================================================


class TestCommitNoOverwriteHandler:
    def test_commit_does_not_overwrite_pre_existing_final(self, test_state):
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID

        # Pre-existing final
        final = resolve_model_path(models_dir, cp_id)
        final.mkdir(parents=True)
        (final / "model.safetensors").write_bytes(b"pre-existing")

        # Staged copy (simulating download result)
        staging = resolve_downloading_target_path(models_dir, cp_id)
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "model.safetensors").write_bytes(b"new-download")

        result = test_state.downloads._commit_staged_checkpoint(cp_id)
        assert result is False
        assert (final / "model.safetensors").read_bytes() == b"pre-existing"
        assert not staging.exists()


# ============================================================
# Handler-level tests: rollback
# ============================================================


class TestRollbackHandler:
    def test_rollback_deletes_committed_file(self, test_state):
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID
        committed = resolve_model_path(models_dir, cp_id)
        committed.mkdir(parents=True)
        (committed / "model.safetensors").write_bytes(b"committed")

        test_state.downloads._rollback_committed_checkpoints([cp_id])
        assert not committed.exists()

    def test_rollback_does_not_delete_preexisting_file(self, test_state):
        models_dir: Path = test_state.config.default_models_dir
        # Pre-existing file NOT in rollback list
        preexisting = resolve_model_path(models_dir, "ltx-2.3-spatial-upscaler-x2-1.0")
        preexisting.parent.mkdir(parents=True, exist_ok=True)
        preexisting.write_bytes(b"pre-existing")

        # Rollback empty list → nothing deleted
        test_state.downloads._rollback_committed_checkpoints([])
        assert preexisting.exists()
        assert preexisting.read_bytes() == b"pre-existing"

    def test_rollback_deletes_committed_preserves_preexisting(self, test_state):
        models_dir: Path = test_state.config.default_models_dir
        committed_cp = IMG_GEN_MODEL_CP_ID
        preexisting_cp = "ltx-2.3-spatial-upscaler-x2-1.0"

        committed_path = resolve_model_path(models_dir, committed_cp)
        committed_path.mkdir(parents=True, exist_ok=True)
        (committed_path / "model.safetensors").write_bytes(b"session-committed")

        preexisting_path = resolve_model_path(models_dir, preexisting_cp)
        preexisting_path.parent.mkdir(parents=True, exist_ok=True)
        preexisting_path.write_bytes(b"pre-existing")

        test_state.downloads._rollback_committed_checkpoints([committed_cp])

        assert not committed_path.exists()
        assert preexisting_path.exists()


# ============================================================
# Handler-level tests: failed download preserves file
# ============================================================


class TestFailedDownloadPreservesFile:
    def test_failed_download_preserves_pre_existing(self, test_state):
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID

        # Pre-existing final at the canonical path: the scanner sees it as
        # installed and skips the download. Simulates user data that must be
        # preserved (no overwrite).
        final = resolve_model_path(models_dir, cp_id)
        final.mkdir(parents=True)
        (final / "model.safetensors").write_bytes(b"user-data")

        # Attempt download — scanner skip sees it as installed, no download happens
        test_state.downloads.start_model_download(download_type="download", cp_ids={cp_id})
        assert len(test_state.model_downloader.calls) == 0
        # User file preserved
        assert (final / "model.safetensors").read_bytes() == b"user-data"


# ============================================================
# Handler-level tests: startup cleanup is non-destructive
# ============================================================


class TestStartupCleanupHandler:
    def test_startup_cleanup_preserves_downloading_dir_contents(self, test_state):
        """cleanup_downloading_dir() must not delete the shared .downloading/
        directory or foreign locks/staging belonging to another process."""
        models_dir: Path = test_state.config.default_models_dir
        downloading_dir = resolve_downloading_dir(models_dir)

        # Simulate another process/session's artifacts
        lock_path = download_lock_path(models_dir, IMG_GEN_MODEL_CP_ID)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("other-process")
        staging = resolve_downloading_target_path(models_dir, IMG_GEN_MODEL_CP_ID)
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "model.safetensors").write_bytes(b"foreign")

        test_state.downloads.cleanup_downloading_dir()

        # Shared dir preserved (not blanket-deleted)
        assert downloading_dir.exists()
        # Foreign lock preserved
        assert lock_path.exists()
        assert lock_path.read_text() == "other-process"
        # Foreign staging preserved
        assert staging.exists()
        assert (staging / "model.safetensors").read_bytes() == b"foreign"

    def test_startup_cleanup_creates_dir_if_missing(self, test_state):
        """cleanup_downloading_dir() is idempotent and ensures the dir exists."""
        import shutil

        models_dir: Path = test_state.config.default_models_dir
        downloading_dir = resolve_downloading_dir(models_dir)
        # AppHandler init creates this dir; remove it to verify (re)creation.
        if downloading_dir.exists():
            shutil.rmtree(downloading_dir)
        assert not downloading_dir.exists()

        test_state.downloads.cleanup_downloading_dir()

        assert downloading_dir.exists()


# ============================================================
# Handler-level tests: cancellation (Phase 3B)
# ============================================================


class TestDownloadCancellation:
    def test_cancel_no_active_session(self, test_state):
        """No active session → no_active_download."""
        result = test_state.downloads.cancel_download()
        assert result.status == "no_active_download"

    def test_cancel_active_session_returns_cancelling(self, test_state):
        session_id = test_state.downloads.start_download({IMG_GEN_MODEL_CP_ID})
        result = test_state.downloads.cancel_download()
        assert result.status == "cancelling"
        assert result.sessionId == str(session_id)
        # cancellation_requested flag set on active session.
        assert test_state.state.downloading_session is not None
        assert test_state.state.downloading_session.cancellation_requested is True

    def test_cancel_then_progress_reports_cancelled(self, test_state):
        session_id = test_state.downloads.start_download({IMG_GEN_MODEL_CP_ID})
        test_state.downloads.cancel_download()
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"

    def test_worker_cancellation_preserves_phase3a_cleanup(self, test_state, monkeypatch):
        """Cancellation triggered mid-worker: no lock leak, session-owned
        staging removed, no promote after cancellation observed."""
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID

        # Trigger cancellation after the staging download completes but
        # before the worker's post-download commit checkpoint.
        real_download = test_state.downloads._download_to_staging

        def cancelling_download(*args: Any, **kwargs: Any) -> None:
            real_download(*args, **kwargs)
            with test_state.downloads._lock:
                session = test_state.state.downloading_session
                if session is not None:
                    session.cancellation_requested = True

        monkeypatch.setattr(test_state.downloads, "_download_to_staging", cancelling_download)

        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={cp_id}
        )

        # Worker finalized as cancelled (not error, not complete).
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"

        # No lock leak: worker released its lock.
        assert not download_lock_path(models_dir, cp_id).exists()
        # Session-owned staging removed.
        assert not resolve_downloading_target_path(models_dir, cp_id).exists()
        # No promote after cancellation observed: final file not created.
        assert not resolve_model_path(models_dir, cp_id).exists()
        # No background error surfaced (cancellation handled internally).
        assert len(test_state.task_runner.errors) == 0

    def test_worker_cancellation_no_error_in_task_runner(self, test_state, monkeypatch):
        """DownloadCancelled is caught inside the worker, never reaches the
        generic task-runner error handler."""
        from handlers.download_handler import DownloadCancelled

        def cancel_immediately(session_id: DownloadSessionId) -> None:
            raise DownloadCancelled(session_id)

        # Cancel before the worker touches the first CP.
        monkeypatch.setattr(
            test_state.downloads,
            "_raise_if_download_cancelled",
            cancel_immediately,
        )
        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={IMG_GEN_MODEL_CP_ID}
        )
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"
        assert len(test_state.task_runner.errors) == 0

    def test_active_session_present_during_cancellation_cleanup(self, test_state, monkeypatch):
        """The active session must stay present (blocking new downloads) until
        cleanup/lock-release finishes; finalization clears it only afterwards."""
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID
        session_present_during_cleanup: list[bool] = []

        # Observe whether the active session is still set when cleanup runs.
        real_cleanup = test_state.downloads._cleanup_session_staging

        def observing_cleanup(*args: Any, **kwargs: Any) -> None:
            with test_state.downloads._lock:
                session_present_during_cleanup.append(
                    test_state.state.downloading_session is not None
                )
            return real_cleanup(*args, **kwargs)

        monkeypatch.setattr(test_state.downloads, "_cleanup_session_staging", observing_cleanup)

        # Trigger cancellation after staging completes, before the commit checkpoint.
        real_download = test_state.downloads._download_to_staging

        def cancelling_download(*args: Any, **kwargs: Any) -> None:
            real_download(*args, **kwargs)
            with test_state.downloads._lock:
                session = test_state.state.downloading_session
                if session is not None:
                    session.cancellation_requested = True

        monkeypatch.setattr(test_state.downloads, "_download_to_staging", cancelling_download)

        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={cp_id}
        )

        # The active session was still present during cleanup (before release).
        assert session_present_during_cleanup == [True]
        # After the worker finishes, the session is cleared (finalized cancelled).
        assert test_state.state.downloading_session is None
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"

    def test_cancel_accepted_during_cleanup_finalizes_cancelled(self, test_state, monkeypatch):
        """If cancel_download() is accepted during cleanup (while the session
        is still active), the terminal outcome must be ``cancelled`` even
        though the worker's precomputed outcome was ``complete``."""
        cp_id = IMG_GEN_MODEL_CP_ID

        # Monkeypatch cleanup to request cancellation mid-cleanup while the
        # active session is still present (before finalization).
        real_cleanup = test_state.downloads._cleanup_session_staging

        def cleanup_that_cancels(*args: Any, **kwargs: Any) -> None:
            test_state.downloads.cancel_download()
            return real_cleanup(*args, **kwargs)

        monkeypatch.setattr(test_state.downloads, "_cleanup_session_staging", cleanup_that_cancels)

        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={cp_id}
        )

        # Download work completed successfully (would normally be "complete"),
        # but cancel arrived during cleanup → terminal must be cancelled.
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"
        # Session was finalized and cleared.
        assert test_state.state.downloading_session is None

    def test_finish_download_with_cancellation_requested_finalizes_cancelled(self, test_state):
        """finish_download(session_id) must write cancelled (not complete) when
        the active matching session has cancellation_requested=True — the
        cancellation check and terminal write are atomic under the lock."""
        session_id = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        test_state.downloads.cancel_download()

        test_state.downloads.finish_download(session_id)

        assert test_state.state.downloading_session is None
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"

    def test_fail_download_with_cancellation_requested_finalizes_cancelled(self, test_state):
        """fail_download(..., session_id) must write cancelled (not error) when
        the active matching session has cancellation_requested=True — the
        cancellation check and terminal write are atomic under the lock."""
        session_id = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        test_state.downloads.cancel_download()

        test_state.downloads.fail_download("boom", "UNKNOWN_ERROR", session_id)

        assert test_state.state.downloading_session is None
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"


# ============================================================
# Handler-level tests: structured error codes (Phase 3B)
# ============================================================


class TestDownloadErrorCodes:
    def test_lock_contention_reports_download_locked(self, test_state):
        models_dir: Path = test_state.config.default_models_dir
        cp_id = IMG_GEN_MODEL_CP_ID
        # Pre-create lock (another session).
        lock_path = download_lock_path(models_dir, cp_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("other-session")

        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={cp_id}
        )
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "error"
        assert progress.error_code == "DOWNLOAD_LOCKED"

    def test_network_failure_reports_network_error(self, test_state):
        test_state.model_downloader.fail_next = http_requests.ConnectionError("connection refused")
        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={IMG_GEN_MODEL_CP_ID}
        )
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "error"
        assert progress.error_code == "NETWORK_ERROR"

    def test_generic_failure_reports_unknown_error(self, test_state):
        test_state.model_downloader.fail_next = RuntimeError("boom")
        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={IMG_GEN_MODEL_CP_ID}
        )
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "error"
        assert progress.error_code == "UNKNOWN_ERROR"

    def test_os_error_maps_to_unknown_error(self, test_state):
        """OSError (filesystem/promote) must NOT be treated as NETWORK_ERROR."""
        test_state.model_downloader.fail_next = OSError("disk read error")
        session_id = test_state.downloads.start_model_download(
            download_type="download", cp_ids={IMG_GEN_MODEL_CP_ID}
        )
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "error"
        assert progress.error_code == "UNKNOWN_ERROR"


# ============================================================
# Handler-level tests: session-id-aware state transitions (Phase 3B)
# ============================================================


class TestSessionIdAwareTransitions:
    def test_stale_worker_finish_does_not_overwrite_newer_session(self, test_state):
        """A stale worker (session A) cannot finalize a newer session (B)."""
        session_a = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        # Simulate A going stale: a newer session B is now active.
        session_b = test_state.downloads.start_download({"ltx-2.3-spatial-upscaler-x2-1.0"})

        # Stale worker A tries to finish — must no-op.
        test_state.downloads.finish_download(session_a)

        # Session B is still the active session.
        assert test_state.state.downloading_session is not None
        assert test_state.state.downloading_session.id == session_b
        # A was not recorded as complete by the stale worker.
        assert session_a not in test_state.state.completed_download_sessions

    def test_stale_worker_fail_does_not_overwrite_newer_session(self, test_state):
        session_a = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        session_b = test_state.downloads.start_download({"ltx-2.3-spatial-upscaler-x2-1.0"})

        test_state.downloads.fail_download("boom", "UNKNOWN_ERROR", session_a)

        assert test_state.state.downloading_session is not None
        assert test_state.state.downloading_session.id == session_b
        assert session_a not in test_state.state.completed_download_sessions

    def test_stale_worker_cancel_finalize_does_not_overwrite_newer_session(self, test_state):
        session_a = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        session_b = test_state.downloads.start_download({"ltx-2.3-spatial-upscaler-x2-1.0"})

        test_state.downloads._finalize_cancelled(session_a)

        assert test_state.state.downloading_session is not None
        assert test_state.state.downloading_session.id == session_b
        assert session_a not in test_state.state.completed_download_sessions

    def test_terminal_cancelled_progress_after_worker_finalizes(self, test_state):
        """Once the worker finalizes a cancelled session, progress reports
        cancelled from the terminal result."""
        session_id = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        test_state.downloads._finalize_cancelled(session_id)
        # Active session is now cleared.
        assert test_state.state.downloading_session is None
        progress = test_state.downloads.get_download_progress(str(session_id))
        assert progress.status == "cancelled"

    def test_background_safety_net_error_cannot_clobber_newer_session(self, test_state):
        """An escaped background error from a stale worker (session A) must be
        session-aware and must NOT fail a newer active session (B)."""
        session_a = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        # Simulate A going stale: a newer session B is now active.
        session_b = test_state.downloads.start_download({"ltx-2.3-spatial-upscaler-x2-1.0"})

        # Stale worker A's escaped error reaches the session-aware safety net.
        test_state.downloads._on_background_download_error(RuntimeError("stale boom"), session_a)

        # Newer session B is untouched / still active.
        assert test_state.state.downloading_session is not None
        assert test_state.state.downloading_session.id == session_b
        # A was not recorded as a terminal error by the stale safety net.
        assert session_a not in test_state.state.completed_download_sessions


# ============================================================
# Endpoint tests: admin guards
# ============================================================


class TestAdminGuards:
    def test_download_requires_admin(self, client):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_delete_requires_admin(self, client):
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_check_access_requires_admin(self, client):
        response = client.post(
            "/api/models/check-access",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_progress_does_not_require_admin(self, client):
        """Progress remains regular auth/session-id based."""
        response = client.get(
            "/api/models/download/progress",
            params={"sessionId": "nonexistent"},
        )
        # 404 for unknown session, NOT 403 for missing admin
        assert response.status_code == 404
