"""Checkpoint download session handler."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING
from uuid import uuid4

import requests as http_requests

from _routes._errors import HTTPError
from api_types import (
    CheckModelAccessResponse,
    DownloadCancelResponse,
    DownloadCancelCancellingResponse,
    DownloadCancelNoActiveResponse,
    DownloadErrorCode,
    DownloadProgressCancelledResponse,
    DownloadProgressCompleteResponse,
    DownloadProgressErrorResponse,
    DownloadProgressResponse,
    DownloadProgressRunningResponse,
    ModelAccessStatus,
    ModelCheckpointID,
)
from handlers.base import StateHandlerBase, with_state_lock
from handlers.hf_auth_utils import require_hf_token
from handlers.models_handler import ModelsHandler
from runtime_config.model_download_specs import (
    ALL_MODEL_CP_IDS,
    get_model_cp_spec,
    is_cp_downloaded,
    resolve_downloading_dir,
    resolve_downloading_path,
    resolve_downloading_target_path,
    resolve_model_path,
)
from services.interfaces import ModelDownloader, TaskRunner
from services.model_downloader.download_transaction import (
    DownloadLock,
    DownloadLockError,
    InsufficientDiskSpaceError,
    acquire_download_lock,
    preflight_disk_space,
    safe_atomic_promote,
    should_skip_download,
)
from services.model_scanner import scan_models
from state.app_state_types import (
    AppState,
    DownloadSessionCancelled,
    DownloadSessionComplete,
    DownloadSessionError,
    DownloadSessionId,
    DownloadingSession,
    FileDownloadRunning,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class DownloadCancelled(Exception):
    """Internal sentinel raised at cancellation checkpoints in the worker.

    Caught inside ``_download_worker`` (never by the generic task-runner error
    handler) so that user-initiated cancellation finalizes as ``cancelled``
    rather than ``error``.
    """

    def __init__(self, session_id: DownloadSessionId) -> None:
        self.session_id = session_id
        super().__init__(f"Download cancelled: {session_id}")


class DownloadHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        models_handler: ModelsHandler,
        model_downloader: ModelDownloader,
        task_runner: TaskRunner,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._models_handler = models_handler
        self._model_downloader = model_downloader
        self._task_runner = task_runner

    def _ordered_cp_ids(self, cp_ids: Iterable[ModelCheckpointID]) -> tuple[ModelCheckpointID, ...]:
        cp_id_set = set(cp_ids)
        return tuple(cp_id for cp_id in ALL_MODEL_CP_IDS if cp_id in cp_id_set)

    @with_state_lock
    def is_download_running(self) -> bool:
        return self.state.downloading_session is not None

    @with_state_lock
    def start_download(self, cp_ids: set[ModelCheckpointID]) -> DownloadSessionId:
        session_id = DownloadSessionId(uuid4().hex)
        self.state.downloading_session = DownloadingSession(
            id=session_id,
            current_running_file=None,
            files_to_download=cp_ids,
            completed_files=set(),
            completed_bytes=0,
        )
        return session_id

    @with_state_lock
    def start_file(
        self,
        cp_id: ModelCheckpointID,
        target: str,
        session_id: DownloadSessionId | None = None,
    ) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        if session_id is not None and session.id != session_id:
            return
        if session.current_running_file is not None:
            session.completed_bytes += session.current_running_file.downloaded_bytes
            session.completed_files.add(session.current_running_file.file_type)
        session.current_running_file = FileDownloadRunning(
            file_type=cp_id,
            target_path=target,
            downloaded_bytes=0,
            speed_bytes_per_sec=0.0,
        )

    @with_state_lock
    def finish_download(self, session_id: DownloadSessionId | None = None) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        if session_id is not None and session.id != session_id:
            return
        # Cancellation-aware terminal write (atomic under the lock): if cancel
        # was accepted while the session was still active, finalize as cancelled
        # rather than complete. This closes the re-check-vs-write race.
        if session.cancellation_requested:
            self.state.completed_download_sessions[session.id] = DownloadSessionCancelled()
            self.state.downloading_session = None
            return
        if session.current_running_file is not None:
            session.completed_bytes += session.current_running_file.downloaded_bytes
            session.completed_files.add(session.current_running_file.file_type)
        self.state.completed_download_sessions[session.id] = DownloadSessionComplete()
        self.state.downloading_session = None

    @with_state_lock
    def update_file_progress(
        self,
        cp_id: ModelCheckpointID,
        downloaded: int,
        speed_bytes_per_sec: float,
        session_id: DownloadSessionId | None = None,
    ) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        if session_id is not None and session.id != session_id:
            return
        current = session.current_running_file
        if current is None or current.file_type != cp_id:
            return
        current.downloaded_bytes = downloaded
        current.speed_bytes_per_sec = speed_bytes_per_sec

    @with_state_lock
    def fail_download(
        self,
        error: str,
        error_code: DownloadErrorCode = "UNKNOWN_ERROR",
        session_id: DownloadSessionId | None = None,
    ) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        if session_id is not None and session.id != session_id:
            return
        # Cancellation-aware terminal write (atomic under the lock): if cancel
        # was accepted while the session was still active, finalize as cancelled
        # rather than error. This closes the re-check-vs-write race.
        if session.cancellation_requested:
            self.state.completed_download_sessions[session.id] = DownloadSessionCancelled()
            self.state.downloading_session = None
            return
        # Log only when actually recording an error outcome, not for stale/no-op
        # or cancellation-wins paths.
        logger.error("Checkpoint download failed: %s", error)
        self.state.completed_download_sessions[session.id] = DownloadSessionError(
            error_message=error,
            error_code=error_code,
        )
        self.state.downloading_session = None

    @with_state_lock
    def _finalize_cancelled(self, session_id: DownloadSessionId) -> None:
        """Record the active session as cancelled and clear it (session-id aware)."""
        session = self.state.downloading_session
        if session is None or session.id != session_id:
            return
        self.state.completed_download_sessions[session.id] = DownloadSessionCancelled()
        self.state.downloading_session = None

    def _make_progress_callback(
        self,
        cp_id: ModelCheckpointID,
        session_id: DownloadSessionId,
    ) -> Callable[[int], None]:
        last_sample_time = time.monotonic()
        last_sample_bytes = 0
        smoothed_speed = 0.0

        def on_progress(downloaded: int) -> None:
            nonlocal last_sample_time, last_sample_bytes, smoothed_speed
            now = time.monotonic()
            elapsed = now - last_sample_time
            if elapsed >= 1.0:
                instant_speed = (downloaded - last_sample_bytes) / elapsed
                if smoothed_speed == 0.0:
                    smoothed_speed = instant_speed
                else:
                    smoothed_speed = 0.3 * instant_speed + 0.7 * smoothed_speed
                last_sample_time = now
                last_sample_bytes = downloaded
            self.update_file_progress(cp_id, downloaded, smoothed_speed, session_id)

        return on_progress

    def _on_background_download_error(self, exc: Exception, session_id: DownloadSessionId) -> None:
        # Safety net only: the worker finalizes its own errors/cancellation
        # internally. If an exception somehow escapes, fail ONLY the
        # originating session so a stale worker cannot clobber a newer active
        # session (fail_download no-ops when the session id no longer matches).
        self.fail_download(str(exc), self._map_error_code(exc), session_id)

    @with_state_lock
    def is_download_cancelled(self, session_id: DownloadSessionId) -> bool:
        session = self.state.downloading_session
        if session is not None and session.id == session_id:
            return session.cancellation_requested
        return False

    def _raise_if_download_cancelled(self, session_id: DownloadSessionId) -> None:
        if self.is_download_cancelled(session_id):
            raise DownloadCancelled(session_id)

    @with_state_lock
    def cancel_download(self) -> DownloadCancelResponse:
        """Request cancellation of the active download session.

        Does NOT clear the active session immediately: the worker must release
        its locks and session-owned staging first. Repeated calls while cleanup
        is pending return the same ``cancelling`` response.
        """
        session = self.state.downloading_session
        if session is None:
            return DownloadCancelNoActiveResponse(status="no_active_download")
        session.cancellation_requested = True
        return DownloadCancelCancellingResponse(status="cancelling", sessionId=str(session.id))

    def _map_error_code(self, exc: Exception) -> DownloadErrorCode:
        """Map a worker exception to a structured error code.

        Deliberately narrow: only well-known network exceptions are mapped to
        ``NETWORK_ERROR``. ``OSError`` and other filesystem/promote errors are
        NOT assumed to be network failures — they surface as ``UNKNOWN_ERROR``.
        """
        if isinstance(exc, DownloadLockError):
            return "DOWNLOAD_LOCKED"
        if isinstance(exc, InsufficientDiskSpaceError):
            return "INSUFFICIENT_DISK_SPACE"
        if isinstance(exc, (http_requests.ConnectionError, http_requests.Timeout)):
            return "NETWORK_ERROR"
        return "UNKNOWN_ERROR"

    @with_state_lock
    def get_download_progress(self, session_id: str) -> DownloadProgressResponse:
        typed_session_id = DownloadSessionId(session_id)
        session = self.state.downloading_session
        if session is not None and session.id == typed_session_id:
            # User-initiated cancellation: report cancelled while the worker
            # is still releasing locks/staging.
            if session.cancellation_requested:
                return DownloadProgressCancelledResponse(status="cancelled")

            current = session.current_running_file
            current_downloaded = current.downloaded_bytes if current else 0
            total_downloaded = session.completed_bytes + current_downloaded
            expected_total_bytes = sum(get_model_cp_spec(cp_id).expected_size_bytes for cp_id in session.files_to_download)

            current_file_progress = 0.0
            if current is not None:
                spec = get_model_cp_spec(current.file_type)
                if spec.expected_size_bytes > 0:
                    current_file_progress = min(99.0, current.downloaded_bytes / spec.expected_size_bytes * 100)

            total_progress = 0.0
            if expected_total_bytes > 0:
                total_progress = min(99.0, total_downloaded / expected_total_bytes * 100)

            return DownloadProgressRunningResponse(
                status="downloading",
                current_downloading_file=current.file_type if current else None,
                current_file_progress=current_file_progress,
                total_progress=total_progress,
                total_downloaded_bytes=total_downloaded,
                expected_total_bytes=expected_total_bytes,
                completed_files=set(session.completed_files),
                all_files=set(session.files_to_download),
                speed_bytes_per_sec=current.speed_bytes_per_sec if current else 0.0,
                error=None,
            )

        result = self.state.completed_download_sessions.get(typed_session_id)
        if result is not None:
            match result:
                case DownloadSessionComplete():
                    return DownloadProgressCompleteResponse(status="complete")
                case DownloadSessionCancelled():
                    return DownloadProgressCancelledResponse(status="cancelled")
                case DownloadSessionError(error_message=error_message, error_code=error_code):
                    return DownloadProgressErrorResponse(
                        status="error", error=error_message, error_code=error_code
                    )

        raise ValueError(f"Unknown download session: {session_id}")

    def cleanup_downloading_dir(self) -> None:
        """Startup no-op for the shared ``.downloading/`` directory.

        Only ensures the directory exists. Must NOT delete it or any locks /
        staging files: those may belong to another process or session that is
        concurrently downloading. Session-owned staging is cleaned by the
        download worker itself.
        """
        downloading_dir = resolve_downloading_dir(self.models_dir)
        downloading_dir.mkdir(parents=True, exist_ok=True)

    def _download_to_staging(
        self,
        cp_id: ModelCheckpointID,
        hf_token: str | None,
        models_dir: Path,
        session_id: DownloadSessionId,
    ) -> None:
        spec = get_model_cp_spec(cp_id)
        self.start_file(cp_id, spec.name, session_id)
        progress_cb = self._make_progress_callback(cp_id, session_id)

        resolve_downloading_dir(models_dir).mkdir(parents=True, exist_ok=True)

        if spec.is_folder:
            self._model_downloader.download_snapshot(
                repo_id=spec.repo_id,
                local_dir=str(resolve_downloading_path(models_dir, cp_id)),
                on_progress=progress_cb,
                token=hf_token,
            )
        else:
            self._model_downloader.download_file(
                repo_id=spec.repo_id,
                filename=spec.name,
                local_dir=str(resolve_downloading_path(models_dir, cp_id)),
                on_progress=progress_cb,
                token=hf_token,
            )

    def _commit_staged_checkpoint(self, cp_id: ModelCheckpointID, models_dir: Path | None = None) -> bool:
        root = models_dir if models_dir is not None else self.models_dir
        src = resolve_downloading_target_path(root, cp_id)
        dst = resolve_model_path(root, cp_id)
        return safe_atomic_promote(src, dst, root)

    def _rollback_committed_checkpoints(
        self,
        cp_ids: Iterable[ModelCheckpointID],
        models_dir: Path | None = None,
    ) -> None:
        root = models_dir if models_dir is not None else self.models_dir
        for cp_id in cp_ids:
            spec = get_model_cp_spec(cp_id)
            path = resolve_model_path(root, cp_id)
            if spec.is_folder:
                if path.exists():
                    shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _discover_download_cp_ids(self, requested_cp_ids: set[ModelCheckpointID]) -> tuple[ModelCheckpointID, ...]:
        """Scanner-aware no-redownload skip.

        Scans the effective models_dir (outside lock) and skips CPs that are
        already installed or duplicate-at-canonical.  Does NOT skip
        wrong_folder_usable — current runtime cannot use those paths.
        """
        with self._lock:
            models_dir = self.models_dir
        catalog = scan_models(models_dir)
        catalog_by_cp = {a.cp_id: a for a in catalog.artifacts if a.cp_id is not None}

        missing: set[ModelCheckpointID] = set()
        for cp_id in requested_cp_ids:
            artifact = catalog_by_cp.get(cp_id)
            if should_skip_download(artifact, models_dir):
                continue
            # Fallback for CPs not covered by catalog
            if artifact is None and is_cp_downloaded(models_dir, cp_id):
                continue
            missing.add(cp_id)
        return self._ordered_cp_ids(missing)

    def _cleanup_session_staging(
        self,
        cp_ids: Iterable[ModelCheckpointID],
        models_dir: Path | None = None,
    ) -> None:
        """Remove only session-owned staging files (not shared .downloading/ dir)."""
        root = models_dir if models_dir is not None else self.models_dir
        for cp_id in cp_ids:
            spec = get_model_cp_spec(cp_id)
            staging_path = resolve_downloading_target_path(root, cp_id)
            try:
                if spec.is_folder:
                    if staging_path.exists():
                        shutil.rmtree(staging_path)
                else:
                    staging_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to clean up staging for %s", cp_id)

    def _download_worker(
        self,
        session_id: DownloadSessionId,
        cp_ids: tuple[ModelCheckpointID, ...],
        *,
        atomic_commit: bool,
    ) -> None:
        if not cp_ids:
            self.finish_download(session_id)
            return

        # Snapshot models_dir once so lock/download/promote/cleanup use a
        # consistent root for the whole worker lifetime (app settings could
        # otherwise change mid-download).
        with self._lock:
            models_dir = self.models_dir

        acquired_locks: list[DownloadLock] = []
        # Only CPs this worker actually acquired a lock for and staged.
        # Cleanup is restricted to these so we never delete another session's
        # staging (e.g. when we lose a lock contention race).
        owned_staging_cp_ids: list[ModelCheckpointID] = []
        # CPs committed by THIS session (atomic upgrade path only) — rolled
        # back on cancellation/error to preserve upgrade atomicity.
        committed_cp_ids: list[ModelCheckpointID] = []

        # Deferred terminal outcome. Finalization is intentionally delayed
        # until AFTER cleanup/lock-release so the active session stays present
        # (blocking new downloads) until all resources are released.
        outcome: str = "complete"
        error_message: str = ""
        error_code: DownloadErrorCode = "UNKNOWN_ERROR"

        try:
            try:
                self._raise_if_download_cancelled(session_id)
                # Token acquisition lives inside the guarded flow so auth/token
                # failures benefit from structured error mapping and
                # cancellation-wins semantics.
                hf_token = (
                    require_hf_token(self.state, self._lock)
                    if self.config.hf_gating_enabled
                    else None
                )
                if atomic_commit:
                    for cp_id in cp_ids:
                        self._raise_if_download_cancelled(session_id)
                        lock = acquire_download_lock(models_dir, cp_id)
                        if not lock.acquired:
                            raise DownloadLockError(str(cp_id))
                        acquired_locks.append(lock)
                        owned_staging_cp_ids.append(cp_id)
                        self._raise_if_download_cancelled(session_id)
                        logger.info("Downloading %s from %s", cp_id, get_model_cp_spec(cp_id).repo_id)
                        self._download_to_staging(cp_id, hf_token, models_dir, session_id)
                        self._raise_if_download_cancelled(session_id)

                    for cp_id in cp_ids:
                        self._raise_if_download_cancelled(session_id)
                        if self._commit_staged_checkpoint(cp_id, models_dir):
                            committed_cp_ids.append(cp_id)
                else:
                    for cp_id in cp_ids:
                        self._raise_if_download_cancelled(session_id)
                        lock = acquire_download_lock(models_dir, cp_id)
                        if not lock.acquired:
                            raise DownloadLockError(str(cp_id))
                        acquired_locks.append(lock)
                        owned_staging_cp_ids.append(cp_id)
                        self._raise_if_download_cancelled(session_id)
                        logger.info("Downloading %s from %s", cp_id, get_model_cp_spec(cp_id).repo_id)
                        self._download_to_staging(cp_id, hf_token, models_dir, session_id)
                        self._raise_if_download_cancelled(session_id)
                        self._commit_staged_checkpoint(cp_id, models_dir)

                # Final cancellation checkpoint before success finalization.
                self._raise_if_download_cancelled(session_id)
            except DownloadCancelled:
                self._rollback_committed_checkpoints(committed_cp_ids, models_dir)
                outcome = "cancelled"
            except Exception as exc:
                self._rollback_committed_checkpoints(committed_cp_ids, models_dir)
                # If cancellation was requested around this error, let cancel win.
                if self.is_download_cancelled(session_id):
                    outcome = "cancelled"
                else:
                    outcome = "error"
                    error_message = str(exc)
                    error_code = self._map_error_code(exc)
        finally:
            # Cleanup + lock release happen BEFORE finalization so the active
            # session remains present (new downloads blocked) until release.
            self._cleanup_session_staging(owned_staging_cp_ids, models_dir)
            for lock in acquired_locks:
                lock.release()

        # Finalize the terminal outcome only after cleanup/release is complete.
        # Re-check cancellation: a cancel accepted during cleanup (while the
        # session was still active) must win over the precomputed outcome, so
        # the terminal state is never complete/error after cancel is accepted.
        if self.is_download_cancelled(session_id):
            outcome = "cancelled"

        if outcome == "cancelled":
            self._finalize_cancelled(session_id)
        elif outcome == "error":
            self.fail_download(error_message, error_code, session_id)
        else:
            self.finish_download(session_id)

    def start_model_download(self, *, download_type: str, cp_ids: set[ModelCheckpointID]) -> DownloadSessionId:
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_DOWNLOADS_DISABLED_IN_FORCE_API_MODE")

        with self._lock:
            if self.state.downloading_session is not None:
                raise HTTPError(409, "DOWNLOAD_ALREADY_RUNNING")

        if download_type == "upgrade":
            resolved_upgrade = self._models_handler.resolve_upgrade_download(cp_ids)
            cp_ids_to_download = set(resolved_upgrade.cp_ids)
            ordered_cp_ids = resolved_upgrade.cp_ids
            atomic_commit = True
        elif download_type == "download":
            cp_ids_to_download = set(cp_ids)
            ordered_cp_ids = self._discover_download_cp_ids(cp_ids_to_download)
            atomic_commit = False
        else:
            raise HTTPError(400, "INVALID_DOWNLOAD_REQUEST")

        # Disk-space preflight before creating session / background task
        if ordered_cp_ids:
            with self._lock:
                models_dir = self.models_dir
            required_bytes = sum(
                get_model_cp_spec(cp_id).expected_size_bytes for cp_id in ordered_cp_ids
            )
            try:
                preflight_disk_space(models_dir, required_bytes)
            except InsufficientDiskSpaceError:
                raise HTTPError(409, "INSUFFICIENT_DISK_SPACE") from None

        session_id = self.start_download(set(ordered_cp_ids))
        self._task_runner.run_background(
            lambda: self._download_worker(session_id, ordered_cp_ids, atomic_commit=atomic_commit),
            task_name="model-download",
            on_error=lambda exc: self._on_background_download_error(exc, session_id),
            daemon=True,
        )
        return session_id

    def check_model_access(self, cp_ids: set[ModelCheckpointID]) -> CheckModelAccessResponse:
        repo_ids = {get_model_cp_spec(cp_id).repo_id for cp_id in cp_ids}

        if not self.config.hf_gating_enabled:
            return CheckModelAccessResponse(access={repo_id: "authorized" for repo_id in repo_ids})

        hf_token = require_hf_token(self.state, self._lock)

        access: dict[str, ModelAccessStatus] = {}
        for repo_id in sorted(repo_ids):
            try:
                response = http_requests.head(
                    f"https://huggingface.co/{repo_id}/resolve/main/.gitattributes",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    allow_redirects=True,
                    timeout=10,
                )
                access[repo_id] = "authorized" if response.status_code == 200 else "not_authorized"
            except Exception:
                access[repo_id] = "not_authorized"

        return CheckModelAccessResponse(access=access)
