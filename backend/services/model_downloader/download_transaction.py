"""Pure transactional helpers for the model downloader (Phase 3A).

Provides:
- root path assertions (no traversal escape);
- disk-space preflight;
- per-CP lock files using ``os.open(O_CREAT|O_EXCL)``;
- safe atomic promote that never overwrites an existing final file;
- scanner-aware no-redownload skip rule.

All filesystem paths are validated to be under the effective ``models_dir``.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from api_types import ModelCheckpointID, ModelLibraryArtifact
from runtime_config.model_download_specs import resolve_downloading_dir


# ============================================================
# Exceptions
# ============================================================


class InsufficientDiskSpaceError(Exception):
    """Raised when available disk space is below the required amount."""

    def __init__(self, required: int, available: int) -> None:
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient disk space: required {required} bytes, available {available} bytes"
        )


class DownloadLockError(Exception):
    """Raised when a per-CP lock cannot be acquired (another session holds it)."""

    def __init__(self, cp_id: str) -> None:
        self.cp_id = cp_id
        super().__init__(f"DOWNLOAD_LOCKED: {cp_id}")


# ============================================================
# Lock
# ============================================================


@dataclass(frozen=True, slots=True)
class DownloadLock:
    """Handle for a per-CP download lock file.

    ``acquired`` is True only when *this* instance created the lock file.
    ``release()`` deletes the lock file only when ``acquired`` is True,
    ensuring we never delete another session's lock.
    """

    path: Path
    acquired: bool

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


# ============================================================
# Root assertion
# ============================================================


def assert_under_root(root: Path, path: Path) -> None:
    """Assert that *path* resolves to a location under *root*.

    Uses lexical resolution (``Path.resolve()`` with ``strict=False``) so it
    works for not-yet-existing paths without requiring filesystem probing
    beyond symlink resolution.
    """
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"Path {path} escapes root {root}") from None


# ============================================================
# Disk-space preflight
# ============================================================


def preflight_disk_space(models_dir: Path, required_bytes: int) -> None:
    """Raise :class:`InsufficientDiskSpaceError` if free space is insufficient.

    A no-op when *required_bytes* ≤ 0.
    """
    if required_bytes <= 0:
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(models_dir)
    if usage.free < required_bytes:
        raise InsufficientDiskSpaceError(required=required_bytes, available=usage.free)


# ============================================================
# Per-CP lock files
# ============================================================


def download_lock_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    """Lock file path for a given CP under ``.downloading/locks/``."""
    downloading_dir = resolve_downloading_dir(models_dir)
    safe_name = cp_id.replace("/", "_").replace("\\", "_")
    return downloading_dir / "locks" / f"{safe_name}.lock"


def acquire_download_lock(models_dir: Path, cp_id: ModelCheckpointID) -> DownloadLock:
    """Try to create a per-CP lock file atomically.

    Returns a :class:`DownloadLock` with ``acquired=True`` if this call created
    the lock, or ``acquired=False`` if another session already holds it.
    """
    lock_path = download_lock_path(models_dir, cp_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return DownloadLock(path=lock_path, acquired=False)
    return DownloadLock(path=lock_path, acquired=True)


# ============================================================
# Safe atomic promote
# ============================================================


def _discard_src(src: Path) -> None:
    """Best-effort removal of *src* (file, directory, or symlink).

    Used when a promote is skipped because *dst* already exists; the staged
    *src* is discarded so it does not linger in ``.downloading/``.
    """
    try:
        if src.is_dir() and not src.is_symlink():
            shutil.rmtree(src)
        else:
            src.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _merge_dir_noclobber(src: Path, dst: Path) -> bool:
    """Merge staging dir *src* into final dir *dst*, never overwriting an
    existing file.

    Used for folder CPs instead of an atomic whole-directory rename because:

    - ``renameat2(RENAME_NOREPLACE)`` fails with ``EINVAL`` on filesystems that
      don't support the flag (NTFS/exFAT/some network mounts — common for large
      model drives), which broke every folder download there.
    - Multi-source folder CPs (e.g. the Gemma GGUF model file + its tokenizer
      files from a second repo) and a sibling file CP that lives inside the same
      folder (the gemma ``mmproj-BF16.gguf``) must COEXIST — a whole-folder
      rename cannot merge into an already-populated ``dst``.

    Per file: place it under *dst* only when nothing is there yet (per-file
    no-clobber). Works on any filesystem (plain rename / move, no special
    flags). Returns True if at least one file was placed.
    """
    placed = False
    if os.path.lexists(dst) and not dst.is_dir():
        # dst exists but is not a real directory (a file, or a broken symlink) —
        # merging files "into" it is ambiguous, so skip (no-clobber) and discard
        # the staged copy rather than risk clobbering it.
        _discard_src(src)
        return False
    for root_dir, _dirs, files in os.walk(src):
        rel_root = Path(root_dir).relative_to(src)
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            src_file = Path(root_dir) / name
            dst_file = target_root / name
            if os.path.lexists(dst_file):
                continue  # no-clobber: keep whatever is already there
            try:
                os.replace(src_file, dst_file)  # same-fs move, no rename flags
            except OSError:
                shutil.move(str(src_file), str(dst_file))  # cross-fs fallback
            placed = True
    _discard_src(src)
    return placed


def safe_atomic_promote(src: Path, dst: Path, root: Path) -> bool:
    """Atomically promote *src* to *dst* with no-clobber semantics.

    Returns ``True`` if promoted, ``False`` if *dst* already exists (skipped).

    **Never overwrites** an existing final file, directory, or symlink. If
    *dst* exists — including a broken symlink, which ``Path.exists()`` misses
    because it follows the link target — *src* is removed and the function
    returns ``False``.

    Implementation (Linux-focused):

    - Regular files: ``os.link(src, dst)`` is an atomic no-overwrite primitive
      on Linux (fails with ``FileExistsError`` if *dst* appears concurrently);
      the source inode is then unlinked. ``os.replace`` is deliberately NOT
      used because it overwrites unconditionally.
    - Directories: ``renameat2(RENAME_NOREPLACE)`` is a single atomic syscall
      that fails with ``EEXIST`` if *dst* exists — including a *dst* that
      appears concurrently between the ``lexists`` fast-path check and the
      rename. Plain ``rename()``/``os.replace`` is NOT used because POSIX
      allows it to silently replace an empty destination directory. If
      ``renameat2`` is unavailable, the function fails safe by raising
      ``OSError`` rather than risking a clobber.
    """
    assert_under_root(root, src)
    assert_under_root(root, dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        # Folder CP: per-file no-clobber merge. Works on filesystems without
        # renameat2(RENAME_NOREPLACE) (NTFS/exFAT/network) and lets multi-source
        # folder CPs + a sibling file CP inside the folder coexist.
        return _merge_dir_noclobber(src, dst)

    # Regular file: no-clobber, then place it. Prefer a hard-link (atomic,
    # never overwrites); the lexists() check preserves no-clobber intent and
    # also covers broken symlinks (which exists() would miss).
    if os.path.lexists(dst):
        _discard_src(src)
        return False
    try:
        os.link(src, dst)
        src.unlink()
    except FileExistsError:
        _discard_src(src)
        return False
    except OSError:
        # No hard-link support (e.g. exFAT) or cross-device link: move instead.
        try:
            os.replace(src, dst)
        except OSError:
            shutil.move(str(src), str(dst))
    return True


# ============================================================
# Scanner-aware no-redownload skip
# ============================================================


def should_skip_download(
    artifact: ModelLibraryArtifact | None,
    models_dir: Path,
) -> bool:
    """Determine whether a CP should be skipped (already available at runtime).

    Skip rules (oracle Phase 3A):
    - ``installed`` → skip.
    - ``duplicate`` → skip only if the current runtime canonical path is among
      ``absolute_paths``; do NOT skip when only wrong-folder copies exist.
    - ``wrong_folder_usable`` / ``missing`` → do NOT skip.
    - ``None`` (not in catalog) → do NOT skip.
    """
    if artifact is None:
        return False

    if artifact.status == "installed":
        return True

    if artifact.status == "duplicate":
        canonical = str(models_dir / artifact.canonical_relative_path)
        return canonical in set(artifact.absolute_paths)

    return False
