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

import ctypes
import ctypes.util
import errno
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


# Linux renameat2 flag: atomically rename only if the destination does not
# already exist (never clobber). See renameat2(2).
_RENAME_NOREPLACE = 0x1
# "Relative to current working directory" dirfd — lets us pass paths directly.
_AT_FDCWD = -100


def _resolve_renameat2() -> Any:
    """Resolve libc ``renameat2`` (glibc >= 2.28), or ``None`` if unavailable.

    Returns the configured ctypes callable, or ``None`` when libc or the
    symbol cannot be found (e.g. non-Linux / old glibc). Callers must fail
    safe in that case — never fall back to an overwriting rename.
    """
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return None
    try:
        libc = ctypes.CDLL(libc_name, use_errno=True)
    except OSError:
        return None
    func = getattr(libc, "renameat2", None)
    if func is None:
        return None
    func.argtypes = (
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    func.restype = ctypes.c_int
    return func


def _atomic_noreplace_rename(src: Path, dst: Path) -> None:
    """Atomically rename *src* -> *dst*, refusing to replace an existing *dst*.

    Backed by Linux ``renameat2(RENAME_NOREPLACE)``: a single atomic syscall
    that fails (``EEXIST``) if *dst* exists — including a *dst* that appears
    concurrently between a prior existence check and this call. This closes
    the check-then-rename race where plain ``rename()`` would silently replace
    an empty destination directory.

    Raises ``OSError(EEXIST)`` when *dst* exists, other ``OSError`` subtypes
    for genuine failures, and ``OSError(ENOSYS)`` if ``renameat2`` is not
    available. **Never falls back to an overwriting rename.**
    """
    func = _resolve_renameat2()
    if func is None:
        raise OSError(errno.ENOSYS, "renameat2 unavailable; refusing to clobber")
    result: int = func(
        _AT_FDCWD, os.fsencode(str(src)),
        _AT_FDCWD, os.fsencode(str(dst)),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(src))


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

    # Fast path: lexists() detects files, dirs, AND broken symlinks (exists()
    # follows the link and returns False for broken symlinks). The atomic
    # syscalls below still guard against a dst appearing concurrently.
    if os.path.lexists(dst):
        _discard_src(src)
        return False

    if src.is_dir():
        try:
            _atomic_noreplace_rename(src, dst)
        except OSError as exc:
            # EEXIST (incl. concurrent empty-dir race) → no-clobber skip.
            # ENOTEMPTY is included defensively in case a kernel/filesystem
            # surfaces a non-empty dst that way. Any other OSError (e.g.
            # ENOSYS when renameat2 is missing) is a genuine failure and
            # propagates rather than risking a clobber.
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                _discard_src(src)
                return False
            raise
        return True

    # Regular file: hard-link (atomic no-overwrite) then unlink the source.
    try:
        os.link(src, dst)
    except FileExistsError:
        _discard_src(src)
        return False
    src.unlink()
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
