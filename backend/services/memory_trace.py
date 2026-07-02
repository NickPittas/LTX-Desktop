"""Process-wide memory tracing utility (JSONL) for memory-strategy diagnostics.

Activation
----------
Tracing is gated on the ``LTX_MEMORY_TRACE_PATH`` environment variable. When it
is absent, every public entry point is a full no-op: no torch/psutil imports,
no CUDA/RSS reads, no file writes, no side effects beyond null context
managers. This keeps the hot request/generation path untouched in production
where tracing is not requested.

When ``LTX_MEMORY_TRACE_PATH`` points at a writable file, each event is appended
as one JSON line carrying the required memory fields plus caller-supplied
extras. ``torch`` and ``psutil`` are imported lazily inside the enabled sampling
path only.

Failures in sampling/serialization are always swallowed so tracing can never
raise into the request or generation path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_TRACE_PATH_ENV = "LTX_MEMORY_TRACE_PATH"
_RUN_ID_ENV = "LTX_MEMORY_TRACE_RUN_ID"
_PROCESS_RUN_ID = "__process__"
_PROCESS_CASE_ID = "__process__"

#: Process/thread lock serializing JSONL writes so interleaved events stay whole.
_write_lock = threading.Lock()

#: Active trace context (run_id / case_id). Inherited by child tasks/threads.
_current_context: ContextVar[MemoryTraceContext | None] = ContextVar(
    "ltx_memory_trace_context", default=None
)

#: In-context flag for one-shot HTTP terminal events within a single context.
#: Note: this is safe within one async/thread context; cross-middleware-boundary
#: coordination (e.g. FastAPI ``BaseHTTPMiddleware`` task isolation) should mirror
#: the flag onto request state by the caller.
_http_terminal_recorded: ContextVar[bool] = ContextVar(
    "ltx_memory_http_terminal_recorded", default=False
)


@dataclass(frozen=True)
class MemoryTraceContext:
    """Trace identity carried through a request/generation run."""

    run_id: str
    case_id: str


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time memory reading for ``label``."""

    label: str
    timestamp: str
    run_id: str
    case_id: str
    pid: int
    cuda_available: bool
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int
    rss_bytes: int
    device_name: str


def is_enabled() -> bool:
    """Return True when tracing is activated via the env path."""
    return bool(os.environ.get(_TRACE_PATH_ENV))


def current_context() -> MemoryTraceContext:
    """Active context, or a process-level fallback derived from the env."""
    ctx = _current_context.get()
    if ctx is not None:
        return ctx
    return MemoryTraceContext(
        run_id=os.environ.get(_RUN_ID_ENV) or _PROCESS_RUN_ID,
        case_id=_PROCESS_CASE_ID,
    )


@contextmanager
def use_context(context: MemoryTraceContext) -> Generator[None, None, None]:
    """Bind ``context`` as the active trace context for the duration of the block."""
    token = _current_context.set(context)
    try:
        yield
    finally:
        _current_context.reset(token)


def run_with_context(
    context: MemoryTraceContext,
    fn: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``fn(*args, **kwargs)`` with ``context`` as the active trace context."""
    with use_context(context):
        return fn(*args, **kwargs)


def mark_http_terminal() -> None:
    """Mark an HTTP terminal event as already recorded for this context."""
    _http_terminal_recorded.set(True)


def http_terminal_recorded() -> bool:
    """Whether an HTTP terminal event was already recorded for this context."""
    return _http_terminal_recorded.get()


def _read_stats() -> dict[str, Any]:
    """Read CUDA + RSS stats. Imported lazily; called only when enabled."""
    stats: dict[str, Any] = {
        "cuda_available": False,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "max_allocated_bytes": 0,
        "max_reserved_bytes": 0,
        "rss_bytes": 0,
        "device_name": "",
    }
    try:
        import torch

        cuda = torch.cuda
        if cuda.is_available():
            stats["cuda_available"] = True
            stats["allocated_bytes"] = int(cuda.memory_allocated())
            stats["reserved_bytes"] = int(cuda.memory_reserved())
            stats["max_allocated_bytes"] = int(cuda.max_memory_allocated())
            stats["max_reserved_bytes"] = int(cuda.max_memory_reserved())
            try:
                stats["device_name"] = str(cuda.get_device_name(0))
            except Exception:
                stats["device_name"] = ""
    except Exception:
        logger.debug("memory_trace: torch/CUDA read failed", exc_info=True)

    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        stats["rss_bytes"] = int(info.rss)
    except Exception:
        logger.debug("memory_trace: psutil RSS read failed", exc_info=True)

    return stats


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_event(event_type: str, label: str, **fields: object) -> None:
    """Append one JSONL event with required memory fields plus caller extras.

    No-op when tracing is disabled. Never raises.
    """
    if not is_enabled():
        return
    try:
        ctx = current_context()
        event: dict[str, object] = {
            "event_type": event_type,
            "label": label,
            "timestamp": _now_iso(),
            "run_id": ctx.run_id,
            "case_id": ctx.case_id,
            "pid": os.getpid(),
        }
        event.update(_read_stats())
        event.update(fields)
        line = json.dumps(event, default=str)
        path = os.environ[_TRACE_PATH_ENV]
        with _write_lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
    except Exception:
        logger.debug("memory_trace: write_event failed", exc_info=True)


def snapshot(label: str) -> MemorySnapshot:
    """Capture a memory snapshot, persist a ``snapshot`` event, and return it."""
    ctx = current_context()
    if not is_enabled():
        return MemorySnapshot(
            label=label,
            timestamp=_now_iso(),
            run_id=ctx.run_id,
            case_id=ctx.case_id,
            pid=os.getpid(),
            cuda_available=False,
            allocated_bytes=0,
            reserved_bytes=0,
            max_allocated_bytes=0,
            max_reserved_bytes=0,
            rss_bytes=0,
            device_name="",
        )

    stats = _read_stats()
    snap = MemorySnapshot(
        label=label,
        timestamp=_now_iso(),
        run_id=ctx.run_id,
        case_id=ctx.case_id,
        pid=os.getpid(),
        cuda_available=bool(stats["cuda_available"]),
        allocated_bytes=int(stats["allocated_bytes"]),
        reserved_bytes=int(stats["reserved_bytes"]),
        max_allocated_bytes=int(stats["max_allocated_bytes"]),
        max_reserved_bytes=int(stats["max_reserved_bytes"]),
        rss_bytes=int(stats["rss_bytes"]),
        device_name=str(stats["device_name"]),
    )
    write_event("snapshot", label)
    return snap


def reset_peak(label: str) -> None:
    """Reset CUDA peak memory stats and record a ``peak_reset`` event."""
    if not is_enabled():
        return
    try:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            logger.debug("memory_trace: reset_peak CUDA reset failed", exc_info=True)
        write_event("peak_reset", label)
    except Exception:
        logger.debug("memory_trace: reset_peak failed", exc_info=True)


@contextmanager
def phase(label: str) -> Generator[None, None, None]:
    """Trace a code phase: emits ``phase_start`` and ``phase_end``/``phase_error``.

    No-op (just yields) when tracing is disabled.
    """
    if not is_enabled():
        yield
        return
    start = time.monotonic()
    write_event("phase_start", label)
    try:
        yield
    except Exception:
        write_event(
            "phase_error",
            label,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
        raise
    write_event(
        "phase_end",
        label,
        duration_ms=int((time.monotonic() - start) * 1000),
    )
