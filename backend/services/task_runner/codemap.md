# backend/services/task_runner/

## Responsibility
Provide the background-execution abstraction for the backend. Owns the `TaskRunner`
contract (Protocol) plus a concrete `ThreadingRunner` implementation that fires
fire-and-forget work onto daemon threads with centralized exception logging and an
optional caller-supplied error callback. Used to offload non-request-critical work
(e.g. async post-download side effects in `download_handler`) off the FastAPI
request thread.

## Design Patterns
- **Protocol-first service**: `task_runner.py` defines `TaskRunner` Protocol;
  `threading_runner.py` provides the only production implementation. The Protocol
  is re-exported via `services/interfaces.py` and `services/__init__.py` so callers
  depend on the interface, not the concrete class.
- **Idempotent, dependency-free constructor**: `ThreadingRunner()` takes no args,
  so `app_handler.py` constructs it inline (`task_runner=ThreadingRunner()`).
- **Defensive error containment**: `ThreadingRunner.run_background` wraps the
  caller's `target()` in try/except. Any `Exception` is routed to
  `logging_policy.log_background_exception(task_name, exc)`. If the caller passed
  `on_error`, it is invoked **inside its own try/except** so a buggy callback
  cannot escape (its failure is logged under `f"{task_name}:error-handler"`).
- **Keyword-only contract**: `task_name` is required; `on_error` and `daemon`
  default to `None` / `True`. Threads are always created with `daemon=True` by
  default so they never block process shutdown.
- **Test seam**: `tests/fakes/services.py` ships `FakeTaskRunner`; `conftest.py`
  injects it via `ServiceBundle` so handler tests never spawn real threads.

## Data & Control Flow
1. Caller invokes `runner.run_background(target, *, task_name, on_error=..., daemon=...)`.
2. `ThreadingRunner` defines an inner `_run()` that:
   a. calls `target()`;
   b. on `Exception` → `log_background_exception(task_name, exc)` then, if present,
      `on_error(exc)` (guarded).
3. A `threading.Thread(target=_run, daemon=daemon)` is created and `.start()`-ed;
   the method returns immediately (no handle returned — strictly fire-and-forget).
4. `log_background_exception` (in `backend/logging_policy.py`) is the single,
   consistent sink for off-thread tracebacks.

## Integration Points
- **`app_handler.py`**: constructs `ThreadingRunner()` in the composition root
  (alongside `from services.task_runner.threading_runner import ThreadingRunner`),
  stores it on `AppHandler.task_runner`, and passes it through to handlers.
- **`handlers/download_handler.py`**: sole production caller —
  `self._task_runner.run_background(...)` (around line 323) for background work
  triggered after a download completes.
- **`services/interfaces.py` / `services/__init__.py`**: re-export `TaskRunner` as
  the canonical import path for the rest of the codebase.
- **`logging_policy.py`**: `log_background_exception(task_name, exc)` — the
  exception reporter this runner depends on.
- **Tests**: `tests/test_logging_policy.py` exercises `ThreadingRunner` directly;
  `tests/fakes/services.py` provides `FakeTaskRunner` for handler-level tests.
