#!/usr/bin/env python3
"""Phase 1 live workflow harness.

A boring, stdlib-only out-of-band harness that either **spawns** the LTX
backend (``--backend auto``) or **attaches** to an already-running one
(``--backend http://127.0.0.1:PORT``), then exercises the Phase 1 generation
contracts (fast / Kijai / GGUF / IC-LoRA ingredients / HDR EXR / retake) and
validates output + memory-trace ownership per concrete case.

Design notes
------------
- **No new dependencies.** HTTP via ``urllib``; ``cv2`` / ``OpenImageIO`` /
  ``OpenEXR`` / the base-video registry are lazy-imported only when needed.
- **Spawn mode** only manages backends it owns. It truncates ``report.json``,
  ``summary.md``, ``backend.log`` and the trace path exactly once before the
  first spawn; per-case restarts *append*. It never deletes unknown files.
- **Attach mode** requires an explicit ``--trace-path`` + token/admin-token and
  never truncates the trace.
- **Memory-trace validation** reads the JSONL trace emitted by the backend
  memory-trace middleware (env ``LTX_MEMORY_TRACE_PATH`` /
  ``LTX_MEMORY_TRACE_RUN_ID``). Until that middleware lands, every routed case
  fails trace validation — this is intentional: the harness exists to prove the
  trace contract is honoured once it is wired in.

Run::

    cd backend && uv run python scripts/live_workflow_test.py [--backend auto|URL] ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# repo_root = .../LTX-Desktop  (this file lives in backend/scripts/).
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
SERVER_SCRIPT = BACKEND_DIR / "ltx2_server.py"

# The script lives in backend/scripts, so sys.path[0] is that directory, not
# backend root. Ensure backend root is importable before lazy imports like
# ``from services.base_video_model_registry import ...`` run.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# --------------------------------------------------------------------------- #
# Atomic case catalogue + request bodies
# --------------------------------------------------------------------------- #
SOURCELESS_ATOMICS = ["fast:default", "kijai:fast", "gguf:fast", "ic-lora:default", "harness:selftest"]
MEDIA_BACKED_ATOMICS = ["hdr:kijai_fp8_split", "hdr:gguf", "retake:default"]
ALL_ATOMICS = SOURCELESS_ATOMICS + MEDIA_BACKED_ATOMICS

KIJAI_ID = "ltx-2.3-22b-distilled-fp8-kijai-v3"
GGUF_ID = "ltx-2.3-22b-distilled-gguf-quantstack-q4-k-m"

# Accepted --force-memory-strategy values (mirror services.local_memory_plan's
# LocalMemoryStrategy Literal). NOTE: only "block_offload" is actually honoured
# by the backend override (LTX_FORCE_LOCAL_MEMORY_STRATEGY); other values are
# accepted on the CLI for discoverability but the backend fails closed on them.
FORCE_MEMORY_STRATEGY_CHOICES = (
    "upstream_streaming",
    "full_resident",
    "block_offload",
    "gguf_lazy",
    "gguf_native_streaming_later",
)
FORCE_MEMORY_STRATEGY_ENV = "LTX_FORCE_LOCAL_MEMORY_STRATEGY"

# Harness/debug HDR VAE tiling knobs (NOT production UI policy). Spawn-time
# backend env only; attach mode rejects them (cannot mutate a running backend).
HDR_VAE_TILE_SIZE_ENV = "LTX_HDR_VAE_TILE_SIZE"
HDR_VAE_TILE_THRESHOLD_ENV = "LTX_HDR_VAE_TILE_THRESHOLD"

# ponytail: prompts are sensible defaults; the "plan" referenced by the spec
# was not provided, so keep them here in one place for easy adjustment.
CITY_PROMPT = "A cinematic city street at golden hour"
INGREDIENTS_PROMPT = "A bowl of fresh fruit on a wooden table"
HDR_PROMPT = ""
RETAKE_PROMPT = "Replace the selected section with a steady cinematic shot"

# Statuses.
S_PASSED = "passed"
S_EXPECTED_FAILURE = "expected_failure"
S_ALLOWED_GATE = "allowed_gate"
S_PREFLIGHT_FAILED = "preflight_failed"
S_FAILED = "failed"

READY_RE = re.compile(r"Server running on http://127\.0\.0\.1:(\d+)")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _resolve_under_repo_root(p: str) -> Path:
    """Resolve a path against repo root only if it is relative."""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _normalize_path(p: str) -> str:
    """Collapse redundant slashes only; never resolves symlinks/mutates files."""
    return re.sub(r"/{2,}", "/", p)


def _http(method: str, url: str, token: str, admin_token: str, *, body: dict[str, Any] | None,
          case_id: str, run_id: str, admin_route: bool, timeout: float) -> tuple[int | None, str, str | None]:
    """Perform an authenticated request. Returns (status, body_text, error_text).

    status is None on a transport-level failure (error_text set). Every request
    carries the memory-trace correlation headers so the (future) middleware can
    attribute events to this run/case.
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if admin_route and admin_token:
        headers["X-Admin-Token"] = admin_token
    headers["X-LTX-Memory-Run-Id"] = run_id
    headers["X-LTX-Memory-Case-Id"] = case_id
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return int(resp.status), raw, None
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            raw = ""
        return int(exc.code), raw, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 — transport / timeout / connection
        return None, "", str(exc)


# --------------------------------------------------------------------------- #
# Backend instance (spawned, owned)
# --------------------------------------------------------------------------- #
class BackendInstance:
    """One spawned backend process + its drained stdout log segment list."""

    def __init__(self, proc: subprocess.Popen[str] | None, pid: int, port: int,
                 log_lines: list[str], log_fh: Any, ready_event: threading.Event,
                 drain_thread: threading.Thread) -> None:
        self.proc = proc
        self.pid = pid
        self.port = port
        self.log_lines = log_lines
        self._log_fh = log_fh
        self._ready_event = ready_event
        self._drain_thread = drain_thread

    def wait_ready(self, port: int, timeout: float) -> tuple[bool, str]:
        assert self.proc is not None  # attach stub (proc=None) never waits
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False, f"backend exited before ready (code={self.proc.returncode})"
            if self._ready_event.wait(timeout=0.5):
                break
        else:
            return False, f"backend did not signal readiness within {timeout:.0f}s"
        # The ready line carries the port; a mismatch is a hard failure.
        for line in self.log_lines:
            m = READY_RE.search(line)
            if m:
                got = int(m.group(1))
                if got != port:
                    return False, f"ready port mismatch: expected {port}, backend reported {got}"
                return True, "ready"
        return False, "ready event set but no parseable ready line found"

    def segment(self, start: int, end: int) -> list[str]:
        return self.log_lines[start:end]

    def teardown(self, token: str, admin_token: str, run_id: str) -> str:
        """Authenticated shutdown → SIGTERM pgroup → SIGKILL. Returns method used."""
        assert self.proc is not None  # attach stub (proc=None) never tears down
        method = "already_exited"
        if self.proc.poll() is None:
            status, _, _ = _http(
                "POST", f"http://127.0.0.1:{self.port}/api/system/shutdown",
                token, admin_token, body={}, case_id="__teardown__", run_id=run_id,
                admin_route=False, timeout=10.0,
            )
            method = "shutdown_api" if status == 200 else "shutdown_api_failed"
            # Wait up to 30s for graceful exit.
            for _ in range(300):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                method = method if method == "shutdown_api" else "terminate"
            except ProcessLookupError:
                pass
            except Exception:  # noqa: BLE001
                pass
            # Wait up to 10s after SIGTERM.
            for _ in range(100):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
        if self.proc.poll() is None:
            try:
                self.proc.kill()
                method = "kill"
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        # Wake the drain thread so it can finish.
        try:
            self._drain_thread.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._log_fh.close()
        except Exception:  # noqa: BLE001
            pass
        return method


def _spawn_backend(port: int, app_data_dir: Path, trace_path: Path | None, run_id: str,
                   token: str, admin_token: str, log_path: Path,
                   force_memory_strategy: str | None = None,
                   hdr_vae_tile_size: int | None = None,
                   hdr_vae_tile_threshold: int | None = None) -> BackendInstance:
    env = os.environ.copy()
    env.update({
        "LTX_PORT": str(port),
        "LTX_APP_DATA_DIR": str(app_data_dir),
        "LTX_AUTH_TOKEN": token,
        "LTX_ADMIN_TOKEN": admin_token,
        "PYTHONUNBUFFERED": "1",
    })
    if trace_path is not None:
        env["LTX_MEMORY_TRACE_PATH"] = str(trace_path)
        env["LTX_MEMORY_TRACE_RUN_ID"] = run_id
    # Benchmark-only: force a local memory strategy in the spawned backend.
    # Only "block_offload" is honoured by the backend override (see
    # services.local_memory_plan); other values fail closed inside the backend.
    if force_memory_strategy:
        env[FORCE_MEMORY_STRATEGY_ENV] = force_memory_strategy
    # Harness/debug HDR VAE tiling knobs (NOT production UI policy); spawn-time
    # backend env consumed by LTXHdrIcLoraPipeline.generate.
    if hdr_vae_tile_size is not None:
        env[HDR_VAE_TILE_SIZE_ENV] = str(hdr_vae_tile_size)
    if hdr_vae_tile_threshold is not None:
        env[HDR_VAE_TILE_THRESHOLD_ENV] = str(hdr_vae_tile_threshold)

    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    log_lines: list[str] = []
    log_fh = log_path.open("a", encoding="utf-8")
    ready_event = threading.Event()

    def _drain() -> None:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            log_lines.append(line)
            try:
                log_fh.write(line)
                log_fh.flush()
            except Exception:  # noqa: BLE001
                pass
            if READY_RE.search(line):
                ready_event.set()

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()
    return BackendInstance(proc, proc.pid, port, log_lines, log_fh, ready_event, drain_thread)


# --------------------------------------------------------------------------- #
# Output + trace validation
# --------------------------------------------------------------------------- #
def _validate_mp4(path: str) -> tuple[bool, str]:
    p = Path(path)
    if not p.is_file():
        return False, "mp4 missing"
    try:
        if p.stat().st_size <= 0:
            return False, "mp4 empty"
    except OSError as exc:
        return False, f"mp4 stat failed: {exc}"
    try:
        import cv2  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return False, f"cv2 unavailable: {exc}"
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return False, "cv2 could not open mp4"
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            return True, f"frames={count}"
        ok, _ = cap.read()
        return bool(ok), "one readable frame" if ok else "no readable frames"
    finally:
        cap.release()


def _open_exr(path: Path) -> bool:
    try:
        import OpenImageIO as oiio  # noqa: PLC0415
        inp = oiio.ImageInput.open(str(path))
        if inp is None:
            return False
        inp.read_image()
        inp.close()
        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import OpenEXR  # noqa: PLC0415, reportMissingImports
        return bool(OpenEXR.InputFile(str(path)))  # pyright: ignore[reportUnknownMemberType]
    except Exception:  # noqa: BLE001
        return False


def _validate_exr(root: str, proxy_path: str | None) -> tuple[bool, str]:
    d = Path(root)
    if not d.exists():
        return False, "exr sequence root missing"
    exrs = sorted(d.glob("*.exr"))
    if not exrs:
        return False, "no .exr frames in root"
    if not _open_exr(exrs[0]):
        return False, f"first exr unreadable: {exrs[0].name}"
    if not proxy_path:
        return False, "HDR response missing proxy_path (required for EXR)"
    ok, msg = _validate_mp4(proxy_path)
    if not ok:
        return False, f"proxy mp4 invalid: {msg}"
    return True, f"{len(exrs)} exr frames; proxy ok"


def _load_trace_events(trace_path: Path | None, run_id: str, case_id: str,
                       expect_pid: int | None) -> list[dict[str, Any]]:
    """Return trace events for this run/case. Stale (other run_id) events excluded."""
    if trace_path is None or not trace_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("run_id") != run_id or ev.get("case_id") != case_id:
            continue
        if expect_pid is not None:
            ep = ev.get("pid")
            if ep is not None and ep != expect_pid:
                continue
        events.append(ev)
    return events


def _event_type(ev: dict[str, Any]) -> str:
    # Real backend events (services/memory_trace.py) use ``event_type``.
    return str(ev.get("event_type") or ev.get("event") or ev.get("type") or ev.get("name") or "")


def _validate_trace(events: list[dict[str, Any]], hdr_create_required: bool) -> tuple[bool, str]:
    types = {_event_type(e) for e in events}
    has_start = "http_start" in types
    has_terminal = "http_end" in types or "http_error" in types
    if not has_start:
        return False, "missing http_start trace event"
    if not has_terminal:
        return False, "missing terminal (http_end/http_error) trace event"
    if hdr_create_required:
        # HDR pipeline construction is traced via memory_trace.phase("hdr_create")
        # / phase("pipeline_create:hdr_ic_lora") → phase_start/phase_end events.
        created = any(
            ("hdr" in lbl) and any(k in lbl for k in ("create", "construct", "build"))
            for lbl in (str(e.get("label", "")).lower() for e in events)
        )
        if not created:
            return False, "missing HDR pipeline create/construct trace event"
    return True, "trace ok"


# --------------------------------------------------------------------------- #
# Concrete-case body builders
# --------------------------------------------------------------------------- #
def _media_source(atomic: str, media: str, assets_dir: Path) -> tuple[str | None, str | None]:
    """Return (source_path, error) for a media-backed atomic."""
    if media == "mov":
        primary = assets_dir / "buildings_day_clean_121 prores4444.mov"
        secondary = assets_dir / "instant_shave_beard_121 prores4444.mov"
        chosen = primary if primary.is_file() else secondary
    else:  # mp4
        chosen = assets_dir / "hdr_input_video.mp4"
    if not chosen.is_file():
        return None, f"media source missing: {chosen}"
    return str(chosen), None


def _build_body(
    atomic: str,
    media: str | None,
    assets_dir: Path,
    hdr_overrides: dict[str, Any] | None = None,
    hdr_source_path: str | None = None,
) -> tuple[str, dict[str, Any], str | None]:
    """Return (route, body, error). error set means media/ingredient missing.

    ``hdr_overrides`` (harness-only) is merged into the body for HDR atomics
    only; ignored for every other atomic. ``hdr_source_path`` (harness-only)
    replaces the media-backed source video for HDR atomics only.
    """
    if atomic == "fast:default":
        return "/api/generate", {
            "prompt": CITY_PROMPT, "resolution": "540p", "model": "fast",
            "model_selection": None, "cameraMotion": "none", "negativePrompt": "",
            "duration": 5, "fps": 24, "audio": False, "imagePath": None,
            "audioPath": None, "aspectRatio": "16:9", "output_format": "mp4",
        }, None
    if atomic == "kijai:fast":
        _, body, _ = _build_body("fast:default", None, assets_dir)
        b = dict(body)
        b["model_selection"] = KIJAI_ID
        return "/api/generate", b, None
    if atomic == "gguf:fast":
        _, body, _ = _build_body("fast:default", None, assets_dir)
        b = dict(body)
        b["model_selection"] = GGUF_ID
        return "/api/generate", b, None
    if atomic == "ic-lora:default":
        img = assets_dir / "ingredients_input.jpg"
        if not img.is_file():
            return "/api/ic-lora/generate", {}, f"ingredients image missing: {img}"
        return "/api/ic-lora/generate", {
            "video_path": None, "conditioning_type": None, "adapter_id": "ingredients",
            "model_selection": None, "prompt": INGREDIENTS_PROMPT,
            "images": [{"path": str(img)}], "width": 704, "height": 1280,
            "num_frames": 121, "frame_rate": 24, "output_format": "mp4",
        }, None
    if atomic.startswith("hdr:"):
        # Harness-only: an explicit --hdr-source-path replaces the media-backed
        # source. HDR generate() derives the padded frame count from the decoded
        # source length, so frame-count limit tests must swap the source (not just
        # override the num_frames request field). When provided, media lookup is
        # skipped entirely.
        if hdr_source_path is not None:
            src = str(hdr_source_path)
        else:
            if media is None:
                return "", {}, "hdr atomic requires media"
            src, err = _media_source(atomic, media, assets_dir)
            if err:
                return "/api/ic-lora/generate", {}, err
        model_sel = KIJAI_ID if atomic == "hdr:kijai_fp8_split" else GGUF_ID
        body: dict[str, Any] = {
            "video_path": src, "conditioning_type": None, "adapter_id": "hdr",
            "model_selection": model_sel, "prompt": HDR_PROMPT, "images": [],
            "width": 704, "height": 1280, "num_frames": 121, "frame_rate": 24,
            "output_format": "exr_zip_float",
        }
        # Harness-only HDR body overrides (width/height/num_frames/frame_rate).
        if hdr_overrides:
            body.update(hdr_overrides)
        return "/api/ic-lora/generate", body, None
    if atomic == "retake:default":
        if media is None:
            return "", {}, "retake atomic requires media"
        src, err = _media_source(atomic, media, assets_dir)
        if err:
            return "/api/retake", {}, err
        return "/api/retake", {
            "video_path": src, "start_time": 0, "duration": 5,
            "prompt": RETAKE_PROMPT, "mode": "replace_video", "output_format": "mp4",
        }, None
    return "", {}, f"unknown atomic: {atomic}"


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class Harness:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = os.environ.get("LTX_MEMORY_TRACE_RUN_ID") or uuid.uuid4().hex
        self.token = args.token or os.environ.get("LTX_AUTH_TOKEN", "")
        self.admin_token = args.admin_token or os.environ.get("LTX_ADMIN_TOKEN", "")
        self.report_dir = _resolve_under_repo_root(args.report_dir)
        self.trace_path: Path | None = (
            _resolve_under_repo_root(args.trace_path) if args.trace_path else None)
        self.attach_mode = args.backend.lower() != "auto"
        # Spawn mode only: if no bearer/admin token was supplied (flag or env),
        # mint a random ephemeral one and use it for BOTH the spawned backend
        # env and every harness request. The admin guard rejects all admin
        # routes when its token is empty, so an unset admin token must become a
        # real ephemeral value — never an empty string. Attach mode keeps
        # requiring explicit tokens (validated in prepare()).
        if not self.attach_mode:
            if not self.token:
                self.token = uuid.uuid4().hex
            if not self.admin_token:
                self.admin_token = uuid.uuid4().hex
        self.results: list[dict[str, Any]] = []
        self.shared: BackendInstance | None = None
        self.started_at = _now_iso()
        self.shutdown_methods: list[str] = []
        self.log_path = self.report_dir / "backend.log"

        # Harness-only HDR body-field overrides (applied only to hdr:* atomics).
        self.hdr_body_overrides: dict[str, Any] = {}
        for _key, _val in (
            ("width", args.hdr_width),
            ("height", args.hdr_height),
            ("num_frames", args.hdr_frames),
            ("frame_rate", args.hdr_fps),
        ):
            if _val is not None:
                self.hdr_body_overrides[_key] = _val
        # Harness-only HDR source video override (hdr:* atomics only); takes the
        # place of the media-backed default source. Validated to exist up front.
        self.hdr_source_path: str | None = args.hdr_source_path

        # expect-failure / allow-gate maps keyed by concrete case id.
        self.expect_failure: dict[str, str] = {}
        for item in (args.expect_failure or []):
            if "=" in item:
                k, v = item.split("=", 1)
                self.expect_failure[k.strip()] = v
        self.allow_gate: dict[str, str] = {}
        for item in (args.allow_gate or []):
            if "=" in item:
                k, v = item.split("=", 1)
                self.allow_gate[k.strip()] = v

    # ----- lifecycle ------------------------------------------------------- #
    def prepare(self) -> str | None:
        """Validate mode preconditions. Returns an error string or None."""
        if self.attach_mode:
            if self.trace_path is None:
                return "attach mode requires explicit --trace-path"
            if not self.token:
                return "attach mode requires --token or LTX_AUTH_TOKEN"
            if not self.admin_token:
                return "attach mode requires --admin-token or LTX_ADMIN_TOKEN"
            if self.args.force_memory_strategy:
                # The override is a spawn-time env var on the backend; an
                # already-running backend cannot be affected. Reject loudly
                # rather than silently ignoring the flag.
                return (
                    "attach mode cannot apply --force-memory-strategy: the "
                    "backend is already running and its memory strategy env was "
                    "fixed at spawn time. Re-run in spawn mode (--) to force a strategy."
                )
            if self.args.hdr_vae_tile_size is not None or self.args.hdr_vae_tile_threshold is not None:
                # VAE tile knobs are spawn-time backend env; a running backend
                # cannot be reconfigured. Fail closed rather than silently ignore.
                return (
                    "attach mode cannot apply --hdr-vae-tile-size/--hdr-vae-tile-threshold: "
                    "the backend is already running and its VAE tiling env was fixed at "
                    "spawn time. Re-run in spawn mode to set these."
                )
            return None
        # spawn mode preflight: app-data-dir must exist + hold settings/profiles.
        add = Path(self.args.app_data_dir)
        if not add.is_dir():
            return "APP_DATA_PRECONDITION_FAILED"
        if not (add / "settings.json").is_file() or not (add / "model_profiles.json").is_file():
            return "APP_DATA_PRECONDITION_FAILED"
        # truncate the four owned artifacts exactly once; never touch others.
        self.report_dir.mkdir(parents=True, exist_ok=True)
        for name in ("report.json", "summary.md", "backend.log"):
            (self.report_dir / name).write_text("")
        if self.trace_path is None:
            self.trace_path = self.report_dir / "memory.jsonl"
        self.trace_path.write_text("")
        return None

    def spawn(self) -> BackendInstance:
        port = int(self.args.port) if self.args.port else _free_port()
        inst = _spawn_backend(
            port=port,
            app_data_dir=Path(self.args.app_data_dir),
            trace_path=self.trace_path,
            run_id=self.run_id,
            token=self.token,
            admin_token=self.admin_token,
            log_path=self.log_path,
            force_memory_strategy=self.args.force_memory_strategy,
            hdr_vae_tile_size=self.args.hdr_vae_tile_size,
            hdr_vae_tile_threshold=self.args.hdr_vae_tile_threshold,
        )
        ok, msg = inst.wait_ready(port, timeout=60.0)
        if not ok:
            # Best-effort teardown so the orphaned process does not linger.
            method = inst.teardown(self.token, self.admin_token, self.run_id)
            self.shutdown_methods.append(method)
            raise RuntimeError(f"backend readiness failed: {msg}")
        return inst

    def attach_probe(self, base_url: str) -> tuple[bool, str]:
        """GET /health as case __probe__; require fresh trace http_start + terminal."""
        status, body, err = _http(
            "GET", f"{base_url}/health", self.token, self.admin_token,
            body=None, case_id="__probe__", run_id=self.run_id,
            admin_route=False, timeout=15.0,
        )
        if status != 200:
            return False, f"/health probe failed: status={status} err={err}"
        events = _load_trace_events(self.trace_path, self.run_id, "__probe__", expect_pid=None)
        ok, msg = _validate_trace(events, hdr_create_required=False)
        return ok, msg

    # ----- preflight per concrete case ------------------------------------- #
    def preflight(self, base_url: str, atomic: str, media: str | None,
                  case_id: str, assets_dir: Path) -> tuple[bool, str, str]:
        """Returns (ok, detail, failure_code). failure_code only when ok is False."""
        code = "PREFLIGHT_FAILED"
        # GET /api/settings (not admin).
        st, body, err = _http(
            "GET", f"{base_url}/api/settings", self.token, self.admin_token,
            body=None, case_id=case_id, run_id=self.run_id, admin_route=False, timeout=15.0,
        )
        if st != 200:
            return False, f"settings GET failed: {st} {err}", code
        try:
            settings = json.loads(body)
        except Exception as exc:  # noqa: BLE001
            return False, f"settings not JSON: {exc}", code
        models_dir = str(settings.get("models_dir") or settings.get("modelsDir") or "")
        if self.args.models_dir:
            if _normalize_path(models_dir) != _normalize_path(str(self.args.models_dir)):
                return False, (f"settings models_dir {models_dir!r} != --models-dir "
                               f"{self.args.models_dir!r}"), code

        # GET /api/model-profiles (admin).
        st, body, err = _http(
            "GET", f"{base_url}/api/model-profiles", self.token, self.admin_token,
            body=None, case_id=case_id, run_id=self.run_id, admin_route=True, timeout=15.0,
        )
        if st != 200:
            return False, f"profiles GET failed: {st} {err}", code
        try:
            profiles_resp = json.loads(body)
        except Exception as exc:  # noqa: BLE001
            return False, f"profiles not JSON: {exc}", code
        active_id = profiles_resp.get("active_model_profile_id")
        profiles = profiles_resp.get("profiles") or []
        active = next((p for p in profiles if p.get("id") == active_id), None)
        if not active:
            return False, "no active model profile", code
        if self.args.profile:
            if active.get("id") != self.args.profile and active.get("name") != self.args.profile:
                return False, (f"active profile {active.get('id')!r} does not match "
                               f"--profile {self.args.profile!r}"), code

        # harness:selftest only needs the settings/profiles preflight — it never
        # touches a model or generation route, so skip model/media validation.
        if atomic == "harness:selftest":
            return True, "preflight ok", code

        # Selected model install check (Kijai / GGUF) via the base-video registry.
        model_sel = KIJAI_ID if atomic in ("kijai:fast", "hdr:kijai_fp8_split") else (
            GGUF_ID if atomic in ("gguf:fast", "hdr:gguf") else None)
        if model_sel:
            installed, detail = _selection_installed(models_dir, model_sel)
            if not installed:
                return False, f"selected model not installed: {model_sel}: {detail}", "MISSING_MODEL_SELECTION"

        # Media / ingredient file existence.
        route, body_dict, berr = _build_body(
            atomic, media, assets_dir, self.hdr_body_overrides, self.hdr_source_path
        )
        if berr:
            return False, berr, code
        # For HDR, also require the active profile's HDR adapter path to exist.
        if atomic.startswith("hdr:"):
            comps = active.get("components") or {}
            hdr_path = comps.get("ic_lora_hdr") or (comps.get("official_adapters") or {}).get("hdr")
            if hdr_path and not Path(_normalize_path(str(hdr_path))).is_file():
                return False, f"HDR adapter missing: {hdr_path}", code
        return True, "preflight ok", code

    # ----- run one case ---------------------------------------------------- #
    def run_case(self, inst: BackendInstance, base_url: str, atomic: str, media: str | None,
                 assets_dir: Path) -> dict[str, Any]:
        case_id = atomic if media is None else f"{atomic}/{media}"
        record: dict[str, Any] = {
            "case_id": case_id, "atomic": atomic, "media": media,
            "pid": inst.pid, "run_id": self.run_id, "started_at": _now_iso(),
        }

        # Preflight.
        ok, detail, fcode = self.preflight(base_url, atomic, media, case_id, assets_dir)
        if not ok:
            record.update(status=S_PREFLIGHT_FAILED, detail=detail, failure_code=fcode,
                          finished_at=_now_iso())
            return record

        method, route, body = self._request_spec(atomic, media, assets_dir)
        seg_start = len(inst.log_lines)
        # selftest hits a cheap /health route; generation cases keep the long timeout.
        timeout = 15.0 if atomic == "harness:selftest" else float(self.args.timeout_minutes) * 60.0
        status, raw, err = _http(
            method, f"{base_url}{route}", self.token, self.admin_token,
            body=body, case_id=case_id, run_id=self.run_id, admin_route=False, timeout=timeout,
        )
        seg_end = len(inst.log_lines)
        segment = inst.segment(seg_start, seg_end)
        record["log_segment"] = "".join(segment)

        # Trace ownership (spawn: exact pid; attach: pid not known).
        events = _load_trace_events(self.trace_path, self.run_id, case_id,
                                    expect_pid=inst.pid if not self.attach_mode else None)
        hdr_required = atomic.startswith("hdr:")
        trace_ok, trace_msg = _validate_trace(events, hdr_create_required=hdr_required)
        record["trace_ok"] = trace_ok
        record["trace_detail"] = trace_msg

        if atomic == "harness:selftest":
            status_outcome, out_detail = self._evaluate_health(status, err)
        else:
            status_outcome, out_detail = self._evaluate(status, raw, err, atomic, case_id, segment)
        record.update(status=status_outcome, detail=out_detail, http_status=status,
                      finished_at=_now_iso())

        # Trace is a hard requirement for every routed case: a passing output
        # without proven trace ownership is downgraded to failed.
        if status_outcome == S_PASSED and not trace_ok:
            record["status"] = S_FAILED
            record["detail"] = f"{out_detail}; trace invalid: {trace_msg}"
        return record

    def _request_spec(self, atomic: str, media: str | None,
                      assets_dir: Path) -> tuple[str, str, dict[str, Any] | None]:
        """Resolve (HTTP method, route, body) for a case.

        ``harness:selftest`` routes to a cheap ``GET /health`` (no generation);
        every other atomic POSTs its generation body from ``_build_body``.
        """
        if atomic == "harness:selftest":
            return "GET", "/health", None
        route, body, _ = _build_body(
            atomic, media, assets_dir, self.hdr_body_overrides, self.hdr_source_path
        )
        return "POST", route, body

    def _evaluate_health(self, status: int | None, err: str | None) -> tuple[str, str]:
        """Selftest success = authenticated GET /health returning 200."""
        if status == 200:
            return S_PASSED, "GET /health 200"
        return S_FAILED, f"GET /health failed: status={status} err={err or '-'}"

    def _evaluate(self, status: int | None, raw: str, err: str | None, atomic: str,
                  case_id: str, segment: list[str]) -> tuple[str, str]:
        # Parse JSON body if any.
        parsed: dict[str, Any] | None = None
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:  # noqa: BLE001
                parsed = None

        resp_code = ""
        if isinstance(parsed, dict):
            resp_code = str(parsed.get("code") or parsed.get("error_code") or "")

        # Determine request success (complete + video_path).
        success = (status == 200 and isinstance(parsed, dict)
                   and parsed.get("status") == "complete"
                   and bool(parsed.get("video_path")))

        # --- output validation on success ---------------------------------- #
        if success:
            assert parsed is not None
            if atomic.startswith("hdr:"):
                ok, msg = _validate_exr(parsed["video_path"], parsed.get("proxy_path"))
            else:
                ok, msg = _validate_mp4(parsed["video_path"])
            if ok:
                return S_PASSED, f"output valid: {msg}"
            return S_FAILED, f"output invalid: {msg}"

        # --- failure path --------------------------------------------------- #
        combined = " ".join(filter(None, [raw, err or ""])).lower()
        seg_text = "".join(segment)

        # allow_gate: a gated feature failing with the expected error code passes.
        gate = self.allow_gate.get(case_id)
        if gate and resp_code == gate:
            return S_ALLOWED_GATE, f"gated by {gate}"

        # expected_failure: substring in response/error, OR routed request
        # crashed (transport failure), provided no output validated and the
        # failure is genuinely this case's (ownership via log segment).
        exp = self.expect_failure.get(case_id)
        if exp:
            substring = exp
            in_response = substring.lower() in combined
            crashed = status is None  # transport-level failure / crash
            # For log-only substrings (e.g. OOM in a crash), require the
            # substring in THIS case's log segment only — never cross-case.
            in_segment = substring.lower() in seg_text.lower()
            owns_log = bool(seg_text.strip())
            # Must NOT be a preflight failure (handled earlier) and must not be
            # a success that already validated (handled above).
            if (in_response or (crashed and in_segment and owns_log)):
                return S_EXPECTED_FAILURE, f"matched expected failure: {substring}"

        return S_FAILED, (f"request failed: status={status} code={resp_code or '-'} "
                          f"err={err or '-'}")


def _selection_installed(models_dir: str, model_selection: str) -> tuple[bool, str]:
    """Resolve a base-video selection via the registry and report exact status.

    Returns ``(installed, detail)``. ``installed`` is True only when a usable
    transformer file is confirmed on disk; ``detail`` names the real outcome so
    preflight failures stop collapsing every cause to a bare False:

    - ``registry import failed: ...``
    - ``registry resolve failed: ...``
    - ``registry installed=false expected=... status=...`` (scanner missed it)
    - ``transformer_path missing/non-file: ...``
    - ``installed: <path>`` (success, optionally via canonical-path fallback)

    Registry-first throughout. The one fallback is narrow: if the registry
    reports ``installed=false`` but the entry's exact canonical absolute path
    exists on disk, treat it as installed (covers a subfolder-resident model the
    scanner missed). No broad filename guessing.
    """
    try:
        from services.base_video_model_registry import (  # noqa: PLC0415
            resolve_base_video_model_selection,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"registry import failed: {exc}"
    try:
        entry = resolve_base_video_model_selection(Path(models_dir), model_selection)
    except Exception as exc:  # noqa: BLE001
        return False, f"registry resolve failed: {exc}"

    status = getattr(entry, "scanner_status", "?")
    installed_flag = bool(getattr(entry, "installed", False))
    expected = getattr(entry, "expected_absolute_path", None)
    tp = getattr(entry, "transformer_path", None)

    # Registry-confirmed install: validate its path(s) actually exist on disk.
    if installed_flag:
        if isinstance(tp, tuple):
            missing = [p for p in tp if not Path(p).is_file()]
            if missing:
                return False, f"transformer_path missing/non-file: {missing}"
            return True, f"installed: {tp}"
        if tp is not None and Path(tp).is_file():
            return True, f"installed: {tp}"
        return False, f"transformer_path missing/non-file: {tp!r}"

    # Registry says not installed. Narrow canonical-path fallback: if the
    # entry's exact canonical absolute path exists, the scanner simply missed a
    # subfolder-resident model — accept it. No guessing beyond this exact path.
    if expected and Path(expected).is_file():
        return True, f"installed: {expected} (canonical-path fallback; scanner_status={status!r})"

    detail = f"registry installed=false expected={expected!r} status={status!r}"
    if tp:
        detail += f" transformer_path={tp!r}"
    return False, detail


# --------------------------------------------------------------------------- #
# Case expansion
# --------------------------------------------------------------------------- #
def expand_cases(matrix: str | None, media_types: list[str]) -> list[tuple[str, str | None]]:
    wanted = {m.strip() for m in matrix.split(",") if m.strip()} if matrix else None
    cases: list[tuple[str, str | None]] = []
    for a in SOURCELESS_ATOMICS:
        cases.append((a, None))
    for a in MEDIA_BACKED_ATOMICS:
        for m in media_types:
            cases.append((a, m))
    if wanted:
        filtered: list[tuple[str, str | None]] = []
        for atomic, media in cases:
            cid = atomic if media is None else f"{atomic}/{media}"
            if atomic in wanted or cid in wanted:
                filtered.append((atomic, media))
        return filtered
    return cases


def _write_reports(report_dir: Path, harness: Harness) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for r in harness.results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    exit_code = 0 if all(r["status"] in (S_PASSED, S_EXPECTED_FAILURE, S_ALLOWED_GATE)
                        for r in harness.results) else 1
    report = {
        "run_id": harness.run_id,
        "backend_mode": "attach" if harness.attach_mode else "spawn",
        "started_at": harness.started_at,
        "finished_at": _now_iso(),
        "shutdown_methods": harness.shutdown_methods,
        "force_memory_strategy": harness.args.force_memory_strategy,
        "summary": counts,
        "case_count": len(harness.results),
        "exit_code": exit_code,
        "cases": harness.results,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        "# Live Workflow Test Report",
        f"- Run: `{harness.run_id}`",
        f"- Mode: {report['backend_mode']}",
        f"- Started: {report['started_at']}  Finished: {report['finished_at']}",
        f"- Total: {report['case_count']}  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        f"- Exit: {exit_code}",
        "",
        "| Case | Status | Detail |",
        "|------|--------|--------|",
    ]
    for r in harness.results:
        lines.append(f"| {r['case_id']} | {r['status']} | {str(r.get('detail','')).replace('|','/')[:200]} |")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 live workflow harness.")
    p.add_argument("--backend", default="auto", help="auto|http://127.0.0.1:PORT")
    p.add_argument("--port", default=None, help="Spawn port (default: free 127.0.0.1 port)")
    p.add_argument("--token", default=None, help="Auth bearer token (or LTX_AUTH_TOKEN)")
    p.add_argument("--admin-token", default=None, help="Admin token (or LTX_ADMIN_TOKEN)")
    p.add_argument("--app-data-dir", default=None, help="Backend app-data dir (spawn mode)")
    p.add_argument("--trace-path", default=None, help="Memory trace JSONL path (relative → repo root)")
    p.add_argument("--profile", default=None, help="Required active profile id or name")
    p.add_argument("--models-dir", default=None, help="Required settings models_dir value")
    p.add_argument("--assets-dir", default=None, help="Dir holding media + ingredient assets")
    p.add_argument("--matrix", default=None, help="Comma-separated atomic/concrete ids to run")
    p.add_argument("--media", default="mp4", help="Comma-separated media types (mp4,mov) for media-backed atomics")
    p.add_argument("--report-dir", default="live-workflow-reports",
                   help="Report dir (relative → repo root)")
    p.add_argument("--timeout-minutes", type=float, default=20.0, help="Per-case request timeout (minutes)")
    p.add_argument("--expect-failure", action="append", default=[], metavar="CASE=SUBSTRING",
                   help="Mark a case as expected to fail with a response/error/log substring")
    p.add_argument("--allow-gate", action="append", default=[], metavar="CASE=ERROR_CODE",
                   help="Allow a case to pass when it fails with a gated error code")
    p.add_argument("--force-memory-strategy", default=None,
                   choices=FORCE_MEMORY_STRATEGY_CHOICES, metavar="STRATEGY",
                   help=("Benchmark-only: force a local memory strategy in the spawned "
                         "backend via LTX_FORCE_LOCAL_MEMORY_STRATEGY. Only 'block_offload' "
                         "is honoured; other choices fail closed inside the backend. "
                         "Spawn mode only (rejected in attach mode)."))
    # Harness/debug HDR knobs (NOT production UI policy). Body-field overrides
    # apply only to HDR atomics; VAE tile knobs are spawn-time backend env, so
    # attach mode rejects them. Width/height must be supplied together; frames
    # must satisfy (frames-1) % 8 == 0.
    p.add_argument("--hdr-width", type=int, default=None,
                   help="Override HDR body width (px); requires --hdr-height")
    p.add_argument("--hdr-height", type=int, default=None,
                   help="Override HDR body height (px); requires --hdr-width")
    p.add_argument("--hdr-frames", type=int, default=None,
                   help="Override HDR body num_frames; requires (frames-1) % 8 == 0. "
                        "NOTE: HDR output frame count is source-driven (generate() pads "
                        "from the decoded source length), so this only overrides the request "
                        "field. Pair with --hdr-source-path to actually limit output frames.")
    p.add_argument("--hdr-fps", type=float, default=None, help="Override HDR body frame_rate")
    p.add_argument("--hdr-source-path", default=None,
                   help="Override the HDR source video path (hdr:* atomics only) instead of "
                        "the media-backed default. Pair with --hdr-frames for real frame-count "
                        "limit tests, since HDR output frame count is derived from the source.")
    p.add_argument("--hdr-vae-tile-size", type=int, default=None,
                   help="HDR VAE spatial tile size in spawned backend "
                        "(0 disables tiled encode; unset keeps backend default). "
                        "Spawn mode only (rejected in attach mode).")
    p.add_argument("--hdr-vae-tile-threshold", type=int, default=None,
                   help="Override HDR VAE tiled-encode pixel threshold in spawned "
                        "backend. Spawn mode only (rejected in attach mode).")
    return p


def _validate_hdr_knobs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail fast on malformed HDR harness flags before running any case."""
    if (args.hdr_width is None) != (args.hdr_height is None):
        parser.error("--hdr-width and --hdr-height must be supplied together")
    if args.hdr_frames is not None and (args.hdr_frames - 1) % 8 != 0:
        parser.error(
            f"--hdr-frames must satisfy (frames-1) % 8 == 0; got {args.hdr_frames}"
        )
    if args.hdr_source_path is not None and not Path(args.hdr_source_path).is_file():
        parser.error(f"--hdr-source-path does not exist: {args.hdr_source_path}")


def main() -> None:
    # pnpm/yarn forward a literal leading ``--`` separator to the script;
    # argparse treats it as end-of-options and then errors on the flags that
    # follow. Strip exactly one leading ``--`` so
    # ``pnpm live:workflow-test -- --backend ...`` parses cleanly.
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_hdr_knobs(parser, args)
    harness = Harness(args)

    err = harness.prepare()
    if err:
        # Could not even reach a backend; record a minimal report and exit.
        harness.report_dir.mkdir(parents=True, exist_ok=True)
        harness.results.append({
            "case_id": "__prepare__", "atomic": None, "media": None,
            "status": S_PREFLIGHT_FAILED, "detail": err, "run_id": harness.run_id,
            "started_at": harness.started_at, "finished_at": _now_iso(),
        })
        _write_reports(harness.report_dir, harness)
        print(f"[live_workflow_test] prepare failed: {err}", file=sys.stderr)
        sys.exit(2)

    media_types = [m.strip() for m in args.media.split(",") if m.strip()]
    cases = expand_cases(args.matrix, media_types)
    assets_dir = Path(args.assets_dir) if args.assets_dir else (REPO_ROOT / "assets")

    base_url: str
    if harness.attach_mode:
        base_url = args.backend.rstrip("/")
        ok, msg = harness.attach_probe(base_url)
        if not ok:
            harness.results.append({
                "case_id": "__probe__", "atomic": None, "media": None,
                "status": S_FAILED, "detail": f"attach probe failed: {msg}",
                "run_id": harness.run_id, "started_at": _now_iso(), "finished_at": _now_iso(),
            })
            _write_reports(harness.report_dir, harness)
            print(f"[live_workflow_test] attach probe failed: {msg}", file=sys.stderr)
            sys.exit(2)
        attach_inst = BackendInstance.__new__(BackendInstance)
        attach_inst.pid = -1  # unknown for an externally-owned backend
        attach_inst.log_lines = []  # no stdout ownership in attach mode
        attach_inst.proc = None  # type: ignore[assignment]
        attach_inst.port = 0
        inst_for_cases: BackendInstance = attach_inst
    else:
        base_url = "http://127.0.0.1"

    try:
        for atomic, media in cases:
            case_id = atomic if media is None else f"{atomic}/{media}"
            # OOM isolation: a case expected to fail with CUDA OOM gets its own
            # fresh backend and is torn down immediately afterwards.
            oom = "cuda out of memory" in harness.expect_failure.get(case_id, "").lower()
            if oom or harness.attach_mode:
                if harness.attach_mode:
                    inst = inst_for_cases
                else:
                    try:
                        inst = harness.spawn()
                    except RuntimeError as exc:
                        harness.results.append(_crash_record(case_id, atomic, media,
                                                             harness.run_id, str(exc)))
                        continue
            else:
                if harness.shared is None:
                    try:
                        harness.shared = harness.spawn()
                    except RuntimeError as exc:
                        harness.results.append(_crash_record(case_id, atomic, media,
                                                             harness.run_id, str(exc)))
                        continue
                inst = harness.shared
            # base URL is per-instance: spawn mode runs each backend on its own
            # port (OOM-isolated cases get a fresh backend on a new port).
            base = base_url if harness.attach_mode else f"http://127.0.0.1:{inst.port}"
            result = harness.run_case(inst, base, atomic, media, assets_dir)
            harness.results.append(result)
            if oom and not harness.attach_mode and inst is not harness.shared:
                harness.shutdown_methods.append(inst.teardown(harness.token, harness.admin_token,
                                                              harness.run_id))
    finally:
        if harness.shared is not None and not harness.attach_mode:
            harness.shutdown_methods.append(harness.shared.teardown(
                harness.token, harness.admin_token, harness.run_id))

    _write_reports(harness.report_dir, harness)
    exit_code = 0 if all(r["status"] in (S_PASSED, S_EXPECTED_FAILURE, S_ALLOWED_GATE)
                        for r in harness.results) else 1
    print(f"[live_workflow_test] wrote {harness.report_dir}/report.json (exit={exit_code})")
    sys.exit(exit_code)


def _crash_record(case_id: str, atomic: str, media: str | None, run_id: str,
                  detail: str) -> dict[str, Any]:
    return {
        "case_id": case_id, "atomic": atomic, "media": media, "run_id": run_id,
        "status": S_FAILED, "detail": f"backend spawn/readiness failed: {detail}",
        "started_at": _now_iso(), "finished_at": _now_iso(),
    }


if __name__ == "__main__":
    main()
