# backend/services/gpu_info/

## Responsibility

Read-only GPU/runtime introspection: device availability (CUDA / MPS), device name, total VRAM, and live VRAM utilization. Feeds the system-info / status UI and gates device-dependent behavior elsewhere.

## Design Patterns

- **Protocol + Impl split.** `gpu_info.py` defines `GpuInfo` (Protocol with 6 query methods) and the `GpuTelemetryPayload` `TypedDict` (`name`, `vram`, `vramUsed` — all in MiB). `gpu_info_impl.py` holds the `torch` / `pynvml` / `sysctl` dependencies.
- **Defensive querying.** Every method wraps runtime calls in `try/except`, logs a warning with `exc_info=True`, and returns a safe default (`False` / `None` / `0` / `"Unknown"`). Queries never raise into the handler layer.
- **Backend cascade.** CUDA is preferred; MPS is the fallback; everything else reports `"Unknown"` / `0`.

## Data & Control Flow

`get_gpu_info()` (the telemetry entry):
1. If `get_cuda_available()` → import `pynvml`, `nvmlInit()`, `nvmlDeviceGetHandleByIndex(0)`, read `nvmlDeviceGetName` (decoded if `bytes`) and `nvmlDeviceGetMemoryInfo`; return `{name, vram: total//MiB, vramUsed: used//MiB}`; `nvmlShutdown()`. On any exception → log + fall back to torch metadata: `get_device_name()` and `get_vram_total_gb()` (so telemetry still returns when NVML is absent).
2. Else if `get_mps_available()` → name from `_get_macos_chip_name()` (suffix `" (MPS)"`), `vram` from `_get_system_ram_mb()`, `vramUsed: 0`.
3. Else → `{"name": "Unknown", "vram": 0, "vramUsed": 0}`.

Availability methods all gate on `torch`: `get_cuda_available()` → `torch.cuda.is_available()`; `get_mps_available()` → `hasattr(torch.backends,"mps") and torch.backends.mps.is_available()`; `get_gpu_available()` → OR of the two. `get_device_name()` returns CUDA device 0 name (or MPS chip name). `get_vram_total_gb()` returns `int(torch.cuda.get_device_properties(0).total_memory // GiB)` on CUDA, or system-RAM GiB on MPS via `os.sysconf(SC_PAGE_SIZE * SC_PHYS_PAGES)`.

`_get_macos_chip_name()` shells out `sysctl -n machdep.cpu.brand_string` (5 s timeout, Darwin-only). `_get_system_ram_mb()` uses `os.sysconf` and returns `0` on win32.

## Integration Points

- **`app_handler.build_default_service_bundle()`** instantiates `GpuInfoImpl()` into `ServiceBundle.gpu_info`.
- **`services/interfaces.py`** re-exports `GpuInfo`, `GpuTelemetryPayload` as the public boundary; `services/__init__.py` re-exports them again.
- **Handlers/Routes** call `gpu_info.get_gpu_info()` to populate the status payload surfaced to the Electron frontend; `get_gpu_available()` / `get_cuda_available()` gate GPU-only code paths.
- **Tests:** `tests/fakes/fake_gpu_info.py::FakeGpuInfo` implements the Protocol; `tests/fakes/services.py` wires it as the default `gpu_info` in the fake `ServiceBundle`; `conftest.py` installs it per test.
