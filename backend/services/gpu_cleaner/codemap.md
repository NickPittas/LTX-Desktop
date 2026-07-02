# backend/services/gpu_cleaner/

## Responsibility

Reclaims GPU/CPU memory between heavy operations (pipeline teardown, post-generation, profile-swap). Thin, intentionally minimal: one `cleanup()` method.

## Design Patterns

- **Protocol + Impl split.** `gpu_cleaner.py` defines the `GpuCleaner` `Protocol` (`cleanup(self) -> None`). `torch_cleaner.py` holds the `torch` + `gc` dependencies and the concrete `TorchCleaner`.
- **Delegation to shared device helpers.** `TorchCleaner.cleanup()` calls `services.services_utils.empty_device_cache(self._device)` (which dispatches to `torch.cuda.empty_cache()` / `torch.mps.empty_cache()` based on `get_device_type()`) and then `gc.collect()` — no direct torch API calls here.
- **Device-parametric.** `TorchCleaner.__init__(device="cpu")` stores the device; the cleaner is useless for `"cpu"` (empty cache is a no-op) but correct for any device string.

## Data & Control Flow

`TorchCleaner(device)` constructed once per `RuntimeConfig.device`. `cleanup()` → `empty_device_cache(device)` (logs + swallows failures) → `gc.collect()`. Order matters: releasing the caching allocator before collecting Python cycles lets tensors held only by Python refs be freed. No return value, never raises (helpers log-and-return).

## Integration Points

- **`app_handler.build_default_service_bundle()`** constructs `TorchCleaner(device=config.device)` and wires it into `ServiceBundle.gpu_cleaner`.
- **`services/interfaces.py`** re-exports `GpuCleaner`; `services/__init__.py` re-exports again.
- **Consumed by handlers** (generation/teardown/profile-swap paths) via `services.gpu_cleaner` to drop VRAM between runs; also used as the fallback path after errors.
- **`services/services_utils.empty_device_cache`** is the actual implementation surface; this folder is the service-injection wrapper around it.
- **Tests:** `tests/fakes/services.py::FakeGpuCleaner` implements the Protocol; wired as the default `gpu_cleaner` in the fake `ServiceBundle`.
