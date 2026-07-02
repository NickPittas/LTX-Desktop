# backend/services/model_downloader/

## Responsibility

Downloads model files / snapshots from Hugging Face Hub into the user-configured models folder (single source of truth per AGENTS.md). Defines the `ModelDownloader` `Protocol` and the `huggingface_hub`-backed implementation with aggregated progress reporting.

## Design Patterns

- **Protocol + Impl split.** `model_downloader.py` defines the `ModelDownloader` `Protocol` (`download_file`, `download_snapshot`) with a `Callable[[int], None] | None` progress callback. `hugging_face_downloader.py` carries the `huggingface_hub` / `tqdm` dependencies.
- **Progress via tqdm subclassing + monkey-patch.** `_make_progress_tqdm_class(callback)` returns a `tqdm.auto.tqdm` subclass whose `update(n)` atomically accumulates `n` into a `shared["downloaded"]` counter (under a `Lock`) and invokes `callback`. For snapshot downloads (multiple concurrent tqdm bars) this yields cross-file aggregate progress.
- **`hf_hub_download` progress injection.** Unlike `snapshot_download`, `hf_hub_download` has no `tqdm_class` kwarg; `_patch_download_progress(callback)` temporarily rebinds `huggingface_hub.file_download.http_get` (and `xet_get` when present) to inject a `_tqdm_bar=<ProgressTqdm(disable=True)>` into the private kwarg consumed by `http_get`/`xet_get`. Context-managed; restores originals on exit.
- **Callback optional.** When `on_progress is None`, both methods use `contextlib.nullcontext()` and call the hub function unmodified.

## Data & Control Flow

`HuggingFaceDownloader.download_file(repo_id, filename, local_dir, token, on_progress)`:
- Build `_patch_download_progress(on_progress)` ctx (or `nullcontext`).
- `with ctx: hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, token=token)` → returns local path string → wrapped in `Path`.

`download_snapshot(repo_id, local_dir, token, on_progress)`:
- Same ctx pattern; `snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token)` → `Path`.

Inside the ctx, each chunk transfer triggers `_ProgressTqdm.update(n)` → `shared["downloaded"] += int(n)` under lock → `callback(shared["downloaded"])`. Snapshot spawns one tqdm per file but they share the mutable `shared` dict, so the callback reports total bytes across the whole snapshot.

No retry, no resume logic, no integrity check here — those belong to `huggingface_hub`.

## Integration Points

- **`app_handler.build_default_service_bundle()`** instantiates `HuggingFaceDownloader()` into `ServiceBundle.model_downloader`.
- **`services/interfaces.py`** re-exports `ModelDownloader`; `services/__init__.py` re-exports again.
- **Handlers** (download/status/scan flows per AGENTS.md "one models folder is source of truth") call `download_file` / `download_snapshot` with the user-set models folder as `local_dir`; the `on_progress` callback streams byte counts to the frontend (Electron IPC) for progress UI.
- **Auth:** `token` (`str | None`) is passed through; sourced from `hf_auth` in `app_handler` (loaded via `hf_auth.load_token()`).
- **Tests:** `tests/fakes/services.py::FakeModelDownloader` implements the Protocol (records calls, writes dummy files); wired as default `model_downloader` in the fake `ServiceBundle`. The real implementation is never exercised in the suite (no network).
- **`test_http_get_accepts_tqdm_bar`** (referenced in `_patch_download_progress` docstring) guards the private `_tqdm_bar` kwarg contract — a `huggingface_hub` upgrade that breaks it requires revisiting this patch.
