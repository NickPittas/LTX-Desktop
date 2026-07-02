# backend/server_utils/

## Responsibility

Two small, side-effect-bounded helpers used at server bootstrap and inside
request handlers. No routers, no state, no business logic — pure filesystem /
media checks plus one startup migration.

- `media_validation.py` — validates user-supplied filesystem paths (image and
  audio inputs) before a handler touches a pipeline. Intentionally
  **handler-oriented**: it raises `_routes._errors.HTTPError(400, ...)` so
  invalid input surfaces as HTTP 400 instead of leaking `PIL`/`OSError`
  exceptions.
- `model_layout_migration.py` — one-shot startup migration that relocates the
  legacy `models/ltx-2/*` layout into `models/*`.

## Design Patterns

- **Fail-closed with typed errors.** `media_validation` never returns a
  sentinel; bad input raises `HTTPError(400, ...)`. Callers in `handlers/` do
  not need try/except — the exception is caught by the `HTTPError` handler in
  `app_factory.py`.
- **Defense in depth for media** — each validator checks: path is a real file
  (`_assert_is_file`), size under a byte cap (`_assert_max_bytes`), and a
  format/content check (PIL `Image.verify` for images; magic-byte sniffing for
  audio). Size/format failures are reported with the same `400` envelope.
- **Content sniffing over extension trust.** `_sniff_audio(header, ext)`
  validates leading bytes (RIFF/WAVE, fLaC, OggS, ID3 / MP3 frame sync, ADIF /
  ADTS, MP4 `ftyp`) keyed to the file extension; unknown extensions only accept
  unambiguous signatures to avoid misclassifying MP4 containers as audio.
- **Non-fatal migration.** `migrate_legacy_models_layout` never overwrites an
  existing target (logs a warning and skips), falls back from `Path.rename` to
  `shutil.move` on cross-device `OSError`, and best-effort removes the emptied
  `ltx-2/` directory. No exception is allowed to block startup.

## Data & Control Flow

### `media_validation.py`
Constants: `_MAX_IMAGE_BYTES = 50 MiB`, `_MAX_AUDIO_BYTES = 100 MiB`,
`_MAX_IMAGE_PIXELS = 50_000_000`,
`_ALLOWED_IMAGE_FORMATS = {PNG, JPEG, WEBP, GIF, BMP, TIFF}`.

- `normalize_optional_path(value: str | None) -> str | None` — collapses
  `None`/`""`/whitespace to `None`; otherwise returns the trimmed string. Used
  to treat "no input supplied" uniformly.
- `validate_image_file(path: str) -> Path`:
  1. `Path(path)` (any failure → `HTTPError(400, "Image file not found: …")`).
  2. `_assert_is_file(..., kind="Image", ...)`.
  3. `_assert_max_bytes(..., _MAX_IMAGE_BYTES, "Image file too large: …")`.
  4. `Image.open` → read `format`, `size`; `img.verify()`
     (any failure → `HTTPError(400, "Invalid image file: …")`).
  5. Reject if `fmt` not in `_ALLOWED_IMAGE_FORMATS`.
  6. Reject if `width<=0` or `height<=0` or `width*height > _MAX_IMAGE_PIXELS`.
  Returns the validated `Path`.
- `validate_audio_file(path: str) -> Path`:
  1. `Path(path)` → file existence → `_assert_max_bytes(_MAX_AUDIO_BYTES)`.
  2. `_read_header(file_path, num_bytes=64)` → `_sniff_audio(header, suffix)`.
  3. Reject (`HTTPError(400, "Invalid audio file: …")`) if signature mismatches.
  Returns the validated `Path`.
- Internal helpers: `_assert_is_file`, `_assert_max_bytes`, `_read_header`,
  `_sniff_audio`.

### `model_layout_migration.py`
`migrate_legacy_models_layout(app_data_dir: Path) -> None`:
1. Resolve `models_root = app_data_dir/"models"`,
   `legacy_root = models_root/"ltx-2"`; return early if `legacy_root` absent.
2. Ensure `models_root` exists; for each entry in `legacy_root`, if
   `models_root/<name>` already exists log a warning and skip, else `rename`
   (fallback `shutil.move` on `OSError`).
3. If `legacy_root` is now empty, `rmdir()` it (best-effort, logged on
   failure).

## Integration Points

- **`ltx2_server.py`** — imports and runs
  `migrate_legacy_models_layout(APP_DATA_DIR)` at module load, before
  `build_initial_state`.
- **`handlers/video_generation_handler.py`** — imports
  `validate_image_file`, `validate_audio_file`, `normalize_optional_path` (and
  `_MAX_*` indirectly) to gate `imagePath`/`audioPath` on `GenerateVideoRequest`
  before video/A2V pipelines run.
- **`handlers/suggest_gap_prompt_handler.py`** — imports
  `normalize_optional_path` and `validate_image_file` for the
  `image-to-video` gap-prompt path (`inputImage`).
- **`_routes/_errors.HTTPError`** — the exception type both modules raise; it
  is translated to an `HTTPErrorResponse` JSON body by the exception handler
  registered in `app_factory.create_app`.
- **Tests** — `tests/test_model_layout_migration.py` exercises the migration
  directly. (No frontend coupling; this package is backend-internal.)
