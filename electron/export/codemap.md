# electron/export/
## Responsibility

The **timeline composition and ffmpeg-based export pipeline**. Renders a multi-track project timeline (video/image/audio clips, speed, reverse, flip, opacity, letterbox, burn-in subtitles) to a single output file. Supports three codecs: **H.264** (`h264`), **Apple ProRes** (`prores`), and **VP9** (`vp9`). Exposes two IPC channels — `exportNative` and `exportCancel` — plus shared ffmpeg utilities consumed by `ipc/file-handlers.ts` (video import transcode, thumbnails, dimensions) and `ipc/video-processing-handlers.ts` (frame extraction).

> **CRITICAL DISTINCTION — this is export-only, NOT primary generation.** The ProRes profile handling here (`prores_ks` with `-profile:v <n>` and `yuva444p10le`) renders a *timeline* the user has already assembled in the editor down to a shareable file. It is **not** the path used when the Python backend decodes/encodes fresh generation output. The upcoming MOV ProRes / EXR primary-output work will produce files directly from the backend's decoded frames; this folder's `runFfmpeg`/`findFfmpegPath` plumbing and the codec-args pattern in `export-handler.ts` are the **reference design** for that encoder, but the primary-output path itself does not yet exist in Electron.

## Design Patterns

- **Three-pass export (video-only → audio-only → mux).** Pass 1 renders directly to the requested target codec and pass 3 always uses `-c:v copy`; no newly created H.264 intermediate may feed ProRes. The existing EXR-directory fallback to its already-existing proxy remains the documented exception because ffmpeg cannot consume a directory.
- **Single global fps conversion.** Per-segment filters do NOT do fps conversion (comment: "real NLEs convert frame rate globally"); `concat` is followed by exactly one `fps=<fps>` filter on the concatenated stream to avoid per-segment duration quantization.
- **Filter graph as a script file, not CLI args.** `buildVideoFilterGraph` returns `{ inputs, filterScript }`; the script is written to a temp file and passed via `-filter_complex_script <file>` to avoid shell-escaping/quoting issues with large timelines.
- **Multi-track timeline flattening (NLE convention).** `flattenTimeline` collects every clip boundary, computes the highest-`trackIndex` clip active at each midpoint, emits contiguous `FlatSegment`s, then merges adjacent segments that share file/speed/flip/opacity/mute/contiguous-trim into single segments (smaller filter graph).
- **PCM mixing in JS, not ffmpeg.** `mixAudioToPcm` extracts each source as raw `s16le` PCM via ffmpeg `pipe:1`, accumulates into a `Float64Array` (no clipping during sum), then clamps to Int16 and writes the mix. This gives sample-accurate placement and per-clip volume control that ffmpeg's `amix` cannot match cleanly.
- **One export-owned `AbortController`** rejects concurrent exports, checks cancellation between phases, and kills only its active ffmpeg/PCM extraction. `runFfmpeg` clears its global process slot only when the exiting process still owns it; isolated preview/import transcodes remain outside this cancellation path.
- **Pure, I/O-free filter graph builder.** `video-filter.ts` does zero filesystem work — only string construction. This keeps it unit-testable in isolation.
- **Codec profile centralization.** `export-codec.ts::buildExportVideoPlan` validates quality and owns direct, BT.709-tagged pass-1 codec/container/pixel-format selection: H.264 MKV, ProRes 422 10-bit MOV (`apl0`, no qscale), or VP9 WebM.

## Data & Control Flow

### `export-handler.ts` / `export-runner.ts` — `exportNative({ clips, outputPath, codec, width, height, fps, quality, letterbox, subtitles })`
1. **Resolve ffmpeg** via `findFfmpegPath()`; bail with `{ success: false, error: 'FFmpeg not found' }` if missing.
2. **Validate every path** — `validatePath(outputPath, getAllowedRoots())` plus `validatePath(clip.path, ...)` for each clip.
3. **Preflight sources** — validate every visual segment path and, for each audible audio/video clip, the effective `audioPath ?? path`.
4. **Flatten** — `flattenTimeline(clips)` → `FlatSegment[]` (gaps allowed). Reject empty timeline.
5. **Allocate temp artifacts** in `os.tmpdir()`: `ltx-export-video-<ts>.mkv`, `ltx-export-audio-<ts>.wav`; define `cleanup()` to unlink both.
6. **Pass 1 — video only:**
   - `buildVideoFilterGraph(segments, { width, height, fps, letterbox, subtitles })` → `{ inputs, filterScript }`.
   - Write `filterScript` to `ltx-filter-v-<ts>.txt`.
   - `runFfmpeg(ffmpegPath, ['-y', ...inputs, '-filter_complex_script', filterFile, '-map', '[outv]', '-an', ...plan.pass1VideoArgs, tmpVideo])`. The requested delivery codec is encoded directly with limited BT.709 tags.
7. **Pass 2 — audio mix:** compute `totalDuration` as `max(seg.startTime + seg.duration)` across both segments and original clips; `mixAudioToPcm(clips, totalDuration, ffmpegPath)` → `{ pcmBuffer, sampleRate, channels }`. Write raw PCM (`ltx-pcm-<ts>.raw`), then `runFfmpeg(['-f', 's16le', '-ar', '<sr>', '-ac', '<ch>', '-i', rawPcm, '-c:a', 'pcm_s16le', tmpAudio])`.
8. **Pass 3 — mux:**
   - H.264, ProRes 4:2:2 10-bit, and VP9 are each encoded directly in pass 1.
   - Every final mux uses `-c:v copy`, codec-specific audio, target muxer/container flags, and limited BT.709 tags.
9. `cleanup()` and return `{ success: true }` or `{ success: false, error }`.

### `export-handler.ts` — `exportCancel({ sessionId })`
Calls `stopExportProcess()` (kills the active ffmpeg child if any) and returns `{ success: true }`. Note: `sessionId` is accepted per the schema but not currently used to disambiguate (there is only one `activeExportProcess`).

### `ffmpeg-utils.ts`
- **`findFfmpegPath()`:** resolves the ffmpeg binary shipped inside the Python venv's `imageio_ffmpeg/binaries` — Windows path is fixed; macOS/Linux scans `lib/python3.X/site-packages/imageio_ffmpeg/binaries` for the matching python dir. Falls back to a system `ffmpeg` on PATH (probed with `execSync('ffmpeg -version')`).
- **`runFfmpeg(ffmpegPath, args)`** → `Promise<{ success, error? }>`: `spawn`s, logs each `frame=`/`Error` line from stderr (truncated to 200 chars), resolves with `{ success: true }` on exit code 0 or `{ success: false, error: 'FFmpeg failed (code N): <last 5 stderr lines>' }`. Stores process in `activeExportProcess`.
- **`fileHasAudio(ffmpegPath, filePath)`**: spawnSync `ffmpeg -i <file> -hide_banner`, true iff combined stdout+stderr contains `'Audio:'`.
- **`getVideoDimensions(videoPath)`**: spawnSync `ffmpeg -hide_banner -i <file>`, parse the `Video:` line for `WxH` via `/(\d{2,5})x(\d{2,5})(?:[,\s\[]|$)/`.
- **`extractVideoFrameToFile({ videoPath, seekTime, width?, quality?, outputPath?, timeoutMs=10000 })`**: builds `ffmpeg -ss <seekTime> -i <video> [-vf scale=<w>:-2] -frames:v 1 [-q:v <q>] -y <out>` via `runFfmpegSyncOrThrow`. Default `outputPath` is a `ltx_frame_<ts>_<rand>.jpg` in `os.tmpdir()`.
- **`stopExportProcess()`**: `kill()` + null-out the module-global.

### `timeline.ts` — `flattenTimeline(clips: ExportClip[]): FlatSegment[]`
1. Filter to `type === 'video' || type === 'image'`.
2. Boundary set = `{0}` ∪ all `startTime` ∪ all `startTime + duration`, sorted.
3. For each `[t0, t1)` sub-interval: pick the clip with the highest `trackIndex` active at the midpoint (NLE convention: top track wins). Push a segment with computed `trimStart = clip.trimStart + offsetInClip * clip.speed`. If none active, push a `type: 'gap'` segment (black).
4. Merge pass: collapse adjacent segments that share `filePath` (non-empty), `speed`, `reversed`, `flipH`, `flipV`, `opacity`, `muted`, `volume`, and whose `prev.trimStart + prev.duration * prev.speed ≈ seg.trimStart` (within 0.01s).

### `video-filter.ts` — `buildVideoFilterGraph(segments, opts): { inputs, filterScript }`
Per-segment chain construction (one `[vN]` label per segment):
- **gap:** input `-f lavfi -i color=c=black:s=WxH:r=fps:d=<dur>`, filter `setsar=1`.
- **image:** input `-loop 1 -framerate <fps> -t <dur> -i <file>`, chain `scale=...:force_original_aspect_ratio=decrease,pad=...,setsar=1` + optional `hflip`/`vflip`.
- **video:** input `-i <file>`, chain `trim=start=...:end=...,setpts=PTS-STARTPTS` + optional `setpts=PTS/<speed>` (only when `speed !== 1`) + optional `reverse` + `scale/pad/setsar` + optional flips. **No fps filter per segment.**

Post-concat:
- `concat=n=<count>:v=1:a=0` → `[concatraw]`.
- `[concatraw]fps=<fps>[fpsout]` — single global framerate conversion.
- **Letterbox/pillarbox** (optional): computes visible height/width from `width/height` vs `letterbox.ratio`, emits `drawbox=...:c=0xRRGGBBAA:t=fill` pairs for the bars.
- **Subtitles** (optional, per-subtitle `drawtext`): escapes `\ ' : % \n`, scales `fontSize` by `height/1080`, supports `top`/`center`/`bottom` `y` expressions, optional background box from 6- or 8-char hex (`@alpha`), gated by `enable='between(t,start,end)'`.
- Final label renamed to `[outv]` if not already.

Returns the script with filter parts joined by `;\n`.

### `audio-mix.ts` — `mixAudioToPcm(clips, totalDuration, ffmpegPath)`
Constants: `SAMPLE_RATE = 48000`, `NUM_CHANNELS = 2`, 16-bit. Skips `muted`/`volume <= 0` clips. Audio clips contribute directly; video clips only if `fileHasAudio` (cached per file). For each source, `extractPcmBuffer` runs ffmpeg with `atrim=start=...:end=...,asetpts=PTS-STARTPTS` + `atempo` chain (decomposed for values outside `[0.5, 2.0]`) + optional `areverse`, outputting `s16le` stereo on `pipe:1`. Samples are summed into a `Float64Array` sized to `totalDuration * 48000 * 2`, scaled by `src.volume`, then clamped to `[-32768, 32767]` and written as Int16LE. Returns `{ pcmBuffer, sampleRate: 48000, channels: 2 }`.

## Integration Points

- **Registered from `main.ts`:** `registerExportHandlers()` is the last registrar wired up before `app.whenReady()`; `stopExportProcess()` is called from `app.on('before-quit')`.
- **Shared ffmpeg binary & runner reused outside export:** `ipc/file-handlers.ts` imports `extractVideoFrameToFile`, `findFfmpegPath`, `getVideoDimensions`, `runFfmpeg` — the import-transcode path (`transcodeVideoInPlace`) uses the same `runFfmpeg` to make H.264 proxies of imported video assets. `ipc/video-processing-handlers.ts` imports `extractVideoFrameToFile` for `extractVideoFrame`. **This is the precedent for the future primary-output proxy generator.**
- **IPC contract:** the `exportClip` and `exportSubtitle` Zod schemas in `shared/electron-api-schema.ts` mirror the `ExportClip` interface here (`timeline.ts`) and the `ExportSubtitle` interface in `video-filter.ts` — they must stay in sync. `exportNative` output is `emptyResult` (the unit `ipcResult({})`).
- **Primary-output design implication (forward-looking):** when generation starts emitting MOV/ProRes or EXR sequences, the encoder should reuse `findFfmpegPath()` + `runFfmpeg()` from this folder. The ProRes args block here (`-c:v prores_ks -profile:v <quality> -pix_fmt yuva444p10le` + `-c:a pcm_s16le`) is the existing profile template; a primary-output encoder would select the ProRes profile directly (no libx264 intermediate) because the input frames are already decoded. Note that the renderer's `<video>` cannot play EXR at all and ProRes playback is OS-dependent — so any primary-output feature must ship alongside a proxy generator (mirroring `transcodeVideoInPlace`) and a renderer-side codec-aware path, neither of which exists yet.
- **Cancellation:** `exportCancel` → `stopExportProcess()`. The `sessionId` parameter exists in the schema but the current implementation only tracks a single `activeExportProcess` — concurrent exports would require lifting this into a session map.
