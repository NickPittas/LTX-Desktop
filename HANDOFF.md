# Handoff — LTX Desktop IC-LoRA, local models, EXR/MOV next
**Date:** 2026-06-27

## Goal
Make LTX Desktop robust for local/offline LTX-2.3 workflows: GGUF/Kijai/local profiles; correct IC-LoRA behavior across standard adapters, in/outpaint, retake, and Ingredients; then implement primary-generation MOV ProRes / EXR outputs and HDR IC-LoRA. This fork must not fake unsupported workflows, must never re-download existing models, and must preserve high-quality data paths (MOV/EXR must be primary outputs from decoded frames, not transcodes from clamped MP4).

## Current State
Latest pushed commit is `239820b fix: improve local LTX workflows` on `main` (`https://github.com/NickPittas/LTX-Desktop.git`). Current working tree has uncommitted but reviewed changes for:

1. **IC-LoRA LoRA weight**
   - Adds real `lora_strength` / LoRA merge weight `0–2`, default `1.0`.
   - Keeps `conditioning_strength` separate as reference/conditioning attention.
   - Pipeline cache rebuilds when LoRA weight changes.
   - Reviewed and approved.

2. **Ingredients IC-LoRA T2V-style flow**
   - Ingredients no longer requires user driving video.
   - Backend `video_path` is optional only for Ingredients.
   - Non-Ingredients IC-LoRAs still require video and derive FPS/dims/frame count from uploaded video.
   - Ingredients uses prompt + reference sheet image + T2V-like output fields.
   - Backend tests for no-video Ingredients, non-Ingredients missing video, conditioning_type rejection, frame snapping, LoRA weight.
   - Reviewed and approved.

3. **Ingredients PromptBar/output settings + Generate no-op fix**
   - Root cause was `GenSpace.tsx` `handleGenerate` still checking `!icLoraInput.videoPath` while `canSubmit` exempted Ingredients. Button looked clickable, click silently returned.
   - Fixed so Ingredients generate is actionable when prompt + reference image are present.
   - FPS/duration/resolution/aspect live in `PromptBar` for Ingredients only.
   - FPS is a free numeric input, not fixed buttons/dropdown, so `25`, `50`, etc. are allowed.
   - V2V LoRAs hide/ignore these output settings and use source video FPS/dims.
   - Reviewed and approved.

Validation already reported by workers/reviewers:
- `backend:test -- tests/test_ic_lora.py`: `49 passed`
- `typecheck:py`: clean
- `typecheck:ts`: clean
- frontend typecheck clean for latest PromptBar changes
- OpenAPI regenerated for backend API changes

Run fresh validation before commit because changes are uncommitted and app is active:
```bash
rtk npx --yes pnpm@10.30.3 openapi:check
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_ic_lora.py
rtk npx --yes pnpm@10.30.3 typecheck:py
rtk npx --yes pnpm@10.30.3 typecheck:ts
rtk npx --yes pnpm@10.30.3 build:frontend
```

## Files in Flight
Tracked modified files to review/commit:
- `backend/api_types.py` — `IcLoraGenerateRequest.video_path` optional; added Ingredients T2V fields; `lora_strength` field.
- `backend/handlers/ic_lora_handler.py` — Ingredients no-video/T2V path; non-Ingredients `video_path` guard; LoRA strength forwarding.
- `backend/handlers/pipelines_handler.py` — cache key/state includes `lora_strength`.
- `backend/services/ic_lora_pipeline/ic_lora_pipeline.py` — protocol accepts `lora_strength` at construction.
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py` — passes `lora_strength` into `LoraPathStrengthAndSDOps`; ponytail note that one weight applies to all LoRAs in stack.
- `backend/state/app_state_types.py` — `ICLoraState.lora_strength`.
- `backend/tests/fakes/services.py` — fake IC-LoRA pipeline records `last_lora_strength`.
- `backend/tests/test_ic_lora.py` — new tests for Ingredients no-video path and LoRA strength.
- `frontend/generated/backend-openapi.json` — regenerated schema.
- `frontend/generated/backend-openapi.ts` — regenerated TS types.
- `frontend/hooks/use-ic-lora.ts` — optional `videoPath`, optional `width/height/numFrames`, `loraStrength`, conditional request body fields.
- `frontend/components/ICLoraPanel.tsx` — Ingredients no driving-video UI; no FPS buttons in panel; sends `videoPath: null`, `conditioningType: null` for Ingredients.
- `frontend/views/GenSpace.tsx` — PromptBar Ingredients settings; Generate no-op fix; derives Ingredients `frame_rate`, `num_frames`, `width`, `height` from PromptBar settings.
- `HANDOFF.md` — this file replaces stale old handoff.

Untracked / do not commit unless explicitly asked:
- `HF_TOKEN` — secret; never commit, never print contents.
- `subagent-artifacts/` — run reports/plans/reviews; useful context but not tracked unless user asks.
- `.task-reports/`, `progress.md`, many locator/plan markdown files, `backend/data/` — local artifacts/state.

## Changed This Session
- Committed and pushed earlier work: `239820b fix: improve local LTX workflows`.
- Killed orphan backend process `PID 1329462` bound to `127.0.0.1:41954`; restarted dev app.
- Current app is running in Herdr pane alias `app` via:
  ```bash
  npx --yes pnpm@10.30.3 dev:debug
  ```
  Backend reported ready: `Server running on http://127.0.0.1:41954`.
- Implemented/reviewed LoRA weight 0–2.
- Implemented/reviewed Ingredients no-video/T2V backend path.
- Implemented/reviewed Ingredients PromptBar settings + Generate no-op fix.
- Created and removed `/tmp/review_diff.txt` after it was used for review.

## Failed Attempts / Problems To Avoid
- ❌ **Initial Ingredients misunderstanding** — We treated Ingredients as V2V or static guide video. User clarified product semantics: Ingredients is **T2V + Ingredients LoRA + reference sheet image**, no user driving video. Internally docs may mention static guide, but UI/API must not require user video.
- ❌ **`video_conditioning=[]` debate** — Official workflow loops reference sheet internally, but current product decision is no user video and minimal T2V-like IC-LoRA path with reference sheet `images[]`. Do not re-open this unless testing proves output wrong.
- ❌ **Generate silent no-op** — `canSubmit` exempted Ingredients from `videoPath`, but `handleGenerate` still returned on missing video. Always align enabled state and click handler conditions.
- ❌ **FPS in wrong place** — Ingredients FPS buttons inside `ICLoraPanel` were wrong. Output FPS/duration/resolution/aspect belong in `PromptBar` when workflow is T2V/I2V-like. V2V LoRAs derive from source video.
- ❌ **Fixed FPS button list was wrong** — User needs free numeric FPS (`25`, `50`, etc.), not a closed list like `16/24/30/60`.
- ❌ **Bad backend startup** — App failed with `[Errno 98] address already in use` because orphan backend was bound to port `41954`. Use `fuser -n tcp 41954` and kill orphan before restart.
- ❌ **Test command got bad `-v` placement** — `pnpm backend:test -- tests/test_ic_lora.py -v` was interpreted incorrectly by repo wrapper. Use `pnpm backend:test -- tests/test_ic_lora.py`.
- ❌ **Subagents sometimes could not see pasted packets/artifacts** — Put exact packet text in prompts or reply over intercom. Prefer inline Task Packets and exact paths.
- ❌ **Planner/reviewer contract conflicts** — If subagent asks about writing plan files, tell it inline-only/no file write unless explicitly desired.
- ❌ **Researcher edited `progress.md` despite read-only** — `progress.md` is untracked; do not commit. Warn workers not to edit unless allowed.
- ❌ **Never claim official parity from ffprobe/mp4 only** — Need semantic visual/metric evidence and reviewer approval.
- ❌ **Do not touch inpaint/outpaint unless explicitly asked** — User says inpaint/in_outpainting is perfect; avoid mask/green guide/sigma changes.

## Next Step
1. **Finish current working tree**:
   - Run fresh validation commands listed above.
   - Inspect `git diff --stat` and critical diffs.
   - Commit current tracked changes if validation passes.
   - Ask explicit structured confirmation before push.

2. **Then move to EXR/MOV primary output implementation**:
   - Must produce MOV/EXR as **primary generation output from decoded frames**, not MP4→MOV/EXR transcode.
   - Requires implementation package before worker.
   - Reviewer already found plan fixes:
     - include retake direct encode path (`retake_pipeline.py` imports/calls `ltx_pipelines.utils.media_io.encode_video` directly; must route through new encoder service).
     - exact ProRes mapping:
       - `proxy`/`lt` → `prores_ks`, profile `proxy`/`lt`, `yuv422p10le`
       - `422`/`422_hq` → profile `standard`/`hq`, `yuv422p10le`
       - `4444`/`4444_xq` → profile `4444`/`4444xq`, `yuva444p10le`
     - add `exr_zip_half` and likely `exr_zip_float` or remove float test.
     - existing project import `transcodeVideoInPlace` destroys primary video in place; must split keep-primary + proxy sidecar for ProRes/EXR.

3. **HDR IC-LoRA after EXR path exists**:
   - HDR currently gated intentionally.
   - Needs scene-embedding injection, HDR decode/postprocess equivalent, EXR primary output, SDR proxy/preview, gate removal only after backend+UI validated.

## Integral Repo Files
- `AGENTS.md` — mandatory repo instructions. Read at start of each turn.
- `package.json` — scripts.
- `backend/api_types.py` — request/response DTOs.
- `backend/handlers/ic_lora_handler.py` — IC-LoRA orchestration and workflow dispatch.
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py` — IC-LoRA pipeline wrapper.
- `backend/services/ltx_pipeline_common.py` — common output encode wrapper.
- `backend/services/retake_pipeline/ltx_retake_pipeline.py` — retake encode path; note direct encode call for EXR/MOV plan.
- `backend/handlers/video_generation_handler.py` — T2V generation path; resolution map; frame-count logic.
- `backend/tests/test_ic_lora.py` — main IC-LoRA tests.
- `backend/tests/test_ltx_ic_lora_pipeline.py` — pipeline wrapper tests.
- `frontend/views/GenSpace.tsx` — main generation UI and embedded `PromptBar`.
- `frontend/components/ICLoraPanel.tsx` — IC-LoRA source/reference controls.
- `frontend/hooks/use-ic-lora.ts` — IC-LoRA request hook.
- `frontend/generated/backend-openapi.json` / `.ts` — generated API schema/types.
- `electron/export/export-handler.ts` — existing timeline/export ProRes support, not primary generation.
- `electron/ipc/file-handlers.ts` — project import/video transcode behavior; important for ProRes/EXR proxy plan.

## External / Non-Repo Important Paths
- App data dir: `/home/npittas/.local/share/LTXDesktop/`
  - `settings.json`
  - `model_profiles.json`
  - `app_state.json`
  - `logs/`
  - `outputs/`
- Correct standalone backend command must set data dir:
  ```bash
  LTX_APP_DATA_DIR=/home/npittas/.local/share/LTXDesktop backend/.venv/bin/python backend/ltx2_server.py
  ```
- Current app/dev command:
  ```bash
  npx --yes pnpm@10.30.3 dev:debug
  ```
- Port: `127.0.0.1:41954`.
- If port stuck:
  ```bash
  fuser -n tcp 41954
  kill <pid>
  ```

### ComfyUI / LTX Examples
- Main local workflow tree:
  `/home/npittas/ComfyUI-Easy-Install/ComfyUI-Easy-Install/ComfyUI/custom_nodes/ComfyUI-LTXVideo/example_workflows/`
- Important official workflow:
  `.../2.3/LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json`
- Previous cloned/reference repo files:
  `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo/`
- Comfy assets used for tests:
  `/mnt/ssd1/ltx-results/Assets/`

### Model / Asset Locations
- User settings model dir:
  `/mnt/ssd1/models/diffusion_models/LTX`
- GGUF transformer:
  `/mnt/ssd1/LTX_models/gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`
- Kijai transformer:
  `/mnt/ssd1/LTX_models/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors`
- Gemma GGUF:
  `/mnt/ssd1/LTX_models/text_encoders/unsloth/gemma-3-12b-it-qat-GGUF/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf`
- x2 spatial upscaler:
  `/mnt/ssd1/LTX_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- Deblur adapter:
  `/mnt/ssd1/LTX_models/adapters/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors`
- In/outpainting adapter:
  `/mnt/ssd1/LTX_models//adapters/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors`
- Depth Union blocked model path:
  `/mnt/ssd1/models/diffusion_models/LTX/dpt-hybrid-midas`
  - Missing; do not download unless approved.
- Smoke/result dir:
  `/mnt/ssd1/ltx-results/ic-matrix/`
- Full 196f inpaint test source/mask:
  `/mnt/ssd1/1920x1080.mp4`
  `/mnt/ssd1/1920x1080_mask.mp4`
- Official downloaded IC-LoRA samples:
  `subagent-artifacts/official-samples/`
  - contains download report and 36 MP4 samples.

## Validated Outputs / Evidence
- Full 196f inpaint stress:
  `/home/npittas/.local/share/LTXDesktop/outputs/ic_lora_20260626_222107_55727c89.mp4`
  - 196f, 1920x1088, peak ~31.4GB VRAM.
- Controlled colorization:
  `subagent-artifacts/ic-lora-colorization-controlled/`
  - output `/home/npittas/.local/share/LTXDesktop/outputs/ic_lora_20260627_023802_3ccadcfd.mp4`
  - reviewer approved; source grayscale → output colorized.
- Controlled deblur:
  `subagent-artifacts/ic-lora-deblur-controlled/`
  - output `/home/npittas/.local/share/LTXDesktop/outputs/ic_lora_20260627_034444_eac5b1b4.mp4`
  - reviewer approved; Laplacian variance higher all sampled frames.
- Kijai/GGUF smoke outputs were copied into `/mnt/ssd1/ltx-results/ic-matrix/` earlier, but invalid same-source IC-LoRA matrix was wiped by user request.

## Commands / Best Practices
### Run app
```bash
npx --yes pnpm@10.30.3 dev:debug
```
Use Herdr for long-running app panes. Current pane alias may be `app`.

### Run tests/checks
```bash
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_ic_lora.py
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_ltx_ic_lora_pipeline.py
rtk npx --yes pnpm@10.30.3 typecheck:py
rtk npx --yes pnpm@10.30.3 typecheck:ts
rtk npx --yes pnpm@10.30.3 build:frontend
rtk npx --yes pnpm@10.30.3 openapi:generate
rtk npx --yes pnpm@10.30.3 openapi:check
```
Full CI expected for PRs:
```bash
rtk npx --yes pnpm@10.30.3 typecheck
rtk npx --yes pnpm@10.30.3 backend:test
rtk npx --yes pnpm@10.30.3 build:frontend
```

### Programmatic API usage
- Backend base when app running: `http://127.0.0.1:41954`.
- Use app endpoints through `backendFetch` in frontend; do not raw `fetch` to app backend in UI code.
- IC-LoRA endpoint: `POST /api/ic-lora/generate`.
  - Ingredients request now omits `video_path`, uses `adapter_id: "ingredients"`, `prompt`, `images`, optional `width`, `height`, `num_frames`, `frame_rate`, `lora_strength`.
  - V2V adapters must send `video_path`; backend derives fps/dims/frame count.
- Never paste HF tokens in prompts/logs. Use existing token file/env only.

## Workflow / Agent Best Practices
- Read `AGENTS.md` every turn.
- Orchestrator delegates; do not self-code in parent except safe file writes like this handoff or trivial cleanup.
- Use codemap-assisted workflow:
  1. locator
  2. implementation Task Packet
  3. reviewer checks packet
  4. scoped worker edits only allowed files
  5. reviewer checks diff
  6. orchestrator validates/reports/commits
- Always pass exact cwd `/home/npittas/LTX-Desktop` to subagents.
- Use async/live subagents and monitor `intercom pending`.
- Reviewers are static review only; orchestrator owns validation execution.
- Subagent prompts must include exact files, commands, timeouts, stop conditions.
- Do not run model workflows from parent; use scoped runners with exact commands/artifacts.
- Before push/delete/tag/release, use structured `ask_user` confirmation.
- Do not download models unless explicitly approved. Official sample MP4 fetches are okay; model weights are not.
- Do not use `/mnt` in research/locator agents unless explicitly scoped; runner agents may use `/mnt` for live smoke/media.
- Do not commit `HF_TOKEN`, `backend/data`, `subagent-artifacts`, temp reports unless explicitly requested.
- Preserve source resolution for IC-LoRA; never add arbitrary 768 pre-downscale.
- Inpaint/in_outpainting is considered working; avoid touching.
- Canny/depth disabled by default; only run when Union Control explicitly enabled.
- Union Control first, then LoRA if both are enabled.
- Empty prompt must remain supported/default for official inpaint; retake requires prompt.
- For visual review, extract full-resolution single frames, not contact sheets.

## EXR/MOV Plan Notes For Next Agents
- Current primary encode bottleneck: `ltx_pipelines/utils/media_io.py::encode_video()` hardcodes H.264/yuv420p.
- App wrapper: `backend/services/ltx_pipeline_common.py::encode_video_output()`.
- Pipelines using encode path:
  - `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
  - `backend/services/a2v_pipeline/ltx_a2v_pipeline.py`
  - `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  - `backend/services/retake_pipeline/ltx_retake_pipeline.py` (**direct encode call; must fix**)
- Handlers hardcode `.mp4` paths:
  - `backend/handlers/video_generation_handler.py`
  - `backend/handlers/ic_lora_handler.py`
  - `backend/handlers/retake_handler.py`
- Existing Electron export ProRes support is timeline/export-only, not primary generation:
  `electron/export/export-handler.ts`.
- Frontend `<video>` cannot play EXR sequences; need proxy MP4 sidecar or frame viewer.
- ProRes playback is OS-dependent; Windows likely needs proxy fallback.
- EXR needs OpenEXR/pyexr or another writer; not currently in `backend/pyproject.toml` / lock.

## HDR IC-LoRA Plan Notes
- HDR workflows are currently gated:
  - `backend/handlers/ic_lora_handler.py` `_UNAVAILABLE_WORKFLOWS`
  - `frontend/components/ICLoraPanel.tsx` workflow unavailable
  - tests assert 400 unavailable.
- HDR requires:
  1. scene embeddings injection (`kind="embeddings"` asset)
  2. HDR decode/postprocess equivalent to `LTXVHDRDecodePostprocess`
  3. EXR primary output
  4. SDR preview/proxy
  5. gate removal only after validated
- Do not implement HDR by faking standard IC-LoRA output.

## Commit / Push Guidance
Before committing current changes:
```bash
rtk git status --short
rtk git diff --stat
rtk npx --yes pnpm@10.30.3 openapi:check
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_ic_lora.py
rtk npx --yes pnpm@10.30.3 typecheck:py
rtk npx --yes pnpm@10.30.3 typecheck:ts
rtk npx --yes pnpm@10.30.3 build:frontend
```
Then commit tracked source changes only. Suggested commit message:
```bash
git add -u
git commit -m "fix: support ingredients ic-lora generation"
```
Do **not** push without structured confirmation from user.
