# Planner Report

## Status
split-required

## Why Split / Parallelize
Official IC-LoRA support touches shared request types, handler dispatch, pipeline generation, UI gating, and tests. Work is sequential because each later adapter un-gates UI/backend behavior on top of shared workflow metadata and pipeline helpers. Inpaint must land first, outpaint second, then remaining official workflows can reuse the same registry and validation surface.

## Interference Check
- parallel safe: no
- shared files or generated outputs: `backend/api_types.py`, `backend/handlers/ic_lora_handler.py`, `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`, `frontend/components/ICLoraPanel.tsx`, `frontend/views/GenSpace.tsx`, `frontend/hooks/use-ic-lora.ts`, `backend/tests/test_ic_lora.py`
- shared validation state: backend pytest fakes and IC-LoRA pipeline cache; frontend strict TS
- worktree isolation required: no, but run phases sequentially
- rationale: all phases edit same adapter/workflow registry and submit/generate contracts. Parallel workers would race on workflow IDs, required inputs, and tests.

## Authoritative Official Evidence Used
- `subagent-artifacts/official-all-iclora-workflows-research.md`: official adapter/workflow inventory. Note: requested `subagent-artifacts/official-inoutpaint-node-implementation-research.md` was missing from repo; plan uses official Lightricks source/workflow files below instead.
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/vanish_nodes.py`: `LTXVInpaintPreprocess` lines 92-143. Green is RGB `(102,255,0)` / `#66FF00`; mask active area becomes green; single-frame mask broadcasts; output trims to shortest frame count.
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/iclora.py`: `LTXAddVideoICLoRAGuideAdvanced` lines 261-430. Official advanced guide inherits video guide encode/latent insertion, then appends guide attention entry with `attention_strength` and optional normalized `attention_mask`; in official in/outpaint workflows the mask input is not linked, widgets set guide `strength=1`, `latent_downscale_factor=1`, `crop='disabled'`, tiled encode false, `attention_strength=1`.
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/pyramid_blending.py`: `_pyramid_blend_chunk` lines 97-114 and `LTXVLaplacianPyramidBlend` lines 178-267. Polarity is `white = image_a`, `black = image_b`; low-res mask dilation defaults differ per workflow.
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json`: source video `LoadVideo`, B/W mask `LoadVideo`; note says **white = inpaint, black = keep**; two-stage pipeline uses green composites, `LTXAddVideoICLoRAGuideAdvanced`, `SamplerCustomAdvanced` twice, and Laplacian blends.
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json`: source video -> `ImagePadForOutpaintTargetSize` -> generated mask; new padded area mask drives green composite and blends; two-stage structure mirrors inpaint.

## Official Inpaint Pipeline To Implement
1. Inputs: source video, B/W mask video/image, optional prompt (empty prompt allowed), In/Outpainting LoRA.
2. Mask polarity: white/1.0 = inpaint/generated region; black/0.0 = keep original.
3. Preprocess: resize source and mask like current resolution path, then create guide frames with official green composite: `source * (1-mask) + #66FF00 * mask`; broadcast single-frame mask to video length; trim to shortest.
4. Stage 1 low generation: condition empty/video latent from source through `LTXVImgToVideoConditionOnly`, then add green-composite guide through `AddVideoICLoRAGuideAdvanced` behavior (`frame_idx=0`, `strength=1`, `latent_downscale_factor=1`, `crop=disabled`, tiled=false, `attention_strength=1`, no attention mask unless explicitly wired later). Run low stage sampler.
5. Stage 1 blend: decode low result and `LTXVLaplacianPyramidBlend(image_a=generated_low, image_b=green_guide_low, mask=dilated_mask_low)`. White mask keeps generated in inpaint region; black keeps green/source guide outside.
6. Stage 2 high/refine generation: upscale/blended low result by multiplier 2, encode tiled, condition stage 2 with source/resized original, run second sampler.
7. Final blend: `LTXVLaplacianPyramidBlend(image_a=generated_high, image_b=green_guide_full, mask=dilated_mask_full)`, then save video/audio. This replaces current alpha composite; do not accept simple `mp4 exists` as correctness.

## Official Outpaint Pipeline To Implement
1. Inputs: source video, target canvas width, target canvas height, optional prompt (empty prompt allowed), In/Outpainting LoRA. No user mask required for basic outpaint.
2. Pad source to target canvas using official `ImagePadForOutpaintTargetSize` semantics: source stays on target canvas, new/padded area becomes mask white. Existing source area stays mask black. Stop if exact node placement/pad behavior cannot be matched from workflow or installed node source.
3. New-area semantics: white/new area = generate/outpaint; black/original source = keep.
4. Preprocess and two-stage generation mirror inpaint, but mask comes from padded canvas instead of user mask. Official workflow uses `ImagePadForOutpaintTargetSize` widgets `[1920, 1088, 0, 'nearest-exact']`, then resize-to-multiple/match-size, green composites, `LTXAddVideoICLoRAGuideAdvanced`, two samplers, and Laplacian blends.
5. Validation must prove output canvas equals requested target and original source pixels remain preserved in black-mask area.

## Adapter Matrix
| Adapter | Official workflow | Required UI inputs | Backend workflow | Release state |
|---|---|---|---|---|
| `in_outpainting` inpaint | `LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json` | source video, mask video/image, optional prompt | special inpaint two-stage green + Laplacian | implement phase 1 |
| `in_outpainting` outpaint | `LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json` | source video, target canvas width/height, optional prompt | special outpaint pad + two-stage green + Laplacian | implement phase 2 |
| `union_control` | `LTX-2.3_ICLoRA_Union_Control_Distilled.json` | source video, conditioning type canny/depth/pose as supported, prompt | special preprocessing, then generic guide; Union LoRA loaded before any extra LoRA | phase 3; keep canny/depth disabled unless selected |
| `motion_track_control` | `LTX-2.3_ICLoRA_Motion_Track_Distilled.json` | reference image/video, sparse track points/path editor, prompt | special `LTXVSparseTrackEditor`/`LTXVDrawTracks`, then generic guide | gated until phase 4 |
| `ingredients` | `LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json` | single composite reference sheet image, prompt in reference-sheet/generated-video format | generic guide with image/reference sheet | phase 5; UI should say single sheet, not fake multi-sheet correctness |
| `hdr` | `LTX-2.3_ICLoRA_HDR_Distilled.json` | SDR source video/image, prompt, HDR output settings/path | generic guide plus `LTXVHDRDecodePostprocess`, HDR scene embeddings/output path | gated until phase 6 |
| `hdr_scene_embeddings` | support asset for HDR | none standalone | not selectable | gated/support only |
| `lipdub` | `LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json` | talking-head/source video, audio/speech track, prompt | special audio encode/ref tokens, two-stage, tiled guide encode true | gated until phase 7 |
| `water_simulation` | generic V2V IC-LoRA workflow only | source video, prompt | generic `LTXAddVideoICLoRAGuide` single-stage | phase 8 |
| `decompression` | generic V2V IC-LoRA workflow only | compressed source video, prompt | generic guide single-stage | phase 8 |
| `deblur` | generic V2V IC-LoRA workflow only | blurry source video, prompt | generic guide single-stage | phase 8 |
| `colorization` | generic V2V IC-LoRA workflow only | grayscale/low-color source video, prompt | generic guide single-stage | phase 8 |
| `day_to_night` | generic V2V IC-LoRA workflow only | daytime source video, prompt | generic guide single-stage | phase 8 |
| `instant_shave` | generic V2V IC-LoRA workflow only | face source video, prompt | generic guide single-stage | phase 8 |
| `cross_eyed` | generic V2V IC-LoRA workflow only | face/source video, prompt | generic guide single-stage | phase 8 |

## Proposed Task Sequence Or Parallel Batch
1. Task name: Phase 1 — official inpaint pipeline
   - purpose: replace current in/outpainting shortcut with official inpaint green-composite/two-stage/Laplacian path, keeping outpaint gated.
   - allowed files: `backend/api_types.py`, `backend/handlers/ic_lora_handler.py`, `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`, `backend/tests/test_ic_lora.py`, `backend/tests/fakes/services.py`, `frontend/components/ICLoraPanel.tsx`, `frontend/views/GenSpace.tsx`, `frontend/hooks/use-ic-lora.ts`
   - validation: targeted backend pytest, TS/Py typecheck, unit assertions for green and blend polarity
   - can run in parallel with: none
2. Task name: Phase 2 — official outpaint pipeline
   - purpose: add outpaint mode with target canvas and generated mask/new-area semantics.
   - allowed files: same shared workflow files as phase 1
   - validation: outpaint canvas/mask unit tests and gating tests
   - can run in parallel with: none
3. Task name: Phase 3 — union control official pass
   - purpose: keep Union Control as explicit workflow, enforce selected canny/depth only, Union first before extra LoRA.
   - allowed files: same shared registry/handler/UI/tests
   - validation: existing order tests plus no implicit canny/depth for non-union adapters
   - can run in parallel with: none
4. Task name: Phase 4 — motion track
   - purpose: implement sparse track inputs/drawn guide or keep adapter gated with exact reason.
   - allowed files: shared registry/handler/UI/tests plus any narrow motion-track helper file if worker proves needed
   - validation: track guide shape/overlay test; live smoke checks object follows path
   - can run in parallel with: none
5. Task name: Phase 5 — ingredients
   - purpose: align UI/backend with official single composite reference sheet workflow.
   - allowed files: shared registry/handler/UI/tests
   - validation: one sheet required, prompt format hint present, no multi-image claim unless composited
   - can run in parallel with: none
6. Task name: Phase 6 — HDR
   - purpose: implement HDR-specific decode/postprocess and support asset handling, or keep gated.
   - allowed files: shared files plus HDR output helper if needed
   - validation: output bit depth/range path, not visual contrast only
   - can run in parallel with: none
7. Task name: Phase 7 — LipDub
   - purpose: add audioPath contract and official two-stage audio ref-token workflow.
   - allowed files: shared files plus audio UI hook wiring
   - validation: audio path required; fake pipeline records audio; live smoke checks mouth movement against audio, not mp4 existence
   - can run in parallel with: none
8. Task name: Phase 8 — standard generic adapters
   - purpose: un-gate generic V2V adapters that use official generic guide only.
   - allowed files: registry/UI/handler/tests only unless generic pipeline bug found
   - validation: each adapter can submit with source video+prompt; no canny/depth preprocessing unless Union explicitly selected
   - can run in parallel with: none

## Task Packets

### Task Packet 1 — Phase 1 Official Inpaint Pipeline

#### User Goal
Implement official LTX 2.3 IC-LoRA inpaint pipeline from Lightricks ComfyUI source: green composite, advanced guide behavior, two-stage generation, Laplacian pyramid blend, white=inpaint/black=keep, empty prompt allowed.

#### Mode
general-coding

#### Relevant Locations
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `class LTXIcLoraPipeline`, `_run_inference`, `_composite_in_outpainting`, `generate`
  approximate lines: 63-305
  stable anchor: `def _composite_in_outpainting(`
  reason: current implementation passes mask as `conditioning_attention_mask` and alpha-composites post-output; must replace with official green/two-stage/Laplacian path for inpaint.
  confidence: high
- file: `backend/handlers/ic_lora_handler.py`
  symbol: `_ADAPTER_WORKFLOW`, `_UNAVAILABLE_WORKFLOWS`, `IcLoraHandler.generate`
  approximate lines: 61-85, 286-463
  stable anchor: `workflow == "in_outpainting"`
  reason: split inpaint vs outpaint workflow validation, allow empty prompt only for in/outpaint, pass workflow/mask/original video to pipeline.
  confidence: high
- file: `backend/api_types.py`
  symbol: `AdapterPipeline`, `IcLoraGenerateRequest`
  approximate lines: 407-422, 606-617
  stable anchor: `class IcLoraGenerateRequest`
  reason: add smallest request field needed to disambiguate inpaint/outpaint, e.g. `workflow_mode: Literal["inpaint", "outpaint"] = "inpaint"`; do not add outpaint canvas fields until phase 2.
  confidence: high
- file: `frontend/components/ICLoraPanel.tsx`
  symbol: `AdapterWorkflow`, `IC_LORA_ADAPTERS`, `onChange`, mask UI
  approximate lines: 36-68, 149-168, 619-642, 704-724
  stable anchor: `workflow === 'in_outpainting'`
  reason: rename UI semantics to inpaint mask, keep outpaint unavailable until phase 2, expose ready state only when mask exists.
  confidence: high
- file: `frontend/views/GenSpace.tsx`
  symbol: `icLoraInput`, `handleGenerate`, `canSubmit`
  approximate lines: 997-1020, 1377-1420, 1690-1740
  stable anchor: `const isInOutpainting = isIcLoraMode && icLoraInput.adapterId === 'in_outpainting'`
  reason: keep empty prompt for inpaint, pass inpaint mode through submit contract.
  confidence: high
- file: `frontend/hooks/use-ic-lora.ts`
  symbol: `IcLoraSubmitParams`, `submitIcLora`
  approximate lines: 7-58
  stable anchor: `mask_path = params.maskPath`
  reason: forward new inpaint/outpaint mode field if added.
  confidence: high
- file: `backend/tests/test_ic_lora.py`
  symbol: `TestIcLoraWorkflowGating`, `TestIcLoraEmptyPromptWorkflow`
  approximate lines: 539-904
  stable anchor: `test_in_outpainting_forwards_original_video_and_mask`
  reason: add/adjust tests for official inpaint request and fake pipeline args.
  confidence: high
- file: `backend/tests/fakes/services.py`
  symbol: `FakeIcLoraPipeline.generate`
  approximate lines: 580-620
  stable anchor: `self.generate_calls.append(kwargs)`
  reason: fake must record new workflow/mode fields; no real video needed.
  confidence: high
- file: `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/vanish_nodes.py`
  symbol: `LTXVInpaintPreprocess`
  approximate lines: 92-143
  stable anchor: `_BG_COLOR_RGB = (102, 255, 0)`
  reason: official green composite source.
  confidence: high
- file: `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/pyramid_blending.py`
  symbol: `LTXVLaplacianPyramidBlend`
  approximate lines: 178-267
  stable anchor: `tooltip="Blend mask (white = image_a, black = image_b)."`
  reason: official blend polarity and low-res dilation.
  confidence: high

#### Allowed Edit Files
- `backend/api_types.py`
- `backend/handlers/ic_lora_handler.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/tests/test_ic_lora.py`
- `backend/tests/fakes/services.py`
- `frontend/components/ICLoraPanel.tsx`
- `frontend/views/GenSpace.tsx`
- `frontend/hooks/use-ic-lora.ts`

#### Read-Only Context Files
- `subagent-artifacts/official-all-iclora-workflows-research.md`
- `subagent-artifacts/ic-lora-workflow-implementation-locator.md`
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/vanish_nodes.py`
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/iclora.py`
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/pyramid_blending.py`
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json`

#### Required Change
- Add explicit inpaint mode to the request/submit path only if needed; keep default as inpaint for backward compatibility with current `in_outpainting` mask UI.
- In backend validation, treat `adapter_id='in_outpainting'` + mode inpaint as requiring `mask_path`, allowing empty prompt, and passing `workflow_mode='inpaint'` to pipeline. Keep outpaint rejected with a clear “Outpaint workflow not implemented yet” until phase 2.
- In `LTXIcLoraPipeline`, replace simple post alpha composite for inpaint with official helper path:
  - helper `green_composite_frames(frames, mask)` matching `LTXVInpaintPreprocess` exactly (`#66FF00`, broadcast single-frame mask, trim shortest, white=inpaint).
  - helper `laplacian_pyramid_blend(image_a, image_b, mask, mask_low_res_dilation)` matching official polarity (`white=image_a/generated`, `black=image_b/original/green guide`). Use torch/kornia if already installed; do not add dependency.
  - branch `generate(... workflow_mode='inpaint', mask_path=..., original_video_path=...)` that prepares green guide and mask and runs the closest implementable two-stage flow with current `ICLoraPipeline`. If current high-level `ICLoraPipeline` cannot return frames/latents or cannot run the second stage without broad rearchitecture, stop and report exact missing API; do not ship fake two-stage.
- Update UI copy: mask label must say `Mask (white = inpaint, black = keep)`; empty prompt allowed only for inpaint.
- Keep standard adapters and unavailable adapters behavior unchanged.

#### Non-Goals
- Do not implement outpaint in phase 1.
- Do not implement motion track, HDR, LipDub, or standard generic adapter changes.
- Do not add new dependencies.
- Do not change model download locations.
- Do not run full live generation in unit tests.

#### Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ic_lora.py`
- `rtk pnpm typecheck:py`
- `rtk pnpm typecheck:ts`

Expected result:
- Green preprocess unit proves `#66FF00` in white mask pixels, source preserved in black pixels, single-frame mask broadcast, white=inpaint/black=keep.
- Laplacian blend unit proves white mask selects generated/image_a and black mask selects original/image_b; no polarity inversion.
- Workflow tests prove inpaint requires mask, accepts empty prompt, passes original video/mask/workflow mode, and no longer accepts “mp4 exists” alone as correctness.

#### Stop Conditions
Stop and report if:
- target symbol is missing
- required fix exceeds allowed files
- validation cannot run
- existing architecture contradicts the requested change
- `ltx_pipelines.ICLoraPipeline` lacks accessible API for two-stage/frame-return implementation without copying a large ComfyUI graph
- exact official mask polarity or green composite cannot be preserved
- task requires product/design judgment not in packet

#### Required Return Contract
Return only a task-focused summary. Include status, files inspected/changed, validation evidence, blockers, and risks. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues.

### Task Packet 2 — Phase 2 Official Outpaint Pipeline

#### User Goal
Implement official LTX 2.3 IC-LoRA outpaint pipeline separately from inpaint: pad source to target canvas, generate mask for new area, use white=new/inpaint black=keep, green composite, two-stage generation, Laplacian blends.

#### Mode
general-coding

#### Relevant Locations
- file: `backend/api_types.py`
  symbol: `IcLoraGenerateRequest`
  approximate lines: 606-617
  stable anchor: `mask_path: str | None = None`
  reason: add `outpaint_target_width` and `outpaint_target_height` (or one nested minimal equivalent if code style prefers) with validation.
  confidence: high
- file: `backend/handlers/ic_lora_handler.py`
  symbol: `IcLoraHandler.generate`
  approximate lines: 298-310, 450-463
  stable anchor: `workflow == "in_outpainting"`
  reason: route `workflow_mode='outpaint'`, require target canvas, do not require mask.
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `LTXIcLoraPipeline.generate`
  approximate lines: 162-305
  stable anchor: `workflow_mode`
  reason: implement outpaint pad-to-target and generated mask semantics, then reuse phase 1 inpaint two-stage helpers.
  confidence: high
- file: `frontend/components/ICLoraPanel.tsx`
  symbol: `AdapterWorkflow`, mask UI area, adapter selector
  approximate lines: 36-68, 619-724
  stable anchor: `workflow === 'in_outpainting'`
  reason: add inpaint/outpaint selector and target canvas width/height inputs for outpaint.
  confidence: high
- file: `frontend/views/GenSpace.tsx`
  symbol: `icLoraInput`, `handleGenerate`
  approximate lines: 997-1020, 1377-1420
  stable anchor: `adapterId: icLoraInput.adapterId`
  reason: persist and submit outpaint target dimensions/mode.
  confidence: high
- file: `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json`
  symbol: `ImagePadForOutpaintTargetSize`, `LTXVInpaintPreprocess`, `LTXVLaplacianPyramidBlend`
  approximate lines: JSON nodes order 31-37 and 40-60
  stable anchor: `ImagePadForOutpaintTargetSize` widgets `[1920, 1088, 0, 'nearest-exact']`
  reason: official outpaint graph evidence.
  confidence: high

#### Allowed Edit Files
- `backend/api_types.py`
- `backend/handlers/ic_lora_handler.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/tests/test_ic_lora.py`
- `backend/tests/fakes/services.py`
- `frontend/components/ICLoraPanel.tsx`
- `frontend/views/GenSpace.tsx`
- `frontend/hooks/use-ic-lora.ts`

#### Read-Only Context Files
- all Phase 1 read-only files
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json`

#### Required Change
- Add outpaint UI mode under In/Outpainting with target canvas width/height fields; default to source dimensions rounded up only if UI can inspect source metadata, otherwise require explicit values.
- Backend rejects outpaint if target canvas is missing or smaller than source.
- Pipeline pads source onto target canvas and creates generated mask where padded/new pixels are white and source region is black.
- Reuse green composite and Laplacian blend helpers from Phase 1. Do not reuse user `mask_path` for outpaint unless later product asks for custom placement masks.

#### Non-Goals
- No arbitrary placement UI unless official node semantics are verified.
- No crop/scale creative controls beyond target canvas.
- No changes to unrelated adapters.

#### Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ic_lora.py`
- `rtk pnpm typecheck:py`
- `rtk pnpm typecheck:ts`

Expected result:
- Outpaint unit proves padded/new area mask is white, original source area black.
- Outpaint generated output dimensions equal target canvas.
- Pixel preservation check proves black/source region is preserved by final blend within codec tolerance.
- Backend/UI gating proves mask is not required for outpaint but target canvas is.

#### Stop Conditions
Stop and report if:
- exact `ImagePadForOutpaintTargetSize` placement semantics cannot be verified
- target canvas validation requires design decision not in packet
- two-stage helpers from Phase 1 are unavailable or incomplete
- required fix exceeds allowed files

#### Required Return Contract
Return task-focused status, files changed, validation evidence, blockers, and risks only.

### Task Packet 3 — Phase 3 Union Control Official Pass

#### User Goal
Ensure Union Control follows official behavior and project rules: canny/depth disabled by default, only run when Union Control is explicitly enabled, and Union Control LoRA loads before any selected extra adapter.

#### Mode
general-coding

#### Relevant Locations
- file: `backend/handlers/ic_lora_handler.py`, lines 61-85, 298-345, anchors `_ADAPTER_WORKFLOW`, `_resolve_base_lora_path`.
- file: `backend/tests/test_ic_lora.py`, lines 480-620, anchors `test_canny_with_adapter_loads_union_then_ingredients`.
- file: `frontend/components/ICLoraPanel.tsx`, lines 74-82, 149-168, conditioning UI anchors.
- official workflow: `LTX-2.3_ICLoRA_Union_Control_Distilled.json`, nodes `CannyEdgePreprocessor`, `DWPreprocessor`, `LoadVideoDepthAnythingModel`, `LTXAddVideoICLoRAGuide`.

#### Allowed Edit Files
- `backend/handlers/ic_lora_handler.py`
- `backend/tests/test_ic_lora.py`
- `frontend/components/ICLoraPanel.tsx`

#### Read-Only Context Files
- `backend/api_types.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_Distilled.json`

#### Required Change
- Keep/verify Union requires explicit `conditioning_type`.
- Keep/verify non-Union adapters do not trigger canny/depth preprocessing unless user selected conditioning.
- Preserve LoRA load order: Union first, selected adapter second.

#### Non-Goals
- Do not add pose unless current app already has complete pose processor path.

#### Validation
Commands: `rtk pnpm backend:test -- tests/test_ic_lora.py`, `rtk pnpm typecheck:ts`, `rtk pnpm typecheck:py`
Expected result: existing and new tests prove no implicit canny/depth and correct Union-first ordering.

#### Stop Conditions
Stop if pose support is requested but missing model/UI path.

#### Required Return Contract
Task-focused status only.

### Task Packet 4 — Phase 4 Motion Track

#### User Goal
Implement or keep honestly gated official Motion Track Control based on `LTXVSparseTrackEditor` + `LTXVDrawTracks` guide requirements.

#### Mode
general-coding

#### Relevant Locations
- file: `frontend/components/ICLoraPanel.tsx`, lines 48-68 adapter registry and disabled reason.
- file: `backend/handlers/ic_lora_handler.py`, lines 78-85 unavailable workflows.
- official workflow: `LTX-2.3_ICLoRA_Motion_Track_Distilled.json`, nodes `LTXVSparseTrackEditor` and `LTXVDrawTracks`.

#### Allowed Edit Files
- `backend/api_types.py`
- `backend/handlers/ic_lora_handler.py`
- `backend/tests/test_ic_lora.py`
- `frontend/components/ICLoraPanel.tsx`
- `frontend/views/GenSpace.tsx`
- `frontend/hooks/use-ic-lora.ts`

#### Read-Only Context Files
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Motion_Track_Distilled.json`
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/sparse_tracks.py`

#### Required Change
- If implementing, add exact required input contract for sparse track points and generate track overlay guide before generic IC-LoRA guide.
- If not implementing in this phase, keep adapter disabled with exact reason and tests proving it returns 400.

#### Non-Goals
- No approximate “motion prompt only” support.

#### Validation
- Unit: track JSON -> guide frames dimensions and nonempty path pixels.
- Smoke: tracked object follows supplied path; background remains stable.

#### Stop Conditions
Stop if UI track editor scope exceeds single narrow component change.

#### Required Return Contract
Task-focused status only.

### Task Packet 5 — Phase 5 Ingredients

#### User Goal
Align Ingredients adapter with official single composite reference sheet workflow.

#### Mode
general-coding

#### Relevant Locations
- file: `frontend/components/ICLoraPanel.tsx`, lines 642-657 ingredient UI.
- file: `backend/handlers/ic_lora_handler.py`, line 309 ingredients validation.
- official workflow: `LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json`, node `LoadImage ingredients_input.jpg`, `LTXAddVideoICLoRAGuide`.

#### Allowed Edit Files
- `backend/handlers/ic_lora_handler.py`
- `backend/tests/test_ic_lora.py`
- `frontend/components/ICLoraPanel.tsx`
- `frontend/views/GenSpace.tsx`

#### Read-Only Context Files
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json`

#### Required Change
- Require one composite reference sheet image, not arbitrary multi-image promise. If multiple images remain accepted, UI/backend must label them as separate refs and not claim official sheet equivalence.
- Add prompt hint: `Reference sheet: ... / Generated video: ...`.

#### Non-Goals
- No automatic collage builder unless explicitly requested.

#### Validation
- Backend rejects ingredients with zero images.
- UI copy states official single composite sheet.
- Fake pipeline records image ref path/frame/strength.

#### Stop Conditions
Stop if automatic sheet composition is requested without design specs.

#### Required Return Contract
Task-focused status only.

### Task Packet 6 — Phase 6 HDR

#### User Goal
Implement or keep gated official HDR IC-LoRA workflow with HDR decode/postprocess and scene embeddings support.

#### Mode
general-coding

#### Relevant Locations
- file: `backend/runtime_config/model_download_specs.py`, lines 221-234 HDR and `hdr_scene_embeddings` entries.
- file: `backend/handlers/ic_lora_handler.py`, lines 78-85 HDR unavailable reason.
- official workflow: `LTX-2.3_ICLoRA_HDR_Distilled.json`, nodes `LTXVHDRDecodePostprocess`, `LTXICLoRALoaderModelOnly` for HDR.

#### Allowed Edit Files
- `backend/api_types.py`
- `backend/runtime_config/model_download_specs.py`
- `backend/handlers/ic_lora_handler.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/tests/test_ic_lora.py`
- `frontend/components/ICLoraPanel.tsx`

#### Read-Only Context Files
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_HDR_Distilled.json`
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/hdr.py`

#### Required Change
- Implement only if HDR output path/bit-depth support exists. Otherwise keep gated and improve message: requires HDR postprocess/output support.

#### Non-Goals
- No fake HDR by contrast/saturation.

#### Validation
- Test checks 16-bit/HDR-aware output path/range metadata if implemented.
- Gating test remains if not implemented.

#### Stop Conditions
Stop if output container/format decision is needed.

#### Required Return Contract
Task-focused status only.

### Task Packet 7 — Phase 7 LipDub

#### User Goal
Implement official LipDub two-stage IC-LoRA with audio conditioning, or keep gated until audio path contract exists.

#### Mode
general-coding

#### Relevant Locations
- file: `backend/api_types.py`, `IcLoraGenerateRequest` lines 606-617.
- file: `frontend/hooks/use-ic-lora.ts`, lines 7-58.
- file: `frontend/views/GenSpace.tsx`, lines 997-1020 and 1377-1420.
- official workflow: `LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json`, nodes `LTXVAudioVAEEncode`, `LTXVSetAudioRefTokens`, `LTXVLatentUpsampler`, two `LTXAddVideoICLoRAGuide` nodes with tiled encode true.

#### Allowed Edit Files
- `backend/api_types.py`
- `backend/handlers/ic_lora_handler.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/tests/test_ic_lora.py`
- `frontend/components/ICLoraPanel.tsx`
- `frontend/views/GenSpace.tsx`
- `frontend/hooks/use-ic-lora.ts`

#### Read-Only Context Files
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json`

#### Required Change
- Add `audio_path` request/submit field only when implementing LipDub.
- Require source video + audio path + prompt; run official audio token conditioning and two-stage generation.
- If not implementing, keep disabled and backend 400.

#### Non-Goals
- No visual-only lip movement without audio conditioning.

#### Validation
- Backend rejects LipDub without audio.
- Fake pipeline records `audio_path`.
- Smoke checks mouth motion/sync against audio; not mp4 existence.

#### Stop Conditions
Stop if audio encode/ref token APIs are not available from current pipeline layer.

#### Required Return Contract
Task-focused status only.

### Task Packet 8 — Phase 8 Standard Generic IC-LoRA Adapters

#### User Goal
Un-gate official generic V2V IC-LoRA adapters that use the official generic `LTXAddVideoICLoRAGuide` workflow and no special preprocessing.

#### Mode
general-coding

#### Relevant Locations
- file: `frontend/components/ICLoraPanel.tsx`, lines 48-68 registry.
- file: `backend/handlers/ic_lora_handler.py`, lines 61-85 workflow map.
- official workflow: `LTX-2.3_V2V_ICLoRA_Single_Stage_Distilled.json`, node `LTXAddVideoICLoRAGuide`.

#### Allowed Edit Files
- `backend/handlers/ic_lora_handler.py`
- `backend/tests/test_ic_lora.py`
- `frontend/components/ICLoraPanel.tsx`

#### Read-Only Context Files
- `/tmp/pi-github-repos/Lightricks/ComfyUI-LTXVideo@master/example_workflows/2.3/LTX-2.3_V2V_ICLoRA_Single_Stage_Distilled.json`

#### Required Change
- Ensure `water_simulation`, `decompression`, `deblur`, `colorization`, `day_to_night`, `instant_shave`, and `cross_eyed` use source video + prompt only.
- Do not run canny/depth for them unless user explicitly also selected Union Control conditioning.

#### Non-Goals
- No adapter-specific fake workflows where official JSON is missing.

#### Validation
- For each adapter, backend accepts source video + prompt and loads only that adapter.
- Empty prompt rejected.
- No canny/depth preprocessing called by default.

#### Stop Conditions
Stop if an adapter has no official model path or download spec.

#### Required Return Contract
Task-focused status only.

## Planner Self-Check
- locator evidence sufficient: yes — high-confidence locator plus direct official source/workflow reads; one requested research file missing but redundant official evidence exists.
- allowed edit files minimal and explicit: yes — all tasks list explicit files; shared files force sequential work.
- read-only context minimal: yes — only named research/locator, official node/workflow files, and current files from locator.
- anchors/lines included: yes — relevant locations include symbols, approximate lines, anchors, reasons, confidence.
- validation concrete: yes — commands and expected test assertions included; smoke criteria reject “mp4 exists” pass.
- parallelization decision explicit and safe: yes — sequential split due shared registry/handler/pipeline/UI/tests.
- non-goals and stop conditions sufficient: yes — block on missing pipeline APIs, outpaint node semantics, product/design scope.
- reviewer findings addressed, if revision: not applicable — no reviewer feedback supplied.
