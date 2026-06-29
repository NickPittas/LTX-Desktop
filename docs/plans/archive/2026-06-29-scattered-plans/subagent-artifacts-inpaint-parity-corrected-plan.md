# Planner Report

## Status
split-required

## Why Split
Current inpaint parity gaps sit mostly in `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`, so workers must run sequentially to avoid same-file collisions. Packets are small and ordered: first lock mask/blend truth, then guide conditioning, then stage2/audio/seed behavior, then acceptance checks.

## Interference Check
- parallel safe: no
- shared files or generated outputs: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`, `backend/tests/test_ltx_ic_lora_pipeline.py`, generated test videos/temp files
- shared validation state: backend pytest, installed local `ltx_pipelines` package
- worktree isolation required: recommended if multiple workers are used; otherwise run sequentially
- rationale: packets 1-3 edit same pipeline/test files; current repo is dirty, so each worker must inspect `git status --short` before edits and avoid overwriting unrelated changes.

## Corrected Official Runtime Inputs
Use linked inputs, not widget defaults:
- Stage1/half-res mask dilation: workflow node `5382` `LTXVDilateVideoMask` feeds `5378 LTXVInpaintPreprocess` and `5266 LTXVLaplacianPyramidBlend`; its `spatial_radius` input link `14467` comes from node `5400 PrimitiveInt [15, "fixed"]`. Effective runtime radius = `15`. Node widget default `[32, 0]` is ignored while linked.
- Stage2/full-res mask dilation: workflow node `5379` `LTXVDilateVideoMask` feeds `5380 LTXVInpaintPreprocess` and `5226 LTXVLaplacianPyramidBlend`; its `spatial_radius` input link `14403` comes from node `5372 ComfyMathExpression ["2*a"]`, where `a` link `14468` comes from node `5400 PrimitiveInt [15, "fixed"]`. Effective runtime radius = `30`. Node widget default `[5, 0]` is ignored while linked.
- Blend low-res dilation controls are separate from mask grow: node `5266 LTXVLaplacianPyramidBlend` widget `[True, 5]` for stage1/inter-stage blend; node `5226 LTXVLaplacianPyramidBlend` widget `[True, 6]` for stage2/final blend.
- Stage1 sampler/noise/sigmas: node `4831 KSamplerSelect = euler_ancestral_cfg_pp`, node `4832 RandomNoise = [43, "fixed"]`, node `5025 ManualSigmas = 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`.
- Stage2 sampler/noise/sigmas: node `5215 KSamplerSelect = euler_cfg_pp`, node `5210 RandomNoise = [42, "fixed"]`, node `5211 ManualSigmas = 0.7250, 0.4219, 0.0`.

## Proposed Task Sequence
1. Mask/blend truth guardrails.
2. Direct IC-LoRA guide conditioning parity.
3. Stage2 sigmas/seed/audio output parity.
4. Visual/manual acceptance gate.

---

# Task Packet 1 — Mask/blend truth guardrails

## Status
ready

## User Goal
Reconcile mask-radius findings and prevent regressions: effective mask radii must come from linked workflow inputs (`15`, `30`), while blend low-res dilation stays separate (`5`, `6`).

## Mode
general-coding

## Relevant Locations
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `derive_stage_radii`
  approximate lines: 19-34
  stable anchor: `def derive_stage_radii(mask_grow_px: int) -> tuple[int, int]:`
  reason: current default `mask_grow_px=30` should resolve to official linked runtime stage radii `(15, 30)`.
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `INPAINT_BLEND1_LOW_RES_DILATION`, `INPAINT_BLEND2_LOW_RES_DILATION`
  approximate lines: 14-17
  stable anchor: `INPAINT_BLEND1_LOW_RES_DILATION = 5`
  reason: these are official Laplacian blend controls, not `LTXVDilateVideoMask` radii.
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `LTXIcLoraPipeline.generate_inpaint`
  approximate lines: 378-410
  stable anchor: `# 2. Dilate masks`
  reason: comments and variable names must match official linked runtime paths: stage1 half=15, stage2 full=30.
  confidence: high
- file: `backend/tests/test_ltx_ic_lora_pipeline.py`
  symbol: `TestDeriveStageRadii`
  approximate lines: 394-421
  stable anchor: `class TestDeriveStageRadii:`
  reason: update/add assertions documenting linked runtime values vs ignored widget defaults.
  confidence: high

## Allowed Edit Files
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/tests/test_ltx_ic_lora_pipeline.py`

## Read-Only Context Files
- `subagent-artifacts/locator-official-inpaint-parity-gaps.md`
- Official workflow URL: `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/2.3/LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json`
- Official source URL: `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/vanish_nodes.py` (`LTXVDilateVideoMask`, `LTXVInpaintPreprocess`)
- Official source URL: `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/pyramid_blending.py` (`LTXVLaplacianPyramidBlend`)

## Required Change
- Do not change `derive_stage_radii` behavior if it already returns `(15, 30)` for default `30`.
- Correct any misleading comments/docstrings that imply widget defaults `5/32` are effective mask radii.
- Add/adjust tests in `TestDeriveStageRadii` to state that default runtime links produce stage1 half-res `15` and stage2 full-res `30`, and that blend low-res dilation constants remain `5` and `6`.
- Do not expose new UI/API settings for blend dilation in this packet.

## Non-Goals
- No sampler changes.
- No guide conditioning changes.
- No source-of-truth JSON parser in production code.
- No UI changes.

## Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ltx_ic_lora_pipeline.py -q`

Expected result:
- Tests pass; `TestDeriveStageRadii` documents linked runtime inputs, not widget defaults.

## Stop Conditions
Stop and report if:
- `derive_stage_radii` no longer exists.
- Fix requires files outside allowed edit files.
- Existing code intentionally maps default `30` to anything other than `(15, 30)` with a documented reason.
- Validation cannot run.

## Required Return Contract
Return status, files inspected/changed, test evidence, and any mismatch between official linked inputs and current code.

---

# Task Packet 2 — Direct IC-LoRA guide conditioning parity

## Status
ready

## User Goal
Move current inpaint closer to official `LTXAddVideoICLoRAGuideAdvanced` guide wiring by feeding the green guide as a direct encoded latent conditioning, not a temp mp4 re-decode path.

## Mode
general-coding

## Relevant Locations
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `LTXIcLoraPipeline.generate_inpaint`
  approximate lines: 430-505
  stable anchor: `# 4. Save green composite as temp video for conditioning`
  reason: current code writes green composite to mp4 and re-decodes through `_encode_video_conditioning`; official node encodes the image/guide tensor directly.
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `_encode_video_conditioning`
  approximate lines: 661-684
  stable anchor: `def _encode_video_conditioning(`
  reason: helper currently path-decodes video; replace or supplement with direct tensor guide encoding using existing `VideoConditionByReferenceLatent`.
  confidence: high
- file: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_core/conditioning/types/reference_video_cond.py`
  symbol: `VideoConditionByReferenceLatent.apply_to`
  approximate lines: 12-91
  stable anchor: `class VideoConditionByReferenceLatent(ConditioningItem):`
  reason: installed dependency already implements IC-LoRA reference-token append semantics equivalent to guide conditioning.
  confidence: high
- file: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_core/conditioning/types/attention_strength_wrapper.py`
  symbol: `ConditioningItemAttentionStrengthWrapper`
  approximate lines: 13-72
  stable anchor: `class ConditioningItemAttentionStrengthWrapper(ConditioningItem):`
  reason: installed dependency covers `LTXAddVideoICLoRAGuideAdvanced` attention strength/mask if needed; official workflow has attention strength `1` and no linked attention mask.
  confidence: high
- file: `backend/tests/test_ltx_ic_lora_pipeline.py`
  symbol: `TestEncodeVideoConditioning`
  approximate lines: 520-590
  stable anchor: `class TestEncodeVideoConditioning:`
  reason: replace path-decode helper test with direct tensor guide-conditioning test.
  confidence: high

## Allowed Edit Files
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/tests/test_ltx_ic_lora_pipeline.py`

## Read-Only Context Files
- Official source URL: `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/iclora.py` (`LTXAddVideoICLoRAGuideAdvanced`, lines ~260-444)
- Official source URL: `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/latents.py` (`LTXVImgToVideoConditionOnly`, lines ~498-575)
- Installed dependency: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_core/conditioning/types/reference_video_cond.py`
- Installed dependency: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_core/conditioning/types/attention_strength_wrapper.py`

## Required Change
- Remove the temp green mp4 conditioning path from `generate_inpaint` for stage1 guide conditioning.
- Encode `green_half` directly inside `self.pipeline.image_conditioner(...)` and append a `VideoConditionByReferenceLatent` with:
  - `latent = enc(green_half)` or tiled equivalent only if the existing encoder requires it,
  - `downscale_factor = 1` (official advanced guide widget `latent_downscale_factor=1`),
  - `strength = conditioning_strength` (official widget strength `1`; preserve existing app parameter).
- Keep ordinary image conditionings from `combined_image_conditionings(...)` unchanged and append guide conditioning after them.
- If adding attention support is a small call to installed `ConditioningItemAttentionStrengthWrapper`, keep default behavior equivalent to official workflow: attention strength `1`, no attention mask. Do not add UI/API params.
- Delete now-unused temp directory/write path only if no other code path uses it. Keep `_write_tensor_video` only if still used elsewhere.
- Add/update a unit test using a fake encoder that asserts the direct guide helper encodes `(1, 3, F, H, W)` green tensor without file I/O and returns a `VideoConditionByReferenceLatent` with expected `strength` and `downscale_factor`.

## Non-Goals
- No custom Comfy node port.
- No broad replacement of `DiffusionStage`.
- No attention-mask UI.
- No sampler/sigma changes in this packet.

## Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ltx_ic_lora_pipeline.py -q`
- `rtk pnpm typecheck:py`

Expected result:
- Unit test proves guide conditioning no longer depends on mp4 temp roundtrip.
- Pyright passes or reports only unrelated pre-existing errors; worker must include evidence.

## Stop Conditions
Stop and report if:
- Direct `enc(green_half)` shape is incompatible with installed `VideoEncoder` and cannot be fixed within this file.
- Worker cannot import `VideoConditionByReferenceLatent` from installed `ltx_core.conditioning`.
- Required change expands into Comfy runtime dependency.
- Validation cannot run.

## Required Return Contract
Return status, exact guide helper/function changed, tests run, and whether temp mp4 path remains with reason.

---

# Task Packet 3 — Stage2 sigmas, seed/noise, sampler honesty, audio preservation

## Status
ready

## User Goal
Match official stage2 runtime inputs where current code diverges: stage2 sigmas `0.7250, 0.4219, 0.0`, seed/noise order matching workflow `stage1=seed+1`, `stage2=seed` for default seed `42`, audio latent carried into stage2, and output encoded with audio instead of cv2 dropping it.

## Mode
general-coding

## Relevant Locations
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `LTXIcLoraPipeline.generate_inpaint`
  approximate lines: 459-607
  stable anchor: `generator = torch.Generator(device=device).manual_seed(seed)`
  reason: current stage1 uses `seed`, stage2 uses `seed + 1`; official workflow uses stage1 `43`, stage2 `42`.
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `stage2_sigmas`
  approximate lines: 580-586
  stable anchor: `stage2_sigmas = STAGE_2_DISTILLED_SIGMAS`
  reason: current four-value constant includes `0.909375`; official stage2 node `5211` uses only `0.7250, 0.4219, 0.0`.
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: `self.pipeline.stage_2(...)`
  approximate lines: 594-624
  stable anchor: `audio=ModalitySpec(`
  reason: current stage2 audio spec lacks `initial_latent=audio_state.latent`; installed official pipeline preserves audio latent across stage2.
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: output encode block
  approximate lines: 641-661
  stable anchor: `# 11. Encode output video`
  reason: current cv2 writer drops audio; use existing `encode_video_output` path with decoded audio.
  confidence: high
- file: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_pipelines/ic_lora.py`
  symbol: `ICLoraPipeline.__call__`
  approximate lines: 248-270
  stable anchor: `initial_latent=audio_state.latent`
  reason: installed pipeline shows stage2 audio preservation pattern.
  confidence: high
- file: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_pipelines/utils/blocks.py`
  symbol: `DiffusionStage.__call__`
  approximate lines: 233-267
  stable anchor: `stepper = EulerDiffusionStep()`
  reason: current installed API exposes Euler stepper default; do not fake Comfy `euler_cfg_pp`/`euler_ancestral_cfg_pp` without an existing compatible implementation.
  confidence: medium
- file: `backend/services/ltx_pipeline_common.py`
  symbol: `encode_video_output`
  approximate lines: 25-38
  stable anchor: `def encode_video_output(`
  reason: existing project helper encodes video plus optional audio.
  confidence: high
- file: `backend/tests/test_ltx_ic_lora_pipeline.py`
  symbol: add `TestInpaintRuntimeParity`
  approximate lines: near `TestDeriveStageRadii` or end of file
  stable anchor: `class TestDeriveStageRadii:`
  reason: add focused checks for stage2 sigmas, seed mapping, and audio-preserving encode path.
  confidence: high

## Allowed Edit Files
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/tests/test_ltx_ic_lora_pipeline.py`

## Read-Only Context Files
- Official workflow URL: `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/2.3/LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json`
- Installed dependency: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_pipelines/ic_lora.py`
- Installed dependency: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_pipelines/utils/blocks.py`
- Installed dependency: `/home/npittas/.local/share/LTXDesktop/python/lib/python3.13/site-packages/ltx_core/components/diffusion_steps.py`
- `backend/services/ltx_pipeline_common.py`

## Required Change
- Add a small explicit stage2 sigma constant in `ltx_ic_lora_pipeline.py`, e.g. values matching official node `5211`: `[0.7250, 0.4219, 0.0]`. Use it for `stage2_sigmas`; do not use `STAGE_2_DISTILLED_SIGMAS` in inpaint stage2.
- Change seed/noise mapping so app `seed=42` reproduces official workflow seeds: stage1 generator/noiser uses `43`, stage2 generator/noiser uses `42`. Implement with minimal helper or inline variables; handle large seeds safely without negative/manual_seed errors.
- Preserve audio into stage2 by passing `initial_latent=audio_state.latent` when `audio_state` is not `None`, matching installed `ICLoraPipeline.__call__`.
- Preserve audio in output by decoding `audio_state_s2.latent` with `self.pipeline.audio_decoder(...)` and calling `encode_video_output(video=video_out_tensor_uint8, audio=decoded_audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=video_chunks_number(...))` instead of cv2-only writing. Ensure `video_out_tensor_uint8` shape is `(F, H, W, 3)` RGB uint8.
- Sampler honesty: check only the installed read-only files listed above for an existing compatible CFG++/ancestral stepper. If none exists, do not implement a custom sampler and do not label the pipeline as exact Comfy sampler parity. Leave `DiffusionStage` default Euler stepper and add a short `ponytail:` comment near stage calls noting sampler parity ceiling and upgrade path.

## Non-Goals
- No custom CFG++ sampler implementation unless an installed compatible stepper already exists.
- No output muxing from original input video audio in this packet; preserve model/audio latent path. If product wants exact source-audio passthrough, request separate decision.
- No frontend/API seed field changes.

## Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ltx_ic_lora_pipeline.py -q`
- `rtk pnpm typecheck:py`

Expected result:
- Tests pass for stage2 sigmas and seed mapping.
- Test or source-level assertion catches reversion to cv2-only output/audio drop.
- Worker reports sampler status honestly: `exact sampler parity not implemented` unless an existing compatible implementation is found.

## Stop Conditions
Stop and report if:
- `encode_video_output` cannot encode the blended tensor shape without a larger refactor.
- `audio_state_s2` can be `None` in normal prompt context and output helper cannot handle it.
- Sampler parity requires writing a new sampler implementation.
- Fix requires frontend/API changes.
- Validation cannot run.

## Required Return Contract
Return status, exact sigma values, seed mapping, audio preservation evidence, sampler finding, and validation output summary.

---

# Task Packet 4 — Visual/manual acceptance gate

## Status
ready

## User Goal
Add concrete visual acceptance checks for inpaint parity: background outside mask preserved, masked region changes, audio stream present after encode, and known remaining sampler gap visible in report.

## Mode
general-coding

## Relevant Locations
- file: `backend/tests/test_ltx_ic_lora_pipeline.py`
  symbol: `TestInpaintBlendOutsideMaskPreservation`
  approximate lines: 607-735
  stable anchor: `class TestInpaintBlendOutsideMaskPreservation:`
  reason: existing fast deterministic visual acceptance tests should use effective runtime stage2 mask and thresholds.
  confidence: high
- file: `backend/tests/test_ltx_ic_lora_pipeline.py`
  symbol: module comments at top
  approximate lines: 14-30
  stable anchor: `Smoke acceptance criterion for inpaint blend outside-mask preservation`
  reason: update manual acceptance criteria to include linked radii, blend dilation controls, and audio stream presence.
  confidence: high

## Allowed Edit Files
- `backend/tests/test_ltx_ic_lora_pipeline.py`

## Read-Only Context Files
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/services/ic_lora_pipeline/official_inpaint.py`

## Required Change
- Update existing acceptance comments/tests so the automated smoke test uses `derive_stage_radii(30)[1]` for final/stage2 mask radius instead of hardcoding a stale value when practical.
- Add a small test or assertion documenting blend controls: stage1 `mask_low_res_dilation=5`, stage2 `mask_low_res_dilation=6` are separate from mask grow radii.
- Add manual acceptance checklist in test comments only (not docs):
  1. Generate fixed inpaint sample with seed `42`.
  2. Export/review triptych `[original_frame | effective_dilated_mask | output_frame]`.
  3. Outside effective stage2 mask mean absolute diff `< 5/255`.
  4. Inside effective stage2 mask mean absolute diff `> 20/255` when prompt should alter masked content.
  5. Output has audio stream when model/audio context produces audio.
  6. Report whether sampler remains default Euler vs Comfy `euler_cfg_pp`/`euler_ancestral_cfg_pp`.
- Do not add a heavy model integration test to pytest.

## Non-Goals
- No golden image assets.
- No frontend E2E.
- No full model download or long GPU test.

## Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ltx_ic_lora_pipeline.py -q`

Expected result:
- Fast unit tests pass and manual acceptance criteria are unambiguous for orchestrator-run visual validation.

## Stop Conditions
Stop and report if:
- Existing acceptance test semantics conflict with official mask polarity (`white=image_a/generated`, `black=image_b/original/green`).
- Validation cannot run.
- Worker needs model files or GPU to finish this packet.

## Required Return Contract
Return status, tests changed, acceptance thresholds, and whether manual live validation still needs orchestrator execution.

---

## Planner Self-Check
- locator evidence sufficient: yes — locator plus official workflow/source confirms paths and node IDs; linked runtime inputs override widget defaults.
- allowed edit files minimal and explicit: yes — packets edit only `ltx_ic_lora_pipeline.py` and/or `test_ltx_ic_lora_pipeline.py`.
- read-only context minimal: yes — official workflow/source URLs and installed dependency files only where needed.
- anchors/lines included: yes — each packet lists path, symbol, approximate lines, stable anchor, reason, confidence.
- validation concrete: yes — targeted backend pytest and pyright where source changes occur; visual gate has explicit manual criteria.
- parallelization decision explicit and safe: yes — sequential split due shared source/test files and dirty worktree.
- non-goals and stop conditions sufficient: yes — prevent UI/API creep, custom sampler invention, heavy GPU tests, and Comfy dependency port.
- reviewer findings addressed, if revision: not applicable — no structured reviewer findings supplied; mask-radius conflict reconciled in plan.

## Reviewer Gate
required
