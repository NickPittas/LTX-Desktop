# Planner Report

## Status
split-required

## Why Split / Parallelize
Four tiny workers match the requested boundaries, but implementation is sequential: Worker A creates the torch dequant helper, Worker B wires `QParam.dequant`, Worker C adds tests against that behavior, and Worker D validates. Shared validation state is fine; source edits are intentionally split so each worker has one concern.

## Interference Check
- parallel safe: no
- shared files or generated outputs: Worker B and C depend on Worker A helper API; Worker C tests depend on Worker B routing for `QParam.dequant` CUDA behavior.
- shared validation state: backend pytest/pyright only; no generated outputs expected.
- worktree isolation required: no
- rationale: sequential order avoids API guesswork and failing tests before implementation exists.

## Proposed Task Sequence Or Parallel Batch
1. Task name: Worker A — torch GGUF dequant helper
   - purpose: add torch-device dequant for only the qtypes used by the timed-out GGUF smoke: `Q4_K` and `Q6_K`.
   - allowed files:
     - `backend/services/patches/gguf_torch_dequant.py`
   - validation:
     - `python -m py_compile backend/services/patches/gguf_torch_dequant.py`
   - can run in parallel with: none
2. Task name: Worker B — wire QParam GPU path
   - purpose: call the torch helper from `QParam.dequant` for CUDA tensors; keep existing gguf/numpy path as CPU and unsupported-qtype fallback.
   - allowed files:
     - `backend/services/patches/gguf_loader_fix.py`
   - validation:
     - `pnpm backend:test -- tests/test_gguf_loader.py`
   - can run in parallel with: none
3. Task name: Worker C — focused tests
   - purpose: add small coverage for `Q4_K`/`Q6_K` torch helper parity and optional CUDA `QParam.dequant` routing.
   - allowed files:
     - `backend/tests/test_gguf_loader.py`
   - validation:
     - `pnpm backend:test -- tests/test_gguf_loader.py`
   - can run in parallel with: none
4. Task name: Worker D/reviewer — validate + smoke
   - purpose: review diff, run backend checks, rerun GGUF smoke that timed out.
   - allowed files: none
   - validation:
     - `pnpm backend:test -- tests/test_gguf_loader.py`
     - `pnpm typecheck:py`
     - rerun the exact prior GGUF smoke command with a timeout high enough to observe stage_1 progress beyond the prior 6/8 step timeout
   - can run in parallel with: none

## Task Packets

# Task Packet — Worker A

## User Goal
Add a torch/GPU dequant helper for exact used GGUF qtypes only, without full-transformer dequant caching.

## Mode
general-coding

## Relevant Locations
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `QParam.dequant`
  approximate lines: 90-101
  stable anchor: `def dequant(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:`
  reason: current CPU numpy dequant path to preserve as fallback.
  confidence: high
- file: `/home/npittas/.local/lib/python3.14/site-packages/gguf/quants.py`
  symbol: `Q4_K.dequantize_blocks`
  approximate lines: 475-522
  stable anchor: `class Q4_K(__Quant, qtype=GGMLQuantizationType.Q4_K):`
  reason: reference math/layout for torch helper.
  confidence: high
- file: `/home/npittas/.local/lib/python3.14/site-packages/gguf/quants.py`
  symbol: `Q6_K.dequantize_blocks`
  approximate lines: 552-572
  stable anchor: `class Q6_K(__Quant, qtype=GGMLQuantizationType.Q6_K):`
  reason: reference math/layout for torch helper.
  confidence: high
- file: `/home/npittas/.local/lib/python3.14/site-packages/gguf/quants.py`
  symbol: `quant_shape_from_byte_shape`
  approximate lines: 21-25
  stable anchor: `def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:`
  reason: output shape must match gguf package behavior.
  confidence: high

## Allowed Edit Files
- `backend/services/patches/gguf_torch_dequant.py`

## Read-Only Context Files
- `backend/services/patches/gguf_loader_fix.py`
- `/home/npittas/.local/lib/python3.14/site-packages/gguf/quants.py`

## Required Change
Create a tiny helper module exporting one function, e.g. `dequantize_gguf_tensor_torch(raw: torch.Tensor, tensor_type: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None`. It must:
- support only `gguf.GGMLQuantizationType.Q4_K` and `gguf.GGMLQuantizationType.Q6_K`;
- return `None` for every other qtype so Worker B can fall back to existing numpy path;
- move only the current quantized raw tensor/block to the target device and dequantize there using torch ops;
- match `gguf.quants.dequantize` output shape and values for Q4_K/Q6_K;
- cast final output to `dtype` if floating, otherwise `torch.float32`.

Use the installed `gguf` package constants for qtype detection and shape/block sizes. Do not cache dequantized tensors or add a global registry/config.

## Non-Goals
- No full transformer dequantization cache.
- No quantized matmul kernel.
- No support for qtypes other than Q4_K and Q6_K.
- No loader/model architecture changes.
- No new dependencies.

## Validation
Commands:
- `python -m py_compile backend/services/patches/gguf_torch_dequant.py`

Expected result:
- helper module compiles.

## Stop Conditions
Stop and report if:
- `gguf.GGMLQuantizationType.Q4_K` or `Q6_K` constants are missing
- torch ops cannot reproduce gguf numpy parity for Q4_K/Q6_K without broad qtype support
- implementation needs full-weight caching or persistent GPU storage
- required fix exceeds allowed files
- validation cannot run
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet

## Required Return Contract
Return only a task-focused summary. Include status, files inspected/changed, validation evidence, blockers, and task-specific risks. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues.

# Task Packet — Worker B

## User Goal
Wire `QParam.dequant` to use the GPU torch path while preserving the existing gguf numpy CPU fallback.

## Mode
general-coding

## Relevant Locations
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `QParam.dequant`
  approximate lines: 90-101
  stable anchor: `def dequant(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:`
  reason: exact method to route CUDA dequant to Worker A helper.
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `GgufLinear.forward`
  approximate lines: 137-148
  stable anchor: `def forward(self, input: torch.Tensor) -> torch.Tensor:`
  reason: caller passes input device/dtype into `QParam.dequant`; avoid touching caller unless needed.
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `GgufStateDictLoader.load`
  approximate lines: 262-273
  stable anchor: `tensor_t: torch.Tensor = QParam(tensor.data, tensor.tensor_type, name=tensor.name)`
  reason: confirms quantized weights remain lazy raw QParams; do not materialize at load.
  confidence: high
- file: `backend/services/patches/gguf_torch_dequant.py`
  symbol: `dequantize_gguf_tensor_torch`
  approximate lines: whole file
  stable anchor: Worker A exported helper function
  reason: helper to call for CUDA dequant.
  confidence: medium until Worker A completes

## Allowed Edit Files
- `backend/services/patches/gguf_loader_fix.py`

## Read-Only Context Files
- `backend/services/patches/gguf_torch_dequant.py`
- `backend/tests/test_gguf_loader.py`

## Required Change
In `QParam.dequant`, before the current `gguf.quants.dequantize(self._raw.numpy(), self._tensor_type)` path, try Worker A's helper only when `device.type == "cuda"`. If helper returns a tensor, return it. If helper returns `None`, continue to the existing gguf/numpy fallback unchanged. Preserve current error message behavior for unsupported qtypes from the fallback.

Keep `GgufLinear.forward` unchanged unless absolutely necessary. Do not move `_raw` permanently to GPU; dequantized result must remain local to the forward call and be discarded after `linear` completes.

## Non-Goals
- No full dequant cache.
- No CPU torch path routing from `QParam.dequant`; CPU stays existing gguf numpy path.
- No qtype expansion beyond Worker A helper.
- No changes to loader, model builder, or pipeline selection.
- No new dependencies.

## Validation
Commands:
- `pnpm backend:test -- tests/test_gguf_loader.py`

Expected result:
- existing GGUF loader tests pass; tests may not yet cover new CUDA path until Worker C.

## Stop Conditions
Stop and report if:
- Worker A helper file or export is missing
- `QParam.dequant` anchor is missing or signature changed
- routing requires editing outside allowed file
- unsupported qtypes would stop falling back to existing gguf/numpy behavior
- implementation would cache full dequantized tensors or keep dequantized weights on GPU between forwards
- validation cannot run
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet

## Required Return Contract
Return only a task-focused summary. Include status, files inspected/changed, validation evidence, blockers, and task-specific risks. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues.

# Task Packet — Worker C

## User Goal
Add small tests for the torch GGUF dequant helper and `QParam.dequant` GPU route.

## Mode
general-coding

## Relevant Locations
- file: `backend/tests/test_gguf_loader.py`
  symbol: imports from `services.patches.gguf_loader_fix`
  approximate lines: 13-22
  stable anchor: `from services.patches.gguf_loader_fix import (`
  reason: add helper import from Worker A module if needed.
  confidence: high
- file: `backend/tests/test_gguf_loader.py`
  symbol: `_write_quantized_gguf`
  approximate lines: 247-253
  stable anchor: `def _write_quantized_gguf(path: str, tensor_name: str = "x.weight") -> np.ndarray:`
  reason: existing quantized GGUF test helper; can mirror pattern for Q4_K/Q6_K raw blocks.
  confidence: high
- file: `backend/tests/test_gguf_loader.py`
  symbol: `test_gguf_loader_load_wraps_quantized_tensor_as_qparam`
  approximate lines: 366-386
  stable anchor: `def test_gguf_loader_load_wraps_quantized_tensor_as_qparam(tmp_path: Path) -> None:`
  reason: existing reference comparison to `gguf.quants.dequantize`.
  confidence: high
- file: `backend/tests/test_gguf_loader.py`
  symbol: `test_qparam_raw_survives_module_to_dtype_bfloat16_and_optional_cuda`
  approximate lines: 314-339
  stable anchor: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")`
  reason: existing optional CUDA test pattern.
  confidence: high
- file: `backend/services/patches/gguf_torch_dequant.py`
  symbol: `dequantize_gguf_tensor_torch`
  approximate lines: whole file
  stable anchor: Worker A exported helper function
  reason: direct parity tests.
  confidence: medium until Worker A completes

## Allowed Edit Files
- `backend/tests/test_gguf_loader.py`

## Read-Only Context Files
- `backend/services/patches/gguf_torch_dequant.py`
- `backend/services/patches/gguf_loader_fix.py`
- `/home/npittas/.local/lib/python3.14/site-packages/gguf/quants.py`

## Required Change
Add focused tests only:
- deterministic parity for `dequantize_gguf_tensor_torch` on `Q4_K` and `Q6_K` by quantizing small float32 matrices with `gguf.quants.quantize`, dequantizing with the torch helper on CPU (for CI availability), and comparing to `gguf.quants.dequantize` with tight tolerance;
- unsupported qtype returns `None` (use one existing qtype not supported by helper, e.g. `Q4_0`, if available in installed gguf);
- optional CUDA test, skipped when CUDA is unavailable, proving `QParam.dequant(device=cuda, dtype=torch.bfloat16)` returns CUDA bf16 output matching numpy reference for Q4_K/Q6_K.

Keep tests in the existing file. Use existing no-mock style; no `unittest.mock`.

## Non-Goals
- No broad smoke/performance tests here.
- No new test framework or fixtures.
- No changes to production code.
- No assertions about speed, only route/output correctness.

## Validation
Commands:
- `pnpm backend:test -- tests/test_gguf_loader.py`

Expected result:
- GGUF loader test file passes on CPU-only machines; CUDA-specific test skips when CUDA unavailable.

## Stop Conditions
Stop and report if:
- Worker A helper export is missing or has a different API
- Worker B did not preserve `QParam.dequant` signature
- installed `gguf.quants.quantize` cannot produce Q4_K/Q6_K samples for small deterministic tensors
- tests require mocks or external GGUF model files
- validation cannot run
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet

## Required Return Contract
Return only a task-focused summary. Include status, files inspected/changed, validation evidence, blockers, and task-specific risks. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues.

# Task Packet — Worker D/reviewer

## User Goal
Validate implementation and rerun the GGUF smoke that previously timed out.

## Mode
general-coding

## Relevant Locations
- file: `backend/services/patches/gguf_torch_dequant.py`
  symbol: `dequantize_gguf_tensor_torch`
  approximate lines: whole file
  stable anchor: Worker A exported helper function
  reason: review helper qtype scope and no caching.
  confidence: medium until Worker A completes
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `QParam.dequant`
  approximate lines: 90-101
  stable anchor: `def dequant(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:`
  reason: review GPU route and fallback behavior.
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `GgufLinear.forward`
  approximate lines: 137-148
  stable anchor: `def forward(self, input: torch.Tensor) -> torch.Tensor:`
  reason: verify dequant remains per-forward/local, not cached.
  confidence: high
- file: `backend/tests/test_gguf_loader.py`
  symbol: GGUF tests
  approximate lines: 1-520
  stable anchor: `# Lazy dequant: QParam / GgufLinear / module op / load wrapping`
  reason: run/review focused coverage.
  confidence: high

## Allowed Edit Files
- none

## Read-Only Context Files
- `backend/services/patches/gguf_torch_dequant.py`
- `backend/services/patches/gguf_loader_fix.py`
- `backend/tests/test_gguf_loader.py`

## Required Change
Review only. Confirm:
- helper supports only Q4_K/Q6_K and returns `None` for other qtypes;
- `QParam.dequant` uses helper only for CUDA and preserves existing CPU gguf/numpy fallback;
- no full transformer or cross-forward dequant cache exists;
- no new dependency was added;
- tests are small and do not require external model files.

Then run validation.

## Non-Goals
- No implementation edits.
- No performance tuning beyond validating smoke behavior.
- No qtype expansion.
- No unrelated backend/frontend checks unless requested by orchestrator.

## Validation
Commands:
- `pnpm backend:test -- tests/test_gguf_loader.py`
- `pnpm typecheck:py`
- rerun the exact prior GGUF smoke command used for the timeout; timeout must be high enough to observe whether stage_1 advances beyond prior 6/8 at ~40.9s/step

Expected result:
- tests pass;
- pyright passes;
- smoke no longer spends forward time on repeated CPU numpy dequant bottleneck, or at minimum stage_1 progresses beyond the prior timeout point without full BF16 transformer caching/OOM.

## Stop Conditions
Stop and report if:
- implementation changed files outside worker packets
- helper supports broad/speculative qtypes
- full dequantized transformer caching appears anywhere
- CPU fallback was removed or changed for unsupported qtypes
- validation cannot run or prior smoke command is unavailable
- smoke OOMs near full BF16 transformer footprint (39.12GiB vs RTX 5090 31.84GiB)
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet

## Required Return Contract
Return only a task-focused summary. Include status, files inspected/changed, validation evidence, smoke timing evidence, blockers, and task-specific risks. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues.

## Planner Self-Check
- locator evidence sufficient: yes — exact current anchors found for `QParam`, `GgufLinear.forward`, existing tests, and installed gguf reference qtype implementations.
- allowed edit files minimal and explicit: yes — Worker A new helper file only, Worker B one production file, Worker C one test file, Worker D none.
- read-only context minimal: yes — only current loader/test files plus installed gguf reference source.
- anchors/lines included: yes — each packet lists path, symbol/anchor, approximate lines, reason, confidence.
- validation concrete: yes — unit tests and pyright concrete; smoke command intentionally references orchestrator's prior exact command because planner was not given the command string.
- parallelization decision explicit and safe: yes — split-required/sequential due helper API and test dependencies.
- non-goals and stop conditions sufficient: yes — qtype scope, no cache, no new deps, fallback preservation, no broad edits.
- reviewer findings addressed, if revision: not applicable — no reviewer findings supplied.
