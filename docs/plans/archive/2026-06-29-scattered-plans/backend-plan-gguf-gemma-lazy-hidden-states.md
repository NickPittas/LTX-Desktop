# Planner Report

## Status
ready

## Rationale
Future task is scoped to one GGUF loader patch and its existing loader test file. Locator evidence is high-confidence: lazy `QParam`/`GgufLinear` already exists, Gemma builder currently disables it, and builder ordering already applies `module_ops` before `load_state_dict`. Minimal fix is to lazily keep only Gemma Linear GGUF weights as `QParam`, eagerly dequantize embeddings/norms/support tensors, then wire existing Gemma `GgufLinear` ModuleOp into the builder.

# Task Packet

## User Goal
Make PyTorch Gemma GGUF hidden-state encoding use lazy `QParam`/`GgufLinear` GPU dequant for Linear weights instead of dense BF16 full-model dequant, while preserving all hidden states consumed by `EmbeddingsProcessor`.

## Mode
general-coding

## Relevant Locations
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `QParam`, `GgufLinear`, `_amend_forward_with_gguf`
  approximate lines: 63-205
  stable anchor: `class QParam`, `class GgufLinear`, `def _amend_forward_with_gguf`
  reason: Existing lazy quantized parameter and Linear module swap path. Reuse; do not add new dequant abstraction.
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `GGUF_GEMMA_DEQUANT_LINEAR_OP`, `GGUF_GEMMA_TEXT_ONLY_OP`
  approximate lines: 214-230
  stable anchor: `GGUF_GEMMA_DEQUANT_LINEAR_OP = ModuleOps(`
  reason: Existing dead Gemma ModuleOp must be included in Gemma builder `module_ops`.
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `GgufStateDictLoader.__init__`, `GgufStateDictLoader.load`
  approximate lines: 490-610
  stable anchor: `lazy_quantized: bool = True`, `if _is_quantized_type(tensor.tensor_type) and self._lazy_quantized:`
  reason: Add optional name whitelist so lazy wrapping applies only to Gemma Linear raw GGUF names; non-Linear quantized tensors remain eager.
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `GgufGemmaSDOps`
  approximate lines: 650-690
  stable anchor: `_layer_suffixes = {`
  reason: Provides raw GGUF Gemma key suffixes; Linear suffix whitelist should match q/k/v/o and ffn gate/up/down weights, not norms or token embeddings.
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `install_gguf_prompt_encoder_patch.patched_init`
  approximate lines: 740-805
  stable anchor: `self._text_encoder_builder = Builder(`
  reason: Gemma builder currently passes `lazy_quantized=False` and omits `GGUF_GEMMA_DEQUANT_LINEAR_OP`; change here only.
  confidence: high
- file: `backend/services/patches/gguf_torch_dequant.py`
  symbol: `dequantize_gguf_tensor_torch`, `_DEQUANTIZE_DISPATCH`
  approximate lines: 1-140
  stable anchor: `_DEQUANTIZE_DISPATCH: dict[object, object] = {`
  reason: Already supports Q4_K/Q5_K/Q6_K GPU dequant used by `QParam.dequant`; read-only unless actual checkpoint uses unsupported qtype.
  confidence: medium
- file: `backend/tests/test_gguf_loader.py`
  symbol: loader/QParam tests, PromptEncoder GGUF patch tests
  approximate lines: 350-560 and 880-1015
  stable anchor: `test_gguf_loader_load_wraps_quantized_tensor_as_qparam`, `test_patched_init_passes_tokenizer_root_not_gguf_file`
  reason: Add focused tests beside existing GGUF loader and Gemma builder tests.
  confidence: high
- file: `backend/.venv/.../ltx_core/loader/single_gpu_model_builder.py`
  symbol: `SingleGPUModelBuilder.build`
  approximate lines: 74-99
  stable anchor: `meta_model = self.meta_model(config, self.module_ops)` before `load_state_dict(..., assign=True)`
  reason: Locator verified ModuleOps run before state dict assignment; no edit needed.
  confidence: high

## Allowed Edit Files
- `backend/services/patches/gguf_loader_fix.py`
- `backend/tests/test_gguf_loader.py`

## Read-Only Context Files
- `backend/locator_report_gemma_gguf_hidden_states.md`
- `backend/services/patches/gguf_torch_dequant.py`

## Required Change
1. Before editing, run `rtk git status --short backend/services/patches/gguf_loader_fix.py backend/tests/test_gguf_loader.py`. This is a future packet because these files may be under active llama.cpp timing-test work; stop if either file has unexpected edits or conflict markers.
2. In `backend/services/patches/gguf_loader_fix.py`, add the smallest loader whitelist for Gemma Linear raw GGUF tensor names:
   - Linear suffixes only: `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`, `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` under `blk.<n>.`.
   - Exclude `token_embd.weight`, `output_norm.weight`, `*norm.weight`, biases, tokenizer/support tensors, and anything unmapped.
   - Keep this as a tiny private helper/predicate near GGUF helpers or Gemma SDOps; no new class hierarchy.
3. Extend `GgufStateDictLoader` minimally with an optional lazy filter, defaulting to current behavior for existing callers:
   - `lazy_quantized_filter: Callable[[str], bool] | None = None` (or equivalent smallest typed callable).
   - In `load()`, wrap as `QParam` only when tensor is quantized, `self._lazy_quantized` is true, and filter is absent or returns true for original GGUF tensor name.
   - If filter returns false, follow existing eager dequant path so norm/embedding/support tensors are normal BF16 tensors before `GgufGemmaSDOps.apply_to_key_value()` subtracts 1 for norms.
4. In Gemma `PromptEncoder` builder inside `install_gguf_prompt_encoder_patch`:
   - Change `GgufStateDictLoader(require_transformer_config=False, lazy_quantized=False)` to lazy enabled with the Gemma Linear-name filter.
   - Add existing `GGUF_GEMMA_DEQUANT_LINEAR_OP` to the Gemma `module_ops` tuple after `GGUF_GEMMA_TEXT_ONLY_OP` and before `_llama_cpp_prompt_enhancer_op(...)` / upstream `module_ops`.
   - Update the stale dense-path `ponytail` comment to state only Gemma Linear weights are lazy; embeddings/norms stay eager because they are not `GgufLinear`.
5. Add tests in `backend/tests/test_gguf_loader.py`:
   - Loader behavior test: write a tiny GGUF with one quantized Gemma Linear raw name (`blk.0.attn_q.weight`) and one quantized Gemma non-Linear raw name (`blk.0.attn_norm.weight` or `token_embd.weight`); load with `GgufGemmaSDOps()` and the Gemma lazy filter; assert mapped Linear key is `QParam`, mapped norm/embedding key is plain `torch.Tensor` BF16, and no QParam placeholder reaches non-Linear keys.
   - Builder wiring test: after `install_gguf_prompt_encoder_patch()`, initialize fake `PromptEncoder` as existing tests do; assert `_text_encoder_builder.model_loader` has lazy quant enabled with the filter, and `_text_encoder_builder.module_ops` contains `GGUF_GEMMA_DEQUANT_LINEAR_OP.name`.
   - Keep tests CPU-only. CUDA parity already belongs to `gguf_torch_dequant.py`; do not require a real Gemma checkpoint in unit tests.
6. Reviewer notes to check:
   - Verify filter uses original GGUF names before `GgufGemmaSDOps` remapping.
   - Verify default loader behavior is unchanged for native LTX transformer GGUF paths.
   - Verify `GgufGemmaSDOps.apply_to_key_value()` never receives norm `QParam` values.
   - Verify no llama.cpp hidden-state shortcut is introduced; PyTorch Gemma path remains source of all hidden states for `EmbeddingsProcessor`.

## Non-Goals
- Do not modify `backend/services/patches/gguf_torch_dequant.py` unless validation proves target Gemma checkpoint uses an unsupported quant type.
- Do not change installed `ltx_core` package files.
- Do not replace `EmbeddingsProcessor` or its hidden-state contract.
- Do not add llama.cpp per-layer hidden-state support; locator says C API does not expose it.
- Do not add broad loader abstractions, new dependencies, or frontend changes.
- Do not touch llama.cpp timing-test work in progress.

## Validation
Commands:
- `rtk git status --short backend/services/patches/gguf_loader_fix.py backend/tests/test_gguf_loader.py`
- `rtk pnpm backend:test -- tests/test_gguf_loader.py`
- `rtk pnpm typecheck:py`

Expected result:
- Status command shows no unexpected pre-existing edits/conflicts before work starts.
- GGUF loader tests pass, including new Gemma lazy Linear whitelist and builder wiring tests.
- Pyright passes.
- Optional manual smoke, only if a real Gemma GGUF checkpoint is available: load prompt encoder and confirm Linear weights are `QParam` with `_raw` on CPU, non-Linear norm/embed weights are BF16 tensors, and encode returns full hidden states accepted by `EmbeddingsProcessor`.

## Stop Conditions
Stop and report if:
- target symbol is missing
- required fix exceeds allowed files
- validation cannot run
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet
- `backend/services/patches/gguf_loader_fix.py` or `backend/tests/test_gguf_loader.py` contains conflict markers or unexpected active-worker edits
- Gemma raw GGUF names differ from locator evidence enough that the Linear whitelist cannot be written safely
- `GgufGemmaSDOps.apply_to_key_value()` would receive any non-Linear `QParam`
- actual required quant type is unsupported by existing torch dequant and would require editing `gguf_torch_dequant.py`

## Planner Self-Check
- locator evidence sufficient: yes — high-confidence locator report plus targeted reads confirm symbols, anchors, builder wiring, and tests.
- allowed edit files minimal and explicit: yes — one source file plus one existing test file.
- read-only context minimal: yes — locator report and torch dequant support file only.
- anchors/lines included: yes — relevant locations include path, symbol, approximate lines, stable anchor, reason, confidence.
- validation concrete: yes — repo test/typecheck commands and optional checkpoint smoke are specified.
- parallelization decision explicit and safe: yes — single future task; not parallel-safe with llama.cpp timing-test worker because same files may be edited.
- non-goals and stop conditions sufficient: yes — prevent llama.cpp hidden-state detour, broad dequant changes, installed package edits, and active-worker conflicts.
- reviewer findings addressed, if revision: not applicable — no previous reviewer findings supplied.

## Required Return Contract
Return only a task-focused summary. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues. Include status, files inspected/changed, validation evidence, blockers, and task-specific risks.
