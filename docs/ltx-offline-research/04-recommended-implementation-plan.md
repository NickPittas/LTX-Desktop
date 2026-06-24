# Recommended Implementation Plan — LTX-2 Offline Standalone App

> **Goal:** Build a single, portable application that can run every official LTX-2 pipeline, apply any LoRA, and fit on GPUs with less than 24 GB VRAM by leveraging GGUF and other low-bit quantization techniques.

## Phase 0 — Foundation & Decision Gates (1–2 days)

Before writing code, lock down the decisions that shape the whole design.

| Decision | Recommended default | Why |
|----------|---------------------|-----|
| Base engine | `ltx-core` + `ltx-pipelines` | Only way to get all 10 official pipelines and their exact inference logic. |
| App entry | Single Python CLI first, Gradio GUI second | Easier to test, automate, and profile. |
| Primary low-VRAM mechanism | GGUF Q4_K_M / Q5_K transformer + `bitsandbytes` 8-bit Gemma | Best community-proven path for 16–24 GB. |
| Secondary mechanism | `optimum-quanto` INT8/INT4 + block streaming | Fallback if GGUF model is unavailable or quality is poor. |
| LoRA strategy | Dual path: fuse for safetensors, patch for GGUF | Covers official, ComfyUI, and Q8-aware LoRAs. |
| Platform target | CUDA 12.x, Python 3.11+ | Match official repo's `uv`/`torch` stack. |

**Gates:**

1. Confirm target VRAM buckets: 12 GB, 16 GB, 20 GB, 24 GB.
2. Confirm whether the app must run **fully offline** or may download models on first run.
3. Confirm UI preference (CLI-only MVP vs. GUI from day one).

## Phase 1 — Minimal Viable Quantized Inference (3–5 days)

Deliver: a script that loads an LTX-2 GGUF transformer and runs the simplest official pipeline (`ti2vid_one_stage` or `distilled`).

### 1.1 Set up project skeleton

```
ltx_offline/
├── pyproject.toml
├── README.md
├── src/ltx_offline/
│   ├── __init__.py
│   ├── cli.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── gguf_transformer.py
│   │   └── text_encoder.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── runner.py
│   └── utils/
│       ├── __init__.py
│       └── memory.py
└── configs/
    └── low_vram_16gb.yaml
```

### 1.2 Implement GGUF transformer loader

- Use the `gguf` Python package to read tensor metadata and quantized weights.
- Adapt dequantization logic from `city96/ComfyUI-GGUF` (`dequant.py`, `ops.py`) or `diffusers` GGUF support.
- Map GGUF tensor names to `ltx_core.model.transformer` state-dict keys.
- Return a state dict that can be loaded into `ltx-core`'s transformer via the existing load path, or build a custom `QuantizationPolicy` with a GGUF-aware `FuseRule`.

### 1.3 Implement 8-bit Gemma loader

- Port `ltx-trainer/gemma_8bit.py` into the app.
- Add CLI flags for `gemma-precision`: `bf16`, `fp8`, `int8-bnb`.

### 1.4 Run first pipeline

- Wrap `TI2VidOneStagePipeline` or `DistilledPipeline` so it accepts a pre-built transformer.
- Add `--transformer-path`, `--gemma-root`, `--prompt`, `--output-path`.
- Verify output video on a 16 GB or 24 GB GPU.

### 1.5 Profile memory

- Record peak VRAM for:
  - `gguf-q4_k_m` transformer
  - `gguf-q5_k` transformer
  - `fp8-cast` transformer (baseline)
  - With/without 8-bit Gemma
  - With/without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Phase 2 — Unified Pipeline Dispatcher (2–3 days)

Deliver: one CLI command that can run any official pipeline.

### 2.1 Pipeline registry

Create a registry that maps a pipeline name to:

- The `ltx_pipelines` class.
- Its argument parser factory.
- Required vs. optional checkpoints (base, distilled, upsampler, etc.).

Example:

```python
PIPELINES = {
    "ti2vid_two_stages": (TI2VidTwoStagesPipeline, default_2_stage_arg_parser),
    "ic_lora": (ICLoraPipeline, ic_lora_arg_parser),
    "a2vid": (A2VidPipelineTwoStage, default_2_stage_arg_parser),
    # ... etc
}
```

### 2.2 Common argument normalization

- `--pipeline` selector.
- `--quantization` unified enum: `bf16`, `fp8-cast`, `fp8-scaled-mm`, `quanto-int8`, `quanto-int4`, `gguf-q8`, `gguf-q4`, etc.
- `--offload` mapped to `OffloadMode`.
- `--lora` repeatable path/strength pairs.
- `--gemma-precision`.

### 2.3 Checkpoint auto-detection

- Mirror `detect_checkpoint_path` logic but for all pipeline types.
- Provide clear error messages when required checkpoints are missing.

## Phase 3 — Universal LoRA Support (3–4 days)

Deliver: any LoRA can be attached regardless of base model format.

### 3.1 Safetensors LoRA path

- Use official `SingleGPUModelBuilder.lora(...)` + `fuse_loras`.
- Support `LTXV_LORA_COMFY_RENAMING_MAP` for ComfyUI LoRAs.

### 3.2 GGUF / runtime-patch LoRA path

- Implement a patch-based LoRA applier modeled on `ComfyUI-GGUF.GGUFModelPatcher`:
  - Keep base weights quantized.
  - Apply `W + alpha * B @ A` on dequantized weights at forward time.
- Cache dequantized + patched weights where VRAM allows.

### 3.3 Q8-aware LoRA path

- Detect whether a LoRA was trained for / matches the Q8 kernel layout.
- If so, apply the Hadamard transform from `ComfyUI-LTXVideo.q8_nodes.py`.
- Else, warn and fall back to standard patching.

### 3.4 Validation

- Test with official LoRAs, ComfyUI LoRAs, and at least one IC-LoRA.
- Verify no key-name mismatches and no OOM during fusion.

## Phase 4 — Advanced Quantization & Streaming (3–5 days)

Deliver: the app can trade quality for VRAM across a wide range of hardware.

### 4.1 `optimum-quanto` integration

- Port/adapt `ltx-trainer/quantization.py` for inference:
  - Load base transformer in bf16.
  - Quantize block-wise on GPU.
  - Freeze weights and run pipeline.
- Add presets: `quanto-int8`, `quanto-int4`, `quanto-fp8`.

### 4.2 Offloading matrix

Implement guardrails:

| Base format | Offload allowed? | Notes |
|-------------|------------------|-------|
| `bf16` | `none`, `cpu`, `disk` | All modes valid. |
| `fp8-cast` / `fp8-scaled-mm` | `none` only | Block streaming disables FP8 in official code. |
| `quanto-*` | `none` preferred; `cpu` possible if patched | Streaming quantized blocks needs testing. |
| `gguf-*` | `none` preferred; `cpu` possible if patched | Same as above. |

### 4.3 VAE tiling + attention optimization

- Expose `--spatial-tile` and `--temporal-tile` knobs.
- Auto-pick defaults based on VRAM tier.
- Optional xFormers / FlashAttention extras.

### 4.4 Low-VRAM profiles

Create YAML presets:

```yaml
# configs/low_vram_12gb.yaml
pipeline: ti2vid_one_stage
quantization: gguf-q4_k_m
gemma_precision: int8-bnb
offload: none
spatial_tile: 64
num_inference_steps: 20
height: 256
width: 384
num_frames: 25
```

Include profiles for 12, 16, 20, and 24 GB.

## Phase 5 — User Interface & Packaging (2–4 days)

Deliver: non-technical users can run the app.

### 5.1 CLI polish

- Subcommands: `generate`, `convert`, `info`, `download`.
- Rich progress bars, VRAM warnings, estimated time.
- `--dry-run` to validate config without inference.

### 5.2 Gradio GUI (optional)

- Pipeline selector dropdown.
- Model/LoRA file pickers.
- Resolution, steps, seed sliders.
- VRAM gauge / recommendation panel.

### 5.3 Packaging

- `uv` lock file for reproducible installs.
- Optional one-folder bundle with embedded Python.
- Docker image for Linux users.

## Phase 6 — Validation & Hardening (ongoing)

- Golden tests for each pipeline at 512x320 / 25 frames.
- Quality regression suite: compare GGUF Q4 vs. fp8 vs. bf16 on a fixed prompt.
- LoRA compatibility matrix (official, ComfyUI, IC-LoRA, HDR, distilled, Q8).
- Memory benchmark matrix across 12/16/20/24 GB GPUs.

## Suggested First Sprint

> Build the smallest possible end-to-end proof of concept that proves GGUF + all-pipeline access is achievable.

**Week 1 deliverables:**

1. Project skeleton with `pyproject.toml` depending on `ltx-core` + `ltx-pipelines`.
2. `gguf_transformer.py` that can load a community LTX-2.3 GGUF into an `ltx-core` transformer.
3. `cli.py` with `--pipeline distilled --transformer-path *.gguf --prompt ... --output-path ...`.
4. A working 5-second video on a 16 GB GPU.
5. Memory profile numbers in `research/profiles/`.

**Week 2+ deliverables:**

- Expand to all 10 pipelines via dispatcher.
- Add LoRA fusion and patching.
- Add quanto and offloading.
- Add GUI.

## Risk Register

| Risk | Mitigation |
|------|------------|
| GGUF key mapping drifts with new LTX-2 releases | Centralize mapping, add unit tests, pin `ltx-core` version. |
| LoRA quality degrades under Q4 GGUF | Test per LoRA; allow `quanto-int8` fallback. |
| `fp8-cast` + offloading incompatibility | Document limitation; prefer GGUF for low VRAM. |
| `q8_kernels` hard to install | Make optional; use quanto/gguf as primary. |
| Diffusers GGUF becomes the "standard" | Keep Diffusers loader as a fallback option. |

## Resources to Keep Handy

- Official repo: `https://github.com/Lightricks/LTX-2`
- ComfyUI nodes: `https://github.com/Lightricks/ComfyUI-LTXVideo`
- GGUF loader: `https://github.com/city96/ComfyUI-GGUF`
- Q8 kernels: `https://github.com/Lightricks/LTX-Video-Q8-Kernels`
- Diffusers GGUF docs: `https://huggingface.co/docs/diffusers/quantization/gguf`
- Community GGUF weights: `Viper-AI-Vaunt/LTX-2.3-DEV-GGUF`, `QuantStack/LTX-2.3-GGUF`
