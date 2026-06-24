# LTX-2 Architecture & Official Stack Findings

## 1. Model Overview

**LTX-2** is a 22B-parameter asymmetric dual-stream diffusion transformer (DiT) that jointly models video and audio in a single forward pass. It is the successor to the earlier 2B-parameter LTX-Video.

| Component | Details |
|-----------|---------|
| Total params | ~22B (dev) / ~22B distilled |
| Video stream | ~14B |
| Audio stream | ~5B |
| Text encoder | Gemma 3 12B-it (also supports QAT Q4_0 unquantized variant) |
| VAE | Video VAE + Audio VAE + neural vocoder |
| Upsampler | Spatial upsampler (x1.5 / x2) for two-stage pipelines |
| Checkpoints | `ltx-2.3-22b-dev.safetensors`, `ltx-2.3-22b-distilled-1.1.safetensors`, plus spatial upscalers |

Source: `packages/ltx-core/README.md`, `packages/ltx-pipelines/README.md`.

## 2. Repository Layout

The official repo (`https://github.com/Lightricks/LTX-2`) is a Python monorepo managed with `uv`:

```
LTX-2/
├── packages/
│   ├── ltx-core/      # Model defs, schedulers, guidance, loaders, quantization, block streaming
│   ├── ltx-pipelines/ # Ready-made inference pipelines
│   └── ltx-trainer/   # LoRA / full fine-tuning / IC-LoRA training
├── pyproject.toml
└── uv.lock
```

### 2.1 `ltx-core` key modules

| Module | Role |
|--------|------|
| `ltx_core.model.transformer` | LTX-2 DiT (video + audio streams) |
| `ltx_core.model.video_vae` | Video VAE encode/decode |
| `ltx_core.model.audio_vae` | Audio VAE + vocoder |
| `ltx_core.model.upsampler` | Spatial latent upsampler |
| `ltx_core.loader` | Safetensors loading, LoRA fusion, model builders |
| `ltx_core.quantization` | FP8 quantization policies |
| `ltx_core.block_streaming` | RAM/disk weight streaming for low VRAM |
| `ltx_core.components` | Schedulers, guiders (CFG, STG, APG), noisers, patchifiers |
| `ltx_core.text_encoders.gemma` | Gemma tokenizer, feature extractor, text encoder |

### 2.2 `ltx-pipelines` available pipelines

All pipelines are runnable as `python -m ltx_pipelines.<module>`:

| Pipeline | Stages | Conditioning | Notes |
|----------|--------|--------------|-------|
| `ti2vid_two_stages` | 2 | Image | Production default |
| `ti2vid_two_stages_hq` | 2 | Image | Higher-quality sampler |
| `ti2vid_one_stage` | 1 | Image | Educational / fast prototyping |
| `distilled` | 2 | Image | 8 sigmas, fastest |
| `ic_lora` | 2 | Image + Video | Video-to-video transformations |
| `keyframe_interpolation` | 2 | Keyframes | Animation/interpolation |
| `a2vid_two_stage` | 2 | Audio + Image | Audio-driven video |
| `retake` | 1 | Source video | Regenerate time region |
| `hdr_ic_lora` | 2 | Video | HDR linear float output (EXR) |
| `lipdub` | 2 | Video + Audio | Lip dubbing / re-voicing |

## 3. Official LoRA Support

- LoRAs are attached via `SingleGPUModelBuilder.lora(path, strength, sd_ops)`.
- Fusion happens at load time using `fuse_loras.py`, with a `FuseRule` per quantization policy.
- The repo includes `LTXV_LORA_COMFY_RENAMING_MAP` in `ltx_core/loader/__init__.py` so official loaders can consume LoRAs produced for ComfyUI.
- Training produces `.safetensors` LoRAs through `ltx-trainer` (IC-LoRA, style, motion, etc.).

## 4. Official Quantization & Memory Footprint

### 4.1 FP8 (inference)

Two policies live in `ltx_core.quantization`:

| Policy | How it works | Requirements |
|--------|--------------|--------------|
| `fp8-cast` | Casts transformer linear weights to FP8 at load, upcasts on the fly. | No extra deps. Works with bf16 checkpoints. |
| `fp8-scaled-mm` | Uses TensorRT-LLM `cublas_scaled_mm` with `.weight_scale` tensors. | `uv sync --frozen --extra fp8-trtllm`; works with fp8 checkpoints. Best on Hopper. |

CLI flag: `--quantization fp8-cast` or `--quantization fp8-scaled-mm`.

### 4.2 Offloading / block streaming

`OffloadMode` in `ltx_pipelines/utils/types.py`:

| Mode | VRAM | RAM | Behavior |
|------|------|-----|----------|
| `none` | ~28 GB | high | All weights on GPU; fastest. |
| `cpu` | ~5 GB | ~36 GB | Weights pinned in CPU RAM, streamed per block. |
| `disk` | ~5 GB | ~5 GB | Weights read from disk on demand; slowest. |

Block streaming is implemented in `ltx_core.block_streaming`. It loads transformer blocks into a small rolling set of GPU buffers. **Important:** block streaming disables FP8 quantization (`README.md` note).

### 4.3 Trainer-side quantization (not exposed in pipelines)

`ltx-trainer/src/ltx_trainer/quantization.py` uses `optimum-quanto` and supports:

- `int8-quanto`
- `int4-quanto`
- `int2-quanto`
- `fp8-quanto`
- `fp8uz-quanto`

It quantizes block-by-block on GPU then moves back to CPU to minimize peak VRAM. This code is currently training-only but is a strong candidate for reuse in a standalone inference app.

`ltx-trainer` also has `gemma_8bit.py` for loading the Gemma text encoder with `bitsandbytes` 8-bit quantization.

## 5. Gaps for a "Universal" Standalone App

| Capability | Status in official repo | Gap |
|------------|------------------------|-----|
| All pipelines | ✅ Exposed as Python modules | Need unified CLI/GUI dispatcher |
| All LoRAs | ✅ Official + ComfyUI naming map | Need runtime LoRA switching without reloading base model |
| GGUF | ❌ Not supported | No native GGUF loader/dequant in `ltx-core` |
| <24 GB VRAM | ⚠️ Partial (FP8 + offloading) | Need stronger quantization (GGUF/Q4/int4) or streaming + quantization combo |

## 6. Key Files to Reference

- `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py`
- `packages/ltx-core/src/ltx_core/loader/fuse_loras.py`
- `packages/ltx-core/src/ltx_core/quantization/fp8_cast.py`
- `packages/ltx-core/src/ltx_core/quantization/fp8_scaled_mm.py`
- `packages/ltx-core/src/ltx_core/block_streaming/builder.py`
- `packages/ltx-pipelines/src/ltx_pipelines/utils/args.py`
- `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py`
- `packages/ltx-trainer/src/ltx_trainer/quantization.py`
- `packages/ltx-trainer/src/ltx_trainer/gemma_8bit.py`
