# Quantization & Low-VRAM Options for LTX-2

Goal: run the 22B LTX-2 model on GPUs with **<24 GB VRAM** while still supporting all pipelines and LoRAs.

## 1. Baseline Memory Reality

- Full 22B model in bf16: ~44 GB of weights alone → needs multi-GPU or heavy optimization.
- Official repo quotes:
  - `OffloadMode.NONE`: ~28 GB VRAM (likely FP8 or partial load; full bf16 is higher).
  - `OffloadMode.CPU` / `DISK`: ~5 GB VRAM by streaming transformer blocks.
- Text encoder (Gemma 3 12B) is a separate ~24 GB in bf16; must be quantized or offloaded independently.

## 2. Official / Semi-Official Methods

### 2.1 FP8 quantization (`ltx-core`)

| Variant | Memory reduction | Speed | Quality | Notes |
|---------|------------------|-------|---------|-------|
| `fp8-cast` | ~50% weights | Moderate | Good | No extra deps; works on any Ampere+ GPU with FP8 support. |
| `fp8-scaled-mm` | ~50% weights | Fast on Hopper | Good | Requires TensorRT-LLM; best on H100/H200. |

Limitations:
- Only transformer weights are quantized.
- Cannot be combined with block streaming (`--offload cpu|disk`).
- Still may not fit comfortably on 16–24 GB cards at high resolution.

### 2.2 Block streaming / offloading

- `OffloadMode.CPU`: pre-load blocks into pinned CPU RAM, copy to GPU on demand. Needs ~36 GB system RAM.
- `OffloadMode.DISK`: read blocks from `.safetensors` on demand. Needs ~5 GB RAM + fast SSD.
- First pass is slowest; subsequent passes reuse CPU cache.
- Disables FP8, so base weights stay bf16. Latency is dominated by PCIe/disk throughput.

### 2.3 Q8 kernels (ComfyUI-LTXVideo)

- Separate repo: `Lightricks/LTX-Video-Q8-Kernels`.
- Provides **blockwise FP8** patching for the transformer via `q8_kernels.integration.patch_transformer`.
- Includes a custom `LTXVQ8LoraModelLoader` that applies Hadamard transforms to LoRA weights so they match the Q8 kernel layout.
- Requires installing `q8_kernels` from source.
- Currently node-based inside ComfyUI-LTXVideo (`q8_nodes.py`). Not exposed as a plain Python API for `ltx-pipelines`.

### 2.4 `optimum-quanto` (from `ltx-trainer`)

- Supports INT8, INT4, INT2, FP8 weight quantization.
- Excludes `patchify_proj`, `proj_out`, `*adaln*`, `*norm*`, caption projection layers.
- Block-by-block GPU quantization then move back to CPU.
- **Not wired into `ltx-pipelines` inference** — only used during training.
- Could be adapted to quantize the transformer once at app startup, then run inference in low precision.

### 2.5 `bitsandbytes` 8-bit for Gemma

- `ltx-trainer/gemma_8bit.py` loads Gemma 3 backbone in 8-bit.
- Essential for <24 GB because the text encoder alone can OOM a 24 GB card.

## 3. Community / Third-Party Methods

### 3.1 GGUF quantization

GGUF is the most promising path for aggressive quantization on consumer GPUs. Several community repos host pre-converted LTX-2 / LTX-Video GGUF weights:

| Source | Notes |
|--------|-------|
| `QuantStack/LTX-2-GGUF` | Community LTX-2 GGUF family |
| `QuantStack/LTX-2.3-GGUF` | LTX-2.3 GGUF family |
| `Viper-AI-Vaunt/LTX-2.3-DEV-GGUF` | Includes `ltx-2.3-22b-dev-Q4_K_M.gguf` |
| `gguf-org/ltx2-gguf` | Alternative community repo |
| `vantagewithai/LTX-2-GGUF` | Includes fp8/fp4 variants + ComfyUI workflows |
| `calcuis/ltxv-gguf` | Older LTX-Video (2B) GGUF; includes Diffusers example |
| `city96/LTX2-gguf` | Mentioned in Diffusers GGUF docs |

Typical GGUF formats: `Q2_K`, `Q3_K`, `Q4_K_M`, `Q4_0`, `Q5_K`, `Q6_K`, `Q8_0`, `fp8_e4m3fn`.

VRAM estimates (transformer only, approximate):

| Format | Relative size | Estimated 22B transformer VRAM |
|--------|---------------|-------------------------------|
| bf16 | 100% | ~44 GB |
| fp8 | 50% | ~22 GB |
| Q8_0 | ~50% | ~22 GB |
| Q6_K | ~38% | ~17 GB |
| Q4_K_M | ~25% | ~11 GB |
| Q3_K | ~19% | ~8 GB |
| Q2_K | ~14% | ~6 GB |

Add VAE, upsampler, text encoder, activations, and LoRA overhead; even Q4_K_M can be tight on 12 GB but feasible with offloading.

### 3.2 GGUF loaders in the wild

#### ComfyUI-GGUF (`city96/ComfyUI-GGUF`)

- Adds `UnetLoaderGGUF` and `CLIPLoaderGGUF` nodes.
- Recognizes `ltxv` in its `IMG_ARCH_LIST` (`loader.py`), so it can load LTX-Video GGUF transformer weights.
- LoRAs are applied through ComfyUI's normal patch mechanism; `GGUFModelPatcher.patch_weight_to_device` dequantizes on the fly and calculates LoRA deltas.
- Does **not** currently have a dedicated GGUF LoRA loader — relies on standard `Load LoRA` node.
- Custom `GGMLOps` / `GGMLTensor` dynamically dequantize during forward.

#### Diffusers GGUF support

- `diffusers` has native `GGUFQuantizationConfig` and `from_single_file` loading for GGUF models.
- Pattern (from community examples):

```python
from diffusers import LTX2Pipeline, GGUFQuantizationConfig, LTX2Transformer3DModel

transformer = LTX2Transformer3DModel.from_single_file(
    "path/to/ltx-2.3-22b-dev-Q4_K_M.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="Lightricks/LTX-2.3",
    subfolder="transformer",
)
pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2.3",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
```

- Diffusers GGUF support currently loads **the transformer only** from GGUF; VAE, text encoder, and upsampler are still loaded from the base repo or separately quantized.
- `LTX2Pipeline` in Diffusers covers T2V / I2V but does **not** cover all official pipelines (no IC-LoRA, retake, lipdub, HDR, audio-to-video, etc.).

## 4. Combining Techniques for <24 GB

Recommended stack for a standalone app targeting 16–24 GB VRAM:

| Component | Technique | Expected footprint |
|-----------|-----------|-------------------|
| Transformer | GGUF Q4_K_M or Q5_K, or `optimum-quanto` INT4 | 11–17 GB weights |
| Text encoder | `bitsandbytes` 8-bit or GGUF Q4 | 6–12 GB |
| VAE / upsampler | bf16 on GPU with tiled decode, or CPU offloaded | 2–6 GB |
| Activations | FP16/BF16 compute, tiled attention | 2–8 GB |
| LoRAs | Load as safetensors, fuse or patch into quantized model | small overhead |

Knobs to expose to the user:

1. **Transformer quantization**: `none`, `fp8-cast`, `fp8-scaled-mm`, `quanto-int8`, `quanto-int4`, `gguf-q8`, `gguf-q4`, etc.
2. **Block streaming / offloading**: `none`, `cpu`, `disk` (mutually exclusive with FP8).
3. **Text encoder precision**: `bf16`, `fp8`, `int8-bnb`, `gguf`.
4. **VAE tiling**: tile size for decode (smaller tiles = lower VRAM).
5. **Max batch size**: for guided denoisers (1–4).
6. **Gradient estimation**: official pipeline option to reduce steps (40 → 20–30).

## 5. Risks / Tradeoffs

| Method | Risk |
|--------|------|
| GGUF Q2/Q3 | Severe quality degradation; Q4_K_M is usually the practical minimum. |
| INT4 / INT2 quanto | May interact badly with some LoRA inits; needs validation per LoRA. |
| FP8 + offloading | Officially cannot be combined. Custom streaming-aware FP8 would need new code. |
| GGUF + official LoRA fusion | Official `fuse_loras.py` expects safetensors and a `FuseRule`. A GGUF-aware fuse rule would be needed, or use patch-at-runtime like ComfyUI. |
| Diffusers GGUF | Limited pipeline coverage; may lag behind official model updates. |
| Q8 kernels | Tied to ComfyUI model patcher; needs porting to work with `ltx-pipelines`. |

## 6. Bottom Line

- **For 24 GB cards**: official `fp8-cast` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + reduced resolution may already work.
- **For 16–20 GB cards**: GGUF Q4_K_M/Q5_K transformer + 8-bit Gemma + tiled VAE is the most practical known path.
- **For 12 GB cards**: GGUF Q4_K_M + block/disk streaming for transformer + CPU-offloaded VAE/text encoder + very small resolutions.
