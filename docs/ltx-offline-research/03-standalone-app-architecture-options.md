# Standalone App Architecture Options for LTX-2

Requirement recap:

1. Standalone application (CLI and/or GUI).
2. Works with **all LoRAs** (official, ComfyUI-named, IC-LoRA, HDR, distilled, Q8-aware, etc.).
3. Works with **all pipelines** (T2V, I2V, V2V, A2V, retake, keyframe, HDR, lipdub, distilled).
4. Runs on **<24 GB VRAM** via GGUF/low-bit quantization.

This document compares four architectural foundations and recommends a hybrid.

## Option A: Build on top of `ltx-pipelines` (official)

### How it works

Use the official Python package as the execution engine. Create a unified dispatcher that imports every pipeline class from `ltx_pipelines` and routes user input to the right one.

```python
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
# ... etc
```

### Pros

- Full fidelity to official inference logic.
- All 10 pipelines available out-of-the-box.
- Native LoRA fusion and FP8 quantization.
- Easy to keep in sync with upstream releases.

### Cons

- **No native GGUF support.** Would have to implement a GGUF loader/dequantizer inside `ltx-core`.
- `OffloadMode` disables FP8 quantization; can't combine streaming + quantization.
- CLI is one-module-per-pipeline; needs a wrapper to feel like a single app.

### Best for

Users who prioritize pipeline completeness and official-model fidelity, and who can accept a custom GGUF integration effort.

---

## Option B: Build on Diffusers

### How it works

Use `diffusers.LTX2Pipeline` / `LTX2Transformer3DModel` with `GGUFQuantizationConfig` for the transformer. Build a thin wrapper that adds I2V, V2V, retake, etc., on top of Diffusers primitives.

```python
from diffusers import LTX2Pipeline, GGUFQuantizationConfig, LTX2Transformer3DModel

transformer = LTX2Transformer3DModel.from_single_file(
    "ltx-2.3-22b-dev-Q4_K_M.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2.3", transformer=transformer)
pipe.enable_model_cpu_offload()
```

### Pros

- **Native GGUF support** via `GGUFQuantizationConfig`.
- Clean, familiar API for T2V/I2V.
- `enable_model_cpu_offload()`, `enable_vae_tiling()`, `enable_sequential_cpu_offload()` built-in.

### Cons

- Diffusers' `LTX2Pipeline` covers only a subset of official capabilities.
- No built-in IC-LoRA, audio-to-video, keyframe interpolation, retake, HDR, lipdub.
- GGUF currently loads transformer only; VAE/text encoder/upsampler still need separate quantization.
- LoRA application in Diffusers may not cover all official/ComfyUI LoRA formats.

### Best for

A quick T2V/I2V-only standalone app with minimal custom code.

---

## Option C: Embed ComfyUI

### How it works

Ship ComfyUI as a backend with `ComfyUI-LTXVideo` + `ComfyUI-GGUF` installed. Build a GUI/CLI that generates and executes ComfyUI workflows.

### Pros

- **All features already exist**: Q8 kernels, GGUF loader, low-VRAM loaders, IC-LoRA, retake, HDR, lipdub, etc.
- Large ecosystem of community nodes.
- GGUF LoRA patching works via `GGUFModelPatcher`.

### Cons

- Heavy runtime dependency (ComfyUI server + model folder conventions).
- Not truly "standalone"; more like a ComfyUI distribution.
- Workflow JSON is brittle; upgrades can break workflows.
- Harder to unit-test and version-control behavior.

### Best for

A GUI-focused app where shipping a full ComfyUI environment is acceptable.

---

## Option D: Hybrid — `ltx-core` + custom quantization/loading plugins

### How it works

Keep `ltx-core` as the model/pipeline execution layer but replace or extend the loading path:

1. **Base loader**: `SingleGPUModelBuilder` for safetensors / FP8.
2. **GGUF loader**: port/adapt ComfyUI-GGUF's `loader.py` + `ops.py` or Diffusers' GGUF dequant to load `.gguf` transformers into `ltx-core` model instances.
3. **Quanto loader**: reuse `ltx-trainer.quantization.quantize_model()` to produce int8/int4 transformers at load time.
4. **Text encoder loader**: use `ltx-trainer.gemma_8bit.load_8bit_gemma()` or a GGUF text encoder loader.
5. **LoRA engine**: keep official `fuse_loras.py` for safetensors, but add a runtime patch path for GGUF models (similar to `GGUFModelPatcher`).
6. **Pipeline dispatcher**: a single CLI/GUI that calls every `ltx_pipelines` class.

### Pros

- **All pipelines** from `ltx-pipelines`.
- **All LoRA formats** via dual path (fuse for safetensors, patch for GGUF).
- **<24 GB VRAM** via GGUF + quanto + 8-bit Gemma + offloading.
- Clean Python API, testable, version-controllable.

### Cons

- Largest implementation effort.
- Need to maintain compatibility across upstream `ltx-core`, `optimum-quanto`, `gguf-py`, and `bitsandbytes` updates.
- Some combinations (e.g., FP8 + block streaming) are inherently incompatible and need clear user messaging.

### Best for

A production-grade standalone app that must support the full feature matrix.

---

## Comparison Matrix

| Criterion | A: `ltx-pipelines` | B: Diffusers | C: ComfyUI | D: Hybrid (recommended) |
|-----------|-------------------|--------------|------------|------------------------|
| All official pipelines | ✅ | ⚠️ partial | ✅ | ✅ |
| All LoRA formats | ✅ official/ComfyUI | ⚠️ limited | ✅ | ✅ |
| Native GGUF | ❌ | ✅ transformer only | ✅ | ✅ after integration |
| <24 GB VRAM | ⚠️ FP8 + offload | ✅ with GGUF | ✅ | ✅ |
| Standalone / portable | ✅ | ✅ | ❌ heavy | ✅ |
| Maintainability | ✅ | ⚠️ lags official | ⚠️ workflow fragility | ⚠️ more code |
| Testability | ✅ | ✅ | ❌ | ✅ |

---

## Recommended Architecture: Option D (Hybrid)

A standalone Python app structured as follows:

```
ltx-offline/
├── ltx_offline/
│   ├── __init__.py
│   ├── cli.py                 # Single entry point (all pipelines)
│   ├── gui.py                 # Optional Gradio / PyQt / web UI
│   ├── pipelines/
│   │   ├── registry.py        # Maps pipeline name -> class + args
│   │   ├── dispatcher.py      # Builds and runs the selected pipeline
│   │   └── wrappers/          # Thin wrappers where needed
│   ├── loaders/
│   │   ├── gguf_loader.py     # GGUF transformer loader (adapted)
│   │   ├── quanto_loader.py   # optimum-quanto loader
│   │   ├── fp8_loader.py      # Official FP8 policies
│   │   └── text_encoder.py    # Gemma 8-bit / GGUF selector
│   ├── lora/
│   │   ├── fuse.py            # Official safetensors fusion
│   │   └── patch.py           # Runtime patch for GGUF models
│   └── memory/
│       ├── offloading.py      # CPU/disk streaming knobs
│       └── vae_tiling.py      # Tile size selection
├── research/
├── configs/
│   └── low_vram_profiles.yaml # Presets for 12/16/20/24 GB
└── pyproject.toml
```

### Runtime flow

1. User selects **pipeline** (e.g., `ti2vid_two_stages`).
2. User selects **model precision**:
   - `bf16` (24+ GB)
   - `fp8-cast` / `fp8-scaled-mm` (20–24 GB)
   - `quanto-int8` / `quanto-int4` (12–20 GB)
   - `gguf-q8_0` / `gguf-q4_k_m` (10–16 GB)
3. User attaches **LoRAs** (paths + strengths).
4. App picks loader:
   - Safetensors base → `SingleGPUModelBuilder` + `fuse_loras`.
   - GGUF base → custom GGUF loader + patch LoRAs at runtime.
   - Quanto → load bf16 then quantize block-wise.
5. App sets **memory mode**:
   - GPU-only if VRAM allows.
   - CPU/disk streaming if transformer is bf16/FP8 and doesn't fit.
6. App runs pipeline and writes output.

### Key libraries to pin

- `torch` (CUDA build, matching GPU generation)
- `ltx-core`, `ltx-pipelines` from the official repo
- `optimum-quanto` (for int8/int4 loaders)
- `bitsandbytes` (for 8-bit Gemma)
- `gguf` (for GGUF parsing)
- `diffusers` (optional fallback for Diffusers-format GGUF and validation)

### UI options

- **CLI first**: single `ltx-offline` command with subcommands.
- **GUI later**: Gradio web UI is easiest; wraps CLI calls.

---

## Open Questions to Resolve Before Coding

1. Do we want to support **both** safetensors and GGUF base models, or standardize on GGUF for the low-VRAM build?
2. Should the app **embed a model manager** to download official + community GGUF checkpoints?
3. Which UI framework: terminal-only, Gradio, PyQt, Tauri/web, or ComfyUI-frontend?
4. Do we need **macOS/ROCm support**, or is CUDA the initial target?
5. Should LoRAs be **fused once at load** (faster inference) or **patched per step** (slower but supports dynamic switching and GGUF)?
