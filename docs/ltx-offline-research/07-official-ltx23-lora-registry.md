# Official LTX-2.3 LoRA / IC-LoRA Registry

## 1. Why This Matters

LTX-Desktop currently includes only one official IC-LoRA checkpoint in its model spec:

- `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`
- file: `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`

That is not enough if the desktop app is meant to expose the wider LTX-2.3 pipeline/workflow set. Many workflows require task-specific official IC-LoRAs: HDR, LipDub, in/outpainting, motion tracking, restoration, VFX, and consistency/reference-sheet control.

Recommendation: add all official LTX-2.3 LoRAs to the model-profile/component registry, but do **not** force-download them at install time. Treat them as optional capability components with per-pipeline validation.

## 2. Official LTX-2.3 Distillation LoRAs

These live in the main `Lightricks/LTX-2.3` Hugging Face repo and are distinct from IC-LoRAs.

| File | Size | Purpose |
|---|---:|---|
| `ltx-2.3-22b-distilled-lora-384.safetensors` | ~7.08 GiB | Distillation LoRA applicable to full model |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | ~7.08 GiB | Updated v1.1 distillation LoRA |

Also present in the same repo:

| File | Size | Purpose |
|---|---:|---|
| `ltx-2.3-22b-distilled.safetensors` | ~42.98 GiB | Full distilled checkpoint |
| `ltx-2.3-22b-distilled-1.1.safetensors` | ~42.98 GiB | Updated full distilled checkpoint |

Design implication:

- For LTX-Desktop’s current “fast” pipeline, the simplest path is the full distilled checkpoint.
- For “dev/full model + distilled refinement” workflows, expose `distilled_lora` as a separate component.
- In model profiles, represent these separately from IC-LoRAs.

## 3. Official LTX-2.3 IC-LoRA Repos and Files

Discovered official repos matching `Lightricks/LTX-2.3-22b-IC-LoRA-*` on Hugging Face.

| Capability | Hugging Face repo | File | Size |
|---|---|---|---:|
| Union Control | `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | ~624.1 MiB |
| Motion Track Control | `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control` | `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` | ~312.1 MiB |
| Ingredients / Reference Sheet | `Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients` | `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` | ~1248.1 MiB |
| Water Simulation | `Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation` | `ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors` | ~864.1 MiB |
| Decompression | `Lightricks/LTX-2.3-22b-IC-LoRA-Decompression` | `ltx-2.3-22b-ic-lora-decompression-0.9.safetensors` | ~864.1 MiB |
| Deblur | `Lightricks/LTX-2.3-22b-IC-LoRA-Deblur` | `ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` | ~864.1 MiB |
| Colorization | `Lightricks/LTX-2.3-22b-IC-LoRA-Colorization` | `ltx-2.3-22b-ic-lora-colorization-0.9.safetensors` | ~864.1 MiB |
| Day to Night | `Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night` | `ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors` | ~312.1 MiB |
| In/Outpainting | `Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting` | `ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` | ~1248.1 MiB |
| Instant Shave | `Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave` | `ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors` | ~624.1 MiB |
| Cross-Eyed | `Lightricks/LTX-2.3-22b-IC-LoRA-Cross-Eyed` | `ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors` | ~312.1 MiB |
| HDR | `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` | `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors` | ~312.1 MiB |
| HDR scene embeddings | `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` | `ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors` | ~12.0 MiB |
| LipDub | `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` | `ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors` | ~2352.4 MiB |

## 4. Functional Categories

Based on LTX documentation and model cards:

### Control IC-LoRAs

| IC-LoRA | Use |
|---|---|
| Union Control | One adapter for depth, canny, and pose control. Useful for IC-LoRA panel currently in LTX-Desktop. |
| Motion Track Control | Sparse spline/trajectory guidance for motion paths. |
| Ingredients | Reference sheet / inventory control for character, prop, and location consistency. |

### Restoration / conversion IC-LoRAs

| IC-LoRA | Use |
|---|---|
| Decompression | Remove heavy compression artifacts, restore cleaner edges/detail. |
| Deblur | Restore spatial defocus blur while preserving scene identity. |
| Colorization | Colorize grayscale/low-color input while preserving structure. |
| HDR | Convert SDR video to HDR / EXR workflow; also needs scene embedding file. |

### VFX / transform IC-LoRAs

| IC-LoRA | Use |
|---|---|
| Water Simulation | Add water/rain/splash/wet-surface effects to existing footage. |
| Day to Night | Re-render scene from daytime to nighttime. |
| In/Outpainting | Fill masked regions or extend canvas beyond original frame. |
| Instant Shave | Remove facial hair while preserving subject/motion. |
| Cross-Eyed | Stylized portrait eye transform. |
| LipDub | Lip dubbing / re-voicing; used by lipdub pipeline/workflow. |

## 5. Required Model-Profile Schema Extension

Add an `official_loras` or `adapters` section to model profiles.

Suggested schema:

```python
class AdapterComponent(BaseModel):
    id: str
    display_name: str
    kind: Literal["lora", "ic_lora", "distilled_lora", "embeddings"]
    source: Literal["official", "kijai", "custom"]
    repo_id: str | None = None
    filename: str
    path: str | None = None
    required_for: set[str] = set()
    optional_for: set[str] = set()
    expected_size_bytes: int | None = None
```

Profile example:

```json
{
  "adapters": {
    "union_control": {
      "kind": "ic_lora",
      "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
      "filename": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
      "required_for": ["ic_lora_depth", "ic_lora_canny", "ic_lora_pose"]
    },
    "lipdub": {
      "kind": "ic_lora",
      "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
      "filename": "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
      "required_for": ["lipdub"]
    },
    "hdr": {
      "kind": "ic_lora",
      "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
      "filename": "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
      "required_for": ["hdr_ic_lora"]
    },
    "hdr_scene_embeddings": {
      "kind": "embeddings",
      "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
      "filename": "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
      "required_for": ["hdr_ic_lora"]
    }
  }
}
```

## 6. Recommended App Behavior

### 6.1 Do not auto-download every official LoRA at install

All official LoRAs together are several GiB. The app should know about them, validate them, and offer download or browse actions, but not force them all during first-run.

Better UX:

- First-run: require only base generation components or valid profile.
- Pipeline panel: if selected pipeline needs a missing LoRA, show “Download official adapter” or “Browse local file”.
- Settings → Models → Official Adapters: show checklist/status for all known adapters.

### 6.2 Pipeline-aware validation

| Pipeline / feature | Required adapters/components |
|---|---|
| Fast T2V/I2V distilled | Full distilled checkpoint or base + distilled LoRA; upsampler if enabled |
| IC-LoRA depth/canny/pose | Union Control IC-LoRA + relevant processor model |
| Motion track | Motion Track Control IC-LoRA + trajectory conditioning UI/preprocessor |
| Ingredients/reference sheet | Ingredients IC-LoRA + reference sheet input |
| HDR | HDR IC-LoRA + HDR scene embeddings + EXR output path |
| LipDub | LipDub IC-LoRA + audio reference/input |
| In/outpainting | In-Outpainting IC-LoRA + mask input/preprocessing |
| Restoration/VFX | Task-specific IC-LoRA + reference video input |

### 6.3 Source priority

For every official LoRA component:

1. User-configured local path in active profile.
2. Existing file under models root with expected filename.
3. Optional download from official Hugging Face repo.
4. Clear missing-component error scoped to the selected pipeline.

## 7. LTX-Desktop Code Impact

Current code has `LtxIcLorasSpec` with only three fields:

```python
class LtxIcLorasSpec:
    depth_cp: ModelCheckpointID
    canny_cp: ModelCheckpointID
    pose_cp: ModelCheckpointID
```

This should be replaced or supplemented with a generic adapter registry. Keeping only `depth/canny/pose` will not scale to HDR, LipDub, in/outpainting, motion tracking, or restoration tasks.

Recommended changes:

- Keep old fields as backward-compatible aliases for Union Control.
- Add a new `AdapterRegistry` independent of `ModelCheckpointID` literals.
- Make model recommendations pipeline-aware rather than globally requiring all adapters.
- Add `adapter_status` endpoint for UI checklist.

## 8. Revised Milestone Placement

Official adapter registry should happen **before** GGUF LoRA patching because:

1. It defines the complete capabilities surface.
2. It gives the UI a stable component model.
3. GGUF support can then plug into the same adapter abstraction.

Suggested revised order:

1. Model profiles + manual official base paths.
2. Official adapter registry + local path/download/browse support.
3. Kijai split safetensors.
4. GGUF transformer.
5. GGUF-safe adapter patching.
