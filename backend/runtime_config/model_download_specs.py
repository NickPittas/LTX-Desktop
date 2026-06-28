"""Canonical checkpoint specs and LTX model relationships."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, cast, get_args

from api_types import (
    AdapterID,
    AdapterKind,
    AdapterPipeline,
    AdapterSource,
    LTXLocalModelId,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    ModelCheckpointID,
)

logger = logging.getLogger(__name__)


ALL_MODEL_CP_IDS = cast(tuple[ModelCheckpointID, ...], get_args(ModelCheckpointID))
ALL_LTX_LOCAL_MODEL_IDS = cast(tuple[LTXLocalModelId, ...], get_args(LTXLocalModelId))


@dataclass(frozen=True, slots=True)
class ModelCheckpointSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str

    @property
    def name(self) -> str:
        return self.relative_path.name


@dataclass(frozen=True, slots=True)
class LTXLocalModelDeprecated:
    pass


@dataclass(frozen=True, slots=True)
class LTXLocalModelRelevant:
    upgrade_messages: dict[LTXLocalModelId, str]


LTXLocalModelRelevance = LTXLocalModelDeprecated | LTXLocalModelRelevant


@dataclass(frozen=True, slots=True)
class LtxIcLorasSpec:
    depth_cp: ModelCheckpointID
    canny_cp: ModelCheckpointID
    pose_cp: ModelCheckpointID


@dataclass(frozen=True, slots=True)
class LTXLocalModelSpec:
    model_cp: ModelCheckpointID
    upscale_cp: ModelCheckpointID
    text_encoder_cp: ModelCheckpointID
    ic_loras_spec: LtxIcLorasSpec
    relevance: LTXLocalModelRelevance
    supported_pipelines: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...]


@dataclass(frozen=True, slots=True)
class AdapterComponent:
    id: AdapterID
    display_name: str
    kind: AdapterKind
    source: AdapterSource
    repo_id: str
    filename: str
    expected_size_bytes: int
    required_for: tuple[AdapterPipeline, ...] = ()
    optional_for: tuple[AdapterPipeline, ...] = ()


def _local_resolution_spec(
    *,
    fps_to_durations: dict[LTXVideoGenFps, tuple[LTXVideoGenDuration, ...]],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        },
    )


IMG_GEN_MODEL_CP_ID: ModelCheckpointID = "z-image-turbo"
DEPTH_PROCESSOR_CP_ID: ModelCheckpointID = "dpt-hybrid-midas"
PERSON_DETECTOR_CP_ID: ModelCheckpointID = "yolox-l-torchscript"
POSE_PROCESSOR_CP_ID: ModelCheckpointID = "dw-ll-ucoco-384-bs5"
# Canonical 2x spatial upscaler checkpoint id. Some legacy profiles stored a
# stale root-level path (e.g. ``<models_dir>//ltx-2.3-spatial-upscaler-x2-1.0.safetensors``);
# the canonical location is ``latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors``.
UPSAMPLER_CP_ID: ModelCheckpointID = "ltx-2.3-spatial-upscaler-x2-1.0"


OFFICIAL_LTX23_ADAPTERS: dict[AdapterID, AdapterComponent] = {
    "distilled_lora_384": AdapterComponent(
        id="distilled_lora_384",
        display_name="Distilled LoRA 384",
        kind="distilled_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3",
        filename="ltx-2.3-22b-distilled-lora-384.safetensors",
        expected_size_bytes=7_080_000_000,
        optional_for=("fast",),
    ),
    "distilled_lora_384_1_1": AdapterComponent(
        id="distilled_lora_384_1_1",
        display_name="Distilled LoRA 384 v1.1",
        kind="distilled_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3",
        filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        expected_size_bytes=7_080_000_000,
        optional_for=("fast",),
    ),
    "union_control": AdapterComponent(
        id="union_control",
        display_name="Union Control",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        expected_size_bytes=310_000_000,
        required_for=("union_control",),
    ),
    "motion_track_control": AdapterComponent(
        id="motion_track_control",
        display_name="Motion Track Control",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        filename="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        expected_size_bytes=310_000_000,
        required_for=("motion_track_control",),
    ),
    "ingredients": AdapterComponent(
        id="ingredients",
        display_name="Ingredients",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
        filename="ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        expected_size_bytes=624_100_000,
        required_for=("ingredients",),
    ),
    "water_simulation": AdapterComponent(
        id="water_simulation",
        display_name="Water Simulation",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation",
        filename="ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors",
        expected_size_bytes=624_100_000,
        required_for=("water_simulation",),
    ),
    "decompression": AdapterComponent(
        id="decompression",
        display_name="Decompression",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Decompression",
        filename="ltx-2.3-22b-ic-lora-decompression-0.9.safetensors",
        expected_size_bytes=312_100_000,
        required_for=("decompression",),
    ),
    "deblur": AdapterComponent(
        id="deblur",
        display_name="Deblur",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Deblur",
        filename="ltx-2.3-22b-ic-lora-deblur-0.9.safetensors",
        expected_size_bytes=312_100_000,
        required_for=("deblur",),
    ),
    "colorization": AdapterComponent(
        id="colorization",
        display_name="Colorization",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Colorization",
        filename="ltx-2.3-22b-ic-lora-colorization-0.9.safetensors",
        expected_size_bytes=312_100_000,
        required_for=("colorization",),
    ),
    "day_to_night": AdapterComponent(
        id="day_to_night",
        display_name="Day to Night",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night",
        filename="ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors",
        expected_size_bytes=312_100_000,
        required_for=("day_to_night",),
    ),
    "in_outpainting": AdapterComponent(
        id="in_outpainting",
        display_name="In/Outpainting",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting",
        filename="ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        expected_size_bytes=624_100_000,
        required_for=("in_outpainting",),
    ),
    "instant_shave": AdapterComponent(
        id="instant_shave",
        display_name="Instant Shave",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave",
        filename="ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors",
        expected_size_bytes=624_100_000,
        required_for=("instant_shave",),
    ),
    "cross_eyed": AdapterComponent(
        id="cross_eyed",
        display_name="Cross Eyed",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Cross-Eyed",
        filename="ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors",
        expected_size_bytes=312_100_000,
        required_for=("cross_eyed",),
    ),
    "hdr": AdapterComponent(
        id="hdr",
        display_name="HDR",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
        filename="ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
        expected_size_bytes=312_100_000,
        required_for=("hdr",),
    ),
    "hdr_scene_embeddings": AdapterComponent(
        id="hdr_scene_embeddings",
        display_name="HDR Scene Embeddings",
        kind="embeddings",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
        filename="ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
        expected_size_bytes=12_000_000,
        required_for=("hdr",),
    ),
    "lipdub": AdapterComponent(
        id="lipdub",
        display_name="LipDub",
        kind="ic_lora",
        source="official",
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
        filename="ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
        expected_size_bytes=2_352_400_000,
        required_for=("lipdub",),
    ),
}


def get_model_cp_spec(cp_id: ModelCheckpointID) -> ModelCheckpointSpec:
    match cp_id:
        case "ltx-2.3-22b-distilled":
            return ModelCheckpointSpec(
                relative_path=Path("diffusion_models/ltx-2.3-22b-distilled.safetensors"),
                expected_size_bytes=43_000_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Main transformer model",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.0":
            return ModelCheckpointSpec(
                relative_path=Path("latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                expected_size_bytes=1_900_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="2x upscaler",
            )
        case "ltx-2.3-22b-ic-lora-union-control-ref0.5":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
                expected_size_bytes=654_465_352,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                description="Union IC-LoRA control model",
            )
        case "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"),
                expected_size_bytes=310_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
                description="Motion Track Control IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-ingredients-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"),
                expected_size_bytes=624_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
                description="Ingredients IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-water-simulation-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors"),
                expected_size_bytes=624_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation",
                description="Water Simulation IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-decompression-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-decompression-0.9.safetensors"),
                expected_size_bytes=312_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Decompression",
                description="Decompression IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-deblur-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"),
                expected_size_bytes=312_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Deblur",
                description="Deblur IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-colorization-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-colorization-0.9.safetensors"),
                expected_size_bytes=312_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Colorization",
                description="Colorization IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-day-to-night-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors"),
                expected_size_bytes=312_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night",
                description="Day to Night IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-in-outpainting-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"),
                expected_size_bytes=624_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting",
                description="In/Outpainting IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-instant-shave-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors"),
                expected_size_bytes=624_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave",
                description="Instant Shave IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-cross-eyed-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors"),
                expected_size_bytes=312_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Cross-Eyed",
                description="Cross Eyed IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-hdr-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"),
                expected_size_bytes=312_100_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
                description="HDR IC-LoRA",
            )
        case "ltx-2.3-22b-ic-lora-hdr-scene-emb":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"),
                expected_size_bytes=12_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
                description="HDR Scene Embeddings",
            )
        case "ltx-2.3-22b-ic-lora-lipdub-0.9":
            return ModelCheckpointSpec(
                relative_path=Path("adapters/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"),
                expected_size_bytes=2_352_400_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
                description="LipDub IC-LoRA",
            )
        case "dpt-hybrid-midas":
            return ModelCheckpointSpec(
                relative_path=Path("depth_processors/dpt-hybrid-midas"),
                expected_size_bytes=500_000_000,
                is_folder=True,
                repo_id="Intel/dpt-hybrid-midas",
                description="DPT-Hybrid MiDaS depth processor",
            )
        case "yolox-l-torchscript":
            return ModelCheckpointSpec(
                relative_path=Path("detectors/yolox_l.torchscript.pt"),
                expected_size_bytes=217_697_649,
                is_folder=False,
                repo_id="hr16/yolox-onnx",
                description="YOLOX person detector for pose preprocessing",
            )
        case "dw-ll-ucoco-384-bs5":
            return ModelCheckpointSpec(
                relative_path=Path("pose_processors/dw-ll_ucoco_384_bs5.torchscript.pt"),
                expected_size_bytes=135_059_124,
                is_folder=False,
                repo_id="hr16/DWPose-TorchScript-BatchSize5",
                description="DW Pose TorchScript processor",
            )
        case "gemma-3-12b-it-qat-q4_0-unquantized":
            return ModelCheckpointSpec(
                relative_path=Path("text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"),
                expected_size_bytes=25_000_000_000,
                is_folder=True,
                repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
                description="Gemma text encoder (bfloat16)",
            )
        case "z-image-turbo":
            return ModelCheckpointSpec(
                relative_path=Path("image_gen/Z-Image-Turbo"),
                expected_size_bytes=31_000_000_000,
                is_folder=True,
                repo_id="Tongyi-MAI/Z-Image-Turbo",
                description="Z-Image-Turbo model for text-to-image generation",
            )
        case _:
            assert_never(cp_id)


def get_ltx_model_spec(model_id: LTXLocalModelId) -> LTXLocalModelSpec:
    match model_id:
        case "ltx-2.3-22b-distilled":
            return LTXLocalModelSpec(
                model_cp="ltx-2.3-22b-distilled",
                upscale_cp="ltx-2.3-spatial-upscaler-x2-1.0",
                text_encoder_cp="gemma-3-12b-it-qat-q4_0-unquantized",
                ic_loras_spec=LtxIcLorasSpec(
                    depth_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
                    canny_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
                    pose_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
                ),
                relevance=LTXLocalModelRelevant(upgrade_messages={}),
                supported_pipelines=(
                    (
                        "fast",
                        LTXVideoGenerationSpec(
                            display_name="LTX 2.3 Fast",
                            supported_resolutions_durations={
                                "540p": _local_resolution_spec(
                                    fps_to_durations={
                                        24: (5, 6, 8, 10, 20),
                                    },
                                ),
                                "720p": _local_resolution_spec(
                                    fps_to_durations={
                                        24: (5, 6, 8, 10),
                                    },
                                ),
                                "1080p": _local_resolution_spec(
                                    fps_to_durations={
                                        24: (5,),
                                    },
                                ),
                            },
                        ),
                    ),
                ),
            )
        case _:
            assert_never(model_id)


def get_ltx_cps() -> set[ModelCheckpointID]:
    cp_ids: set[ModelCheckpointID] = set()
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        cp_ids.add(get_ltx_model_spec(model_id).model_cp)
    return cp_ids


def get_latest_ltx_model_id() -> LTXLocalModelId:
    relevant: list[LTXLocalModelId] = []
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if isinstance(get_ltx_model_spec(model_id).relevance, LTXLocalModelRelevant):
            relevant.append(model_id)
    if len(relevant) != 1:
        raise RuntimeError(f"Expected exactly one relevant LTX model, found {len(relevant)}")
    return relevant[0]


def get_ltx_model_id_for_cp(cp_id: ModelCheckpointID) -> LTXLocalModelId | None:
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if get_ltx_model_spec(model_id).model_cp == cp_id:
            return model_id
    return None


# ponytail: explicit dict, no new class. Add entry when an adapter has a downloadable CP spec.
ADAPTER_TO_CP_ID: dict[AdapterID, ModelCheckpointID] = {
    "union_control": "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "motion_track_control": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5",
    "ingredients": "ltx-2.3-22b-ic-lora-ingredients-0.9",
    "water_simulation": "ltx-2.3-22b-ic-lora-water-simulation-0.9",
    "decompression": "ltx-2.3-22b-ic-lora-decompression-0.9",
    "deblur": "ltx-2.3-22b-ic-lora-deblur-0.9",
    "colorization": "ltx-2.3-22b-ic-lora-colorization-0.9",
    "day_to_night": "ltx-2.3-22b-ic-lora-day-to-night-0.9",
    "in_outpainting": "ltx-2.3-22b-ic-lora-in-outpainting-0.9",
    "instant_shave": "ltx-2.3-22b-ic-lora-instant-shave-0.9",
    "cross_eyed": "ltx-2.3-22b-ic-lora-cross-eyed-0.9",
    "hdr": "ltx-2.3-22b-ic-lora-hdr-0.9",
    "hdr_scene_embeddings": "ltx-2.3-22b-ic-lora-hdr-scene-emb",
    "lipdub": "ltx-2.3-22b-ic-lora-lipdub-0.9",
}


def get_ic_loras_cp_ids(ic_loras_spec: LtxIcLorasSpec) -> tuple[ModelCheckpointID, ...]:
    return tuple(dict.fromkeys((ic_loras_spec.depth_cp, ic_loras_spec.canny_cp, ic_loras_spec.pose_cp)))


def get_ltx_model_cp_ids(model_id: LTXLocalModelId) -> tuple[ModelCheckpointID, ...]:
    spec = get_ltx_model_spec(model_id)
    return (
        spec.model_cp,
        spec.upscale_cp,
        spec.text_encoder_cp,
        *get_ic_loras_cp_ids(spec.ic_loras_spec),
    )


def _normalized_relative_path(cp_id: ModelCheckpointID) -> Path:
    relative_path = get_model_cp_spec(cp_id).relative_path
    if relative_path.is_absolute():
        raise ValueError(f"Model path for {cp_id} must be relative: {relative_path}")

    normalized_parts = [part for part in relative_path.parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError(f"Model path for {cp_id} cannot be empty: {relative_path}")
    if ".." in normalized_parts:
        raise ValueError(f"Model path for {cp_id} cannot traverse parents: {relative_path}")

    return Path(*normalized_parts)


def resolve_model_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    return models_dir / _normalized_relative_path(cp_id)


def resolve_downloading_dir(models_dir: Path) -> Path:
    return models_dir / ".downloading"


def resolve_downloading_target_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    return resolve_downloading_dir(models_dir) / _normalized_relative_path(cp_id)


def resolve_downloading_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    spec = get_model_cp_spec(cp_id)
    relative_path = _normalized_relative_path(cp_id)
    downloading_dir = resolve_downloading_dir(models_dir)
    if spec.is_folder:
        return downloading_dir / relative_path
    parent = relative_path.parent
    if parent == Path("."):
        return downloading_dir
    return downloading_dir / parent


def is_cp_downloaded(models_dir: Path, cp_id: ModelCheckpointID) -> bool:
    path = resolve_model_path(models_dir, cp_id)
    spec = get_model_cp_spec(cp_id)
    if spec.is_folder:
        return path.exists() and any(path.iterdir())
    return path.exists()


def get_existing_cp_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    path = resolve_model_path(models_dir, cp_id)
    if not is_cp_downloaded(models_dir, cp_id):
        raise FileNotFoundError(f"Checkpoint not found: {cp_id} at {path}")
    return path


def delete_cp_path(models_dir: Path, cp_id: ModelCheckpointID) -> None:
    path = resolve_model_path(models_dir, cp_id)
    spec = get_model_cp_spec(cp_id)
    if spec.is_folder:
        if path.exists():
            import shutil

            shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def get_downloaded_ltx_model_id(models_dir: Path) -> LTXLocalModelId | None:
    downloaded: list[LTXLocalModelId] = []
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if is_cp_downloaded(models_dir, get_ltx_model_spec(model_id).model_cp):
            downloaded.append(model_id)
    if not downloaded:
        return None
    if len(downloaded) == 1:
        return downloaded[0]

    logger.warning("Multiple LTX model checkpoints detected: %s", ", ".join(downloaded))
    relevant: list[LTXLocalModelId] = []
    for model_id in downloaded:
        if isinstance(get_ltx_model_spec(model_id).relevance, LTXLocalModelRelevant):
            relevant.append(model_id)
    if len(relevant) == 1:
        return relevant[0]
    if len(relevant) > 1:
        logger.warning("Multiple relevant LTX models detected; selecting the first available: %s", relevant[0])
        return relevant[0]
    logger.warning("Multiple deprecated LTX models detected; selecting the first available: %s", downloaded[0])
    return downloaded[0]


def _validate_model_cp_specs() -> None:
    relative_paths: dict[Path, ModelCheckpointID] = {}
    for cp_id in ALL_MODEL_CP_IDS:
        normalized = _normalized_relative_path(cp_id)
        existing = relative_paths.get(normalized)
        if existing is not None:
            raise RuntimeError(f"Duplicate checkpoint path mapping: {existing} and {cp_id} -> {normalized}")
        relative_paths[normalized] = cp_id


def _validate_ltx_specs() -> None:
    ltx_cps = get_ltx_cps()
    if len(ltx_cps) != len(ALL_LTX_LOCAL_MODEL_IDS):
        raise RuntimeError("LTX model primary checkpoints must map 1:1 with LTX model ids")
    _ = get_latest_ltx_model_id()


_validate_model_cp_specs()
_validate_ltx_specs()
