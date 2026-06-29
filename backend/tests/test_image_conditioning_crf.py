"""Targeted tests for the centralized image-conditioning CRF override (plan §11).

Upstream ``ltx_pipelines`` defaults image-conditioning CRF to
``DEFAULT_IMAGE_CRF`` (=33, lossy). The app overrides this to a near-lossless
CRF of 18 for every image-conditioning input it constructs, routed through a
single helper (``make_ltx_image_conditioning_input``) so the override is
centralized across the fast / distilled-native / IC-LoRA entry points.

These tests are mock-free (hand-rolled fakes only), per the repo's no-mock
policy enforced by ``test_no_mock_usage.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import (
    IMAGE_CONDITIONING_CRF,
    make_ltx_image_conditioning_input,
)

_BACKEND = Path(__file__).resolve().parents[1]


def test_image_conditioning_crf_constant_is_18() -> None:
    assert IMAGE_CONDITIONING_CRF == 18


def test_helper_overrides_crf_to_18() -> None:
    result = make_ltx_image_conditioning_input("/path/frame.png", 4, 0.5)
    assert result.crf == 18
    # Guard against accidental revert to upstream default (33).
    assert result.crf != 33


def test_helper_preserves_path_frame_idx_strength() -> None:
    result = make_ltx_image_conditioning_input("/p/a.png", 7, 0.8)
    assert result.path == "/p/a.png"
    assert result.frame_idx == 7
    assert result.strength == pytest.approx(0.8)


def test_helper_returns_upstream_named_tuple_with_crf_field() -> None:
    result = make_ltx_image_conditioning_input("a.png", 0, 1.0)
    # Upstream ImageConditioningInput is a 4-field NamedTuple
    # (path, frame_idx, strength, crf) — confirms the override targets the
    # correct field rather than falling back to a positional default.
    assert result._fields == ("path", "frame_idx", "strength", "crf")


class _CapturingPipeline:
    """Fake upstream pipeline capturing the kwargs passed to ``__call__``."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> tuple[Any, Any]:
        self.last_kwargs = kwargs
        return (object(), None)


def test_fast_pipeline_run_inference_forwards_crf_18() -> None:
    from services.fast_video_pipeline.ltx_fast_video_pipeline import (
        LTXFastVideoPipeline,
    )
    from services.ltx_pipeline_common import default_tiling_config

    pipe = LTXFastVideoPipeline.__new__(LTXFastVideoPipeline)
    fake = _CapturingPipeline()
    pipe.pipeline = fake  # type: ignore[assignment]
    pipe._streaming_prefetch_count = None  # type: ignore[attr-defined]
    # Phase 3D: _run_inference branches on base_family. The CRF override is
    # shared by both routes; test the distilled route (default) here.
    pipe._base_family = "distilled"  # type: ignore[attr-defined]

    images = [
        ImageConditioningInput(path="a.png", frame_idx=0, strength=1.0),
        ImageConditioningInput(path="b.png", frame_idx=4, strength=0.5),
    ]
    pipe._run_inference(
        prompt="p",
        seed=1,
        height=64,
        width=64,
        num_frames=9,
        frame_rate=8.0,
        images=images,
        tiling_config=default_tiling_config(),
        enhance_prompt=False,
    )

    assert fake.last_kwargs is not None
    forwarded = fake.last_kwargs["images"]
    assert len(forwarded) == 2
    assert [i.crf for i in forwarded] == [18, 18]
    assert [i.path for i in forwarded] == ["a.png", "b.png"]
    assert [i.frame_idx for i in forwarded] == [0, 4]


def test_fast_pipeline_dev_run_inference_forwards_negative_prompt_and_crf_18() -> None:
    """Phase 3D: dev route forwards negative_prompt + LTX_2_3_PARAMS guider
    params + num_inference_steps, and still applies the centralized CRF=18
    override on image conditioning inputs.
    """
    from services.fast_video_pipeline.ltx_fast_video_pipeline import (
        LTXFastVideoPipeline,
    )
    from services.ltx_pipeline_common import default_tiling_config

    pipe = LTXFastVideoPipeline.__new__(LTXFastVideoPipeline)
    fake = _CapturingPipeline()
    pipe.pipeline = fake  # type: ignore[assignment]
    pipe._streaming_prefetch_count = None  # type: ignore[attr-defined]
    pipe._base_family = "dev"  # type: ignore[attr-defined]

    images = [ImageConditioningInput(path="a.png", frame_idx=0, strength=1.0)]
    pipe._run_inference(
        prompt="p",
        seed=1,
        height=64,
        width=64,
        num_frames=9,
        frame_rate=8.0,
        images=images,
        tiling_config=default_tiling_config(),
        enhance_prompt=False,
        negative_prompt="blurry, low quality",
    )

    from ltx_pipelines.utils.constants import LTX_2_3_PARAMS

    assert fake.last_kwargs is not None
    kw = fake.last_kwargs
    # Dev route must forward the negative prompt for CFG.
    assert kw["negative_prompt"] == "blurry, low quality"
    # Upstream LTX_2_3_PARAMS steps + guider params are passed through.
    assert kw["num_inference_steps"] == LTX_2_3_PARAMS.num_inference_steps
    assert kw["video_guider_params"] is LTX_2_3_PARAMS.video_guider_params
    assert kw["audio_guider_params"] is LTX_2_3_PARAMS.audio_guider_params
    # CRF override still applies on the dev route.
    forwarded = kw["images"]
    assert len(forwarded) == 1
    assert forwarded[0].crf == 18


def test_ic_lora_run_inference_forwards_crf_18() -> None:
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    from services.ltx_pipeline_common import default_tiling_config

    pipe = LTXIcLoraPipeline.__new__(LTXIcLoraPipeline)
    fake = _CapturingPipeline()
    pipe.pipeline = fake  # type: ignore[assignment]
    pipe._streaming_prefetch_count = None  # type: ignore[attr-defined]

    images = [ImageConditioningInput(path="i.png", frame_idx=0, strength=1.0)]
    pipe._run_inference(
        prompt="p",
        seed=2,
        height=128,
        width=128,
        num_frames=9,
        frame_rate=8.0,
        images=images,
        video_conditioning=[("cond.mp4", 1.0)],
        tiling_config=default_tiling_config(),
        # mask_path=None + no hdr_video_context → simple self.pipeline(**kwargs) path
    )

    assert fake.last_kwargs is not None
    forwarded = fake.last_kwargs["images"]
    assert len(forwarded) == 1
    assert forwarded[0].crf == 18
    assert forwarded[0].path == "i.png"
    assert fake.last_kwargs["video_conditioning"] == [("cond.mp4", 1.0)]


# --- Source-text regression guards ---
#
# Ensures the app pipelines never go back to constructing the upstream
# ImageConditioningInput directly (which would silently reintroduce the
# default CRF of 33). The helper is the single chokepoint.


@pytest.mark.parametrize(
    "rel",
    [
        "services/fast_video_pipeline/ltx_fast_video_pipeline.py",
        "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        "services/a2v_pipeline/distilled_a2v_pipeline.py",
    ],
)
def test_pipelines_route_through_crf_helper(rel: str) -> None:
    text = (_BACKEND / rel).read_text(encoding="utf-8")
    assert "_LtxImageInput(" not in text, (
        f"{rel} constructs upstream ImageConditioningInput directly — "
        "must route through make_ltx_image_conditioning_input (centralized CRF)"
    )
    assert "LtxImageInput(" not in text, (
        f"{rel} constructs upstream ImageConditioningInput directly — "
        "must route through make_ltx_image_conditioning_input (centralized CRF)"
    )
    assert "make_ltx_image_conditioning_input(" in text, (
        f"{rel} must use make_ltx_image_conditioning_input for image conditioning"
    )


def test_a2v_tuple_unpacking_pattern_yields_crf_18() -> None:
    """A2V builds image inputs by unpacking 3-tuples (path, frame_idx, strength).

    ``DistilledA2VPipeline.__call__`` receives ``list[tuple[str, int, float]]``
    from ``ltx_a2v_pipeline._run_inference`` and constructs one upstream image
    input per tuple via the helper. Prove that construction pattern produces
    ``crf == 18`` for every tuple. (A full runtime forwarding test is not
    practical here — ``__call__`` is monolithic: it decodes audio and runs two
    denoising stages before/around image conditioning — so the helper-level
    check plus the source-text guard above cover the A2V path.)
    """
    image_tuples: list[tuple[str, int, float]] = [
        ("a.png", 0, 1.0),
        ("b.png", 4, 0.5),
    ]
    ltx_images = [
        make_ltx_image_conditioning_input(path, frame_idx, strength)
        for path, frame_idx, strength in image_tuples
    ]
    assert [i.crf for i in ltx_images] == [18, 18]
    assert [i.path for i in ltx_images] == ["a.png", "b.png"]
    assert [i.frame_idx for i in ltx_images] == [0, 4]


def test_common_module_owns_the_single_override_site() -> None:
    text = (_BACKEND / "services/ltx_pipeline_common.py").read_text(encoding="utf-8")
    assert "IMAGE_CONDITIONING_CRF: int = 18" in text
    # The helper is the ONE place that builds the upstream NamedTuple, and it
    # pins crf to the app constant.
    assert text.count("_LtxImageInput(") == 1
    assert "crf=IMAGE_CONDITIONING_CRF" in text
