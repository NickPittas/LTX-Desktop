"""HDR IC-LoRA pipeline package.

Dedicated two-stage HDR IC-LoRA pipeline service. Implements the official
LTX-2 / ComfyUI HDR workflow shape (source video/sequence input → HDR IC-LoRA
two-stage → HDR decode postprocess → linear EXR primary + SDR Reinhard proxy),
distinct from the generic ``services.ic_lora_pipeline`` no-input/T2V-capable
path. See ``hdr_ic_lora_pipeline.py`` for the Protocol and
``ltx_hdr_ic_lora_pipeline.py`` for the concrete wrapper.
"""

from __future__ import annotations

__all__: list[str] = []
