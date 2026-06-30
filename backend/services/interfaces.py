"""Compatibility re-exports for service interfaces."""

from __future__ import annotations

from typing import Literal

from services.a2v_pipeline.a2v_pipeline import A2VPipeline
from services.depth_processor_pipeline.depth_processor_pipeline import DepthProcessorPipeline
from services.fast_video_pipeline.fast_video_pipeline import FastVideoPipeline
from services.zit_api_client.zit_api_client import ZitAPIClient
from services.gpu_cleaner.gpu_cleaner import GpuCleaner
from services.gpu_info.gpu_info import GpuInfo, GpuTelemetryPayload
from services.hdr_ic_lora_pipeline.hdr_ic_lora_pipeline import HdrIcLoraPipeline
from services.http_client.http_client import HTTPClient, HttpResponseLike, HttpTimeoutError
from services.ic_lora_pipeline.ic_lora_pipeline import IcLoraPipeline
from services.image_generation_pipeline.image_generation_pipeline import ImageGenerationPipeline
from services.ltx_api_client.ltx_api_client import LTXAPIClient
from services.media_encoder.media_encoder import MediaEncoder
from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.model_downloader.model_downloader import ModelDownloader
from services.pose_processor_pipeline.pose_processor_pipeline import PoseProcessorPipeline
from services.services_utils import JSONScalar, JSONValue
from services.system_info.system_info import SystemInfo, SystemTelemetry
from services.task_runner.task_runner import TaskRunner
from services.text_encoder.text_encoder import TextEncoder
from services.video_processor.video_processor import VideoInfoPayload, VideoProcessor

#: Local video pipeline type. Both ``"fast"`` (distilled) and ``"full"``
#: (dev/full GGUF) families run on the same ``FastVideoPipeline`` service
#: (``pipeline_kind == "fast"``); the cache key (which carries
#: ``model_selection`` + the effective distilled LoRA path) differentiates the
#: two builds. Kept distinct from the API-only ``"pro"`` request model.
VideoPipelineModelType = Literal["fast", "full"]

__all__ = [
    "A2VPipeline",
    "JSONScalar",
    "JSONValue",
    "GpuTelemetryPayload",
    "SystemTelemetry",
    "VideoInfoPayload",
    "HttpTimeoutError",
    "HttpResponseLike",
    "HTTPClient",
    "MediaEncoder",
    "ModelDownloader",
    "GpuCleaner",
    "GpuInfo",
    "SystemInfo",
    "VideoProcessor",
    "DepthProcessorPipeline",
    "PoseProcessorPipeline",
    "TaskRunner",
    "VideoPipelineModelType",
    "FastVideoPipeline",
    "ZitAPIClient",
    "ImageGenerationPipeline",
    "IcLoraPipeline",
    "HdrIcLoraPipeline",
    "LTXAPIClient",
    "RetakePipeline",
    "TextEncoder",
]
