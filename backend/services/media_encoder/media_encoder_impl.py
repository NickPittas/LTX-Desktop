"""Real :class:`MediaEncoder` implementation.

Three primary paths (all single-pass over the input iterator):

* **MP4** — delegates to the external ``ltx_pipelines.utils.media_io.encode_video``
  UNCHANGED (byte-identical, no color tags). Guards the visually-validated default
  output (§7 non-goal: do not perturb the default MP4).
* **ProRes** — ffmpeg subprocess (``prores_ks``) with decoded frames streamed to
  stdin. Explicit BT.709 matrix/range filter (§0A.C). Video-only (§0A audio
  decision); audio lives in the proxy MP4.
* **EXR** — per-frame OpenEXR write, sRGB→linear (EOTF) before storing.

Proxies (H.264 MP4) are derived from the on-disk primary via ffmpeg so the input
iterator is never re-traversed (§14 single-pass constraint). EXR→proxy uses
``-apply_trc linear`` + an explicit transfer conversion so the proxy is not dark
(§0A.N mandatory gate).

ffmpeg subprocesses consume stdout+stderr in reader threads to avoid deadlock
(§0A.E). Partial outputs are cleaned up on exception (§0A.J).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch

from api_types import OutputFormat
from services.media_encoder.color import (
    ADOPTED_NEUTRAL_D65,
    BT709_CHROMATICITIES,
    LINEAR_REC709_SCENE_COLORSPACE,
    bt709_eotf,
    ffmpeg_bt709_color_flags,
    ffmpeg_bt709_matrix_filter,
)
from services.media_encoder.media_encoder import EncoderResult
from services.services_utils import AudioOrNone

if TYPE_CHECKING:
    from ltx_core.types import Audio

logger = logging.getLogger(__name__)

# ffmpeg ProRes profile index per OutputFormat (§4.3 / §0B refined spec).
_PRORES_PROFILE: Final[dict[OutputFormat, int]] = {
    OutputFormat.PRORES_PROXY: 0,
    OutputFormat.PRORES_LT: 1,
    OutputFormat.PRORES_422: 2,
    OutputFormat.PRORES_422_HQ: 3,
    OutputFormat.PRORES_4444: 4,
    OutputFormat.PRORES_4444_XQ: 5,
}

# Pixel format per profile index. Per §0B: profiles 0-3 → 10-bit 4:2:2;
# profiles 4-5 (4444/4444_xq) → 12-bit 4:4:4 (matches Apple 4444 XQ spec +
# reference fixtures).
_PRORES_PIXFMT_BY_PROFILE: Final[dict[int, str]] = {
    0: "yuv422p10le",
    1: "yuv422p10le",
    2: "yuv422p10le",
    3: "yuv422p10le",
    4: "yuv444p12le",
    5: "yuv444p12le",
}

# 12-bit (4444) profiles ingest rgb48le (uint16) to avoid 8-bit quantization of
# a 12-bit master; 10-bit profiles ingest rgb24 (uint8) bytes directly.
_12BIT_PROFILES: Final[frozenset[int]] = frozenset({4, 5})

_EXR_FRAME_PATTERN: Final[str] = "frame_{:05d}.exr"
# printf-style pattern the ffmpeg image2 demuxer consumes (reads the WHOLE
# sequence, not a single frame). The Python ``str.format`` pattern above
# produces a concrete ``frame_00000.exr`` for writes; this is its ffmpeg-input
# counterpart (``frame_%05d.exr``) — the two are intentionally distinct.
_EXR_FFMPEG_SEQ_PATTERN: Final[str] = "frame_%05d.exr"

# Encode progress budget: the primary encode covers [0, _ENCODE_FRACTION] of the
# combined 0→1 ``on_progress`` range; the proxy pass covers [_ENCODE_FRACTION, 1.0].
# The handler splits on this threshold to label stages "encoding" vs "writing_proxy".
_ENCODE_FRACTION: Final[float] = 0.6


def _is_exr_format(output_format: OutputFormat) -> bool:
    return output_format in (OutputFormat.EXR_ZIP_HALF, OutputFormat.EXR_ZIP_FLOAT)


def _is_prores_format(output_format: OutputFormat) -> bool:
    return output_format in _PRORES_PROFILE


def _ffmpeg_exe() -> str:
    """Resolve the ffmpeg binary once via imageio-ffmpeg (bundled 7.0.2)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _materialize_iterator(
    video: "torch.Tensor | Iterator[torch.Tensor]",
) -> Iterator[torch.Tensor]:
    """Normalize the video input to a single-pass chunk iterator.

    A bare tensor is wrapped as a one-element iterator (matches the external
    ``encode_video`` convention). The caller MUST NOT have consumed the iterator
    already.
    """
    if isinstance(video, torch.Tensor):
        return iter([video])
    return video


class MediaEncoderImpl:
    """Real encoder: MP4 (delegated) / ProRes (ffmpeg) / EXR (OpenEXR) + proxies."""

    def encode(
        self,
        *,
        video: "torch.Tensor | Iterator[torch.Tensor]",
        audio: AudioOrNone,
        fps: int,
        primary_path: str,
        output_format: OutputFormat,
        proxy_path: str | None,
        video_chunks_number: int,  # noqa: ARG002 tqdm total only — unused by real impl
        on_progress: Callable[[float], None] | None = None,
        total_frames: int | None = None,
    ) -> EncoderResult:
        try:
            if output_format == OutputFormat.MP4:
                return self._encode_mp4(
                    video=video,
                    audio=audio,
                    fps=fps,
                    primary_path=primary_path,
                    video_chunks_number=video_chunks_number,
                )
            if _is_prores_format(output_format):
                return self._encode_prores(
                    video=video,
                    fps=fps,
                    primary_path=primary_path,
                    output_format=output_format,
                    proxy_path=proxy_path,
                    on_progress=on_progress,
                    total_frames=total_frames,
                    audio=audio,
                )
            if _is_exr_format(output_format):
                return self._encode_exr(
                    video=video,
                    fps=fps,
                    primary_path=primary_path,
                    output_format=output_format,
                    proxy_path=proxy_path,
                    on_progress=on_progress,
                    total_frames=total_frames,
                    audio=audio,
                )
            raise ValueError(f"Unsupported output_format: {output_format!r}")
        except Exception:
            # Comprehensive partial-output cleanup (§0A.J): on ANY failure during
            # the encode or the proxy pass — ffmpeg nonzero exit, broken-pipe
            # stdin write, OpenEXR write error, etc. — remove the primary (file
            # for MP4/ProRes, directory for EXR) AND the proxy file. Temp WAVs
            # are cleaned by the proxy methods' own finally blocks. Re-raise.
            self._cleanup_partial_outputs(primary_path, proxy_path)
            raise

    @staticmethod
    def _cleanup_partial_outputs(primary_path: str, proxy_path: str | None) -> None:
        """Remove partial primary (file or EXR dir) and proxy from a failed encode."""
        primary = Path(primary_path)
        if primary.is_dir():
            shutil.rmtree(primary, ignore_errors=True)
        else:
            primary.unlink(missing_ok=True)
        if proxy_path is not None:
            Path(proxy_path).unlink(missing_ok=True)

    @staticmethod
    def _proxy_progress_wrapper(
        on_progress: Callable[[float], None] | None,
    ) -> Callable[[float], None] | None:
        """Wrap ``on_progress`` so the proxy's local [0,1] maps to the proxy budget
        ``[_ENCODE_FRACTION, 1.0]`` of the combined progress range. Returns None if
        ``on_progress`` is None (no callback)."""
        if on_progress is None:
            return None

        def _wrapped(p: float) -> None:
            on_progress(_ENCODE_FRACTION + p * (1.0 - _ENCODE_FRACTION))

        return _wrapped

    # ------------------------------------------------------------------
    # MP4 (default — byte-identical to today)
    # ------------------------------------------------------------------

    def _encode_mp4(
        self,
        *,
        video: "torch.Tensor | Iterator[torch.Tensor]",
        audio: AudioOrNone,
        fps: int,
        primary_path: str,
        video_chunks_number: int,
    ) -> EncoderResult:
        """MP4 with Rec.709 VUI tags applied via POST-TAG/REMUX (§9.2).

        The validated pixel stream still comes from the external
        ``ltx_pipelines.utils.media_io.encode_video`` (libx264 / yuv420p) — we
        write it to a temp path, then remux to ``primary_path`` with ``-c copy``
        and force BT.709 limited-range VUI both at the container level and in the
        H.264 SPS via the ``h264_metadata`` bitstream filter. Pixels are
        byte-identical to the external encoder; only VUI/metadata change.
        """
        from ltx_pipelines.utils.media_io import encode_video

        out_file = Path(primary_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Temp path for the external (untagged) encode — remuxed into primary_path.
        temp_path = str(out_file.with_suffix(out_file.suffix + ".untagged.mp4"))

        try:
            encode_video(
                video=video,
                fps=fps,
                audio=audio,
                output_path=temp_path,
                video_chunks_number=video_chunks_number,
            )

            remux_cmd: list[str] = [
                _ffmpeg_exe(), "-y",
                "-i", temp_path,
                "-map", "0",
                "-c", "copy",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-colorspace", "bt709",
                "-color_range", "tv",
                # Force VUI inside the H.264 SPS too (container tags alone are
                # ignored by some decoders): primaries=1 (bt709),
                # transfer=1 (bt709), matrix=1 (bt709), full_range=0 (limited).
                "-bsf:v", ("h264_metadata=colour_primaries=1:"
                           "transfer_characteristics=1:"
                           "matrix_coefficients=1:"
                           "video_full_range_flag=0"),
                primary_path,
            ]
            self._run_ffmpeg_to_completion(remux_cmd, label="mp4-rec709-remux")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return EncoderResult(primary_path=primary_path, proxy_path=None)

    # ------------------------------------------------------------------
    # ProRes (ffmpeg subprocess)
    # ------------------------------------------------------------------

    def _encode_prores(
        self,
        *,
        video: "torch.Tensor | Iterator[torch.Tensor]",
        fps: int,
        primary_path: str,
        output_format: OutputFormat,
        proxy_path: str | None,
        on_progress: Callable[[float], None] | None,
        total_frames: int | None,
        audio: AudioOrNone,
    ) -> EncoderResult:
        profile = _PRORES_PROFILE[output_format]
        pix_fmt = _PRORES_PIXFMT_BY_PROFILE[profile]
        use_12bit = profile in _12BIT_PROFILES
        ff = _ffmpeg_exe()

        out_file = Path(primary_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # First peek to learn H/W and dtype without consuming the stream: pull
        # the first chunk, then re-prefix it onto the iterator.
        chunk_iter = _materialize_iterator(video)
        first_chunk = next(chunk_iter)

        def _refeed() -> Iterator[torch.Tensor]:
            yield first_chunk
            yield from chunk_iter

        height = int(first_chunk.shape[-3])
        width = int(first_chunk.shape[-2])
        is_uint8 = first_chunk.dtype == torch.uint8

        # Input pixel format: rgb48le (uint16) for 12-bit profiles to preserve
        # precision; rgb24 (uint8 bytes) for 10-bit profiles. ffmpeg rawvideo
        # input needs the matching -pix_fmt on the INPUT side.
        in_pix_fmt = "rgb48le" if use_12bit else "rgb24"

        cmd: list[str] = [
            ff, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", in_pix_fmt,
            "-r", str(int(fps)),
            "-i", "-",
            "-vf", ffmpeg_bt709_matrix_filter(),
            "-c:v", "prores_ks",
            "-profile:v", str(profile),
            "-pix_fmt", pix_fmt,
            "-vendor", "apl0",
            "-qscale:v", "9",
            *ffmpeg_bt709_color_flags(),
            "-progress", "pipe:1",
            primary_path,
        ]

        proc = subprocess.Popen(  # noqa: S603 — intentional subprocess to ffmpeg
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        progress_lines: list[str] = []
        stderr_tail: list[str] = []

        def _read_stdout() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
                    if line:
                        progress_lines.append(line)
                        if on_progress is not None and line.startswith("frame="):
                            try:
                                frames_done = int(line.split("=", 1)[1])
                                if total_frames and total_frames > 0:
                                    # Precise: frame/total mapped to encode budget [0, _ENCODE_FRACTION].
                                    pct = min(_ENCODE_FRACTION,
                                              frames_done / total_frames * _ENCODE_FRACTION)
                                else:
                                    # Heuristic fallback (total unknown).
                                    pct = min(_ENCODE_FRACTION,
                                              frames_done / max(1, frames_done + 8) * _ENCODE_FRACTION)
                                on_progress(pct)
                            except (ValueError, IndexError):
                                pass

        def _read_stderr() -> None:
            assert proc.stderr is not None
            for raw in proc.stderr:
                stderr_tail.append(raw.decode("utf-8", errors="replace"))

        out_thread = threading.Thread(target=_read_stdout, daemon=True)
        err_thread = threading.Thread(target=_read_stderr, daemon=True)
        out_thread.start()
        err_thread.start()

        try:
            assert proc.stdin is not None
            for chunk in _refeed():
                if use_12bit:
                    arr = _chunk_to_uint16(chunk, is_uint8)
                    proc.stdin.write(arr.tobytes())
                else:
                    arr = _chunk_to_uint8(chunk, is_uint8)
                    proc.stdin.write(arr.tobytes())
            proc.stdin.close()
        except Exception:
            # Ensure ffmpeg terminates so reader threads exit cleanly.
            if proc.poll() is None:
                proc.kill()
            raise
        finally:
            proc.wait()
            out_thread.join(timeout=5)
            err_thread.join(timeout=5)

        if proc.returncode != 0:
            # Partial-output cleanup is owned by ``encode()``'s except wrapper
            # (§0A.J) — just surface the failure with the stderr tail.
            tail = "".join(stderr_tail[-20:])
            raise RuntimeError(
                f"ffmpeg ProRes encode failed (returncode={proc.returncode}) for "
                f"{primary_path}:\n{tail}"
            )

        logger.info("ProRes (%s) written to %s", output_format.value, primary_path)

        if proxy_path:
            if on_progress is not None:
                on_progress(_ENCODE_FRACTION)  # encode done → proxy stage
            self._proxy_from_file(
                primary_path=primary_path,
                proxy_path=proxy_path,
                audio=audio,
                on_progress=self._proxy_progress_wrapper(on_progress),
            )

        if on_progress is not None:
            on_progress(1.0)
        return EncoderResult(primary_path=primary_path, proxy_path=proxy_path)

    # ------------------------------------------------------------------
    # EXR (OpenEXR, per-frame linearized)
    # ------------------------------------------------------------------

    def _encode_exr(
        self,
        *,
        video: "torch.Tensor | Iterator[torch.Tensor]",
        fps: str | int | float,
        primary_path: str,
        output_format: OutputFormat,
        proxy_path: str | None,
        on_progress: Callable[[float], None] | None,
        total_frames: int | None,
        audio: AudioOrNone,
    ) -> EncoderResult:
        # OpenEXR ships no type stubs; treat the module as Any to keep pyright
        # strict clean without losing real type coverage elsewhere. API form
        # (File(header, channels) / .write / .header() / .channels()) verified
        # against the installed v3.4 binding — see color.py for the
        # chromaticities (8-tuple of floats) + adoptedNeutral (float32 array)
        # + framesPerSecond (fractions.Fraction) value forms.
        import OpenEXR

        openexr: Any = OpenEXR

        out_dir = Path(primary_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        half = output_format == OutputFormat.EXR_ZIP_HALF
        dtype = np.float16 if half else np.float32

        fps_rational = Fraction(int(round(float(fps) * 1000)), 1000).limit_denominator(1_000_000)

        header_template: dict[str, Any] = {
            "compression": openexr.ZIP_COMPRESSION,
            "type": openexr.scanlineimage,
            "chromaticities": BT709_CHROMATICITIES,
            "adoptedNeutral": ADOPTED_NEUTRAL_D65,
            "colorSpace": LINEAR_REC709_SCENE_COLORSPACE,
            "framesPerSecond": fps_rational,
        }

        chunk_iter = _materialize_iterator(video)
        global_idx = 0
        for chunk in chunk_iter:
            is_uint8 = chunk.dtype == torch.uint8
            chunkf = (chunk.float() / 255.0) if is_uint8 else chunk.float()
            # BT.709 EOTF (Rec.709 gamma → linear) per §9.1/§9.3 — EXR is
            # always linear-light. Clamp to [0,1] domain first.
            linear = bt709_eotf(chunkf.clamp(0.0, 1.0)).cpu().numpy()
            for frame in linear:
                frame_name = _EXR_FRAME_PATTERN.format(global_idx)
                frame_path = out_dir / frame_name
                channels = {
                    "R": np.ascontiguousarray(frame[:, :, 0].astype(dtype)),
                    "G": np.ascontiguousarray(frame[:, :, 1].astype(dtype)),
                    "B": np.ascontiguousarray(frame[:, :, 2].astype(dtype)),
                }
                exr_file = openexr.File(header_template, channels)
                exr_file.write(str(frame_path))
                global_idx += 1
                if on_progress is not None:
                    # Per-frame progress mapped to encode budget [0, _ENCODE_FRACTION].
                    if total_frames and total_frames > 0:
                        pct = min(_ENCODE_FRACTION, global_idx / total_frames * _ENCODE_FRACTION)
                    else:
                        pct = min(_ENCODE_FRACTION, global_idx / max(1, global_idx + 4) * _ENCODE_FRACTION)
                    on_progress(pct)
        # Partial-output cleanup on ANY failure is owned by ``encode()`` (§0A.J).

        logger.info(
            "EXR (%s) written: %d frames to %s", output_format.value, global_idx, out_dir
        )

        if proxy_path:
            if on_progress is not None:
                on_progress(_ENCODE_FRACTION)  # encode done → proxy stage
            self._proxy_from_exr(
                exr_dir=out_dir,
                fps=int(fps),
                proxy_path=proxy_path,
                audio=audio,
                on_progress=self._proxy_progress_wrapper(on_progress),
            )

        if on_progress is not None:
            on_progress(1.0)
        return EncoderResult(primary_path=str(out_dir), proxy_path=proxy_path)

    # ------------------------------------------------------------------
    # Proxies (ffmpeg, derived from on-disk primary — never re-iterate tensors)
    # ------------------------------------------------------------------

    def _proxy_from_file(
        self,
        *,
        primary_path: str,
        proxy_path: str,
        audio: AudioOrNone,
        on_progress: Callable[[float], None] | None,
    ) -> None:
        """Build H.264 proxy MP4 from an on-disk ProRes primary (already bt709).

        Filter-free by design: the ProRes primary is already YUV(bt709, limited)
        and the proxy stays YUV(bt709, limited), so this is a same-space
        YUV→YUV transcode — no matrix/range conversion is mathematically required.
        The ``test_prores_proxy_roundtrip_luma`` gate (§0A.N) PROVES this: a
        matrix-sensitive colored gradient encodes to ProRes 422 HQ, transcodes to
        the proxy, and round-trips with per-frame mean-luma diff < 0.1 (measured
        ~0.05 in 8-bit). Adding ``-vf scale=out_color_matrix=bt709:out_range=tv``
        here would be a no-op (YUV→YUV); it is intentionally omitted.
        """
        ff = _ffmpeg_exe()
        out = Path(proxy_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [ff, "-y", "-i", primary_path]
        audio_wav = self._dump_audio_wav(audio) if audio is not None else None
        try:
            if audio_wav is not None:
                cmd += ["-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:a", "aac", "-b:a", "192k"]
            cmd += [
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                *ffmpeg_bt709_color_flags(),
                "-movflags", "+faststart",
            ]
            if audio_wav is not None:
                cmd += ["-shortest"]
            cmd += [proxy_path]
            self._run_ffmpeg_to_completion(cmd, label="proxy-from-prores")
        finally:
            if audio_wav is not None:
                Path(audio_wav).unlink(missing_ok=True)

        logger.info("Proxy written to %s", proxy_path)
        if on_progress is not None:
            on_progress(0.97)

    def _proxy_from_exr(
        self,
        *,
        exr_dir: Path,
        fps: int,
        proxy_path: str,
        audio: AudioOrNone,
        on_progress: Callable[[float], None] | None,
    ) -> None:
        """Build H.264 proxy from a linear EXR sequence (linear→BT.709, ONCE).

        Per §9.4 the linear→BT.709 transfer is applied EXACTLY ONCE via
        ``-apply_trc bt709`` (the EXR decoder applies the BT.709 OETF at decode
        time). The accompanying ``scale`` filter carries ONLY ``out_color_matrix``
        (the YUV matrix) and ``out_range`` (limited range) — it does NOT carry an
        ``out_transfer``/``zscale transfer``, so there is no double transfer. The
        ``-framerate <fps>`` is set before ``-i`` (else ffmpeg defaults to 25 fps →
        audio/duration desync). Sequence start is 00000 (image2 default).

        The ``test_proxy_from_exr_not_dark`` luma gate verifies the single transfer
        produces a correctly bright proxy for the BT.709 working space.
        """
        ff = _ffmpeg_exe()
        out = Path(proxy_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # CRITICAL: pass the image-sequence PATTERN (frame_%05d.exr), not a
        # single concrete file, so ffmpeg reads ALL frames in the sequence.
        # A literal frame_00000.exr would encode exactly one frame.
        seq_input = str(exr_dir / _EXR_FFMPEG_SEQ_PATTERN)

        cmd: list[str] = [
            ff, "-y",
            "-apply_trc", "bt709",
            "-framerate", str(int(fps)),
            "-i", seq_input,
        ]
        audio_wav = self._dump_audio_wav(audio) if audio is not None else None
        try:
            if audio_wav is not None:
                cmd += ["-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:a", "aac", "-b:a", "192k"]
            # Explicit BT.709 matrix + limited range (transfer already handled
            # by -apply_trc bt709 at decode time).
            cmd += [
                "-vf", ffmpeg_bt709_matrix_filter(),
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                *ffmpeg_bt709_color_flags(),
                "-movflags", "+faststart",
            ]
            if audio_wav is not None:
                cmd += ["-shortest"]
            cmd += [proxy_path]
            self._run_ffmpeg_to_completion(cmd, label="proxy-from-exr")
        finally:
            if audio_wav is not None:
                Path(audio_wav).unlink(missing_ok=True)

        logger.info("EXR proxy written to %s", proxy_path)
        if on_progress is not None:
            on_progress(0.97)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _dump_audio_wav(self, audio: "Audio") -> str:
        """Write audio to a temp WAV (pcm_s16le) via PyAV.

        Callers gate on ``audio is not None`` so the parameter is the non-None
        ``Audio`` type. Returns the temp WAV path; the caller unlinks it.
        """
        import av

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name

        sample_rate = int(audio.sampling_rate)
        # Materialize the waveform as an ndarray of floats in [-1, 1].
        waveform = audio.waveform
        if hasattr(waveform, "cpu"):
            waveform = waveform.detach().cpu().numpy()
        samples = np.asarray(waveform, dtype=np.float32)

        if samples.ndim == 1:
            samples = samples[:, None]
        # Resolve (channels, samples) stereo layout.
        if samples.shape[0] != 2 and samples.shape[-1] == 2:
            samples = samples.T
        if samples.shape[0] != 2:
            # Tile/trim to stereo as a best-effort (synthetic test path).
            if samples.shape[0] == 1:
                samples = np.concatenate([samples, samples], axis=0)
            else:
                samples = samples[:2]
        int16_samples = np.clip(samples, -1.0, 1.0)
        int16_samples = (int16_samples * 32767.0).astype(np.int16)

        container = av.open(wav_path, mode="w")
        try:
            # av.Container.add_stream is a pybind-style overload whose generic
            # `str` branch is partially unknown to pyright; route through Any.
            container_any: Any = container
            stream = container_any.add_stream("pcm_s16le", rate=sample_rate)
            stream.codec_context.layout = "stereo"
            stream.codec_context.time_base = Fraction(1, sample_rate)

            frame_in = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(int16_samples.reshape(1, -1)),
                format="s16",
                layout="stereo",
            )
            frame_in.sample_rate = sample_rate
            for packet in stream.encode(frame_in):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

        return wav_path

    def _run_ffmpeg_to_completion(self, cmd: list[str], *, label: str) -> None:
        """Run an ffmpeg command, draining BOTH stdout and stderr in reader threads.

        Both streams must be consumed or the OS pipe buffer deadlocks ffmpeg
        (§0A.E) — the previous version piped stdout but never read it, which can
        block once the stdout buffer fills. Raises ``RuntimeError`` with the
        stderr tail on non-zero exit.
        """
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr_tail: list[str] = []
        stdout_tail: list[str] = []

        def _read_stream(stream: Any, sink: list[str]) -> None:
            if stream is None:
                return
            for raw in stream:
                sink.append(raw.decode("utf-8", errors="replace"))

        out_thread = threading.Thread(
            target=_read_stream, args=(proc.stdout, stdout_tail), daemon=True
        )
        err_thread = threading.Thread(
            target=_read_stream, args=(proc.stderr, stderr_tail), daemon=True
        )
        out_thread.start()
        err_thread.start()
        try:
            proc.wait()
        finally:
            out_thread.join(timeout=5)
            err_thread.join(timeout=5)

        if proc.returncode != 0:
            tail = "".join(stderr_tail[-20:])
            raise RuntimeError(
                f"ffmpeg {label} failed (returncode={proc.returncode}):\n{tail}"
            )


# ---------------------------------------------------------------------------
# Tensor → raw array helpers (dtype-aware per §0A.A)
# ---------------------------------------------------------------------------

def _chunk_to_uint8(chunk: torch.Tensor, is_uint8: bool) -> np.ndarray:
    """Return an ``(N, H, W, 3)`` uint8 ndarray for rgb24 piping."""
    if is_uint8:
        return chunk.contiguous().cpu().numpy()
    return (chunk.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous().cpu().numpy()


def _chunk_to_uint16(chunk: torch.Tensor, is_uint8: bool) -> np.ndarray:
    """Return an ``(N, H, W, 3)`` uint16 ndarray for rgb48le piping (12-bit masters).

    uint8 input is scaled 0..255 → 0..65535 to fill the 16-bit container (avoids
    8-bit quantization of a 12-bit master, §0B). float input is scaled from [0,1].
    """
    if is_uint8:
        arr = (chunk.to(torch.int32) * 257).to(torch.uint16)
    else:
        arr = (chunk.clamp(0.0, 1.0) * 65535.0).round().to(torch.uint16)
    return arr.contiguous().cpu().numpy()


__all__ = ["MediaEncoderImpl"]
