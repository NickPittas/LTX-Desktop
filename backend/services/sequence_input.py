"""Image-sequence input resolution + decode (CM-1b rework).

Authoritative architecture (per VFX supervisor): ``video_path`` is a SINGLE FILE
from a sequence (e.g. ``…/EXR/Instant_Share_Beard_0001.exr``), NEVER a directory.
The sequence is detected from the filename's trailing digit-run; everything else
(prefix, suffix, extension — INCLUDING version segments like ``_v001``/``v01``/
``V1``) stays LITERAL so co-existing sequences in the same folder never bleed.

This module is pure resolution + decode. It does NOT touch the model and does NO
network I/O. Decoded frames are color-transformed to the model working space
(Rec.709 gamma) via :mod:`services.color_management`, matching the yield
shape/dtype/device of ``ltx_pipelines.utils.media_io.decode_video_by_frame``
(``(1, H, W, 3)`` uint8) so the downstream ``video_preprocess`` path is unchanged.

Sequence frames reach the model through the SAME path the model uses for video:
``decode_video_by_frame`` is monkey-patched (see
:mod:`services.patches.sequence_decode_patch`) so a sequence-file path yields the
decoded+color-transformed frames and any normal video file falls through to the
ORIGINAL function (byte-identical). No temp-MP4 detour, no directory path.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from services.color_management import color_to_model_space, detect_colorspace
from services.exr_input import decode_exr_image

logger = logging.getLogger(__name__)

# Image-sequence extensions. Video containers (.mov/.mp4/…) and unnumbered files
# return False from is_sequence_file WITHOUT any directory scan.
_SEQUENCE_EXTS: frozenset[str] = frozenset(
    {".exr", ".dpx", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
)

# The LAST digit-run in the stem is the frame number; only non-digits may follow
# it up to the end of the stem. Everything before/after stays LITERAL.
_TRAILING_DIGITS_RE = re.compile(r"(\d+)(?=[^\d]*$)")

_DEFAULT_FPS: float = 24.0


@dataclass(frozen=True)
class SequenceSpec:
    """A resolved image sequence.

    ``frame_numbers`` and ``files`` are sorted ascending by frame number and
    cover every present frame in the half-open range [min, max] that exists on
    disk (gaps are preserved — no contiguity assumption).
    """

    dir: str
    prefix: str
    suffix_stem: str
    ext: str  # includes the dot, e.g. ".exr" (original case preserved)
    pad: int
    frame_numbers: tuple[int, ...]
    files: tuple[str, ...]


# ---------------------------------------------------------------------------
# Predicates — PURE-STRING fast path (zero directory I/O)
# ---------------------------------------------------------------------------

def is_sequence_file(path: str) -> bool:
    """Cheap, pure-string check: could ``path`` be a file in an image sequence?

    True iff the extension is an image-sequence format (``.exr/.dpx/.tif/.tiff/
    .png/.jpg/.jpeg``) AND the stem ends with a digit-run. Performs NO directory
    scan — a single image (``img_0001.png`` with no siblings) returns True here
    but :func:`resolve_sequence` returns None for it. ``.mov``/``.mp4`` and any
    unnumbered file return False instantly (identity invariant preserved).
    """
    p = Path(path)
    if p.suffix.lower() not in _SEQUENCE_EXTS:
        return False
    return _TRAILING_DIGITS_RE.search(p.stem) is not None


# ---------------------------------------------------------------------------
# Resolution — strict sibling match (version segments preserved LITERAL)
# ---------------------------------------------------------------------------

def resolve_sequence(path: str) -> SequenceSpec | None:
    """Resolve the sequence that ``path`` belongs to, or None if it is standalone.

    The last digit-run in the stem is the frame number; ``prefix``/``suffix_stem``/
    ``ext`` are matched LITERALLY against siblings so version segments
    (``_v001``/``v01``/``V1``) and case differences isolate co-existing sequences
    in the same folder. Returns None when the path isn't a sequence file OR when
    the only matching sibling is the input file itself (a single image, not a
    sequence). The result covers every present frame (gaps tolerated).
    """
    if not is_sequence_file(path):
        return None

    p = Path(path)
    stem = p.stem
    ext = p.suffix  # original case preserved; matched literally
    directory = p.parent

    match = _TRAILING_DIGITS_RE.search(stem)
    if match is None:
        return None  # is_sequence_file already guarantees a run; defensive

    run = match.group(1)
    start, end = match.start(), match.end()
    pad = len(run)
    prefix = stem[:start]
    suffix_stem = stem[end:]

    # Strict sibling regex: prefix + EXACTLY pad digits + suffix_stem + ext.
    # prefix/suffix_stem/ext are re.escape'd (LITERAL); only the digit run varies.
    sibling_re = re.compile(
        r"^" + re.escape(prefix) + r"(\d{" + str(pad) + r"})"
        + re.escape(suffix_stem) + re.escape(ext) + r"$"
    )

    frame_to_file: dict[int, str] = {}
    try:
        children = list(directory.iterdir())
    except (OSError, FileNotFoundError):
        return None

    for child in children:
        if not child.is_file():
            continue
        sibling_match = sibling_re.match(child.name)
        if sibling_match is None:
            continue
        frame_num = int(sibling_match.group(1))
        frame_to_file[frame_num] = str(child)

    if len(frame_to_file) <= 1:
        # Only the input file itself (or nothing matched) → not a sequence.
        return None

    frame_numbers = tuple(sorted(frame_to_file))
    files = tuple(frame_to_file[num] for num in frame_numbers)
    return SequenceSpec(
        dir=str(directory),
        prefix=prefix,
        suffix_stem=suffix_stem,
        ext=ext,
        pad=pad,
        frame_numbers=frame_numbers,
        files=files,
    )


# ---------------------------------------------------------------------------
# Directory-based sequences — transparent fallback for system-generated EXR assets
# ---------------------------------------------------------------------------
#
# When a generated EXR asset (a DIRECTORY like ``..._exr/`` containing
# ``frame_00000.exr…frame_00120.exr``) is re-used as input to another workflow,
# the backend receives the DIR path. The single-file ``is_sequence_file`` /
# ``resolve_sequence`` return False/None for a directory, so the decode path
# would fall through to PyAV → ``av.open(dir)`` → ``[Errno 21] Is a directory``.
#
# These helpers provide a transparent fallback: given a DIRECTORY, they resolve
# the dominant image sequence from the files INSIDE it. The user-facing input
# mechanism stays single-file (user picks one frame); this fallback only kicks
# in when the backend receives a directory (a system-generated EXR asset).
#
# Identity invariant preserved: ``is_sequence_dir`` is ``os.path.isdir`` first,
# which is False for files → instant, no I/O beyond one stat → non-dir inputs
# never enter this branch.


def _classify_seq_name(name: str) -> tuple[str, int, str, str] | None:
    """Classify a dir entry as a sequence frame: ``(prefix, pad, suffix_stem, ext)``.

    Returns None when ``name`` is not an image-sequence file (ext not in
    :data:`_SEQUENCE_EXTS` OR no trailing digit-run in the stem). ``ext`` keeps
    its original case (matched literally downstream). Uses the same trailing
    digit-run rule as :func:`resolve_sequence`, so the patterns line up.
    """
    p = Path(name)
    ext = p.suffix
    if ext.lower() not in _SEQUENCE_EXTS:
        return None
    stem = p.stem
    match = _TRAILING_DIGITS_RE.search(stem)
    if match is None:
        return None
    return stem[: match.start()], len(match.group(1)), stem[match.end() :], ext


def is_sequence_dir(path: str) -> bool:
    """Cheap check: is ``path`` a directory containing ≥2 image-sequence files?

    Non-dir paths return False instantly (a single ``os.path.isdir`` stat — no
    listing). For an actual directory, a SINGLE listing is taken and the helper
    returns True iff at least 2 entries classify as image-sequence files
    (``.exr/.dpx/.tif/.tiff/.png/.jpg/.jpeg`` + trailing digit-run in the stem).
    A single-image dir, an empty dir, a video file, and a missing path all
    return False.
    """
    if not os.path.isdir(path):
        return False
    try:
        children = os.listdir(path)
    except (OSError, FileNotFoundError):
        return False
    count = 0
    for name in children:
        if _classify_seq_name(name) is not None:
            count += 1
            if count >= 2:
                return True
    return False


def resolve_sequence_from_dir(dir_path: str) -> SequenceSpec | None:
    """Resolve the DOMINANT image sequence inside ``dir_path``, or None.

    Groups every dir entry by its literal ``(prefix, pad, suffix_stem, ext)``
    pattern (the same strict trailing-digit + literal-affix rule as
    :func:`resolve_sequence`, applied to the dir's files rather than derived
    from a single file's stem). The dominant pattern is the one with the MOST
    files; ties are broken by the lexicographically smallest pattern tuple
    (deterministic). Returns None when no pattern has ≥2 files (a single
    image, an empty dir, or a dir with no image-sequence files). Gaps in frame
    numbers are preserved (no contiguity assumption) — the result covers every
    present frame in the half-open range, sorted ascending.
    """
    try:
        children = os.listdir(dir_path)
    except (OSError, FileNotFoundError):
        return None

    # Pattern → {frame_number: full_path}. Insertion order is dict order; the
    # tie-break sort below makes the final pick deterministic regardless.
    groups: dict[tuple[str, int, str, str], dict[int, str]] = {}
    for name in children:
        key = _classify_seq_name(name)
        if key is None:
            continue
        # Re-extract the frame number with the same regex _classify_seq_name used
        # (the strict sibling pattern is implicit: the trailing digit run of the
        # SAME file we just classified is by construction exactly pad wide).
        match = _TRAILING_DIGITS_RE.search(Path(name).stem)
        if match is None:
            continue  # defensive — _classify_seq_name already guaranteed a run
        frame_num = int(match.group(1))
        groups.setdefault(key, {})[frame_num] = os.path.join(dir_path, name)

    # Dominant pattern: most files; ties → lexicographically smallest key tuple.
    candidates = [(k, frames) for k, frames in groups.items() if len(frames) >= 2]
    if not candidates:
        return None
    candidates.sort(key=lambda kf: (-len(kf[1]),) + tuple(str(part) for part in kf[0]))
    (prefix, pad, suffix_stem, ext), frame_map = candidates[0]

    frame_numbers = tuple(sorted(frame_map))
    files = tuple(frame_map[num] for num in frame_numbers)
    return SequenceSpec(
        dir=dir_path,
        prefix=prefix,
        suffix_stem=suffix_stem,
        ext=ext,
        pad=pad,
        frame_numbers=frame_numbers,
        files=files,
    )


# ---------------------------------------------------------------------------
# Decode — one frame at a time, transferred to Rec.709 (model domain)
# ---------------------------------------------------------------------------

def _to_uint8_model_domain(src: np.ndarray, src_path: str) -> np.ndarray:
    """Transfer ``src`` (float [0,1] linear OR uint8 [0,255] gamma) → uint8 Rec.709.

    Detects the frame's colorspace and applies ``color_to_model_space``
    (input CS → Rec.709 gamma). For Rec.709/sRGB-domain inputs this is identity;
    for linear EXR it applies the BT.709 OETF. Clips to [0,1] and re-quantizes to
    uint8 — the same domain ``decode_video_by_frame`` yields for video frames.
    """
    cs = detect_colorspace(src_path)
    if src.dtype == np.uint8:
        src = src.astype(np.float32) / 255.0
    transferred = color_to_model_space(src, cs)
    if isinstance(transferred, torch.Tensor):
        transferred = transferred.cpu().numpy()
    arr = np.clip(np.asarray(transferred, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def _decode_non_exr_frame(path: str) -> np.ndarray:
    """Decode a single non-EXR image frame → ``(H, W, 3)`` uint8 RGB.

    Uses cv2 (BGR → RGB). Raises ValueError if the file cannot be decoded.
    """
    import cv2  # noqa: PLC0415 — optional import keeps module import cheap

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode sequence frame: {path}")
    # BGR → RGB; contiguous copy so torch.as_tensor gets a clean buffer.
    return np.ascontiguousarray(bgr[:, :, ::-1])


def decode_sequence_frames(
    spec: SequenceSpec,
    frame_cap: int | None = None,
    device: Any = None,
) -> Iterator[torch.Tensor]:
    """Yield ``(1, H, W, 3)`` uint8 Rec.709-gamma tensors for each frame in order.

    Mirrors ``decode_video_by_frame``'s yield shape/dtype/device so downstream
    ``video_preprocess`` + ``normalize_latent`` is unchanged. EXR frames decode
    via OpenEXR (linear float); other formats via cv2 (uint8). Each frame is
    transferred to the Rec.709 model domain via ``color_to_model_space``.
    ``frame_cap`` caps the count (mirrors ``decode_video_by_frame``).
    """
    if device is None:
        device = torch.device("cpu")

    files = spec.files
    if frame_cap is not None:
        files = files[: max(0, int(frame_cap))]

    for frame_file in files:
        if frame_file.lower().endswith(".exr"):
            linear = decode_exr_image(frame_file)  # (H, W, 3) float32 linear
            uint8 = _to_uint8_model_domain(linear, frame_file)
        else:
            rgb = _decode_non_exr_frame(frame_file)  # (H, W, 3) uint8 RGB
            uint8 = _to_uint8_model_domain(rgb, frame_file)
        yield torch.as_tensor(uint8, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Metadata — dims / count / fps for a sequence (avoids av.open on EXR/PNG)
# ---------------------------------------------------------------------------

def _exr_fps(path: str) -> float:
    """Read the EXR ``framesPerSecond`` header, falling back to 24.0.

    OpenEXR returns the value as a 2-element array ``[numerator, denominator]``;
    older writers may use a scalar. Anything malformed → 24.0 (sequence default).
    """
    try:
        import OpenEXR  # noqa: PLC0415

        openexr: Any = OpenEXR
        header: Any = openexr.File(path, header_only=True).header()
        fps_header = header.get("framesPerSecond")
        if fps_header is None:
            return _DEFAULT_FPS
        arr = np.asarray(fps_header)
        if arr.size >= 2 and int(arr[1]) != 0:
            return float(int(arr[0])) / float(int(arr[1]))
        if arr.size == 1:
            return float(int(arr[0]))
        return float(fps_header)  # scalar fallback
    except Exception:
        logger.debug("EXR fps header unreadable on %s; defaulting to %s", path, _DEFAULT_FPS)
        return _DEFAULT_FPS


def _frame_dims(frame_file: str) -> tuple[int, int]:
    """Return (width, height) of a single frame file."""
    if frame_file.lower().endswith(".exr"):
        rgb = decode_exr_image(frame_file)
        return int(rgb.shape[1]), int(rgb.shape[0])
    import cv2  # noqa: PLC0415

    img = cv2.imread(frame_file, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode sequence frame for dims: {frame_file}")
    return int(img.shape[1]), int(img.shape[0])


def sequence_metadata(path: str) -> tuple[int, int, int, float]:
    """Return ``(width, height, frame_count, fps)`` for the sequence containing ``path``.

    Dimensions come from frame 0; count = number of present frames; fps = the EXR
    ``framesPerSecond`` header if present, else 24.0. Raises ValueError if
    ``path`` is not a resolvable sequence file.
    """
    spec = resolve_sequence(path)
    if spec is None:
        raise ValueError(f"Not a sequence file (no siblings): {path}")

    first = spec.files[0]
    width, height = _frame_dims(first)
    count = len(spec.files)
    fps = _exr_fps(first) if first.lower().endswith(".exr") else _DEFAULT_FPS
    return width, height, count, fps


def sequence_metadata_from_dir(dir_path: str) -> tuple[int, int, int, float]:
    """Return ``(width, height, frame_count, fps)`` for the dominant sequence in ``dir_path``.

    Mirrors :func:`sequence_metadata` but for a DIRECTORY input (a system-
    generated EXR asset). Dims come from the first frame of the dominant
    sequence (see :func:`resolve_sequence_from_dir`); count = number of frames
    in that sequence; fps = the EXR ``framesPerSecond`` header of the first
    frame if present, else 24.0. Raises ValueError if the dir holds no
    resolvable image sequence (≥2 matching files).
    """
    spec = resolve_sequence_from_dir(dir_path)
    if spec is None:
        raise ValueError(f"No image sequence in directory: {dir_path}")

    first = spec.files[0]
    width, height = _frame_dims(first)
    count = len(spec.files)
    fps = _exr_fps(first) if first.lower().endswith(".exr") else _DEFAULT_FPS
    return width, height, count, fps


__all__ = [
    "SequenceSpec",
    "decode_sequence_frames",
    "is_sequence_dir",
    "is_sequence_file",
    "resolve_sequence",
    "resolve_sequence_from_dir",
    "sequence_metadata",
    "sequence_metadata_from_dir",
]
