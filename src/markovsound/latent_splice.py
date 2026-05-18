"""Latent-space crossfades for smoothing Markov-code dead-end jumps."""
from __future__ import annotations

from array import array

_LATENT_CHANNELS = 64
_LATENT_FRAME_BYTES = _LATENT_CHANNELS * 4
_LATENT_FPS = 25


def frame_count(latents: bytes) -> int:
    if len(latents) % _LATENT_FRAME_BYTES:
        raise ValueError("latent payload is not a whole number of [64] f32 frames")
    return len(latents) // _LATENT_FRAME_BYTES


def crossfade_latents(left: bytes, right: bytes, overlap_seconds: float) -> tuple[bytes, int]:
    """Join two latent clips with a linear crossfade overlap."""
    left_frames = frame_count(left)
    right_frames = frame_count(right)
    overlap = min(round(overlap_seconds * _LATENT_FPS), left_frames, right_frames)
    if overlap <= 0:
        return left + right, 0
    a = array("f"); a.frombytes(left)
    b = array("f"); b.frombytes(right)
    out = array("f", a[: (left_frames - overlap) * _LATENT_CHANNELS])
    for frame in range(overlap):
        alpha = frame / (overlap - 1) if overlap > 1 else 0.5
        abase = (left_frames - overlap + frame) * _LATENT_CHANNELS
        bbase = frame * _LATENT_CHANNELS
        for ch in range(_LATENT_CHANNELS):
            out.append((1.0 - alpha) * a[abase + ch] + alpha * b[bbase + ch])
    out.extend(b[overlap * _LATENT_CHANNELS :])
    return out.tobytes(), overlap


def splice_latent_segments(segments: list[bytes], overlap_seconds: float) -> tuple[bytes, list[int]]:
    """Crossfade each adjacent segment and return used overlap frame counts."""
    if not segments:
        raise ValueError("at least one latent segment is required")
    merged = segments[0]
    overlaps: list[int] = []
    for segment in segments[1:]:
        merged, used = crossfade_latents(merged, segment, overlap_seconds)
        overlaps.append(used)
    return merged, overlaps
