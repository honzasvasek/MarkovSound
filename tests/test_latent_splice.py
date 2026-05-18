from __future__ import annotations

from array import array

from markovsound.latent_splice import crossfade_latents, frame_count, splice_latent_segments


def _latent(values: list[float]) -> bytes:
    data = array("f")
    for value in values:
        data.extend([value] * 64)
    return data.tobytes()


def test_crossfade_latents_blends_overlap_and_shortens_once():
    merged, overlap = crossfade_latents(_latent([0, 1, 2]), _latent([10, 11, 12]), overlap_seconds=2 / 25)
    frames = array("f"); frames.frombytes(merged)

    assert overlap == 2
    assert frame_count(merged) == 4
    assert [frames[i * 64] for i in range(4)] == [0.0, 1.0, 11.0, 12.0]


def test_splice_latent_segments_reports_each_overlap():
    merged, overlaps = splice_latent_segments([_latent([0, 1]), _latent([2, 3]), _latent([4, 5])], 1 / 25)

    assert overlaps == [1, 1]
    assert frame_count(merged) == 4
