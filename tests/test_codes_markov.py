from __future__ import annotations

from markovsound.codes_markov import build_codes_chain, format_codes, parse_codes, sample_codes, untrain_codes


def test_parse_and_format_codes_roundtrip():
    assert parse_codes("1, 2,3") == [1, 2, 3]
    assert format_codes([1, 2, 3]) == "1,2,3"


def test_sample_codes_returns_requested_length_and_dead_end_count():
    chain = build_codes_chain([[1, 2, 1, 2]], order=1)

    sampled, dead_end_positions = sample_codes(chain, order=1, length=5)

    assert len(sampled) == 5
    assert isinstance(dead_end_positions, list)


def test_untrain_codes_removes_transitions():
    chain = build_codes_chain([[1, 2, 3]], order=1)

    assert untrain_codes(chain, [1, 2, 3], order=1) == 2
    assert chain == {}
