from __future__ import annotations

from markovsound.markov import build_chain, detokenize, tokenize, train_text, untrain_text


def test_tokenize_and_detokenize_roundtrip_caption_shape():
    tokens = tokenize("Bright, odd music!")

    assert tokens == ["bright", ",", "odd", "music", "!"]
    assert detokenize(tokens) == "Bright, odd music!"


def test_train_and_untrain_restore_empty_chain():
    chain = {}

    assert train_text(chain, "alpha beta", order=1) == 3
    assert chain
    assert untrain_text(chain, "alpha beta", order=1) == 3
    assert chain == {}


def test_build_chain_keeps_sentence_boundaries():
    chain = build_chain(["alpha beta"], order=1)

    assert chain[("__START__",)]["alpha"] == 1
    assert chain[("beta",)]["__END__"] == 1
