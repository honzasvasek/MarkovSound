from __future__ import annotations

from markovsound.composition import FormArc, shape_caption
from markovsound.runtime_config import RuntimeConfig


FORM = FormArc(
    name="test form",
    trajectory="begin plainly and end altered",
    opening_take="expose the source",
    developing_take="displace the source",
    closing_take="leave only its residue",
)


def test_shape_caption_keeps_source_and_adds_take_specific_form():
    shaped = shape_caption(
        "Bowed metal and dry percussion",
        FORM,
        take_number=3,
        total_takes=3,
    )

    assert shaped.startswith("Bowed metal and dry percussion. Form arc (test form):")
    assert "Take 3 of 3: leave only its residue." in shaped


def test_shape_caption_is_noop_without_a_form():
    assert shape_caption("  Unstable brass.  ", None) == "Unstable brass."


def test_plan_song_reuses_one_form_across_related_takes(monkeypatch):
    from markovsound import cycle

    monkeypatch.setattr(cycle, "generate_caption", lambda *_args, **_kwargs: "Prepared piano pulses")
    monkeypatch.setattr(cycle, "choose_form", lambda: FORM)
    monkeypatch.setattr(
        cycle,
        "_experimental_metadata",
        lambda: {"bpm": 60, "keyscale": "C minor", "timesignature": "5"},
    )
    monkeypatch.setattr(cycle.random, "random", lambda: 0.5)
    state = {"caption": None, "takes_left": 0, "take_no": 0}
    kwargs = {
        "chain": {},
        "order": 3,
        "paths": object(),
        "runtime_cfg": RuntimeConfig(form_arc_prob=1.0, vocal_prob=0.0),
        "requested_duration": 90,
        "song_state": state,
        "takes_per_song": 3,
        "random_duration_seconds": lambda _cfg: 120,
        "read_preset": lambda _paths: "",
        "log": lambda _message: None,
    }

    first = cycle.plan_song(**kwargs)
    second = cycle.plan_song(**kwargs)

    assert first is not None and second is not None
    assert first.caption == second.caption == "Prepared piano pulses"
    assert first.form_arc == second.form_arc == FORM.to_dict()
    assert "Take 1 of 3: expose the source." in first.prompt_caption
    assert "Take 2 of 3: displace the source." in second.prompt_caption
