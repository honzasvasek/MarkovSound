"""Ephemeral formal direction layered on top of learned song captions.

The caption chain supplies the material of a piece.  This module supplies a
temporary shape for that material, without training the hand-written shape
language back into the chain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any


@dataclass(frozen=True)
class FormArc:
    """A whole-track trajectory plus ways to vary it across related takes."""

    name: str
    trajectory: str
    opening_take: str
    developing_take: str
    closing_take: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "FormArc | None":
        if not isinstance(value, dict):
            return None
        try:
            return cls(**{field: str(value[field]) for field in cls.__dataclass_fields__})
        except KeyError:
            return None


# These describe temporal behavior instead of genre.  They can therefore act
# on whatever instruments, textures, or idioms the Markov chain happens to
# propose without turning the output into a small set of canned styles.
_FORM_ARCS = (
    FormArc(
        name="accumulation",
        trajectory=(
            "begin with one exposed sound; let layers gather at unequal rates; "
            "reach maximum density late, then stop before the texture resolves"
        ),
        opening_take="make each new layer easy to hear and leave generous gaps",
        developing_take="compress the gaps and let secondary details become structural",
        closing_take="overload the accumulated material until only one stubborn layer remains",
    ),
    FormArc(
        name="erosion",
        trajectory=(
            "state a recognizable figure early; gradually remove its pulse, pitch, "
            "and edges; end with a damaged but audible residue"
        ),
        opening_take="keep the original figure unusually plain before it begins to fray",
        developing_take="interrupt the figure and replace missing parts with noise or silence",
        closing_take="treat the source as a half-remembered trace rather than a theme",
    ),
    FormArc(
        name="false return",
        trajectory=(
            "move away from the opening material, appear to return to it after the midpoint, "
            "then reveal that its rhythm or harmonic center has changed"
        ),
        opening_take="make the opening identity strong enough that its altered return is legible",
        developing_take="hide the change inside repetition and delayed accents",
        closing_take="make the return uncanny, sparse, and less stable than the departure",
    ),
    FormArc(
        name="rupture",
        trajectory=(
            "establish a patient continuity, break it once with a sharply contrasting event, "
            "and let the remainder carry audible consequences of that break"
        ),
        opening_take="delay the rupture and avoid foreshadowing it too clearly",
        developing_take="let the rupture arrive earlier and contaminate the surrounding material",
        closing_take="begin among the consequences, with the original continuity barely recoverable",
    ),
    FormArc(
        name="unstable orbit",
        trajectory=(
            "circle a recurring sonic object without repeating it exactly; change distance, "
            "density, and perspective on every pass; never settle into a loop"
        ),
        opening_take="keep the recurring object near the foreground and vary its surroundings",
        developing_take="push the object in and out of focus while preserving its contour",
        closing_take="let the orbit widen until recurrence is sensed more than heard",
    ),
    FormArc(
        name="figure-ground reversal",
        trajectory=(
            "begin with a clear foreground gesture over a restrained field; slowly give the field "
            "more agency; end after background and foreground have exchanged roles"
        ),
        opening_take="separate figure and field with obvious differences in register and density",
        developing_take="allow the field to imitate and obstruct the foreground gesture",
        closing_take="bury the former figure inside the field without erasing it completely",
    ),
)


def choose_form(rng: Any = random) -> FormArc:
    """Choose one formal archetype; injectable randomness keeps tests exact."""
    return rng.choice(_FORM_ARCS)


def shape_caption(
    caption: str,
    form: FormArc | None,
    *,
    take_number: int = 1,
    total_takes: int = 1,
) -> str:
    """Add a formal instruction while preserving the learned source caption."""
    base = caption.strip()
    if form is None:
        return base
    total = max(1, total_takes)
    take = min(max(1, take_number), total)
    if total == 1:
        variation = form.developing_take
        take_label = ""
    elif take == 1:
        variation = form.opening_take
        take_label = f" Take {take} of {total}:"
    elif take == total:
        variation = form.closing_take
        take_label = f" Take {take} of {total}:"
    else:
        variation = form.developing_take
        take_label = f" Take {take} of {total}:"
    punctuation = "" if base.endswith((".", "!", "?")) else "."
    return (
        f"{base}{punctuation} Form arc ({form.name}): {form.trajectory}."
        f"{take_label} {variation}."
    )
