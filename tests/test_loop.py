from __future__ import annotations

from markovsound.loop import log


def test_loop_logger_is_available(capsys):
    log("hello")

    assert capsys.readouterr().out.endswith("hello\n")
