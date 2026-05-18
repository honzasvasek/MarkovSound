from __future__ import annotations

from markovsound import cli_transcribe


class _Response:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("bad response")


def test_wake_only_posts_when_server_is_sleeping(monkeypatch):
    posts = []
    monkeypatch.setattr(cli_transcribe, "is_sleeping", lambda _url: True)
    monkeypatch.setattr(
        cli_transcribe.requests,
        "post",
        lambda *args, **kwargs: posts.append((args, kwargs)) or _Response({}),
    )

    assert cli_transcribe.wake("http://x") is True
    assert posts[0][0][0] == "http://x/wake_up"


def test_sleep_only_posts_when_server_is_awake(monkeypatch):
    posts = []
    monkeypatch.setattr(cli_transcribe, "is_sleeping", lambda _url: False)
    monkeypatch.setattr(
        cli_transcribe.requests,
        "post",
        lambda *args, **kwargs: posts.append((args, kwargs)) or _Response({}),
    )

    assert cli_transcribe.sleep("http://x") is True
    assert posts[0][0][0] == "http://x/sleep"
    assert posts[0][1]["params"] == {"level": 1}
