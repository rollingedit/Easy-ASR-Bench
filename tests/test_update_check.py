import json
import urllib.error

from app.update_check import check_for_updates, check_for_updates_from_config


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_update_check_reports_current_release(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr("app.update_check.urllib.request.urlopen", lambda request, timeout: FakeResponse({"tag_name": "v0.4.0"}))

    result = check_for_updates(current_tag="v0.4.0", print_func=messages.append)

    assert result["status"] == "current"
    assert "up to date" in messages[0]


def test_update_check_reports_newer_release(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr("app.update_check.urllib.request.urlopen", lambda request, timeout: FakeResponse({"tag_name": "v0.4.1"}))

    result = check_for_updates(current_tag="v0.4.0", print_func=messages.append)

    assert result["status"] == "update_available"
    assert result["latest"] == "v0.4.1"
    assert "releases/latest" in messages[0]


def test_update_check_reports_offline_as_unavailable(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr("app.update_check.urllib.request.urlopen", lambda request, timeout: (_ for _ in ()).throw(urllib.error.URLError("offline")))

    result = check_for_updates(current_tag="v0.4.0", print_func=messages.append)

    assert result["status"] == "unavailable"
    assert "Update check unavailable" in messages[0]


def test_update_check_reports_malformed_response(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr("app.update_check.urllib.request.urlopen", lambda request, timeout: FakeResponse({"name": "latest"}))

    result = check_for_updates(current_tag="v0.4.0", print_func=messages.append)

    assert result["status"] == "unavailable"
    assert "tag_name" in messages[0]


def test_update_check_honors_setup_and_run_config_flags(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("app.update_check.check_for_updates", lambda print_func=print: calls.append("called") or {"status": "current"})

    assert check_for_updates_from_config({"app": {"check_for_updates_on_setup": True}}, context="setup") == {"status": "current"}
    assert check_for_updates_from_config({"app": {"check_for_updates_on_run": True}}, context="run") == {"status": "current"}
    assert check_for_updates_from_config({"app": {"check_for_updates_on_run": False}}, context="run") is None
    assert calls == ["called", "called"]
