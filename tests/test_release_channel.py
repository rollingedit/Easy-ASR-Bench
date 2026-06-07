from app.config import DEFAULT_CONFIG
from app.doctor import run_doctor
from app.results_writer import runtime_environment
from app.version import RELEASE_CHANNEL


def test_default_release_channel_is_not_stable_for_prerelease_build():
    assert DEFAULT_CONFIG["app"]["version_channel"] == "prerelease"
    assert RELEASE_CHANNEL == "prerelease"


def test_runtime_environment_reports_release_channel_and_commit():
    environment = runtime_environment()
    assert environment["release_channel"] == "prerelease"
    assert "release_commit" in environment


def test_doctor_uses_build_channel_over_stale_config(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "app": {"version_channel": "stable"},
  "folders": {"input": "Input", "models": "Models", "output": "Results", "logs": "Logs"},
  "advanced": {"logs_folder": "Logs"},
  "runtime": {},
  "dependencies": {"auto_install_optional": false}
}
""".strip(),
        encoding="utf-8",
    )

    assert run_doctor(config_path) == 0
    output = capsys.readouterr().out
    assert "Release channel: prerelease" in output
    assert "Config channel note: config.json says stable" in output
