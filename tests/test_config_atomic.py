from pathlib import Path

import pytest

from app.config import load_config, save_config, save_default_config


def test_save_config_replaces_atomically_without_partial(tmp_path):
    path = tmp_path / "config.json"

    save_config(path, {"app": {"version": "test"}})

    assert '"version": "test"' in path.read_text(encoding="utf-8")
    assert not path.with_suffix(".json.partial").exists()


def test_save_config_replace_failure_leaves_old_config_intact(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"app":{"version":"old"}}\n', encoding="utf-8")
    original_replace = Path.replace

    def fail_replace(self, target):
        if self.name == "config.json.partial":
            raise OSError("replace blocked")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace blocked"):
        save_config(path, {"app": {"version": "new"}})

    assert path.read_text(encoding="utf-8") == '{"app":{"version":"old"}}\n'
    assert not path.with_suffix(".json.partial").exists()


def test_save_default_config_uses_atomic_writer(tmp_path):
    path = tmp_path / "config.json"

    save_default_config(path)

    config = load_config(path)
    assert config["folders"]["models"] == "Models"
    assert not path.with_suffix(".json.partial").exists()
