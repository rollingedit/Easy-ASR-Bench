import json
from pathlib import Path

from app.hf_model_downloader import HFModelRef, build_smart_download_choices


FIXTURES = Path(__file__).parent / "fixtures" / "hf_filelists" / "downloader_layouts.json"


def test_hf_downloader_fixture_layouts_choose_one_safe_package():
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))

    for fixture in fixtures:
        ref, choices = build_smart_download_choices(
            fixture["files"],
            HFModelRef(fixture["repo"], "main", fixture["subfolder"]),
        )
        assert ref.subfolder == fixture["expected_subfolder"], fixture["name"]
        matches = [
            choice
            for choice in choices
            if choice.kind == fixture["expected_kind"] and choice.task_hint == fixture["expected_task_hint"]
        ]
        assert matches, fixture["name"]
        selected = next(
            (choice for choice in matches if all(filename in choice.files for filename in fixture["must_include"])),
            matches[0],
        )
        for filename in fixture["must_include"]:
            assert filename in selected.files, fixture["name"]
        for filename in fixture["must_exclude"]:
            assert filename not in selected.files, fixture["name"]
        if fixture["expected_task_hint"] == "unknown":
            assert any("not treated as runnable" in note for note in selected.notes), fixture["name"]
