from pathlib import Path


def test_windows_matrix_scripts_define_required_rows():
    script = Path("qa/windows_matrix/run_release_matrix.ps1").read_text(encoding="utf-8")

    assert "win11_clean_no_python_setup" in script
    assert "gguf_asr_mmproj_pair" in script
    assert "dependency_install_declined" in script
    assert 'Status "not_run"' in script


def test_windows_matrix_collector_writes_environment_and_hashes():
    script = Path("qa/windows_matrix/collect_release_evidence.ps1").read_text(encoding="utf-8")

    assert "environment_summary" in script
    assert "Get-FileHash" in script
    assert "row.json" in script
