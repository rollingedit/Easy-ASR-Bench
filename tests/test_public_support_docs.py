from pathlib import Path


def test_bug_report_template_requests_release_support_evidence():
    text = Path(".github/ISSUE_TEMPLATE/bug_report.yml").read_text(encoding="utf-8")

    assert "Run.bat --doctor --json" in text
    assert "setup.bat --doctor --json" in text
    assert "Windows version" in text
    assert "Install mode" in text
    assert "Model and runtime type" in text
    assert "latest run/setup/crash log" in text


def test_public_docs_explain_smartscreen_and_antivirus_without_disable_security():
    readme = Path("README.md").read_text(encoding="utf-8")
    setup_docs = Path("docs/what_setup_installs.md").read_text(encoding="utf-8")

    assert "SmartScreen, Defender, or antivirus warning" in readme
    assert "GitHub bug report template" in readme
    assert "SmartScreen, Defender, And Antivirus Notes" in setup_docs
    assert "unsigned `.bat` and PowerShell files" in setup_docs
    assert "verify the release checksums" in setup_docs
    assert "do not disable protection globally" in setup_docs
    assert "User media and local model files stay on the machine" in setup_docs
