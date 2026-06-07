from pathlib import Path

from app.dependency_specs import CORE_IMPORTS, EXPLICITLY_UNCHECKED_REQUIREMENTS, parse_requirement_packages
from app.dependency_manager import DEPENDENCY_GROUPS


ROOT = Path(__file__).resolve().parents[1]


def test_core_requirement_packages_have_import_checks_or_documented_exclusions():
    requirements = parse_requirement_packages((ROOT / "requirements" / "core.txt").read_text(encoding="utf-8"))
    checked = set(CORE_IMPORTS)
    unchecked = set(EXPLICITLY_UNCHECKED_REQUIREMENTS)
    missing = [item.package for item in requirements if item.package not in checked and item.package not in unchecked]

    assert missing == []


def test_core_dependency_group_uses_central_import_mapping():
    assert set(DEPENDENCY_GROUPS["core"].modules) == set(CORE_IMPORTS.values())
