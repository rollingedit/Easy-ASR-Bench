from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstallPlan:
    dependency_group: str
    reason: str
    packages: list[str]
    requirement_files: list[str]
    index_urls: list[str]
    install_location: str
    network_destinations: list[str]
    system_path_changes: list[str]
    estimated_size_class: str
    confirmation_prompt: str
    fallback_if_declined: str
    log_path: str


def _requirement_packages(path: Path) -> list[str]:
    if not path.exists():
        return []
    packages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(("-f ", "--find-links")):
            continue
        if stripped.startswith(("--index-url", "--extra-index-url")):
            continue
        packages.append(stripped)
    return packages


def _requirement_indexes(path: Path) -> list[str]:
    if not path.exists():
        return []
    indexes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("--index-url", "--extra-index-url")):
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                indexes.append(parts[1])
    return indexes


def build_install_plan(group: str, project_root: Path, config: dict, candidate_names: list[str] | None = None) -> InstallPlan:
    from .dependency_manager import DEPENDENCY_GROUPS, acceleration_install_decision

    decision = acceleration_install_decision(config, group)
    if decision.get("use_accelerator"):
        requirement_files = list(decision.get("requirement_files", []))
        accelerator = str(decision.get("accelerator", "")).upper()
        size = "large" if accelerator in {"CUDA", "VULKAN"} else "medium"
    else:
        requirement_files = [DEPENDENCY_GROUPS[group].requirement_file]
        size = "small-to-medium"
    paths = [project_root / item for item in requirement_files]
    packages = sorted({package for path in paths for package in _requirement_packages(path)})
    indexes = sorted({index for path in paths for index in _requirement_indexes(path)})
    selected = ", ".join(candidate_names or []) or "selected model(s)"
    return InstallPlan(
        dependency_group=group,
        reason=f"{selected} require {DEPENDENCY_GROUPS[group].description}. {decision.get('reason', '')}".strip(),
        packages=packages,
        requirement_files=requirement_files,
        index_urls=indexes or ["PyPI default index"],
        install_location=str(Path(sys.executable).parent.parent),
        network_destinations=indexes or ["pypi.org / files.pythonhosted.org"],
        system_path_changes=[],
        estimated_size_class=size,
        confirmation_prompt="Press Enter to install, or type s to skip this group.",
        fallback_if_declined="Only models requiring this dependency group are skipped; other selected models continue.",
        log_path="Logs/setup.log",
    )


def format_install_plan(plan: InstallPlan) -> str:
    lines = [
        "Runtime install plan",
        f"  Needed group: {plan.dependency_group}",
        f"  Reason: {plan.reason}",
        f"  Requirement files: {', '.join(plan.requirement_files)}",
        f"  Packages: {', '.join(plan.packages) or 'unknown until pip resolves'}",
        f"  Index URLs: {', '.join(plan.index_urls)}",
        f"  Install location: {plan.install_location}",
        f"  Network destinations: {', '.join(plan.network_destinations)}",
        f"  System PATH changes: {', '.join(plan.system_path_changes) or 'none'}",
        f"  Estimated size: {plan.estimated_size_class}",
        f"  Fallback if declined: {plan.fallback_if_declined}",
        f"  Log path: {plan.log_path}",
        f"  {plan.confirmation_prompt}",
    ]
    return "\n".join(lines)
