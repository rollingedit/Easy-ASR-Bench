from __future__ import annotations

from dataclasses import dataclass


CORE_IMPORTS = {
    "numpy": "numpy",
    "soundfile": "soundfile",
    "librosa": "librosa",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "psutil": "psutil",
    "huggingface-hub": "huggingface_hub",
}


EXPLICITLY_UNCHECKED_REQUIREMENTS: dict[str, str] = {}


@dataclass(frozen=True)
class RequirementLine:
    package: str
    raw: str


def parse_requirement_packages(text: str) -> list[RequirementLine]:
    packages: list[RequirementLine] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("--index-url", "--extra-index-url", "-f ", "--find-links")):
            continue
        package = line.split(";", 1)[0].strip()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            if separator in package:
                package = package.split(separator, 1)[0]
        packages.append(RequirementLine(package=package.strip().lower().replace("_", "-"), raw=line))
    return packages
