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


def test_openvino_requirement_is_pinned_to_ort_compatibility():
    text = (ROOT / "requirements" / "onnx_openvino.txt").read_text(encoding="utf-8")

    assert "onnxruntime-openvino" in text
    assert "openvino==2025.4.1" in text


def test_onnx_requirement_variants_keep_transformers_safe_tokenizers_bound():
    for filename in ["onnx.txt", "onnx_cuda.txt", "onnx_directml.txt", "onnx_openvino.txt"]:
        text = (ROOT / "requirements" / filename).read_text(encoding="utf-8")

        assert "tokenizers>=0.22,<0.23.1" in text


def test_onnx_requirement_variants_keep_openvino_safe_numpy_bound():
    for filename in ["onnx.txt", "onnx_cuda.txt", "onnx_directml.txt", "onnx_openvino.txt"]:
        text = (ROOT / "requirements" / filename).read_text(encoding="utf-8")

        assert "numpy>=1.26,<2.4" in text
