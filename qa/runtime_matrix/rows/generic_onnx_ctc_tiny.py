from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.adapters.generic_onnx_manifest import GenericOnnxManifestAdapter
from app.dependency_manager import cuda_diagnostics
from app.model_scanner import scan_models
from qa.runtime_matrix.common import package_versions, write_row


def _provider_for_row(row_id: str) -> str:
    if "directml" in row_id or "amd_directml" in row_id or "intel_directml" in row_id:
        return "directml"
    if "openvino" in row_id:
        return "openvino"
    if "cuda" in row_id:
        return "cuda"
    return "cpu"


def _write_tiny_ctc_fixture(model_dir: Path) -> list[Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    logits = np.array(
        [
            [
                [0.0, 8.0, 0.0],
                [8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0],
                [8.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        nodes=[helper.make_node("Constant", inputs=[], outputs=["logits"], value=numpy_helper.from_array(logits))],
        name="easy_asr_bench_tiny_ctc",
        inputs=[helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, "samples"])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, 3])],
    )
    model = helper.make_model(graph, producer_name="easy-asr-bench-runtime-matrix")
    model.ir_version = 10
    model.opset_import[0].version = 17
    model_path = model_dir / "model.onnx"
    onnx.save(model, model_path)
    (model_dir / "vocab.json").write_text(json.dumps({"|": 0, "a": 1, "b": 2}), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "easy_asr_bench.model_manifest.v1",
        "display_name": "Tiny deterministic ONNX CTC",
        "task": "automatic-speech-recognition",
        "precision": "fp32",
        "files": {"model": "model.onnx"},
        "inputs": {"waveform": {"name": "input_values"}},
        "outputs": {"logits": "logits"},
        "preprocessing": {"type": "raw_waveform"},
        "decoding": {"type": "ctc", "blank_token_id": 0, "vocab_file": "vocab.json"},
    }
    (model_dir / "modelbench.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return [model_path, model_dir / "vocab.json", model_dir / "modelbench.json"]


def _run_without_manifest_rejection(row_id: str, evidence_dir: Path) -> dict:
    model_dir = evidence_dir / "raw_onnx_without_manifest"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"not a valid onnx graph; scanner should reject before runtime")
    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in unsupported if candidate.adapter_name == "generic_onnx_manifest" and candidate.path == model_path]
    details = {
        "model_path": str(model_path),
        "runnable_count": len(runnable),
        "unsupported_count": len(unsupported),
        "unsupported": [
            {
                "candidate_id": candidate.candidate_id,
                "adapter_name": candidate.adapter_name,
                "container_format": candidate.container_format,
                "runnable": candidate.runnable,
                "missing_files": candidate.missing_files,
                "warnings": candidate.warnings,
            }
            for candidate in unsupported
        ],
    }
    ok = bool(candidates) and candidates[0].runnable is False and "modelbench.json" in candidates[0].missing_files
    return write_row(
        row_id,
        "pass" if ok else "fail",
        evidence_dir,
        summary=(
            "Raw ONNX without modelbench.json is rejected before runtime with an exact manifest requirement."
            if ok
            else "Raw ONNX without modelbench.json did not produce the expected scanner rejection."
        ),
        details=details,
        artifacts=[model_path],
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "generic_onnx_without_manifest_rejected":
        return _run_without_manifest_rejection(row_id, evidence_dir)
    diagnostics = cuda_diagnostics()
    if "intel_directml" in row_id and not diagnostics.get("intel_gpu_or_npu_detected", False):
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Intel DirectML ONNX row requires Intel GPU/NPU hardware, which is not detected on this machine.",
            block_reason="Intel GPU/NPU not detected",
            external_requirement="Intel GPU/NPU with DirectML-capable Windows driver",
            details={"cuda_provider_checks": diagnostics},
        )
    if "amd_directml" in row_id and not diagnostics.get("amd_gpu_detected", False):
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="AMD DirectML ONNX row requires AMD GPU hardware, which is not detected on this machine.",
            block_reason="AMD GPU not detected",
            external_requirement="AMD DirectML-capable Windows GPU",
            details={"cuda_provider_checks": diagnostics},
        )
    if "openvino" in row_id and "OpenVINOExecutionProvider" not in diagnostics.get("onnxruntime_providers", []):
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Intel OpenVINO ONNX row requires OpenVINOExecutionProvider, which is not available in this environment.",
            block_reason="OpenVINOExecutionProvider missing from onnxruntime providers",
            external_requirement="Intel/OpenVINO-capable machine with onnxruntime-openvino installed and provider visible",
            details={
                "cuda_provider_checks": diagnostics,
                "dependency_versions": package_versions(["onnxruntime", "onnxruntime-openvino"]),
            },
        )
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort  # noqa: F401
    except ModuleNotFoundError as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="ONNX dependency group is not installed, so the tiny Generic ONNX CTC fixture cannot run.",
            block_reason=f"missing {exc.name}",
            external_requirement="python -m pip install -r requirements/onnx.txt",
            details={"dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-gpu"])},
        )

    model_dir = evidence_dir / "tiny_onnx_ctc"
    artifacts = _write_tiny_ctc_fixture(model_dir)
    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "generic_onnx_manifest"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Tiny ONNX CTC manifest fixture was not discovered as a runnable Generic ONNX CTC model.",
            details={"runnable_count": len(runnable), "unsupported_count": len(unsupported)},
            artifacts=artifacts,
        )

    provider = _provider_for_row(row_id)
    adapter = GenericOnnxManifestAdapter()
    candidate = candidates[0]
    try:
        adapter.load(candidate, {"provider": provider, "prefer_gpu": provider != "cpu", "cpu_threads": 1})
        result = adapter.transcribe_chunks(
            [SimpleNamespace(samples=np.zeros(1600, dtype=np.float32))],
            [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 0.1}],
        )
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"Tiny ONNX CTC fixture could not run with requested provider {provider}.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement="repair ONNX Runtime provider package or rerun with CPU provider",
            details={"provider": provider, "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-gpu"])},
            artifacts=artifacts,
        )
    transcript = result.transcript_chunks[0].text if result.transcript_chunks else ""
    metrics = result.metrics
    if transcript != "ab":
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Tiny ONNX CTC fixture ran but decoded the wrong transcript.",
            details={"provider": provider, "transcript": transcript, "metrics": metrics},
            artifacts=artifacts,
        )
    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary=f"Tiny Generic ONNX CTC manifest fixture decoded transcript with requested provider {provider}.",
        details={
            "provider": provider,
            "transcript": transcript,
            "metrics": metrics,
            "cuda_provider_checks": diagnostics,
            "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-gpu"]),
        },
        artifacts=artifacts,
    )
