from __future__ import annotations

from pathlib import Path

from app.adapters.gguf_llm_reference import GGUFLLMReferenceAdapter
from app.model_scanner import scan_models
from qa.runtime_matrix.common import package_versions, write_row


SMOLLM_PATH = Path("Temp/real_tiny_llm_smoke/Models/SmolLM-135M-GGUF/SmolLM-135M.Q4_K_M.gguf")


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )
    try:
        import llama_cpp  # noqa: F401
    except ModuleNotFoundError:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama-cpp-python is not installed, so SmolLM GGUF cannot be loaded.",
            block_reason="missing llama_cpp import",
            external_requirement="install llama_cpp dependency group",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )

    runnable, unsupported = scan_models(SMOLLM_PATH.parent)
    reference_candidates = [candidate for candidate in unsupported if candidate.adapter_name == "gguf_llm_reference"]
    if not reference_candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference LLM candidate.",
            details={"runnable_count": len(runnable), "unsupported_count": len(unsupported)},
            artifacts=[SMOLLM_PATH],
        )

    adapter = GGUFLLMReferenceAdapter()
    candidate = reference_candidates[0]
    llm = adapter.load(candidate, {"provider": "cpu", "prefer_gpu": False, "llm_context_tokens": 256})
    response = llm("Answer with the word pass.", max_tokens=8, temperature=0.0)
    text = response["choices"][0]["text"].strip() if isinstance(response, dict) else str(response).strip()
    if not text:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF loaded but produced empty text.",
            details={"candidate_id": candidate.candidate_id, "dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )
    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary="SmolLM 135M GGUF loaded as a reference LLM and generated non-empty text.",
        details={
            "candidate_id": candidate.candidate_id,
            "classification": "reference_llm_not_direct_asr",
            "generated_text": text,
            "dependency_versions": package_versions(["llama-cpp-python"]),
        },
        artifacts=[SMOLLM_PATH],
    )

