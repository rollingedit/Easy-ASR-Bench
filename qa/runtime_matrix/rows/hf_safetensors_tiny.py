from __future__ import annotations

import fnmatch
import shutil
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.config import load_config
from app.dependency_manager import install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.model_scanner import scan_models
from qa.runtime_matrix.common import package_versions, write_row


HF_WHISPER_REPO = "optimum-internal-testing/tiny-random-whisper"
HF_CTC_REPO = "undermachine/wav2vec2-small-finetuned"


def _repo_for_row(row_id: str) -> tuple[str, str]:
    if "whisper" in row_id:
        return HF_WHISPER_REPO, "hf_whisper_asr"
    return HF_CTC_REPO, "hf_transformers_asr"


def _download_repo(repo_id: str, destination: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

    destination.mkdir(parents=True, exist_ok=True)
    patterns = [
        "*.safetensors",
        "*.json",
        "*.txt",
        "*.model",
        "merges.txt",
        "vocab.*",
        "tokenizer.*",
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
    ]
    files = [name for name in HfApi().list_repo_files(repo_id) if any(fnmatch.fnmatch(Path(name).name, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)]
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            cached = Path(hf_hub_download(repo_id, name, cache_dir=destination / ".hf_cache"))
            shutil.copy2(cached, target)
        except FileNotFoundError:
            request = urllib.request.Request(hf_hub_url(repo_id, name), headers={"User-Agent": "Easy-ASR-Bench-runtime-matrix"})
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    repo_id, adapter_name = _repo_for_row(row_id)
    config = load_config(Path("config.json"))
    config["runtime"]["provider"] = "cpu"
    config["runtime"]["prefer_gpu"] = False
    missing = missing_modules_for_config("transformers_cpu", config)
    repair_command = recovery_command_for_config("transformers_cpu", config)
    repair_log = evidence_dir / "transformers_cpu_repair.log"
    if missing and install_deps:
        try:
            install_group_for_config("transformers_cpu", Path.cwd(), config, log_path=repair_log)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="Transformers/Safetensors dependency repair failed.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=repair_command,
                details={"missing_before": missing, "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"])},
                artifacts=[repair_log],
            )
        missing = missing_modules_for_config("transformers_cpu", config)
    if missing:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Transformers/Safetensors dependency group is not currently runnable.",
            block_reason=", ".join(missing),
            external_requirement=repair_command,
            details={"missing": missing, "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"])},
            artifacts=[repair_log],
        )

    model_dir = evidence_dir / "Models" / repo_id.replace("/", "__")
    if not list(model_dir.glob("*.safetensors")):
        if not allow_downloads:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary=f"{repo_id} Safetensors fixture is not cached locally.",
                block_reason=f"missing local fixture {model_dir}",
                external_requirement=f"rerun with --allow-downloads to download {repo_id}",
                details={"repo_id": repo_id, "adapter_name": adapter_name},
                artifacts=[repair_log],
            )
        try:
            _download_repo(repo_id, model_dir)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary=f"Could not download {repo_id} Safetensors fixture.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=f"network access to https://huggingface.co/{repo_id}",
                details={"repo_id": repo_id, "adapter_name": adapter_name},
                artifacts=[repair_log],
            )

    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == adapter_name]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{repo_id} did not scan as a runnable {adapter_name} Safetensors model.",
            details={
                "repo_id": repo_id,
                "adapter_name": adapter_name,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "missing": candidate.missing_files, "warnings": candidate.warnings} for candidate in unsupported],
            },
            artifacts=[*model_dir.glob("*.safetensors"), repair_log],
        )

    from app.main import adapter_for

    candidate = candidates[0]
    adapter = adapter_for(candidate)
    try:
        adapter.load(candidate, {"provider": "cpu", "prefer_gpu": False, "language": "en", "task": "transcribe"})
        result = adapter.transcribe_chunks(
            [SimpleNamespace(samples=np.zeros(16000, dtype=np.float32))],
            [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0}],
        )
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"{repo_id} Safetensors fixture could not load or run through Transformers.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement="repair transformers_cpu dependency group or use a different complete HF ASR Safetensors fixture",
            details={"repo_id": repo_id, "adapter_name": adapter_name, "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"])},
            artifacts=[*model_dir.glob("*.safetensors"), repair_log],
        )
    transcript = result.transcript_chunks[0].text if result.transcript_chunks else ""
    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary=f"{repo_id} Safetensors fixture loaded through {adapter_name} and completed one CPU inference call.",
        details={
            "repo_id": repo_id,
            "adapter_name": adapter_name,
            "transcript": transcript,
            "metrics": result.metrics,
            "errors": result.errors,
            "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"]),
        },
        artifacts=[*model_dir.glob("*.safetensors"), repair_log],
    )
