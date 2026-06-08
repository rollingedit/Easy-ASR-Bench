from __future__ import annotations

import shutil
from pathlib import Path

from app.adapters.base import ModelCandidate
from app import hf_model_downloader
from app.hf_model_downloader import (
    HFModelRef,
    DownloadChoice,
    build_download_choices,
    build_missing_file_repair_plan,
    build_smart_download_choices,
    download_choice,
    download_hf_model_from_ref,
    list_repo_files,
    offer_missing_file_repair,
)
from app.model_scanner import scan_models
from qa.runtime_matrix.common import ROOT, sha256, write_row
from qa.runtime_matrix.rows.gguf_asr_mmproj import run_qwen3_asr_model_dir


QWEN3_ASR_GGUF_FILES = [
    "Qwen3-ASR-0.6B.Q4_K_M.gguf",
    "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf",
    "Qwen3-ASR-0.6B.Q2_K.gguf",
]

QWEN3_ASR_GGUF_LIVE_LIKE_FILES = [
    "Qwen3-ASR-0.6B.IQ4_XS.gguf",
    "Qwen3-ASR-0.6B.Q2_K.gguf",
    "Qwen3-ASR-0.6B.Q3_K_L.gguf",
    "Qwen3-ASR-0.6B.Q3_K_M.gguf",
    "Qwen3-ASR-0.6B.Q3_K_S.gguf",
    "Qwen3-ASR-0.6B.Q4_K_M.gguf",
    "Qwen3-ASR-0.6B.Q4_K_S.gguf",
    "Qwen3-ASR-0.6B.Q5_K_M.gguf",
    "Qwen3-ASR-0.6B.Q5_K_S.gguf",
    "Qwen3-ASR-0.6B.Q6_K.gguf",
    "Qwen3-ASR-0.6B.Q8_0.gguf",
    "Qwen3-ASR-0.6B.f16.gguf",
    "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf",
    "Qwen3-ASR-0.6B.mmproj-f16.gguf",
]


def _fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
    target = destination / (relative_name or filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"gguf")
    return target


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "hf_downloader_qwen3_asr_gguf_mmproj_public_listing":
        return _run_public_listing(row_id, evidence_dir, _allow_downloads)
    if row_id == "hf_downloader_qwen3_asr_gguf_mmproj_cached_materialization":
        return _run_cached_materialization(row_id, evidence_dir)
    if row_id == "hf_downloader_qwen3_asr_gguf_mmproj_noninteractive_flow":
        return _run_noninteractive_flow(row_id, evidence_dir)
    if row_id == "hf_downloader_qwen3_asr_gguf_mmproj_public_noninteractive_flow":
        return _run_public_noninteractive_flow(row_id, evidence_dir, _allow_downloads)
    if row_id == "hf_downloader_qwen3_asr_gguf_mmproj_public_download_to_asr":
        return _run_public_download_to_asr(row_id, evidence_dir, _install_deps, _allow_downloads)
    if row_id == "hf_downloader_qwen3_asr_gguf_mmproj_public_real_download_to_asr":
        return _run_public_real_download_to_asr(row_id, evidence_dir, _install_deps, _allow_downloads)
    if row_id == "hf_downloader_supported_outcome_taxonomy":
        return _run_supported_outcome_taxonomy(row_id, evidence_dir)
    if row_id != "hf_downloader_qwen3_asr_gguf_mmproj_layout":
        return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled HF downloader layout row: {row_id}")
    return _run_materialized_layout(row_id, evidence_dir)


def _find_qwen_choice(files: list[str], ref: HFModelRef):
    choices = build_download_choices(files, ref)
    qwen_choice = next(
        (
            choice
            for choice in choices
            if choice.task_hint == "asr_audio"
            and "Qwen3-ASR-0.6B.Q4_K_M.gguf" in choice.files
            and "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf" in choice.files
        ),
        None,
    )
    return choices, qwen_choice


def _run_public_listing(row_id: str, evidence_dir: Path, allow_downloads: bool) -> dict:
    ref = HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF")
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="HF public repo listing validation requires network access and --allow-downloads.",
            block_reason="network downloads are disabled for this row",
            external_requirement="Run with --allow-downloads on a machine that can reach huggingface.co.",
            details={"repo_id": ref.repo_id, "expected_files": QWEN3_ASR_GGUF_FILES},
        )

    try:
        files = list_repo_files(ref)
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="HF public repo listing could not be fetched.",
            block_reason=f"Hugging Face repo listing failed: {exc}",
            external_requirement="Network access to huggingface.co and the huggingface_hub package.",
            details={"repo_id": ref.repo_id, "error": str(exc)},
        )

    choices, qwen_choice = _find_qwen_choice(files, ref)
    present = {name: name in files for name in QWEN3_ASR_GGUF_FILES}
    failures = [name for name, exists in present.items() if not exists]
    if qwen_choice is None:
        failures.append("Q4_K_M GGUF plus infix mmproj-Q8_0 projector choice was not built")

    return write_row(
        row_id,
        "fail" if failures else "pass",
        evidence_dir,
        summary=(
            "HF downloader matched the live Qwen3 ASR GGUF+mmproj public repo listing without downloading large model bytes."
            if not failures
            else "HF downloader did not match the live Qwen3 ASR GGUF+mmproj public repo listing."
        ),
        details={
            "repo_id": ref.repo_id,
            "file_count": len(files),
            "expected_files_present": present,
            "choice": qwen_choice.__dict__ if qwen_choice else None,
            "asr_audio_choices": [choice.__dict__ for choice in choices if choice.task_hint == "asr_audio"],
            "failures": failures,
        },
    )


def _run_materialized_layout(row_id: str, evidence_dir: Path) -> dict:
    ref = HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF")
    choices, qwen_choice = _find_qwen_choice(QWEN3_ASR_GGUF_FILES, ref)
    if qwen_choice is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="HF downloader did not build a Qwen3 ASR GGUF+mmproj choice from the public repo file layout.",
            details={"choices": [choice.__dict__ for choice in choices]},
        )

    destination = evidence_dir / "Models" / "mradermacher__Qwen3-ASR-0.6B-GGUF__Qwen3-ASR-0.6B.Q4_K_M"
    original_download = hf_model_downloader._download_file
    hf_model_downloader._download_file = _fake_download
    try:
        downloaded = download_choice(ref, qwen_choice, destination, print_func=lambda _text: None)
    finally:
        hf_model_downloader._download_file = original_download

    manifest = destination / "model_package.json"
    runnable, unsupported = scan_models(destination.parent)
    candidates = [candidate for candidate in [*runnable, *unsupported] if candidate.adapter_name == "gguf_asr_mmproj"]
    failures: list[str] = []
    if not manifest.exists():
        failures.append("model_package.json was not written")
    if not candidates:
        failures.append("downloaded folder did not scan as GGUF ASR+mmproj")
    elif candidates[0].missing_files:
        failures.append("downloaded GGUF ASR+mmproj candidate still reports missing files")

    return write_row(
        row_id,
        "fail" if failures else "pass",
        evidence_dir,
        summary=(
            "HF downloader materialized the real Qwen3 ASR GGUF+mmproj repo layout with an exact pairing manifest."
            if not failures
            else "HF downloader Qwen3 ASR GGUF+mmproj layout validation failed."
        ),
        details={
            "repo_id": ref.repo_id,
            "choice": qwen_choice.__dict__,
            "downloaded": [str(path) for path in downloaded],
            "failures": failures,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "adapter_name": candidate.adapter_name,
                    "container_format": candidate.container_format,
                    "missing_files": candidate.missing_files,
                    "metadata": candidate.metadata,
                }
                for candidate in candidates
            ],
        },
        artifacts=list(dict.fromkeys([*downloaded, manifest])),
    )


def _find_cached_real_qwen_files() -> dict[str, Path]:
    found: dict[str, Path] = {}
    temp = ROOT / "Temp"
    if not temp.exists():
        return found
    for name in ["Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"]:
        matches = sorted(
            (path for path in temp.rglob(name) if path.is_file() and path.stat().st_size > 1_000_000),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            found[name] = matches[0]
    return found


def _run_cached_materialization(row_id: str, evidence_dir: Path) -> dict:
    ref = HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF")
    cached = _find_cached_real_qwen_files()
    missing_cache = [name for name in ["Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"] if name not in cached]
    if missing_cache:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Real Qwen3 ASR GGUF+mmproj cached bytes are not present for downloader materialization proof.",
            block_reason="cached real Qwen3 ASR GGUF bytes are missing",
            external_requirement="Run audio_asr_gguf_mmproj with --allow-downloads first, or allow this row to be extended to download the selected files.",
            details={"repo_id": ref.repo_id, "missing_cache": missing_cache, "searched_root": str(ROOT / "Temp")},
        )

    choices, qwen_choice = _find_qwen_choice(QWEN3_ASR_GGUF_FILES, ref)
    if qwen_choice is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="HF downloader did not build the cached Qwen3 ASR GGUF+mmproj choice.",
            details={"choices": [choice.__dict__ for choice in choices]},
        )

    def copy_cached(_repo_id: str, _revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        source = cached[filename]
        target = destination / (relative_name or filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    destination = evidence_dir / "Models" / "mradermacher__Qwen3-ASR-0.6B-GGUF__Qwen3-ASR-0.6B.Q4_K_M"
    original_download = hf_model_downloader._download_file
    hf_model_downloader._download_file = copy_cached
    try:
        downloaded = download_choice(ref, qwen_choice, destination, print_func=lambda _text: None)
    finally:
        hf_model_downloader._download_file = original_download

    manifest = destination / "model_package.json"
    runnable, unsupported = scan_models(destination.parent)
    candidates = [candidate for candidate in [*runnable, *unsupported] if candidate.adapter_name == "gguf_asr_mmproj"]
    materialized = {path.name: path for path in downloaded if path.name in cached}
    failures: list[str] = []
    if not manifest.exists():
        failures.append("model_package.json was not written")
    if set(materialized) != set(cached):
        failures.append("real cached bytes were not materialized for every selected file")
    if not candidates:
        failures.append("materialized real-byte folder did not scan as GGUF ASR+mmproj")
    elif candidates[0].missing_files:
        failures.append("materialized GGUF ASR+mmproj candidate still reports missing files")

    return write_row(
        row_id,
        "fail" if failures else "pass",
        evidence_dir,
        summary=(
            "HF downloader materialized cached real Qwen3 ASR GGUF+mmproj bytes with an exact manifest."
            if not failures
            else "HF downloader cached real-byte materialization failed."
        ),
        details={
            "repo_id": ref.repo_id,
            "choice": qwen_choice.__dict__,
            "cached_sources": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in cached.items()
            },
            "materialized": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in materialized.items()
            },
            "manifest": str(manifest),
            "failures": failures,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "adapter_name": candidate.adapter_name,
                    "container_format": candidate.container_format,
                    "missing_files": candidate.missing_files,
                    "metadata": candidate.metadata,
                }
                for candidate in candidates
            ],
        },
        artifacts=[manifest],
    )


def _qwen_q4_q8_choice_index(files: list[str], ref: HFModelRef) -> int | None:
    _resolved_ref, choices = build_smart_download_choices(files, ref)
    for index, choice in enumerate(choices, 1):
        if (
            choice.task_hint == "asr_audio"
            and set(choice.files) == {"Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"}
        ):
            return index
    return None


def _run_noninteractive_flow(row_id: str, evidence_dir: Path) -> dict:
    return _run_product_downloader_flow(row_id, evidence_dir, files=QWEN3_ASR_GGUF_LIVE_LIKE_FILES, use_public_listing=False)


def _run_public_noninteractive_flow(row_id: str, evidence_dir: Path, allow_downloads: bool) -> dict:
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Product HF downloader public-listing flow requires network access and --allow-downloads.",
            block_reason="network downloads are disabled for this row",
            external_requirement="Run with --allow-downloads on a machine that can reach huggingface.co.",
            details={"repo_id": "mradermacher/Qwen3-ASR-0.6B-GGUF"},
        )
    return _run_product_downloader_flow(row_id, evidence_dir, files=None, use_public_listing=True)


def _run_public_download_to_asr(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Product HF downloader-to-ASR flow requires network access and --allow-downloads for live repo inspection.",
            block_reason="network downloads are disabled for this row",
            external_requirement="Run with --allow-downloads on a machine that can reach huggingface.co.",
            details={"repo_id": "mradermacher/Qwen3-ASR-0.6B-GGUF"},
        )
    download_stage_dir = evidence_dir / "download_stage"
    download_stage = _run_product_downloader_flow(
        f"{row_id}_download_stage",
        download_stage_dir,
        files=None,
        use_public_listing=True,
    )
    if download_stage.get("status") != "pass":
        return write_row(
            row_id,
            download_stage.get("status", "fail") if download_stage.get("status") in {"fail", "blocked"} else "fail",
            evidence_dir,
            summary="Product HF downloader-to-ASR flow could not materialize the selected Qwen3 ASR package.",
            block_reason=download_stage.get("block_reason", ""),
            external_requirement=download_stage.get("external_requirement", ""),
            details={"download_stage": download_stage},
            artifacts=[download_stage_dir / "row.json"],
        )
    destination = Path(download_stage.get("details", {}).get("destination", ""))
    if not destination.exists():
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Product HF downloader-to-ASR flow produced a passing download stage but the destination folder is missing.",
            details={"download_stage": download_stage},
            artifacts=[download_stage_dir / "row.json"],
        )
    return run_qwen3_asr_model_dir(
        row_id,
        evidence_dir,
        destination,
        install_deps,
        extra_details={
            "hf_downloader_stage": {
                "row_json": str(download_stage_dir / "row.json"),
                "repo_listing_source": download_stage.get("details", {}).get("repo_listing_source"),
                "repo_file_count": download_stage.get("details", {}).get("repo_file_count"),
                "selected_choice_index": download_stage.get("details", {}).get("selected_choice_index"),
                "destination": str(destination),
                "materialized": download_stage.get("details", {}).get("materialized", {}),
            }
        },
    )


def _run_public_real_download_to_asr(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Product HF downloader real-byte-to-ASR flow requires network access and --allow-downloads.",
            block_reason="network downloads are disabled for this row",
            external_requirement="Run with --allow-downloads on a machine that can reach huggingface.co.",
            details={"repo_id": "mradermacher/Qwen3-ASR-0.6B-GGUF"},
        )
    download_stage_dir = evidence_dir / "download_stage"
    download_stage = _run_product_downloader_flow(
        f"{row_id}_download_stage",
        download_stage_dir,
        files=None,
        use_public_listing=True,
        use_cached_bytes=False,
    )
    if download_stage.get("status") != "pass":
        return write_row(
            row_id,
            download_stage.get("status", "fail") if download_stage.get("status") in {"fail", "blocked"} else "fail",
            evidence_dir,
            summary="Product HF downloader real-byte-to-ASR flow could not materialize the selected Qwen3 ASR package.",
            block_reason=download_stage.get("block_reason", ""),
            external_requirement=download_stage.get("external_requirement", ""),
            details={"download_stage": download_stage},
            artifacts=[download_stage_dir / "row.json"],
        )
    destination = Path(download_stage.get("details", {}).get("destination", ""))
    if not destination.exists():
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Product HF downloader real-byte-to-ASR flow produced a passing download stage but the destination folder is missing.",
            details={"download_stage": download_stage},
            artifacts=[download_stage_dir / "row.json"],
        )
    return run_qwen3_asr_model_dir(
        row_id,
        evidence_dir,
        destination,
        install_deps,
        extra_details={
            "hf_downloader_stage": {
                "row_json": str(download_stage_dir / "row.json"),
                "repo_listing_source": download_stage.get("details", {}).get("repo_listing_source"),
                "repo_file_count": download_stage.get("details", {}).get("repo_file_count"),
                "selected_choice_index": download_stage.get("details", {}).get("selected_choice_index"),
                "byte_source": download_stage.get("details", {}).get("byte_source"),
                "destination": str(destination),
                "materialized": download_stage.get("details", {}).get("materialized", {}),
            }
        },
    )


def _run_product_downloader_flow(
    row_id: str,
    evidence_dir: Path,
    *,
    files: list[str] | None,
    use_public_listing: bool,
    use_cached_bytes: bool = True,
) -> dict:
    ref = HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF")
    cached = _find_cached_real_qwen_files()
    missing_cache = [name for name in ["Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"] if name not in cached]
    if missing_cache and use_cached_bytes:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Real Qwen3 ASR GGUF+mmproj cached bytes are not present for product downloader flow proof.",
            block_reason="cached real Qwen3 ASR GGUF bytes are missing",
            external_requirement="Run audio_asr_gguf_mmproj with --allow-downloads first, or allow this row to be extended to download the selected files.",
            details={"repo_id": ref.repo_id, "missing_cache": missing_cache, "searched_root": str(ROOT / "Temp")},
        )

    if files is None:
        try:
            files = list_repo_files(ref)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="Product HF downloader flow could not fetch the public Qwen3 repo listing.",
                block_reason=f"Hugging Face repo listing failed: {exc}",
                external_requirement="Network access to huggingface.co and the huggingface_hub package.",
                details={"repo_id": ref.repo_id, "error": str(exc)},
            )

    selected_index = _qwen_q4_q8_choice_index(files, ref)
    if selected_index is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="The product HF downloader flow did not expose a Q4_K_M plus mmproj-Q8_0 choice.",
            details={"repo_id": ref.repo_id, "files": files},
        )

    def copy_cached(_repo_id: str, _revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        if filename not in cached:
            raise RuntimeError(f"unexpected downloader request for {filename}")
        source = cached[filename]
        target = destination / (relative_name or filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    prompts: list[str] = []
    messages: list[str] = []

    def input_func(prompt: str = "") -> str:
        prompts.append(prompt)
        return str(selected_index)

    models_root = evidence_dir / "Models"
    original_download = hf_model_downloader._download_file
    original_list_repo_files = hf_model_downloader.list_repo_files
    if use_cached_bytes:
        hf_model_downloader._download_file = copy_cached
    if not use_public_listing:
        hf_model_downloader.list_repo_files = lambda _ref: list(files)
    try:
        destination = download_hf_model_from_ref(
            models_root,
            "https://huggingface.co/mradermacher/Qwen3-ASR-0.6B-GGUF",
            input_func=input_func,
            print_func=messages.append,
        )
    finally:
        hf_model_downloader._download_file = original_download
        hf_model_downloader.list_repo_files = original_list_repo_files

    failures: list[str] = []
    candidates = []
    manifest = None
    materialized: dict[str, Path] = {}
    if destination is None:
        failures.append("download_hf_model_from_ref returned no destination")
    else:
        manifest = destination / "model_package.json"
        materialized = {
            name: destination / name
            for name in ["Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"]
        }
        if not manifest.exists():
            failures.append("product downloader flow did not write model_package.json")
        for name, path in materialized.items():
            if not path.exists():
                failures.append(f"product downloader flow did not materialize {name}")
        runnable, unsupported = scan_models(destination)
        candidates = [candidate for candidate in [*runnable, *unsupported] if candidate.adapter_name == "gguf_asr_mmproj"]
        if not candidates:
            failures.append("product downloader flow output did not scan as GGUF ASR+mmproj")
        elif candidates[0].missing_files:
            failures.append("product downloader flow output still reports missing GGUF ASR+mmproj files")

    return write_row(
        row_id,
        "fail" if failures else "pass",
        evidence_dir,
        summary=(
            "Product HF downloader flow selected and materialized cached real Qwen3 ASR GGUF+mmproj bytes."
            if not failures
            else "Product HF downloader flow failed to materialize a runnable Qwen3 ASR GGUF+mmproj folder."
        ),
        details={
            "repo_id": ref.repo_id,
            "repo_listing_source": "public_huggingface" if use_public_listing else "live_like_fixture",
            "repo_file_count": len(files),
            "byte_source": "cached_redirect" if use_cached_bytes else "product_download",
            "selected_choice_index": selected_index,
            "prompts": prompts,
            "messages": messages,
            "destination": str(destination) if destination else "",
            "manifest": str(manifest) if manifest else "",
            "cached_sources": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in cached.items()
            },
            "materialized": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in materialized.items()
                if path.exists()
            },
            "failures": failures,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "adapter_name": candidate.adapter_name,
                    "container_format": candidate.container_format,
                    "missing_files": candidate.missing_files,
                    "metadata": candidate.metadata,
                }
                for candidate in candidates
            ],
        },
        artifacts=[manifest] if manifest else [],
    )


def _patched_downloader_flow(
    evidence_dir: Path,
    *,
    repo_id: str,
    files: list[str],
    selected_choice: str | None = None,
    repair_answers: list[str] | None = None,
    scan_override=None,
) -> tuple[Path | None, list[str], list[str]]:
    answers = iter([item for item in [selected_choice, *(repair_answers or [])] if item is not None])
    messages: list[str] = []
    prompts: list[str] = []

    def input_func(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(answers, "")

    def fake_download(_repo_id: str, _revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        target = destination / (relative_name or filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".json"):
            target.write_text("{}", encoding="utf-8")
        else:
            target.write_bytes(f"fixture:{filename}".encode("utf-8"))
        return target

    original_download = hf_model_downloader._download_file
    original_list_repo_files = hf_model_downloader.list_repo_files
    original_scan = None
    hf_model_downloader._download_file = fake_download
    hf_model_downloader.list_repo_files = lambda _ref: list(files)
    if scan_override is not None:
        import app.model_scanner as model_scanner

        original_scan = model_scanner.scan_models
        model_scanner.scan_models = scan_override
    try:
        destination = download_hf_model_from_ref(
            evidence_dir / "Models",
            repo_id,
            input_func=input_func,
            print_func=messages.append,
        )
    finally:
        hf_model_downloader._download_file = original_download
        hf_model_downloader.list_repo_files = original_list_repo_files
        if original_scan is not None:
            import app.model_scanner as model_scanner

            model_scanner.scan_models = original_scan
    return destination, prompts, messages


def _candidate_payloads(destination: Path | None) -> tuple[list[dict], list[dict]]:
    if destination is None or not destination.exists():
        return [], []
    runnable, unsupported = scan_models(destination.parent)

    def payload(candidate):
        return {
            "candidate_id": candidate.candidate_id,
            "adapter_name": candidate.adapter_name,
            "container_format": candidate.container_format,
            "task": candidate.task,
            "runnable": candidate.runnable,
            "missing_files": candidate.missing_files,
            "warnings": candidate.warnings,
        }

    return [payload(candidate) for candidate in runnable], [payload(candidate) for candidate in unsupported]


def _run_supported_outcome_taxonomy(row_id: str, evidence_dir: Path) -> dict:
    cases: dict[str, dict] = {}
    failures: list[str] = []

    runnable_dest, runnable_prompts, runnable_messages = _patched_downloader_flow(
        evidence_dir / "runnable_asr",
        repo_id="owner/fw-asr",
        files=["model.bin", "config.json", "tokenizer.json"],
    )
    runnable, unsupported = _candidate_payloads(runnable_dest)
    cases["complete_runnable_asr"] = {
        "destination": str(runnable_dest) if runnable_dest else "",
        "prompts": runnable_prompts,
        "messages": runnable_messages,
        "runnable": runnable,
        "unsupported": unsupported,
    }
    if not any(item["adapter_name"] == "faster_whisper" and item["runnable"] for item in runnable):
        failures.append("complete runnable ASR package did not scan as faster-whisper")

    repair_dir = evidence_dir / "missing_sidecar_repair" / "model"
    repair_dir.mkdir(parents=True, exist_ok=True)
    (repair_dir / "model.safetensors").write_bytes(b"fixture:model.safetensors")
    repair_choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("model.safetensors",),
        task_hint="metadata_required",
    )
    repair_prompts: list[str] = []
    repair_messages: list[str] = []
    repair_state = {"calls": 0}

    def repair_input(prompt: str = "") -> str:
        repair_prompts.append(prompt)
        return "y"

    def repair_download(_repo_id: str, _revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        target = destination / (relative_name or filename)
        target.write_text("{}", encoding="utf-8")
        return target

    def repair_scan(root: Path):
        repair_state["calls"] += 1
        return [], [
            ModelCandidate(
                candidate_id="incomplete",
                display_name="Incomplete",
                family_name="Incomplete",
                backend="transformers",
                container_format="safetensors",
                task="automatic-speech-recognition",
                precision="unknown",
                quantization_label="Unknown precision",
                path=root,
                adapter_name="hf_transformers_asr",
                runnable=False,
                missing_files=["config.json", "preprocessor_config.json"],
            )
        ]

    original_download = hf_model_downloader._download_file
    hf_model_downloader._download_file = repair_download
    import app.model_scanner as model_scanner

    original_scan = model_scanner.scan_models
    model_scanner.scan_models = repair_scan
    try:
        interactive_repair_plan = offer_missing_file_repair(
            HFModelRef("owner/hf-asr"),
            repair_choice,
            ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json"],
            repair_dir,
            input_func=repair_input,
            print_func=repair_messages.append,
        )
    finally:
        hf_model_downloader._download_file = original_download
        model_scanner.scan_models = original_scan

    structured_repair_plan = build_missing_file_repair_plan(
        HFModelRef("owner/hf-asr"),
        repair_choice,
        ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json"],
        repair_dir,
        [
            ModelCandidate(
                candidate_id="incomplete",
                display_name="Incomplete",
                family_name="Incomplete",
                backend="transformers",
                container_format="safetensors",
                task="automatic-speech-recognition",
                precision="unknown",
                quantization_label="Unknown precision",
                path=repair_dir,
                adapter_name="hf_transformers_asr",
                runnable=False,
                missing_files=["config.json", "preprocessor_config.json"],
            )
        ],
    )
    repair_files = sorted(path.name for path in repair_dir.iterdir())
    repair_plan_path = repair_dir / "hf_model_layout_repair_plan.json"
    cases["missing_sidecar_repair"] = {
        "destination": str(repair_dir),
        "prompts": repair_prompts,
        "messages": repair_messages,
        "files": repair_files,
        "repair_plan_path": str(repair_plan_path),
        "repair_plan_file_exists": repair_plan_path.exists(),
        "scan_calls": repair_state["calls"],
        "interactive_repair_plan": interactive_repair_plan,
        "structured_repair_plan": structured_repair_plan,
    }
    if not {"config.json", "preprocessor_config.json"} <= set(repair_files):
        failures.append("missing-sidecar repair did not download exact metadata matches")
    if not repair_plan_path.exists():
        failures.append("missing-sidecar repair did not persist model-layout repair plan")
    plan_record = structured_repair_plan["records"][0] if structured_repair_plan["records"] else {}
    if structured_repair_plan.get("schema") != "easy_asr_bench.model_layout_repair_plan.v1":
        failures.append("missing-sidecar repair plan used the wrong schema")
    if plan_record.get("repair_action") != "download_exact_missing_files":
        failures.append("missing-sidecar repair plan did not choose exact missing-file repair")
    if set(plan_record.get("safe_download_files", [])) != {"config.json", "preprocessor_config.json"}:
        failures.append("missing-sidecar repair plan did not record exact safe sidecar downloads")
    execution = interactive_repair_plan.get("last_execution", {}) if isinstance(interactive_repair_plan, dict) else {}
    if execution.get("schema") != "easy_asr_bench.model_layout_repair_execution.v1":
        failures.append("missing-sidecar repair did not record structured execution evidence")
    if execution.get("summary", {}).get("repaired") != 1:
        failures.append("missing-sidecar repair execution did not record one repaired issue")

    reference_dest, reference_prompts, reference_messages = _patched_downloader_flow(
        evidence_dir / "reference_llm",
        repo_id="owner/llm",
        files=["Model-Q4_K_M.gguf"],
    )
    runnable, unsupported = _candidate_payloads(reference_dest)
    cases["gguf_reference_llm"] = {
        "destination": str(reference_dest) if reference_dest else "",
        "prompts": reference_prompts,
        "messages": reference_messages,
        "runnable": runnable,
        "unsupported": unsupported,
    }
    if not any(item["adapter_name"] == "gguf_llm_reference" and item["task"] == "llm-corrected-reference" for item in unsupported):
        failures.append("GGUF reference LLM was not routed as reference-only")

    unsafe_dest, unsafe_prompts, unsafe_messages = _patched_downloader_flow(
        evidence_dir / "unsafe_unknown",
        repo_id="https://huggingface.co/owner/unsafe/tree/main/custom",
        files=["custom/custom.pt"],
        repair_answers=["DOWNLOAD"],
    )
    runnable, unsupported = _candidate_payloads(unsafe_dest)
    cases["unsafe_or_unknown_inspection"] = {
        "destination": str(unsafe_dest) if unsafe_dest else "",
        "prompts": unsafe_prompts,
        "messages": unsafe_messages,
        "runnable": runnable,
        "unsupported": unsupported,
    }
    if unsafe_dest is None or not any("unknown package layout" in message for message in unsafe_messages):
        failures.append("unknown/unsafe repo did not require inspection confirmation")
    if runnable:
        failures.append("unknown/unsafe repo produced runnable candidates")

    artifacts = [repair_plan_path] if repair_plan_path.exists() else []
    return write_row(
        row_id,
        "fail" if failures else "pass",
        evidence_dir,
        summary=(
            "HF downloader routed supported repo outcomes into runnable ASR, sidecar repair, reference LLM, and inspection-only paths."
            if not failures
            else "HF downloader supported-outcome taxonomy validation failed."
        ),
        details={"cases": cases, "failures": failures},
        artifacts=artifacts,
    )
