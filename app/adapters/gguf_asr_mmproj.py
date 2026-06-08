from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import soundfile as sf

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import process_memory_mb
from ..precision_detector import detect_from_path
from ..runtime_plan import resolve_runtime_plan


class GGUFASRMMProjAdapter:
    name = "gguf_asr_mmproj"

    def __init__(self) -> None:
        self.candidate: ModelCandidate | None = None
        self.runtime_config: dict = {}
        self.model = None
        self.backend = ""

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for folder in [models_root, *[path for path in models_root.rglob("*") if path.is_dir()]]:
            pair = find_gguf_asr_pair(folder)
            if pair is None:
                continue
            main_model, projector, missing, warnings = pair
            raw, bucket = detect_from_path(main_model if main_model else folder)
            complete_pair = main_model is not None and projector is not None and not missing
            runnable = complete_pair
            warnings = list(warnings)
            candidates.append(
                ModelCandidate(
                    candidate_id=f"gguf_asr_mmproj__{folder.name}".lower().replace(" ", "_"),
                    display_name=f"{folder.name} (GGUF ASR + mmproj)",
                    family_name=folder.name,
                    backend="llama.cpp",
                    container_format="gguf+mmproj",
                    task="automatic-speech-recognition",
                    precision=raw,
                    quantization_label=bucket,
                    path=folder,
                    adapter_name=self.name,
                    runnable=runnable,
                    runnable_after_dependency_install=runnable,
                    dependency_groups=["llama_cpp", "llama_mtmd"],
                    missing_files=missing,
                    warnings=warnings if (warnings or complete_pair) else ["Audio/ASR GGUF package is incomplete or has no matching mmproj."],
                    help_text=(
                        "Recognized ASR GGUF/projector package. Keep the main .gguf and exact mmproj .gguf together; "
                        "Easy ASR Bench will use llama-mtmd-cli or llama-cpp-python Qwen3ASRChatHandler when available."
                    ),
                    metadata={
                        "model_path": str(main_model) if main_model else "",
                        "mmproj_path": str(projector) if projector else "",
                        "model_status": "",
                    },
                )
            )
        return candidates

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["llama_cpp", "llama_mtmd"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        self.candidate = candidate
        self.runtime_config = runtime_config
        model_path = Path(candidate.metadata.get("model_path", ""))
        mmproj_path = Path(candidate.metadata.get("mmproj_path", ""))
        if not model_path.exists() or not mmproj_path.exists():
            raise RuntimeError("GGUF ASR requires both the main .gguf model and matching mmproj .gguf projector.")
        python_backend = load_python_backend(model_path, mmproj_path, runtime_config)
        if python_backend is not None:
            self.model = python_backend
            self.backend = "llama-cpp-python-qwen3-asr"
            return self
        cli = llama_mtmd_cli_path(runtime_config, model_path.parent)
        if cli:
            self.model = {"cli": cli, "model_path": model_path, "mmproj_path": mmproj_path}
            self.backend = "llama-mtmd-cli"
            return self
        raise RuntimeError(
            "GGUF ASR requires llama-cpp-python with Qwen3ASRChatHandler or llama-mtmd-cli from llama.cpp on PATH/in the model folder."
        )

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        if self.candidate is None or self.model is None:
            raise RuntimeError("Adapter is not loaded")
        out: list[ChunkTranscript] = []
        errors: list[str] = []
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        with tempfile.TemporaryDirectory(prefix="easy-asr-gguf-") as temp:
            temp_dir = Path(temp)
            for chunk, metadata in zip(chunks, chunk_metadata):
                started = time.perf_counter()
                audio_path = temp_dir / f"{metadata['chunk_id']}.wav"
                try:
                    sf.write(audio_path, chunk.samples, 16000)
                    text = self._transcribe_audio(audio_path)
                except Exception as exc:
                    text = f"[ERROR: chunk failed: {exc}]"
                    errors.append(f"{metadata['chunk_id']}: {exc}")
                inference_seconds += time.perf_counter() - started
                peak_ram = max(peak_ram, process_memory_mb())
                out.append(ChunkTranscript(str(metadata["chunk_id"]), float(metadata["start_seconds"]), float(metadata["end_seconds"]), text.strip()))
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        return ModelRunResult(
            self.candidate,
            out,
            {
                "provider": self.backend,
                "device": "auto",
                "audio_seconds": audio_seconds,
                "chunk_count": len(chunks),
                "inference_seconds": inference_seconds,
                "total_wall_seconds": inference_seconds,
                "peak_process_memory_mb": peak_ram,
                "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds),
            },
            errors,
        )

    def _transcribe_audio(self, audio_path: Path) -> str:
        if self.backend == "llama-cpp-python-qwen3-asr":
            return transcribe_with_python_backend(self.model, audio_path, self.runtime_config)
        return transcribe_with_cli_backend(self.model, audio_path, self.runtime_config)

    def unload(self) -> None:
        self.candidate = None
        self.runtime_config = {}
        self.model = None
        self.backend = ""


def find_gguf_asr_pair(folder: Path) -> tuple[Path | None, Path | None, list[str], list[str]] | None:
    ggufs = [path for path in folder.glob("*.gguf") if path.is_file()]
    if not ggufs:
        return None
    manifest_pair = find_manifest_gguf_pair(folder)
    if manifest_pair is not None:
        return manifest_pair
    projectors = [path for path in ggufs if is_mmproj_gguf(path)]
    main_models = [path for path in ggufs if path not in projectors]
    asr_named = any(any(signal in path.name.lower() for signal in ["asr", "audio", "whisper"]) for path in ggufs)
    if not projectors and not asr_named:
        return None
    matching = [(main, projector) for main in main_models for projector in projectors if gguf_projector_matches(main, projector, main_models, projectors)]
    missing = []
    warnings = []
    if not main_models:
        missing.append("main ASR/audio model .gguf")
    if not matching:
        missing.append("matching mmproj .gguf")
    if len(matching) > 1:
        missing.append("model_package.json exact GGUF ASR pairing manifest")
        warnings.append("Multiple plausible GGUF ASR/mmproj pairs were found; add model_package.json to choose the exact pair.")
        return None, None, sorted(set(missing)), warnings
    main_model, projector = matching[0] if matching else ((main_models[0] if main_models else None), None)
    return main_model, projector, missing, warnings


def find_manifest_gguf_pair(folder: Path) -> tuple[Path | None, Path | None, list[str], list[str]] | None:
    manifest = folder / "model_package.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, ["valid model_package.json"], [f"model_package.json could not be parsed: {exc}"]
    artifacts = data.get("artifacts", {}) if isinstance(data, dict) else {}
    main_name = artifacts.get("main_model")
    projector_name = artifacts.get("projector")
    if not isinstance(main_name, str) or not isinstance(projector_name, str):
        return None, None, ["artifacts.main_model", "artifacts.projector"], ["model_package.json must name the exact main_model and projector."]
    main = folder / main_name
    projector = folder / projector_name
    missing = []
    if not main.exists():
        missing.append(main_name)
    if not projector.exists():
        missing.append(projector_name)
    return (main if main.exists() else None), (projector if projector.exists() else None), missing, []


def gguf_projector_matches(main_model: Path, projector: Path, main_models: list[Path], projectors: list[Path]) -> bool:
    main_stem = main_model.stem.lower()
    projector_stem = projector.stem.lower()
    if main_stem in projector_stem:
        return True
    main_quant = gguf_quant_label(main_model.name)
    projector_quant = gguf_quant_label(projector.name)
    if main_quant and projector_quant and main_quant != projector_quant:
        return False
    if main_quant and projector_quant and main_quant == projector_quant:
        return True
    return len(main_models) == 1 and len(projectors) == 1


def is_mmproj_gguf(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".gguf") and (name.startswith(("mmproj", "mmproj-")) or "mmproj" in name)


def gguf_quant_label(name: str) -> str:
    import re

    match = re.search(
        r"(IQ[1-4]_[A-Z0-9_]+|Q[2-8]_[A-Z0-9_]+|Q[2-8]|F16|F32|BF16|BF8|INT[2-8]|NF4|NVFP4|NVP4|FP4|FP8)",
        name,
        re.IGNORECASE,
    )
    return match.group(1).lower() if match else ""


def load_python_backend(model_path: Path, mmproj_path: Path, runtime_config: dict):
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen3ASRChatHandler
    except Exception:
        return None
    plan = resolve_runtime_plan("llama_cpp", runtime_config)
    n_gpu_layers = -1 if plan.actual_provider in {"cuda", "vulkan", "hip"} and plan.backend_verified else 0
    return Llama(
        model_path=str(model_path),
        chat_handler=Qwen3ASRChatHandler(clip_model_path=str(mmproj_path), verbose=False),
        n_gpu_layers=n_gpu_layers,
        n_ctx=int(runtime_config.get("llama_cpp", {}).get("context_size", 10240)),
        verbose=False,
    )


def llama_mtmd_cli_path(runtime_config: dict, model_folder: Path) -> str:
    configured = runtime_config.get("llama_cpp", {}).get("mtmd_cli_path", "")
    candidates = [configured] if configured else []
    candidates.extend(
        [
            str(model_folder / "llama-mtmd-cli.exe"),
            str(model_folder / "llama-mtmd-cli"),
            shutil.which("llama-mtmd-cli.exe") or "",
            shutil.which("llama-mtmd-cli") or "",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def transcribe_with_python_backend(model, audio_path: Path, runtime_config: dict) -> str:
    prompt = runtime_value(runtime_config, "ar_prompt", "Transcribe the audio.")
    with audio_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a speech-to-text model. Return only the transcript."},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}, {"type": "text", "text": prompt}]},
        ],
        temperature=float(runtime_value(runtime_config, "temperature", 0.0)),
        max_tokens=int(runtime_value(runtime_config, "ar_max_new_tokens", 1024)),
    )
    return response["choices"][0]["message"]["content"]


def transcribe_with_cli_backend(model: dict, audio_path: Path, runtime_config: dict) -> str:
    prompt = runtime_value(runtime_config, "ar_prompt", "Transcribe the audio.")
    command = [
        str(model["cli"]),
        "-m",
        str(model["model_path"]),
        "--mmproj",
        str(model["mmproj_path"]),
        "--audio",
        str(audio_path),
        "-p",
        prompt,
        "-n",
        str(int(runtime_value(runtime_config, "ar_max_new_tokens", 1024))),
        "--temp",
        str(float(runtime_value(runtime_config, "temperature", 0.0))),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=int(runtime_value(runtime_config, "timeout_seconds", 600, "llama_cpp")))
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or f"llama-mtmd-cli exited {completed.returncode}").strip())
    return extract_cli_transcript(completed.stdout or completed.stderr)


def runtime_value(runtime_config: dict, key: str, default, nested_section: str = "transcription"):
    if key in runtime_config:
        return runtime_config[key]
    nested = runtime_config.get(nested_section, {})
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return default


def extract_cli_transcript(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    transcript_lines = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith(("llama_", "main:", "system_info:", "sampler", "generate:", "common_", "mtmd_", "init_", "load:", "print_info:", "warn:")):
            continue
        if re.match(r"^\d+(?:\.\d+){3}\s+[iwe]\s+", line, re.IGNORECASE):
            continue
        if lowered.startswith(("https://", "http://")):
            continue
        line = re.sub(r"^language\s+\w+\s*<asr_text>", "", line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        transcript_lines.append(line)
    return "\n".join(transcript_lines).strip() or output.strip()
