from __future__ import annotations

from pathlib import Path

from .base import ModelCandidate
from ..llm_reference_prompt import build_llm_reference_prompt
from ..precision_detector import detect_from_path


class GGUFLLMReferenceAdapter:
    name = "gguf_llm_reference"

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for path in models_root.rglob("*.gguf"):
            candidates.append(self.candidate_for_file(path))
        return candidates

    def discover_path(self, path: Path) -> list[ModelCandidate]:
        path = path.expanduser()
        if path.is_file() and path.suffix.lower() == ".gguf":
            return [self.candidate_for_file(path)]
        if path.is_dir():
            return self.discover(path)
        return []

    def candidate_for_file(self, path: Path) -> ModelCandidate:
        raw, bucket = detect_from_path(path)
        candidate_key = str(path.resolve()).lower().replace("\\", "_").replace(":", "")
        return ModelCandidate(
            candidate_id=f"gguf_reference__{candidate_key}",
            display_name=path.name,
            family_name=path.stem,
            backend="llama.cpp",
            container_format="gguf",
            task="llm-corrected-reference",
            precision=raw,
            quantization_label=bucket,
            path=path,
            adapter_name=self.name,
            runnable=False,
            category="reference_llm",
            runnable_after_dependency_install=True,
            warnings=["GGUF text LLMs are supported as optional reference/correction models, not direct ASR models."],
            help_text="Install llama_cpp support to use this local LLM for transcript correction/reference generation.",
        )

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["llama_cpp"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as exc:
            raise RuntimeError("GGUF reference support requires llama-cpp-python. Install requirements/llama_cpp.txt.") from exc
        return Llama(model_path=str(candidate.path), n_ctx=int(runtime_config.get("llm_context_tokens", 8192)))

    def generate_reference(self, candidate: ModelCandidate, runtime_config: dict, results: dict) -> dict:
        llm = self.load(candidate, runtime_config)
        prompt = build_llm_reference_prompt(results)
        response = llm(
            prompt,
            max_tokens=int(runtime_config.get("llm_reference_max_tokens", 2048)),
            temperature=float(runtime_config.get("llm_reference_temperature", 0.1)),
            stop=["END_RESULTS_JSON"],
        )
        text = response["choices"][0]["text"] if isinstance(response, dict) else str(response)
        return {
            "candidate_id": candidate.candidate_id,
            "display_name": candidate.display_name,
            "raw_response": text.strip(),
            "status": "generated",
        }

    def discover_reference_only(self) -> bool:
        return True
