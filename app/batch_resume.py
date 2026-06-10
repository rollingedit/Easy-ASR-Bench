from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .adapters.base import ModelCandidate
from .utils import file_key, read_json, write_json_atomic


SCHEMA = "easy_asr_bench.batch_resume_manifest.v1"


def batch_resume_path(config: dict) -> Path:
    logs = Path(str(config.get("folders", {}).get("logs", "Logs")))
    return logs / "batch_resume_manifest.json"


def batch_signature(config: dict, selected: list[ModelCandidate], reference_llm: ModelCandidate | None = None) -> str:
    relevant_config = {
        "runtime": config.get("runtime", {}),
        "transcription": config.get("transcription", {}),
        "whisper": config.get("whisper", {}),
        "chunking": config.get("chunking", {}),
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "reference_llm_candidate_id": reference_llm.candidate_id if reference_llm else "",
    }
    data = json.dumps(relevant_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class BatchResumeManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = read_json(path, {"schema": SCHEMA, "pairs": []})
        if self.data.get("schema") != SCHEMA:
            self.data = {"schema": SCHEMA, "pairs": []}

    def save(self) -> None:
        write_json_atomic(self.path, self.data)

    def completed_output_for(self, source: Path, selected: list[ModelCandidate], signature: str) -> str:
        try:
            key = file_key(source)
        except OSError:
            return ""
        wanted = {candidate.candidate_id for candidate in selected}
        if not wanted:
            return ""
        matched = [
            pair
            for pair in self.data.get("pairs", [])
            if pair.get("source_fast_key") == key
            and pair.get("batch_signature") == signature
            and pair.get("candidate_id") in wanted
            and pair.get("status") == "done"
            and pair.get("output_path")
            and (Path(str(pair.get("output_path"))) / "results.json").exists()
        ]
        if {pair.get("candidate_id") for pair in matched} != wanted:
            return ""
        outputs = {str(pair.get("output_path")) for pair in matched}
        return sorted(outputs)[0] if len(outputs) == 1 else ""

    def record_file(self, source: Path, selected: list[ModelCandidate], signature: str, output_path: Path | None, status: str) -> None:
        try:
            key = file_key(source)
        except OSError:
            return
        resolved = str(source.resolve())
        output = str(output_path or "")
        pairs = self.data.setdefault("pairs", [])
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for candidate in selected:
            record = {
                "source_path": resolved,
                "source_fast_key": key,
                "candidate_id": candidate.candidate_id,
                "batch_signature": signature,
                "status": status,
                "output_path": output,
                "updated_at": now,
            }
            for existing in pairs:
                if (
                    existing.get("source_fast_key") == key
                    and existing.get("candidate_id") == candidate.candidate_id
                    and existing.get("batch_signature") == signature
                ):
                    existing.update(record)
                    break
            else:
                record["created_at"] = now
                pairs.append(record)
        self.save()
