import json

from qa.runtime_matrix.common import write_row


def test_runtime_matrix_row_evidence_schema_for_pass(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")

    row = write_row("example", "pass", tmp_path / "evidence", summary="passed", artifacts=[artifact])

    written = json.loads((tmp_path / "evidence" / "row.json").read_text(encoding="utf-8"))
    assert written == row
    assert written["schema"] == "easy_asr_bench.runtime_matrix.row.v1"
    assert written["status"] == "pass"
    assert written["artifacts"][0]["sha256"].startswith("sha256:")


def test_runtime_matrix_row_evidence_schema_for_blocked(tmp_path):
    row = write_row(
        "nvidia_cuda_torch_onnx_faster_whisper_llama",
        "blocked",
        tmp_path / "evidence",
        summary="No NVIDIA GPU",
        block_reason="No NVIDIA GPU was detected.",
        external_requirement="NVIDIA CUDA GPU",
    )

    assert row["status"] == "blocked"
    assert row["block_reason"] == "No NVIDIA GPU was detected."
    assert row["external_requirement"] == "NVIDIA CUDA GPU"
