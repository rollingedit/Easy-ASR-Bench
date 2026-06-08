from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.runtime_matrix.registry import ROWS
from qa.runtime_matrix.common import write_row


NETWORK_HARDWARE = {"network"}
EXTERNAL_HARDWARE = {
    "clean_windows_vm",
    "win10_vm",
    "interactive_windows_shell",
    "nvidia_cuda",
}


def _row_plan(include_network: bool, include_external: bool, selected_rows: set[str] | None = None) -> list[dict]:
    plan: list[dict] = []
    for row_id, definition in sorted(ROWS.items()):
        if selected_rows is not None and row_id not in selected_rows:
            continue
        hardware = definition.hardware
        if hardware in NETWORK_HARDWARE and not include_network:
            action = "blocked_without_network"
        elif hardware in EXTERNAL_HARDWARE and not include_external:
            action = "run_for_blocked_evidence"
        else:
            action = "run"
        plan.append(
            {
                "id": row_id,
                "module": definition.module,
                "hardware": hardware,
                "action": action,
            }
        )
    return plan


def _run_row(row_id: str, workdir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    command = [
        sys.executable,
        "qa/runtime_matrix/run_row.py",
        "--row",
        row_id,
        "--workdir",
        str(workdir),
    ]
    if install_deps:
        command.append("--install-deps")
    if allow_downloads:
        command.append("--allow-downloads")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=3600)
    row_json = workdir / row_id / "row.json"
    row_status = "missing"
    row_payload = {}
    if row_json.exists():
        try:
            row_payload = json.loads(row_json.read_text(encoding="utf-8"))
            row_status = str(row_payload.get("status", "missing"))
        except json.JSONDecodeError:
            row_status = "invalid_json"
    return {
        "id": row_id,
        "exit_code": completed.returncode,
        "status": row_status,
        "row_json": str(row_json),
        "summary": row_payload.get("summary", ""),
        "block_reason": row_payload.get("block_reason", ""),
        "external_requirement": row_payload.get("external_requirement", ""),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _write_network_blocked(row_id: str, workdir: Path) -> dict:
    evidence_dir = workdir / row_id
    row = write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="Network-gated row was not run because local matrix execution did not include network downloads.",
        block_reason="Network downloads were not enabled for this local matrix run.",
        external_requirement="rerun with --include-network --allow-downloads",
        details={"local_runner_action": "blocked_without_network"},
    )
    return {
        "id": row_id,
        "exit_code": 0,
        "status": row["status"],
        "row_json": str(evidence_dir / "row.json"),
        "summary": row["summary"],
        "block_reason": row.get("block_reason", ""),
        "external_requirement": row.get("external_requirement", ""),
        "stdout_tail": "",
        "stderr_tail": "",
    }


def write_plan(workdir: Path, include_network: bool, include_external: bool, selected_rows: set[str] | None = None) -> dict:
    plan = {
        "schema": "easy_asr_bench.runtime_matrix.local_plan.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "include_network": include_network,
        "include_external": include_external,
        "selected_rows": sorted(selected_rows) if selected_rows is not None else [],
        "rows": _row_plan(include_network, include_external, selected_rows),
    }
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Easy ASR Bench runtime-matrix rows suitable for local Windows evidence.")
    parser.add_argument("--workdir", type=Path, default=ROOT / "Temp" / "runtime_matrix_local")
    parser.add_argument("--install-deps", action="store_true", help="Allow rows to invoke product dependency repair/install paths.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow rows to download model/media fixtures.")
    parser.add_argument("--include-network", action="store_true", help="Run network-gated rows. Requires --allow-downloads for rows that fetch public fixtures.")
    parser.add_argument("--include-external", action="store_true", help="Run VM/interactive/CUDA rows instead of keeping them as local blocked evidence.")
    parser.add_argument("--row", action="append", choices=sorted(ROWS), help="Run only this row. May be passed multiple times.")
    parser.add_argument("--plan-only", action="store_true", help="Write plan.json without running rows.")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    selected_rows = set(args.row) if args.row else None
    plan = write_plan(args.workdir, args.include_network, args.include_external, selected_rows)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0

    results = []
    for row in plan["rows"]:
        if row["action"] == "blocked_without_network":
            results.append(_write_network_blocked(row["id"], args.workdir))
            continue
        results.append(_run_row(row["id"], args.workdir, args.install_deps, args.allow_downloads))

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    summary = {
        "schema": "easy_asr_bench.runtime_matrix.local_summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workdir": str(args.workdir),
        "install_deps": args.install_deps,
        "allow_downloads": args.allow_downloads,
        "include_network": args.include_network,
        "include_external": args.include_external,
        "planned_rows": len(plan["rows"]),
        "run_rows": len(results),
        "network_blocked_rows": sum(1 for row in plan["rows"] if row["action"] == "blocked_without_network"),
        "status_counts": counts,
        "results": results,
    }
    summary_path = args.workdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(result["status"] in {"pass", "blocked"} and result["exit_code"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
