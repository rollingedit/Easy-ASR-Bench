# Release Verification

Release verification is artifact-first. A release is not considered ready because source tests pass; the downloaded GitHub assets must also validate.

## Local Pre-Publish Gate

Run from a clean working tree after the version bump and final metadata regeneration:

```powershell
python scripts\validate_release_files.py
python scripts\validate_physical_files.py --repo .
python scripts\check_release_version_coherence.py --tag vX.Y.Z
python -m compileall app tests scripts
python -m pytest --basetemp .pytest_tmp -p no:cacheprovider
python scripts\build_release_zip.py --version vX.Y.Z --strict-checksums
python scripts\validate_physical_files.py --zip dist\Easy-ASR-Bench-vX.Y.Z-win.zip
python scripts\write_release_smoke.py --tag vX.Y.Z --commit <commit> --output release-smoke-vX.Y.Z.json
python scripts\validate_release_smoke.py --smoke release-smoke-vX.Y.Z.json --required tests\fixtures\release_required_rows_v2.json
cmd /c setup.bat --dry-run --local
cmd /c setup.bat --dry-run --local --json
python qa\runtime_matrix\run_row.py --row windows_vc_runtime_repair_contract --workdir Temp\runtime_matrix_vc_runtime_repair_contract
python qa\runtime_matrix\run_row.py --row python_packaging_tools_repair_contract --workdir Temp\runtime_matrix_python_packaging_repair_contract
python qa\runtime_matrix\run_row.py --row transformers_cpu_dependency_repair_contract --workdir Temp\runtime_matrix_transformers_cpu_dependency_repair_contract
python qa\runtime_matrix\run_row.py --row media_tools_dependency_repair_contract --workdir Temp\runtime_matrix_media_tools_dependency_repair_contract
python qa\runtime_matrix\run_row.py --row llama_mtmd_dependency_repair_contract --workdir Temp\runtime_matrix_llama_mtmd_dependency_repair_contract
python qa\runtime_matrix\run_row.py --row directml_provider_conflict_repair --workdir Temp\runtime_matrix_directml_conflict_repair
python qa\runtime_matrix\run_row.py --row faster_whisper_pkg_resources_repair --workdir Temp\runtime_matrix_faster_whisper_pkg_resources_repair
python qa\runtime_matrix\run_row.py --row faster_whisper_ctranslate2_candidate_fallback_repair --workdir Temp\runtime_matrix_faster_whisper_ctranslate2_candidate_fallback_repair
python qa\runtime_matrix\run_row.py --row faster_whisper_vc_runtime_repair --workdir Temp\runtime_matrix_faster_whisper_vc_runtime_repair
python qa\runtime_matrix\run_row.py --row transformers_cuda_unavailable_cpu_fallback --workdir Temp\runtime_matrix_transformers_cuda_unavailable_cpu_fallback
python qa\runtime_matrix\run_row.py --row openai_whisper_cuda_unavailable_cpu_fallback --workdir Temp\runtime_matrix_openai_whisper_cuda_unavailable_cpu_fallback
python qa\runtime_matrix\run_row.py --row setup_repair_all_safe --workdir Temp\runtime_matrix_setup_repair_all_safe
python qa\runtime_matrix\run_row.py --row setup_dry_run_json --workdir Temp\runtime_matrix_setup_dry_run_json
python qa\runtime_matrix\run_row.py --row repair_all_safe_failure_isolation --workdir Temp\runtime_matrix_repair_all_safe_failure_isolation
python qa\runtime_matrix\run_row.py --row repair_all_safe_stale_cached_resolution --workdir Temp\runtime_matrix_repair_all_safe_stale_cached_resolution
python qa\runtime_matrix\run_row.py --row repair_plan_issue_classification_contract --workdir Temp\runtime_matrix_repair_plan_issue_classification_contract
python qa\runtime_matrix\run_row.py --row setup_repair_model_layouts --workdir Temp\runtime_matrix_setup_repair_model_layouts
python qa\runtime_matrix\run_row.py --row clean_vm_zero_dependency_bootstrap --workdir Temp\runtime_matrix_clean_vm_bootstrap
python qa\runtime_matrix\run_row.py --row report_atomic_write_failure_cleanup --workdir Temp\runtime_matrix_report_atomic_write_failure_cleanup
python qa\runtime_matrix\run_row.py --row model_fixture_quality_claims --workdir Temp\runtime_matrix_model_fixture_quality_claims
python qa\runtime_matrix\run_row.py --row hf_downloader_package_variant_taxonomy --workdir Temp\runtime_matrix_hf_downloader_package_variant_taxonomy
python qa\runtime_matrix\run_row.py --row generic_onnx_openvino_unavailable_cpu_fallback --workdir Temp\runtime_matrix_generic_onnx_openvino_unavailable_cpu_fallback
python -m app.doctor --config config.json --validate-real-smoke
Remove-Item -LiteralPath dist\release-assets -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path dist\release-assets
Copy-Item dist\Easy-ASR-Bench-vX.Y.Z-win.zip dist\release-assets\
Copy-Item setup.bat dist\release-assets\
Copy-Item installer\install.ps1 dist\release-assets\install.ps1
Copy-Item installer\manifest.json dist\release-assets\
Copy-Item installer\checksums.json dist\release-assets\
cmd /c setup.bat --dry-run --verify-release --asset-dir dist\release-assets
python -m app.doctor --config config.json --strict
```

The staged `--asset-dir` check must run before a draft release is published. It validates the same `setup.bat`, `install.ps1`, `manifest.json`, `checksums.json`, and ZIP bytes that will become release assets, without depending on public release URLs.

`setup.bat --dry-run --json` emits a final `easy_asr_bench.setup_dry_run.v1` JSON object after the normal human-readable dry-run output. The `setup_dry_run_json` runtime-matrix row parses that object and verifies setup dry-run machine-readable fields such as mode, version, exit code, installer presence, install folder, and `no_files_modified=true`.

The `windows_vc_runtime_repair_contract` runtime-matrix row is a non-destructive repair contract for native Windows ASR dependencies. It simulates a missing Microsoft Visual C++ 2015-2022 Redistributable x64 state, verifies the bootstrap repair path invokes `winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements`, and verifies the post-repair redistributable probe passes.

The `python_packaging_tools_repair_contract` runtime-matrix row is a non-destructive repair contract for the packaging layer that drives all pip-based dependency repair. It simulates missing `pkg_resources`, routes through the shared `app.repair_plan.execute_repair_plan` path for the `python_packaging` dependency group, verifies `requirements\python_packaging.txt` is the repair target, reprobes `pip`, `setuptools`, and `pkg_resources`, and persists a runtime-resolution record.

The `transformers_cpu_dependency_repair_contract` runtime-matrix row is a non-destructive repair contract for the Hugging Face Transformers ASR dependency group. It simulates a mixed missing/outdated `transformers_cpu` stack, routes through `app.repair_plan.execute_repair_plan`, verifies `requirements\transformers_cpu.txt` is the repair target, reprobes `torch`, `transformers`, `safetensors`, `sentencepiece`, `google.protobuf`, and `torchaudio`, and persists a runtime-resolution record.

The `media_tools_dependency_repair_contract` runtime-matrix row is a non-destructive repair contract for audio/video conversion prerequisites. It simulates missing `imageio_ffmpeg` and FFmpeg executable evidence, routes through `app.repair_plan.execute_repair_plan`, verifies `requirements\core.txt` is the repair target, reprobes the media-tools FFmpeg runtime, records the FFmpeg path, and persists a runtime-resolution record used by real audio/video smoke rows.

The `llama_mtmd_dependency_repair_contract` runtime-matrix row is a non-destructive repair contract for the native ASR GGUF+mmproj runtime. It simulates missing `llama-mtmd-cli` / Qwen3 ASR handler evidence, routes through `app.repair_plan.execute_repair_plan`, verifies the dependency group remains a `native_tool` repair with the llama.cpp `winget` recovery command, reprobes `llama_mtmd_runtime_probe`, records the resolved CLI path, and persists a runtime-resolution record used by ASR GGUF+mmproj rows.

The `directml_provider_conflict_repair` runtime-matrix row is a non-destructive repair contract for the reproduced ONNX Runtime DirectML conflict. It simulates plain `onnxruntime` blocking `DmlExecutionProvider`, routes through the product `install_group_for_config("onnx", ...)` repair executor with commands captured instead of executed, and verifies the repair removes plain `onnxruntime`, installs `requirements\onnx_directml.txt`, force-probes a compatible `onnxruntime-directml` package, and clears the missing-provider state.

The `faster_whisper_pkg_resources_repair` runtime-matrix row is a non-destructive repair contract for the reproduced CTranslate2 import failure. It simulates `pkg_resources` missing during faster-whisper/CTranslate2 native load, routes through the product native-stack repair helper, verifies the first repair attempt force-reinstalls the bounded `requirements\faster_whisper.txt` stack, and verifies the faster-whisper native load probe is rerun before the row can pass.

The `faster_whisper_ctranslate2_candidate_fallback_repair` runtime-matrix row is a non-destructive repair contract for the CTranslate2 candidate resolver. It simulates a bounded requirements reinstall that still fails native load, then verifies the product repair helper discovers CTranslate2 versions dynamically, skips versions outside `requirements\faster_whisper.txt`, tries newer in-range candidates before older fallbacks, logs failed probes, and passes only after a fallback candidate probe succeeds. Version `4.4` is allowed to be a fallback candidate, not a hard-coded primary install.

The `faster_whisper_vc_runtime_repair` row is a non-destructive repair contract for CTranslate2 missing-DLL native-load failures. It simulates a missing Microsoft Visual C++ Redistributable state, verifies the product faster-whisper native-stack repair path tries the `Microsoft.VCRedist.2015+.x64` winget repair before replacing CTranslate2 packages, and verifies the native load probe is rerun before any CTranslate2 candidate fallback is attempted.

The `transformers_cuda_unavailable_cpu_fallback` row is a non-destructive runtime-plan contract for Hugging Face Transformers ASR CUDA fallback. It requests CUDA, records the conservative safe recovery command plus explicit manual CUDA commands for `requirements\torch_cuda_cu128.txt` and `requirements\transformers_cpu.txt`, and passes only if Torch CUDA is verified or the runtime plan records an explicit CPU fallback because Torch CUDA is not verified.

The `openai_whisper_cuda_unavailable_cpu_fallback` row is a non-destructive runtime-plan contract for OpenAI Whisper `.pt` CUDA fallback. It requests CUDA, records the conservative safe recovery command plus explicit manual CUDA commands for `requirements\torch_cuda_cu128.txt` and `requirements\openai_whisper.txt`, and passes only if Torch CUDA is verified or the runtime plan records an explicit CPU fallback because Torch CUDA is not verified.

The `setup_repair_all_safe` runtime-matrix row preflights `app.doctor --repair-plan` and then runs `app.doctor --repair-all-safe` only when safe. If the plan would install dependencies, run it with `--install-deps` only on a machine where dependency repair is intentionally being exercised. The row writes `row.json`, `repair_plan.json`, and `repair_all_safe.json` with backend and accelerator probe summaries. Usable backend records also include `runtime_resolution_path` values pointing to `Logs\dependency_resolution_<group>.json`, which capture the persisted working backend/provider resolution and any requested-but-unverified accelerator state. The repair summary must include `previous_runtime_resolution_valid`, `previous_runtime_resolution_stale`, and `cached_runtime_resolutions` so release evidence shows whether saved resolutions still matched the current package/provider/config state before refresh and whether any valid saved resolutions were consumed.

The `repair_all_safe_failure_isolation` runtime-matrix row is a non-destructive repair contract for failed dependency repair isolation. It routes through `app.repair_plan.execute_repair_plan`, simulates one failed dependency-group repair followed by one repairable group, and verifies the failed group is capped at one logged attempt while the later group still repairs, probes, and persists runtime-resolution evidence.

The `repair_all_safe_stale_cached_resolution` row is a non-destructive cached-resolution repair contract. It seeds a stale saved dependency resolution with mismatched package-version evidence, routes through `app.repair_plan.execute_repair_plan`, and verifies repair-all-safe marks the cache stale, reruns the backend probe, refreshes the persisted runtime-resolution file, and does not reinstall packages unnecessarily.

The `repair_plan_issue_classification_contract` row is a non-destructive repair-plan contract for bootstrapper troubleshooting language. It verifies `app.repair_plan.build_repair_plan` distinguishes corrupt installs, incompatible package/runtime stacks, provider conflicts, outdated packages, missing native tools, and manual blockers while retaining concrete repair commands for auto-repairable groups.

The `setup_repair_model_layouts` runtime-matrix row exercises the shared model-layout repair sweep used by setup doctor. It writes a persisted `hf_model_layout_repair_plan.json`, runs the sweep with controlled downloads, verifies a missing sidecar is written, and records `model_layout_repair_sweep.json` plus the updated plan `last_execution` evidence. This is the release-row proof that incomplete Hugging Face model packages can be repaired through the bootstrapper path instead of only through interactive downloader prompts.

The `clean_vm_zero_dependency_bootstrap` row is the final fresh-machine proof. On a normal development laptop it must block with an external requirement. In a fresh Windows 11 VM/Sandbox, set `EASY_ASR_BENCH_CLEAN_VM_BOOTSTRAP_PROOF=1` and run it with `--install-deps --allow-downloads`; it then runs `setup.bat --doctor --repair-all-safe`, `setup.bat --doctor --repair-model-layouts --allow-downloads`, the `setup_repair_all_safe` subrow, the `setup_repair_model_layouts` subrow, the same-media multi-model SmolLM benchmark, and first-run smoke, writing each subrow/transcript as evidence. The first-run smoke JSON must include the repair-plan schema, repair-plan summary, and setup/model-layout repair commands so a fresh-machine pass proves first-run guidance is connected to the same bootstrapper diagnostics.

`app.doctor --validate-real-smoke` is the local post-repair smoke wrapper. It emits `easy_asr_bench.real_smoke_validation.v1`, includes the `repair_all_safe` JSON, and records each runtime-matrix row status plus its `row.json` path. Use `--no-network` for explicit offline diagnostics, or `--allow-downloads` only when model/media downloads are intentionally permitted.

The `report_atomic_write_failure_cleanup` row forces simulated replace failures in report text and CSV atomic-write paths and verifies failed `.partial` files are removed while the previous complete artifact remains intact. This guards against partial report artifacts after disk, permission, antivirus, or interrupted-write failures.

The `model_fixture_quality_claims` row validates `qa\runtime_matrix\model_fixtures.json` so structural tiny/random/generated fixtures are explicitly marked as not quality-bearing, same-media mixed rows disclose structural inclusions, and quality-bearing fixtures have WER/transcript-oriented runtime rows. This prevents structural smoke fixtures from being counted as accuracy evidence.

The `hf_downloader_package_variant_taxonomy` row validates fixture-backed Hugging Face package selection for sharded Safetensors indexes, multi-variant ONNX folders, and split GGUF reference LLMs. It verifies sharded Safetensors indexes expand into shard downloads, ONNX precision/quant variants do not pull sibling variants, and split GGUF parts stay grouped as one reference/correction LLM choice.

The `generic_onnx_openvino_unavailable_cpu_fallback` row requests OpenVINO for the tiny Generic ONNX CTC fixture and validates provider fallback metadata. It passes when OpenVINO is active on an OpenVINO machine, or when OpenVINO is unavailable and the adapter records `requested_runtime_provider=openvino`, `openvino_requested=true`, `provider_fallback=true`, active CPU providers, and the expected transcript.

Public-asset Windows smoke should also capture machine-readable app state from the installed app:

```powershell
qa\windows_matrix\run_public_asset_smoke.ps1 -Tag vX.Y.Z
qa\windows_matrix\run_public_asset_smoke.ps1 -Tag vX.Y.Z -Install
%LOCALAPPDATA%\Easy-ASR-Bench\Run.bat --doctor --json > doctor.json
%LOCALAPPDATA%\Easy-ASR-Bench\Run.bat --first-run-smoke > first-run-smoke.json
%LOCALAPPDATA%\Easy-ASR-Bench\setup.bat --doctor --json > doctor-from-setup.json
```

The public-asset smoke runner downloads release assets with `gh release download`, verifies the staged setup path, records asset hashes, and writes evidence rows. With `-Install`, it also runs the installed app and captures `doctor.json` plus `first-run-smoke.json`. `doctor.json` records release identity, dependency groups, provider diagnostics, and checked folders. `first-run-smoke.json` verifies the first-run state has actionable next steps without using network or requiring interactive input, and includes the repair-plan summary plus `setup.bat --doctor --repair-all-safe` command for recoverable dependency issues.

After collecting evidence rows, merge them into the release smoke artifact that release notes and strict gates consume:

```powershell
python scripts\merge_release_evidence.py --smoke release-smoke-vX.Y.Z.json --evidence-dir qa\windows_matrix\evidence --output release-smoke-vX.Y.Z.json
python scripts\validate_release_smoke.py --smoke release-smoke-vX.Y.Z.json --required tests\fixtures\release_required_rows_v2.json --require-log-hashes --require-environment-summary
```

`validate_release_smoke.py` verifies that every required release QA row exists. Public release publication and release-gate verification use `--require-all-pass --require-log-hashes --require-environment-summary`; that means every required row must be `pass` and must include app version, release commit, log/result hash evidence, and environment summary. If real Windows/model/provider/media evidence has not been collected, the correct status is `not_run`, and the release must remain unpublished or draft.

Run the real tiny model smoke before claiming model inference has been verified. This downloads or reuses the recommended faster-whisper baseline, generates a short Windows SAPI speech WAV, runs the normal app report pipeline, and fails unless `results.json`, `results.txt`, `benchmark.csv`, `compare.html`, a non-empty transcript, normalized WER against the known phrase at or below the configured threshold, and VRAM measurement source metadata are produced:

```powershell
python qa\run_real_tiny_model_smoke.py --install-deps --clean
python qa\run_real_tiny_model_smoke.py --provider cuda
```

The first command is the CPU-safe baseline. The CUDA command is an additional hardware row and must be run only on a CUDA machine when claiming GPU smoke coverage.

## GitHub Release Asset Gate

After GitHub Actions publishes the release, verify the actual uploaded assets:

```powershell
python scripts\verify_github_release.py --repo rollingedit/Easy-ASR-Bench --tag vX.Y.Z --expected-commit <commit> --write-transcript release-verification-vX.Y.Z.txt
python scripts\write_release_verification_manifest.py --tag vX.Y.Z --transcript release-verification-vX.Y.Z.txt --assets-dir release-asset --output release-verification-manifest-vX.Y.Z.json
python scripts\verify_release_transcript.py --assets-dir release-asset --checksums release-asset\checksums.json --transcript release-asset\release-verification-vX.Y.Z.txt --detached-manifest release-verification-manifest-vX.Y.Z.json --strict
```

For pushed commits, CI also fetches selected files from `raw.githubusercontent.com` at the exact commit SHA:

```powershell
python scripts\validate_raw_github_files.py --repo rollingedit/Easy-ASR-Bench --ref vX.Y.Z
```

Raw validation prints byte counts, CRLF/LF/bare-CR counts, physical line counts, and first/last byte hex for each critical public file. Those diagnostics are required because raw GitHub bytes, release assets, and ZIP contents are the trust boundary.
GitHub raw URLs serve canonical Git blob bytes, which are normally LF-normalized even for files that check out as CRLF on Windows because of `.gitattributes`. Physical line counts are diagnostics only; truncation protection comes from required file presence, required text markers, parse/compile checks, release ZIP comparison, and setup/bootstrap hashes.
When a release ZIP is available, run raw validation with `--zip` so critical raw GitHub files are byte-compared against the matching ZIP copies:

```powershell
python scripts\validate_raw_github_files.py --repo rollingedit/Easy-ASR-Bench --ref vX.Y.Z --zip dist\Easy-ASR-Bench-vX.Y.Z-win.zip
```

The release must upload:

- `setup.bat`
- `install.ps1`
- `manifest.json`
- `checksums.json`
- `Easy-ASR-Bench-vX.Y.Z-win.zip`
- `release-smoke-vX.Y.Z.json`
- `release-verification-vX.Y.Z.txt`
- `release-verification-manifest-vX.Y.Z.json`

`setup.bat` must verify `install.ps1` before executing it. The app ZIP hash must match `checksums.json`.
The smoke JSON records automated release-candidate checks and leaves Windows VM, GPU/provider, model, and media rows as `not_run` until they are actually executed. Do not edit those rows to `pass` unless the matching downloaded-release test was run.
The verification transcript records actual downloaded release asset hashes and validation steps; it does not convert unrun manual matrix rows into passes.
The transcript must not contain a hash of itself. `release-verification-manifest-vX.Y.Z.json` is the detached metadata that records the transcript hash after the transcript exists.

## Manual Windows QA

These rows require real machines, VMs, hardware, or model files and cannot be proven by static CI alone:

- clean Windows 10/11 install with no Python
- Windows with existing Python 3.11/3.12
- install path with spaces
- repair, update, uninstall, and destructive uninstall confirmation
- empty `Models`
- complete HF ASR Safetensors, HF Whisper, faster-whisper, whisper.cpp, generic ONNX manifest, GGUF reference LLM
- real tiny faster-whisper report smoke using `qa\run_real_tiny_model_smoke.py`
- unsupported standalone Safetensors, generic ONNX without manifest, HF text LLM Safetensors, unsafe `.pt`
- WAV, MP3, MP4 with audio, MP4 without audio, corrupt media, long audio
- CPU-only, NVIDIA CUDA, AMD/Intel DirectML, Intel OpenVINO, Vulkan runtime without SDK, Vulkan runtime with SDK

Record manual results separately from automated gates. Do not describe an unrun hardware row as verified.
