# Changelog

## Unreleased

## v0.3.8

Release QA automation:
- Added `Run.bat --doctor --json` / `python -m app.main --doctor --json` so Windows public-asset QA can capture release identity, dependency status, provider diagnostics, and folder paths as machine-readable evidence.
- Changed `setup.bat --doctor` to forward extra doctor flags such as `--json` and `--strict`, keeping setup as a single QA/user entrypoint.
- Added `Run.bat --first-run-smoke` / `python -m app.main --first-run-smoke` so QA can prove the first-run state has actionable next steps without network use or interactive input.
- Added `qa/windows_matrix/run_public_asset_smoke.ps1` to download public release assets, verify staged setup, record asset hashes, and optionally install/capture installed-app `doctor.json` plus `first-run-smoke.json`.
- Added `scripts/merge_release_evidence.py` so collected `row.json` evidence can update the `release-smoke-vX.Y.Z.json` artifact consumed by release notes and strict validation.
- Documented the new public-asset runner and installed-app JSON capture commands in release verification guidance.

Setup/runtime hardening:
- Changed public install flow so `installer/install.ps1` runs local setup with `--no-post-setup-menu`; only the outer setup shows the final Run/Paste/Open/Quit menu after installed-app validation completes.
- Changed setup's Hugging Face paste option to route through `Run.bat --first-run --download-model-first`, so a successful model download rescans and continues into the app instead of telling the user to relaunch.
- Removed the normal-launch doctor wall from `Run.bat` and `Drop_Audio_Or_Folders_Here.bat`; full diagnostics now run only through explicit `--doctor` or setup/repair paths.
- Added structured model-level failure reports for model-load/inference/unload failures with stage, model id/path, provider request, dependency group, likely causes, next actions, repair command, log path, and traceback detail.
- Changed first-run recommended baseline copy to disclose `Systran/faster-whisper-tiny.en`, the `faster-whisper / CTranslate2` runtime, and CPU-default behavior before download.

Dependency/model support truth:
- Replaced the hard-coded llama.cpp CUDA 12.4 install decision with a prebuilt wheel resolver that can select `cu118`, `cu121`, `cu122`, `cu123`, `cu124`, `cu125`, `cu130`, or `cu132` indexes and falls back to CPU when the selected index is unreachable.
- Changed llama.cpp Vulkan setup to try the Vulkan prebuilt wheel first when a Vulkan runtime is detected; source builds now require explicit opt-in plus SDK/build tooling.
- Changed Windows repair command text for llama.cpp accelerator installs to emit PowerShell and `cmd.exe` environment syntax instead of POSIX-style `KEY=value` prefixes.
- Demoted Audio/ASR GGUF+`mmproj` packages from stable runnable ASR to recognized experimental until a real ASR GGUF smoke fixture proves transcription through this app; text GGUF models remain reference/correction LLMs only.
- Updated README and support/setup docs to match the first-run setup menu, CPU-safe baseline disclosure, llama.cpp wheel resolver, Vulkan behavior, and ASR GGUF experimental scope.

## v0.3.7

User-path hardening:
- Changed prerelease builds to report `prerelease` from runtime environment and doctor output instead of trusting stale `config.json` channel values.
- Added failed-file report generation for pre-model failures so bad media, preprocessing errors, or similar file-level failures still write `results.json`, `results.txt`, `benchmark.csv`, and `compare.html` with cause and next-action text.
- Changed batch/queue status detection so structured failed-file reports are marked `failed` in queue state and batch dashboards instead of being mislabeled as completed just because an output folder exists.
- Added more specific failed-file stages for no-audio/media-probe, FFmpeg/decode, and path/permission failures.
- Expanded setup completion into direct first-run choices for running the app, pasting a Hugging Face model link, opening `Models`, opening `Input`, or quitting.
- Added a dedicated first-run wizard with a recommended CPU baseline download path, Hugging Face paste flow, folder-open actions, and automatic continuation into the app after a successful model download.
- Added faster-whisper/CTranslate2 package detection to the Hugging Face downloader so baseline ASR repos with `model.bin`, `config.json`, and tokenizer/vocabulary files can be downloaded as one runnable package.
- Added app commands for model download and folder-opening actions so setup can route users through product actions without exposing Python commands.
- Reworked optional dependency recovery prompts into explicit install, skip affected models, show repair command, and quit-batch choices.
- Changed queue discovery and direct batch processing to enqueue by fast file identity before full SHA256 hashing, avoiding long startup stalls on large media while preserving result-level source hashes.
- Added focused regression coverage for release-channel truth, stale-config doctor output, first-run guidance, setup choices, dependency prompt choices, and failed-file reports.

## v0.3.6

Release proof hardening:
- Changed release-note generation so it no longer writes unconditional "passed" claims from static prose; automated checks now come from the smoke artifact, and releases with any non-pass manual rows are labeled as not all-pass manual smoke releases.
- Changed changelog extraction for release notes so "still not claimable" caveat bullets are not accidentally promoted into the "What changed" section.
- Clarified raw GitHub byte diagnostics so canonical Git blob LF line endings are not mistaken for collapsed files when `physical_line_count_universal` is correct.
- Expanded the Windows release matrix helper to include all current manual smoke rows, not just a subset.
- Hardened Windows evidence collection so rows marked `pass` must include app version, release commit, environment summary, and log/result artifact hashes.
- Tightened GitHub release verification so v2 smoke artifacts must include explicit `manual_rows`, even when those rows are `not_run`.
- Documented that README model support describes complete code-supported packages, while per-release verification is controlled by the smoke artifact.

## v0.3.5 candidate

Audit follow-up:
- Fixed `compare.html` and batch dashboard JSON embedding so JSON script tags contain parseable JSON with script-breakout-safe escaping instead of HTML entity escaping.
- Added Python-side LLM reference import/scoring helpers that validate `easy_asr_bench.llm_reference.v1` JSON and score per chunk, allowing large references to be scored without relying on browser-wide dynamic programming.
- Added `compare.html` support for precomputed LLM-corrected reference scores and a browser scoring guard for oversized pasted references.
- Added CTranslate2-specific CUDA probing so faster-whisper CUDA is used only when the CTranslate2 backend is verified, not merely when Torch CUDA or NVIDIA marker packages are present.
- Centralized Hugging Face Transformers ASR and HF Whisper runtime planning so CUDA requests record requested provider, actual provider, backend verification, and CPU retry warnings.
- Narrowed Hugging Face text-LLM safetensors detection so explicit CausalLM/text/quant formats such as GPTQ, AWQ, and EXL2 are rejected as local ASR/reference-runtime folders, while generic unknown Transformer folders remain ASR probe-required instead of disappearing.
- Changed optional runtime dependency installs to skip safely in noninteractive terminals and write dependency-specific logs such as `Logs/dependency_install_<group>_<timestamp>.log`.
- Added regression coverage for safe JSON embedding, large reference scoring/import, HF Whisper/Transformers CUDA fallback, CTranslate2 probing, noninteractive dependency prompts, and safetensors text-vs-ASR classification.

Still not claimable without external release QA:
- A public v0.3.5 app release with downloadable installer assets.
- Clean Windows setup proof from public release URLs.
- Strict all-pass release smoke matrix and QA evidence bundle from real Windows/GPU/media/model runs.
- Real browser execution screenshots/console proof for `compare.html` beyond source-level fixture tests.

## v0.3.4

Release proof:
- Added `scripts/validate_release_smoke.py` plus `tests/fixtures/release_required_rows_v2.json` so release smoke artifacts must contain the required Windows, media, provider, model, failure-isolation, and dependency-decline rows.
- Added `scripts/verify_release_transcript.py` so release verification transcripts are checked against downloaded assets and `checksums.json`, and transcript self-hashes are rejected.
- Added detached `release-verification-manifest-vX.Y.Z.json` generation and validation so the transcript hash is recorded after the transcript exists, without embedding a self-hash inside the transcript.
- Updated release workflows to require all mandatory smoke rows to pass with app version, release commit, log/result hash evidence, and environment summaries before a public release can be published or accepted by the release gate.
- Removed transcript self-hashes from generated release verification transcripts; transcript hashes belong in detached release metadata, not inside the transcript being hashed.
- Changed release-note generation to read the smoke artifact and list pass rows under verified and all other rows under not verified.

Runtime/provider behavior:
- Added `app/runtime_plan.py` with explicit hardware facts and resolved runtime plans for faster-whisper and llama.cpp/GGUF paths.
- Changed ONNX `auto` provider order to prefer CUDA on NVIDIA, OpenVINO before DirectML on Intel, DirectML on AMD/generic Windows GPUs, then CPU.
- Changed faster-whisper loading so `prefer_gpu` alone does not prove CUDA; CUDA load failures retry CPU when fallback is allowed and record the actual provider/device.
- Changed GGUF ASR llama.cpp loading so full GPU offload is used only when the backend is verified GPU-capable; otherwise it uses CPU-safe `n_gpu_layers=0`.

Model detection and install UX:
- Added manifest-first GGUF ASR pairing through `model_package.json`; mixed folders with multiple plausible GGUF/mmproj pairs are now ambiguous instead of runnable.
- Changed complete unknown Hugging Face Safetensors folders to `asr_probe_required` rather than runnable ASR unless config metadata identifies an ASR architecture.
- Added a deliberate probe option for complete unknown Safetensors ASR folders so users can try a runtime probe without the scanner pretending the model is already runnable.
- Added explicit user-facing model-state buckets for runnable ASR, dependency-needed, ASR probe-required, reference LLM, incomplete, unsafe, and unsupported packages.
- Added `app/install_plan.py` and changed optional dependency installation prompts to show packages, requirement files, indexes, install location, network destinations, PATH changes, size class, and fallback behavior before the user presses Enter to install or `s` to skip.
- Added a central core dependency import map so doctor/core health checks cover every package in `requirements/core.txt` or explicitly document exclusions.

Reporting and QA:
- Added a 500-chunk offline `compare.html` fixture test proving the large-report page remains self-contained and paginated.
- Polished `compare.html` with clearer tabs, table wrappers, model filters for transcript/chunk views, better status badges, and scroll-safe layout for many models and chunks.
- Added a multi-file batch dashboard at `Output/batch__*/index.html` so runs with many audio/video files show paged side-by-side file cards, per-file model summaries, search/filtering, and links to full per-file `compare.html` reports.

## v0.3.3 candidate

Model support:
- Added Audio/ASR GGUF+`mmproj` adapter support. Complete matching packages now appear as runnable ASR candidates instead of recognized-unsupported entries.
- The GGUF ASR adapter uses llama-cpp-python Qwen3 ASR chat handling when available and falls back to `llama-mtmd-cli` from llama.cpp when present.
- GGUF text LLMs remain reference/correction models only; Audio/ASR GGUF packages are separated from text reference LLM discovery so they are not misclassified.
- Incomplete or mismatched Audio/ASR GGUF packages stay out of the runnable list and report exact missing/nonmatching `mmproj` requirements.
- Expanded sharded Safetensors validation to handle noncanonical index names such as `model.safetensors.index.fp32.json`, not only `model.safetensors.index.json`.
- Complete sharded Hugging Face Whisper/Transformers Safetensors folders are treated as ASR candidates when the index and all shard files are present; incomplete folders report the missing shard names.

Dependency and safety fixes:
- Changed GGUF llama.cpp CUDA dependency bootstrap from the unavailable llama-cpp-python `cu125` wheel index to the available `cu124` wheel index.
- Added a llama-cpp-python CUDA wheel-index probe during optional dependency install; if the prebuilt CUDA wheel index is unavailable, setup falls back to the CPU package with a visible reason instead of attempting a local source build on a normal user's machine.
- Removed `torchcodec` from the Hugging Face Transformers ASR dependency group because Easy ASR Bench pre-decodes audio and passes arrays to the pipeline, avoiding TorchCodec's Windows FFmpeg shared-library hazard.
- Populated the OpenAI Whisper `.pt` SHA256 allowlist from the official Whisper model URLs so the advertised checksum-verified safe path is real while unknown or wrong-hash `.pt` files remain blocked by default.
- Changed Hugging Face large or unknown package confirmation from default-yes to typed `DOWNLOAD`, so Enter and `y` cancel instead of accidentally downloading ambiguous or multi-GB packages.

Docs and validation:
- Added raw GitHub byte diagnostics with CRLF/LF/bare-CR counts, physical line counts, byte counts, and first/last byte hex for critical public files.
- Added raw GitHub versus release-ZIP byte comparison for critical public files in the release gate.
- Added release verification transcript generation via `scripts/verify_github_release.py --write-transcript`, recording resolved release commit, release state, downloaded asset hashes, and completed automated checks.
- Added staged release asset verification with `setup.bat --dry-run --verify-release --asset-dir <dir>` so the exact local assets can be checked before a draft release is made public.
- Hardened release publishing so it refuses to replace assets on an already-public release unless `allow_replace_public=true` is explicitly selected.
- Updated release publishing to upload and re-download `release-verification-vX.Y.Z.txt` before marking the release public/latest.
- Expanded the release-smoke manual matrix into explicit Windows, install/update/repair/uninstall, bad-checksum, media, provider, and model-family rows so unrun runtime coverage cannot hide behind broad buckets.
- Updated README and supported-model docs to remove vague "not runnable yet" wording and distinguish runnable ASR, incomplete packages, platform-specific packages, and blocked unsafe formats.
- Updated setup docs and doctor wording so `llama_cpp` is described as both GGUF ASR+`mmproj` support and GGUF text reference/correction support.
- Added regression coverage for GGUF ASR+`mmproj` discovery, matching-projector validation, llama-cpp-python transcription path, CLI output cleanup, sharded Safetensors missing-file detection, llama CUDA fallback, the HF dependency list, official Whisper `.pt` checksum verification, raw byte diagnostics, raw-vs-ZIP comparison, release transcript output, staged setup verification, public-release clobber refusal, expanded smoke matrix rows, and typed large-download confirmation.
- Local candidate validation has passed: full pytest suite, release-file validation, repo/ZIP physical validation, version coherence, strict checksum build, setup dry-run, and strict doctor.

## v0.3.2

Release and installer safety:
- Hardened standalone setup bootstrap: public setup now downloads `install.ps1` from the versioned GitHub release asset path instead of raw GitHub content and verifies its SHA256 before executing PowerShell.
- Added `setup.bat --dry-run --verify-release` to download and validate release manifest, checksums, app ZIP hash, ZIP layout, and release-file validators without installing.
- Extended `--verify-release` for v2 manifests so it also verifies uploaded `manifest.json`, `setup.bat`, and `install.ps1` assets, while keeping legacy validation compatible with the already-published `v0.3.0` assets.
- Updated GitHub release publishing to upload `install.ps1` as a release asset and verify uploaded release asset hashes after publication.
- Changed release publishing so uploaded draft assets are verified before the release is marked public/latest.
- Added a public `scripts/verify_github_release.py` verifier for downloaded GitHub release assets, including annotated-tag handling for expected-commit checks.
- Added `release-smoke-vX.Y.Z.json` generation and release verification so every new release carries machine-readable automated validation evidence without pretending unrun VM/GPU/model rows passed.
- Added release physical-file validation for repo and ZIP bytes, including CRLF launcher checks, LF source/docs/config checks, YAML/JSON/Python parsing, requirements formatting, and minimum physical line counts for critical files.
- Added raw GitHub source validation for pushed commits so collapsed or CR-only public bytes are caught from `raw.githubusercontent.com`, not only from the checkout.
- Added release version-coherence validation so `app.__version__`, `setup.bat`, `install.ps1`, manifest, checksums, and ZIP names must agree.
- Added `app/version.py` as the release version source used by `app.__version__` and release coherence checks.
- Made release CI use strict committed checksum validation and fail if release metadata is generated but not committed.
- Kept checksum validation scoped to release/bootstrap artifacts and unsafe checkpoint allowlists; normal user-supplied ASR/LLM models are still structure/runtime validated rather than hash-policed.
- Hardened uninstall behavior: standalone `setup.bat --uninstall` can locate the installed uninstaller, and destructive user-data removal requires an explicit confirmation string.
- Changed destructive user-data removal confirmation to the app-specific phrase `DELETE EASY ASR BENCH USER DATA`.
- Added PowerShell staging validation before activation so clean systems without Python still get basic ZIP layout, line-ending, and critical-file line-count checks before local setup runs.
- Added installed-app validation after local setup so systems that bootstrap Python during install still run the Python release validator before setup reports success.

Runtime and report correctness:
- Improved media error reporting for no-audio videos, ffprobe failures, and FFmpeg conversion failures.
- Reported Granite AR/NAR chunk exceptions in structured run errors instead of only embedding them in transcript text.
- Configured ONNX Runtime DirectML sessions with DirectML-safe session options: memory pattern disabled and sequential execution.
- Fixed faster-whisper CPU fallback so requested FP16 uses the effective CPU-safe compute type when constructing the model.
- Added a whisper.cpp runtime probe for the `pywhispercpp.Model.transcribe` API shape before loading a model.
- Tightened Generic ONNX support to explicit "Generic ONNX CTC manifest v1" wording and rejects non-CTC manifests before runtime.
- Added transcript and alignment paging in `compare.html` so long reports do not inject full transcript/alignment content into the DOM at once.
- Added detected precision vs runtime precision support fields in result model metadata.
- Added optional `char_for_cjk` scoring tokenization for CJK/Thai-style no-space text while keeping the default tokenizer unchanged.
- Improved Unicode scoring normalization and long HTML report guards.

Validation and documentation:
- Added `docs/release_verification.md` with local gates, GitHub release asset verification, and manual Windows QA rows that must not be claimed as verified unless actually run.
- Added `docs/hf_downloader_validation.md` and fixture file-list tests for split GGUF, ASR GGUF+projector, sharded Safetensors, multi-variant ONNX, and unknown inspection-only layouts.
- Added fixture E2E pipeline tests with fake adapters so one model can fail while reports still write JSON/TXT/CSV/HTML and media preparation failures return cleanly.
- Added regression coverage for release bootstrap assets, version coherence, raw source validation, publish workflow ordering, release-smoke artifacts, annotated Git tag verification, OpenAI Whisper `.pt` safety, whisper.cpp runtime probing, Generic ONNX CTC manifest scope, precision-vs-runtime support, multilingual scoring edge cases, long HTML transcript/alignment pagination, LLM reference validation guards, Windows pasted path parsing, candidate-ID uniqueness for nested model folders, dependency fallback skipping, and queue fast-key skipping.

## v0.3.0

- Added a smart Hugging Face downloader to the interactive model menu. Choose `D`, paste a Hugging Face model URL or `owner/model` repo id, pick one detected package/weight variant, and the app downloads it into `Models` before rescanning.
- Accepted forgiving Hugging Face URL shapes: repo root links, `/tree/<revision>` links, nested folder links, file/blob/resolve links, links with extra trailing slashes, and folders that need parent-walking to find the nearest actual model package.
- Added package-level download choices so repositories with many weights or quantizations do not download the whole repo by default.
- Added failsafes for large and unknown download choices: the app shows the selected file count, asks before downloading large/unknown packages, and treats unknown folders as inspection-only until the scanner recognizes the downloaded files.
- Added post-download missing-file repair: if a downloaded package rescans as incomplete and exact missing files exist in the Hugging Face repo, the app lists those files and asks before downloading them. Ambiguous missing files are reported instead of guessed.
- Added a structured LLM/file-audit request package for ambiguous incomplete downloads. When exact repair matches are not available, the app writes `hf_missing_file_request.json` and `hf_missing_file_prompt.txt` beside the package so a local or external LLM can recommend exact repo filenames without the downloader inventing files.
- Existing files in the target model package folder are skipped during Hugging Face downloads and repair downloads, so rerunning the same choice does not overwrite files that are already present.
- Added one-local-folder packaging for selected downloads so nested HF folders such as `BF16/...gguf` become one clean local model folder instead of duplicating the remote folder layout.
- Added GGUF package grouping for quantized LLM repos, including split GGUF files such as `00001-of-00002`, so all required parts are selected together.
- Added GGUF ASR/audio distinction for packages with ASR/audio signals or matching `mmproj` files. Qwen-style ASR GGUF packages are not treated as text reference LLMs when they require a projector; ordinary GGUF models remain reference/correction LLM choices.
- Added Safetensors download grouping for standalone weights, sharded indexes, unusual index names such as `model.safetensors.index.fp32.json`, split Safetensors parts, and missing-shard expansion from downloaded index files.
- Added ONNX package grouping by variant inside shared `onnx/` folders: default, FP16, Q4, Q4F16, quantized, INT8, and related sidecars are separated so one selected precision/quant path does not pull every ONNX variant.
- Added shared metadata selection for downloaded packages, including tokenizer, processor, preprocessor, config, generation config, chat templates, Tekken tokenizer files, token files, and parent-folder metadata when the selected weight package is nested.
- Added real-world no-weight-download stress checks for representative HF repos: Unsloth Qwen/Gemma/Qwopus GGUF repos, OpenAI Whisper, Voxtral, Fun-ASR, NVIDIA Canary, Cohere Transcribe, Granite speech GGUF/ONNX, ONNX Community variants, and Cohere/Granite multi-variant ONNX folders. These are representative edge cases, not a claim that every future HF layout is known; the downloader still uses conservative confirmation and inspection-only fallbacks for unknown layouts.
- Added arrow-key interactive menus on real Windows terminals for model selection, precision buckets, LLM reference choices, and local reference LLM selection, with the existing typed prompts kept as a fallback for non-interactive or captured terminals.
- Added colored action keys and prompt labels for typed fallback prompts so `D`, `R`, `A`, numeric choices, `Y/n`, `q`, and other required input markers are harder to miss.
- Added broader ASR package recognition for known but not-yet-runnable packages: NeMo `.nemo`, FunASR folders, ORT edge graphs, Core ML / WhisperKit `.mlmodelc`, sherpa-onnx packages, split Whisper/Transformers.js ONNX, Granite-style split ONNX, Qwen split ONNX, and ASR GGUF+`mmproj`.
- Added partial-package recognition and missing-file reporting so incomplete known layouts list expected sibling files instead of falling into generic ONNX or unknown-file noise.
- Added runtime-probe behavior for complete non-text-generation Hugging Face Safetensors folders when metadata is unfamiliar, so plausible ASR folders are allowed to run and report real runtime errors instead of being blocked only by weak scanner signals.
- Added sharded Safetensors missing-file validation to Hugging Face ASR adapters and unsupported scanner output.
- Added faster-whisper `vocabulary.txt` discovery support.
- Updated README and supported-model docs with the Hugging Face downloader flow, recognized-but-not-runnable model families, large-download confirmation behavior, unknown-folder inspection behavior, and GGUF ASR-vs-reference-LLM distinction.
- Updated the external audit prompt to require web-backed edge-case review of the smart downloader, smart detector, large-download safeguards, arrow-key menus, typed fallbacks, and real HF package layouts.
- Added regression coverage for Hugging Face URL parsing, parent-folder fallback, package-level download grouping, split GGUF/Safetensors, sharded Safetensors indexes, ONNX variant sidecars, unknown-folder failsafes, large-choice cancellation, scanner recognition for researched ASR layouts, runtime-probe behavior, and menu rescan after download.

## v0.2.9

- Made the runtime GPU-first by default while keeping explicit CPU fallback reporting when an accelerator path cannot safely run.
- Added accelerator-aware dependency repair/install decisions instead of only CPU-safe package suggestions.
- Added NVIDIA CUDA dependency paths for ONNX Runtime, Hugging Face/Transformers ASR, OpenAI Whisper `.pt`, faster-whisper/CTranslate2, and GGUF llama.cpp reference LLMs.
- Added ONNX DirectML dependency support for broad Windows GPU acceleration, including AMD, Intel, and NVIDIA DirectX 12-capable hardware.
- Added ONNX OpenVINO dependency support and changed automatic ONNX routing to prefer OpenVINO on Intel hardware before generic DirectML.
- Added GGUF llama.cpp Vulkan build support for AMD/Intel/NVIDIA systems, gated on detectable Vulkan runtime and Vulkan SDK build tooling so setup does not offer a known-bad source build.
- Added `cmake` and `ninja` to the Vulkan llama.cpp requirements so source builds bootstrap more of their Python-side build chain.
- Added accelerator diagnostics to `setup.bat --doctor`: NVIDIA, AMD, Intel, Windows GPU, Vulkan runtime, Vulkan SDK, Torch CUDA, and ONNX Runtime provider state.
- Added dependency status checks that repair CPU-only Torch, missing ONNX providers, missing faster-whisper NVIDIA CUDA runtime wheels, and llama.cpp builds without GPU offload when a GPU path is selected.
- Added CUDA/provider diagnostics into result environment metadata so reports show whether GPU support was actually available.
- Added provider fallback metadata for CUDA, DirectML, and OpenVINO ONNX runs.
- Added runtime warnings when requested/preferred GPU execution will fall back to CPU.
- Added `Open_Latest_Report.bat` and included it in installer manifests, release ZIP entrypoints, and uninstall cleanup.
- Made `Run.bat` print the `compare.html` path after report generation so users can find the HTML report immediately.
- Expanded required dependency groups for ASR/LLM runtimes: Transformers includes `accelerate`, `tokenizers`, `sentencepiece`, `protobuf`, and `torchaudio`; faster-whisper CUDA includes NVIDIA cuBLAS/cuDNN wheels; CUDA Torch uses the PyTorch CUDA 12.8 wheel index.
- Improved optional dependency failure handling so a failed optional install skips only the affected model and prints the exact manual repair command.
- Added broad precision and quantization label detection for INT2/3/4/5/6/8, FP4/8/16/32, BF8, BF16/bfloat16, NF4, NVFP4/NVP4, Q2-Q8, K_M/K_S/K_L, and IQ variants.
- Added native 32-bit support throughout model discovery and reporting: `fp32`, `f32`, `float32`, and safetensors `torch_dtype: "float32"` now map to `32-bit / FP32`.
- Fixed safetensors `torch_dtype: "bfloat16"` so it maps to `16-bit / BF16` instead of an unknown transformed label.
- Added multi-file ONNX AR/NAR support for `fp32`, `f32`, and `float32` precision folders.
- Renamed user-facing multi-file ONNX AR/NAR labels so they no longer imply Granite/IBM is the only ONNX path; internal adapter names remain unchanged for compatibility.
- Added faster-whisper precision alias detection for `fp32`, `f32`, `float32`, `fp16`, `f16`, and `float16`, with CTranslate2 compute-type normalization.
- Added whisper.cpp precision detection from filenames such as `ggml-large-f32.bin`.
- Clarified GGUF local LLM support: `.gguf` is treated as the complete local reference/correction LLM artifact because tokenizer/model metadata is normally embedded.
- Added scanner handling for Hugging Face text/non-ASR safetensors folders: they are reported as unsupported local LLMs with a clear requirement for a GGUF export or the manual ChatGPT/Claude workflow.
- Replaced brittle text-LLM family-only scanning with structural detection for unknown/custom Hugging Face text models, including `*ForCausalLM` architectures and generic transformer configs with vocab/hidden/layer/head fields.
- Kept ASR safetensors folders from being misclassified as LLMs by recognizing ASR config signals such as Whisper, Wav2Vec2, HuBERT, CTC, speech, Seamless, Moonshine, and ASR metadata.
- Improved unsupported-model explanations for standalone safetensors, GGUF typo files, generic ONNX files without `modelbench.json`, incomplete multi-file ONNX folders, and unsupported local LLM formats.
- Updated README and supported-model docs to separate ASR safetensors support from local LLM support, and to state that local text LLM loading is GGUF-only in this app.
- Updated setup documentation with the actual dependency groups, accelerator package paths, hardware routing rules, and remaining CPU-only packaged paths.
- Added regression coverage for accelerator routing, CUDA/DirectML/OpenVINO/Vulkan install decisions, doctor diagnostics, runtime fallback warnings, dependency repair commands, GGUF discovery, text-LLM safetensors explanations, broad quant labels, native FP32, BF16/BF8/NF4/NVFP4 labels, and report environment diagnostics.

## v0.2.8

- Added HTML visual word-diff rendering for LLM-corrected reference scoring.
- Added chunk pagination in `compare.html` so large reports do not render every chunk at once.
- Added balanced score and stricter reference timestamp validation in the HTML report.
- Added ONNX provider fallback metadata for generic ONNX manifest runs.
- Added faster-whisper requested/effective compute type and device reporting, including CPU float16 relabeling.

## v0.2.7

- Added a version bump helper so release version strings are updated consistently.
- Added clean Markdown release-note generation and a GitHub Actions publish workflow that builds and uploads release assets from GitHub.
- Updated agent instructions for CI failure classification, release-note formatting, GitHub-generated assets, and model checksum philosophy.
- Added real CUDA VRAM peak sampling for model runs when Torch CUDA is available.
- Added source-hash validation for LLM-corrected references and validation-aware reference merging.
- Added readable unsupported-input extension reporting and no-audio video precheck errors.
- Cleaned up installer secondary issues: `-Doctor` path test binding, readable download failures, and successful backup removal.
- Added workflow YAML parsing to release validation.

## v0.2.6

- Added the `release.published` workflow trigger so the release gate verifies uploaded GitHub release assets after a release is published.
- Updated the release asset verification step to use the published release tag from the release event.

## v0.2.5

- Fixed release packaging so ZIP bytes come from the working tree, manifest metadata is written before ZIP hashing, and CI verifies committed release metadata instead of regenerating it.
- Added CI artifact upload and tag-only published asset checksum verification.
- Hardened installer downloads for Windows PowerShell 5.1 by forcing TLS 1.2 and using `-UseBasicParsing`.
- Moved OpenAI Whisper `.pt` out of runnable README claims unless a checksum is allowlisted or unsafe trusted loading is explicitly enabled.
- Removed the misleading `transformers_cuda` dependency group until a real CUDA wheel strategy is implemented.
- Dedupe scanner output so HF Whisper folders and incomplete Granite folders are not reported twice.
- Added real chunk `cut_reason` and `rms_db` metadata plus non-stub environment and dependency version reporting.
- Updated HTML scoring to use Unicode-aware normalization and escape interpolated model metadata.
- Removed unused config keys that were not consumed by runtime code.

## v0.2.4

- Fixed Generic ONNX CTC inference using an undefined `candidate` variable when loading vocab files.
- Removed the ambiguous Generic ONNX raw-waveform first-input fallback; manifests must build valid declared inputs.
- Hardened installer user-data preservation by enumerating files explicitly, failing on preservation errors, and writing `install-preservation-report.json`.
- Changed uninstall behavior to preserve user data by default; destructive uninstall now requires `-RemoveUserData`.
- Restored exact generated release ZIP checksum enforcement in the release builder.
- Fixed standalone `setup.bat` version metadata for the v0.2.4 release line.

## v0.2.3

- Hardened installer updates to preserve `Models`, `Input`, `Output`, `Logs`, `Cache`, `Temp`, and `config.json` across installs and updates.
- Added installer rollback handling if the atomic directory swap fails.
- Removed bare `python` usage from the PowerShell installer staging validator.
- Added a repeatable release ZIP builder and CI ZIP validation.
- Blocked local OpenAI Whisper `.pt` files by default unless SHA256-allowlisted or explicitly trusted with unsafe pickle loading enabled.
- Passed HF Whisper long-form chunking options through the Transformers pipeline call.
- Required vocab metadata for generic ONNX CTC manifests so numeric token IDs are not emitted as transcripts.
- Made recommended model selection family-balanced instead of Granite-only.
- Improved Unicode-aware scoring normalization and multi-path paste parsing.
- Made dependency install failures skip only affected models instead of crashing the whole run.
- Made `benchmark.csv` writes atomic and watch queue skip completed files by fast size/mtime key before hashing.

## v0.2.2

- Added a complete LLM reference menu for detected GGUF models, saved external GGUF paths, manual ChatGPT/Claude workflows, and skipping reference scoring.
- Added `llm_reference.custom_model_paths` so GGUF files or folders used by another app can be saved and rescanned on future runs.
- Updated version metadata and docs for the v0.2.2 release pass.
- Kept GGUF wording explicit: GGUF text LLMs are reference/correction helpers, not direct ASR models.

## v0.2.0

- Added canonical `results.json`, `results.txt`, `benchmark.csv`, and offline `compare.html` output.
- Added LLM-corrected reference prompt and browser-side scoring workflow.
- Added Hugging Face Safetensors ASR adapter.
- Added generic ONNX CTC manifest adapter.
- Added GGUF local LLM reference/correction candidate handling.
- Added dependency group repair support and doctor command.
- Improved model scanner grouping and unsupported-model explanations.
- Updated all user-facing docs for product-ready wording.

## v0.1

- Initial Windows app shell, model scanner, Granite ONNX adapter foundation, and release BAT.
