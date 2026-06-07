# Changelog

## v0.3.2 candidate

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
- Made release CI use strict committed checksum validation and fail if release metadata is generated but not committed.
- Kept checksum validation scoped to release/bootstrap artifacts and unsafe checkpoint allowlists; normal user-supplied ASR/LLM models are still structure/runtime validated rather than hash-policed.
- Hardened uninstall behavior: standalone `setup.bat --uninstall` can locate the installed uninstaller, and destructive user-data removal requires an explicit confirmation string.
- Added PowerShell staging validation before activation so clean systems without Python still get basic ZIP layout, line-ending, and critical-file line-count checks before local setup runs.
- Added installed-app validation after local setup so systems that bootstrap Python during install still run the Python release validator before setup reports success.

Runtime and report correctness:
- Improved media error reporting for no-audio videos, ffprobe failures, and FFmpeg conversion failures.
- Reported Granite AR/NAR chunk exceptions in structured run errors instead of only embedding them in transcript text.
- Configured ONNX Runtime DirectML sessions with DirectML-safe session options: memory pattern disabled and sequential execution.
- Fixed faster-whisper CPU fallback so requested FP16 uses the effective CPU-safe compute type when constructing the model.
- Improved Unicode scoring normalization and long HTML report guards.

Validation and documentation:
- Added `docs/release_verification.md` with local gates, GitHub release asset verification, and manual Windows QA rows that must not be claimed as verified unless actually run.
- Added `docs/hf_downloader_validation.md` and fixture file-list tests for split GGUF, ASR GGUF+projector, sharded Safetensors, multi-variant ONNX, and unknown inspection-only layouts.
- Added fixture E2E pipeline tests with fake adapters so one model can fail while reports still write JSON/TXT/CSV/HTML and media preparation failures return cleanly.
- Added regression coverage for release bootstrap assets, version coherence, raw source validation, publish workflow ordering, release-smoke artifacts, annotated Git tag verification, Unicode scoring edge cases, long HTML report pagination guards, LLM reference validation guards, Windows pasted path parsing, candidate-ID uniqueness for nested model folders, dependency fallback skipping, and queue fast-key skipping.

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
- Expanded required dependency groups for ASR/LLM runtimes: Transformers now includes `accelerate`, `tokenizers`, `sentencepiece`, `protobuf`, `torchaudio`, and `torchcodec`; faster-whisper CUDA includes NVIDIA cuBLAS/cuDNN wheels; CUDA Torch uses the PyTorch CUDA 12.8 wheel index.
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
