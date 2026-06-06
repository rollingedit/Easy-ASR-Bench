# Changelog

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
