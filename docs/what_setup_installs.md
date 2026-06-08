# What Setup Installs

`setup.bat` installs Easy ASR Bench into:

```text
%LOCALAPPDATA%\Easy-ASR-Bench
```

It creates a local Python virtual environment and these folders:

- `Models`
- `Input`
- `Output`
- `Logs`
- `Temp`
- `Cache`

Default setup installs only `requirements/core.txt`: `numpy`, `soundfile`, `librosa`, `imageio-ffmpeg`, `psutil`, and `huggingface_hub`, with bounded version ranges. Heavier ASR/LLM packages stay in optional runtime groups and are installed only when a selected model needs them.

Optional runtimes are installed only when selected:

- ONNX: `requirements/onnx.txt`
- ONNX with CUDA, when NVIDIA is detected: `requirements/onnx_cuda.txt`
- ONNX with OpenVINO on Intel hardware: `requirements/onnx_openvino.txt`
- ONNX with DirectML on Windows GPUs, including AMD, Intel, and NVIDIA: `requirements/onnx_directml.txt`
- Hugging Face / HF Whisper: `requirements/transformers_cpu.txt`
- Hugging Face / HF Whisper with CUDA, when NVIDIA is detected: `requirements/torch_cuda_cu128.txt` followed by `requirements/transformers_cpu.txt`
- faster-whisper: `requirements/faster_whisper.txt`
- faster-whisper with CUDA, when NVIDIA is detected: `requirements/faster_whisper_cuda.txt`
- whisper.cpp: `requirements/whisper_cpp.txt`
- OpenAI Whisper `.pt`: `requirements/openai_whisper.txt`
- OpenAI Whisper `.pt` with CUDA, when NVIDIA is detected: `requirements/torch_cuda_cu128.txt` followed by `requirements/openai_whisper.txt`
- GGUF reference LLM: `requirements/llama_cpp.txt`
- GGUF reference LLM with CUDA, when NVIDIA is detected and a compatible prebuilt wheel index is reachable: `llama-cpp-python` from the selected `cu118`, `cu121`, `cu122`, `cu123`, `cu124`, `cu125`, `cu130`, or `cu132` wheel index
- GGUF reference LLM with Vulkan, when a Vulkan runtime is detected: `llama-cpp-python` from the Vulkan prebuilt wheel index; source builds require explicit opt-in plus SDK/build tooling

When a selected model needs one of these groups, Easy ASR Bench prompts before installing it. If an optional install fails, the affected model is skipped, other runnable models continue, and the console prints the manual repair command.

`setup.bat --doctor` reports each dependency group, what it enables, missing modules, and the matching `pip install -r requirements/...` repair command.
`setup.bat --doctor --repair-plan` emits the same dependency repair information as structured JSON so QA scripts can assert exact repair commands, affected dependency groups, and blocked requirements before any install is attempted.
`setup.bat --doctor --repair-all-safe` executes the safe repairs from that plan, attempts each repairable dependency group once, writes install logs under `Logs`, rechecks missing modules/providers afterward, and emits structured JSON after-state evidence. A failed optional group is recorded without stopping other repairable groups. The after-state also includes lightweight backend probes, such as ONNX provider visibility, isolated CTranslate2/faster-whisper imports, Transformers/OpenAI Whisper import probes, whisper.cpp `Model.transcribe` availability, llama.cpp import status, llama.cpp GPU-offload status, and `llama-mtmd-cli` or Qwen3 ASR handler visibility. CPU backend usability and accelerator verification are reported separately: a backend can be usable on CPU while `accelerator_probe.ok=false` records an unverified CUDA, DirectML, OpenVINO, or Vulkan path. For every backend that probes usable, repair writes `Logs\dependency_resolution_<group>.json` with schema `easy_asr_bench.runtime_resolution.v1`, package/provider versions where available, the selected accelerator expectation, and whether the accelerator was actually verified. If a resolution file already exists, repair records whether it was still valid or stale before refreshing it by comparing schema, dependency group, runtime config, probe kind, status, versions, providers, runtime path, and accelerator verification. Valid saved resolutions may be reused for already-ok dependency groups when current dependency checks still pass. Python package groups require matching saved package versions; ONNX provider resolutions require the saved providers to still be listed by a lightweight provider probe; verified accelerator resolutions require a lightweight live provider/backend recheck such as ONNX provider visibility, CTranslate2 CUDA availability, or llama.cpp GPU/offload support; requested-but-unverified accelerator resolutions are stale by policy and force a fresh probe. Native `llama-mtmd-cli` resolutions require the saved CLI path to still exist. Reused entries are reported as `already_ok_cached_resolution` and counted in `cached_runtime_resolutions`. Each safe repair run also writes `Logs\repair_all_safe_last.json`; benchmark report environment metadata summarizes those saved dependency resolutions and the last safe-repair run so cached, stale, and requested-but-unverified accelerator states are visible in report artifacts.

`setup.bat --doctor --validate-real-smoke` runs safe repair-plan execution first, then executes configured runtime-matrix smoke rows and emits `easy_asr_bench.real_smoke_validation.v1` JSON. By default it runs `setup_repair_all_safe`, `cpu_model_smoke`, and `compare_html_offline`. Add `--install-deps` and `--allow-downloads` only when the validation machine is intentionally allowed to repair optional runtimes or fetch model/media fixtures. Use `--no-network` for an explicit offline validation policy; it is recorded in JSON and prevents `--allow-downloads` from being forwarded to row scripts.

The first-run baseline is CPU-safe and discloses its optional runtime before install. `Run.bat --first-run-smoke` emits noninteractive JSON with the first-run action state plus the same repair-plan schema/summary and `setup.bat --doctor --repair-all-safe` command used by setup diagnostics, so installed-app and clean-VM QA can prove first-run guidance is not disconnected from bootstrap repair. GPU acceleration is offered only after a selected model has a supported provider path. The default config allows accelerator installs, but CPU-safe packages remain the fallback when no supported provider is detected, an accelerator package install fails, or the adapter cannot safely use GPU.

Easy ASR Bench only uses CUDA requirement files when `nvidia-smi` detects an NVIDIA GPU. If no NVIDIA GPU is detected, doctor prints the reason and tries other supported accelerator paths where practical. ONNX uses OpenVINO on Intel hardware and DirectML on other Windows GPU paths, including AMD. DirectML ONNX sessions use DirectML-safe options: memory pattern disabled and sequential execution. GGUF LLM CUDA installs select a llama-cpp-python prebuilt wheel index from the detected NVIDIA driver/Python runtime and fall back to the CPU package if that index is not reachable. GGUF LLM Vulkan installs try the Vulkan prebuilt wheel first when a Vulkan runtime is detected; local source builds require explicit opt-in plus SDK/build tooling. Hugging Face/OpenAI Whisper GPU acceleration is packaged for NVIDIA CUDA. AMD's Windows ROCm PyTorch path exists only for AMD's supported Python/GPU matrix, so setup reports AMD detection but does not silently install ROCm wheels on unsupported AMD systems. faster-whisper's AMD path requires a ROCm CTranslate2 build path, not the packaged Windows pip flow. `whisper.cpp` through `pywhispercpp` is CPU-only in the packaged flow because its GPU path requires source/build flags rather than a stable simple wheel path.

Precision labels are detected broadly for reporting and selection, including INT4/5/6/8, FP4, NF4, NVFP4/NVP4, FP8, BF8, BF16/bfloat16, FP16/32, Q2-Q8, K_M/K_S/K_L variants, and IQ quantization names. Runtime support still depends on the selected backend: GGUF quantization is handled by llama.cpp, ONNX precision depends on the ONNX model/export, and Hugging Face safetensors quantization depends on the model config and installed Transformers/Torch stack.

Setup modes:

```text
setup.bat
setup.bat --repair
setup.bat --update
setup.bat --uninstall
setup.bat --doctor
setup.bat --local
setup.bat --dry-run
setup.bat --dry-run --verify-release
```

`setup.bat --dry-run` is local and does not download or install files. `setup.bat --dry-run --verify-release` validates the public release path: it verifies the installer script before execution, downloads release manifest/checksums/app ZIP to temp, verifies release hashes, checks ZIP layout, and exits without installing.

Release checksums protect the setup/bootstrap assets that Easy ASR Bench publishes on GitHub. They are not a requirement for normal user-supplied ASR or LLM models; those are validated by file structure, safe format handling, runtime dependency checks, and clear errors.

Setup writes logs to `Logs/setup.log`.

Default uninstall removes app/runtime files but preserves user data folders and `config.json`. A standalone downloaded `setup.bat --uninstall` can use the installed uninstaller when the installer script is not beside `setup.bat`. Destructive user-data removal requires the explicit PowerShell installer flags `-RemoveUserData -ConfirmRemoveUserData "DELETE EASY ASR BENCH USER DATA"`.

Updates move preserved user folders and `config.json` into the new install instead of recursively copying large model folders. They write `Logs/install-preservation-report.json` listing preserved user folders, file counts, byte counts, and the `move_without_model_copy` method. If activation fails, rollback moves those preserved folders back with the previous install.

For release-asset validation commands and manual Windows QA rows, see `docs/release_verification.md`.
