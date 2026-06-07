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

Default setup installs only `requirements/core.txt`.

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
- GGUF reference LLM with CUDA, when NVIDIA is detected: `requirements/llama_cpp_cuda_cu125.txt`
- GGUF reference LLM with Vulkan build path for AMD/Intel/NVIDIA GPUs, when the Vulkan runtime and SDK build tools are detected: `requirements/llama_cpp_vulkan.txt`

When a selected model needs one of these groups, Easy ASR Bench prompts before installing it. If an optional install fails, the affected model is skipped, other runnable models continue, and the console prints the manual repair command.

`setup.bat --doctor` reports each dependency group, what it enables, missing modules, and the matching `pip install -r requirements/...` repair command.

GPU acceleration is attempted first when a supported provider is detected. The default config sets `runtime.prefer_gpu=true`, `dependency_install.allow_cuda_install=true`, and `dependency_install.allow_accelerator_install=true`. CPU-safe packages remain the fallback when no supported provider is detected, an accelerator package install fails, or the adapter cannot safely use GPU.

Easy ASR Bench only uses CUDA requirement files when `nvidia-smi` detects an NVIDIA GPU. If no NVIDIA GPU is detected, doctor prints the reason and tries other supported accelerator paths where practical. ONNX uses OpenVINO on Intel hardware and DirectML on other Windows GPU paths, including AMD. DirectML ONNX sessions use DirectML-safe options: memory pattern disabled and sequential execution. GGUF LLMs use llama-cpp-python CUDA wheels on NVIDIA and expose a Vulkan build path when the Vulkan runtime and Vulkan SDK build tools are detected. Hugging Face/OpenAI Whisper GPU acceleration is packaged for NVIDIA CUDA. AMD's Windows ROCm PyTorch path exists only for AMD's supported Python/GPU matrix, so setup reports AMD detection but does not silently install ROCm wheels on unsupported AMD systems. faster-whisper's AMD path requires a ROCm CTranslate2 build path, not the packaged Windows pip flow. `whisper.cpp` through `pywhispercpp` is CPU-only in the packaged flow because its GPU path requires source/build flags rather than a stable simple wheel path.

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

Updates write `Logs/install-preservation-report.json` listing preserved user folders, file counts, and byte counts.

For release-asset validation commands and manual Windows QA rows, see `docs/release_verification.md`.
