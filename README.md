# Easy ASR Bench

Easy ASR Bench is a Windows drag-and-drop benchmark app for comparing local speech-to-text models on your own audio and video files.

Drop models into `Models`, drop media into `Input` or onto the launcher, choose the models from a numbered menu, and get a complete report folder with:

- `results.txt`: readable transcript and benchmark report
- `results.json`: canonical machine-readable run data
- `benchmark.csv`: spreadsheet-friendly performance rows
- `compare.html`: offline visual comparison page with LLM-corrected reference scoring

## What It Does

Easy ASR Bench gives you a practical answer to:

```text
Which local transcription model is best for my files, on my machine?
```

It runs every selected ASR model on the same normalized audio and the same chunk boundaries, then compares transcript differences, speed, RAM, VRAM when CUDA metrics are available, and errors.

## Supported Model Types

### Runnable ASR Models

- Known multi-file ONNX ASR layouts, including AR folders with `int8`, `fp16w`, `fp32`, `f32`, or `float32` precision folders
- Known multi-file ONNX ASR layouts, including NAR folders with `int8`, `fp16w`, `fp32`, `f32`, or `float32` precision folders
- Hugging Face Transformers ASR folders using `.safetensors` weights, including native FP32/float32 weights
- Hugging Face Whisper and Transformers Safetensors folders, including sharded folders when the index and all shard files are present
- faster-whisper / CTranslate2 folders
- whisper.cpp GGML `.bin` models
- Generic ONNX CTC manifest v1 models with a valid `modelbench.json` manifest using the built-in CTC recipe
- Audio/ASR GGUF packages with a matching `mmproj` projector through llama-cpp-python Qwen3 ASR support or `llama-mtmd-cli`

Precision and quantization labels such as INT4, INT5, INT6, INT8, FP4, NF4, NVFP4/NVP4, FP8, BF8, BF16/bfloat16, FP16, FP32, Q4/Q5/Q6/Q8, K_M/K_S variants, and IQ variants are detected for grouping/reporting when the underlying model backend supports them. Easy ASR Bench does not invent a runtime for an unsupported model format; it labels the model accurately and installs the dependency group for the matching backend.

Complete Hugging Face Safetensors folders that do not look like text-generation LLMs but also do not identify a known ASR architecture are listed as **ASR probe required**, not as normal runnable models. In interactive mode you can deliberately choose the probe option. If the model cannot actually run, that failure is captured in the report with the traceback and the rest of the batch continues.

### Blocked By Default

- OpenAI Whisper `.pt` checkpoint files are detected, but blocked by default because `.pt` uses pickle-backed loading. They run only when a checksum is explicitly allowlisted by the app or when you deliberately enable unsafe trusted-file loading in `config.json`.

### Recognized, Explained, And Not Misclassified

The scanner also recognizes common model packages that should not be silently treated as unknown files or as the wrong model type. When one of these layouts is incomplete or outside the app's packaged runtime scope, Easy ASR Bench reports the exact package family, expected sibling files, and the concrete next action instead of pretending it is runnable.

- NeMo `.nemo` archives
- FunASR folders with `model.pt`, `config.yaml`, `am.mvn`, and tokenizer/BPE files
- sherpa-onnx Whisper packages with matching encoder, decoder, tokens, and optional weights
- Split Whisper/Transformers.js ONNX, Granite-style split ONNX, Qwen split ONNX, and ORT edge graph packages
- Core ML / WhisperKit `.mlmodelc` packages, which are not runnable in this Windows-first app
- Incomplete or mismatched Audio/ASR GGUF packages, including missing or nonmatching `mmproj` projectors
- Incomplete sharded Hugging Face Safetensors folders, including missing shard detection from Safetensors index JSON files

### GGUF Reference Models

GGUF text LLMs are supported as local reference/correction models, not direct transcription models.

That means a GGUF LLM can be used to help create an LLM-corrected reference transcript from multiple ASR outputs. The product never labels that as human ground truth.

Local reference/correction LLM loading currently supports `.gguf` through llama.cpp. Hugging Face `.safetensors` folders are supported for ASR models, not as local text LLMs in this app. Other LLM package formats such as GPTQ/AWQ safetensors, EXL2, ONNX LLMs, TensorRT-LLM engines, or raw PyTorch checkpoints are not loaded as local reference LLMs; use a GGUF export or the manual ChatGPT/Claude workflow.

For local LLMs, a `.gguf` file is the required runnable artifact. The tokenizer and metadata are normally embedded in the GGUF. If the scanner sees a Hugging Face text-generation safetensors folder, it reports that a GGUF export is needed for local reference/correction instead of treating the folder as runnable.

## Quick Start

1. Download `setup.bat` from the latest release.
2. Double-click `setup.bat`.
3. Open the installed folder.
4. Put local ASR model folders or files in `Models`.
5. Put audio/video files in `Input`, or drag them onto `Drop_Audio_Or_Folders_Here.bat`.
6. Double-click `Run.bat`.
7. Choose ASR models and precision buckets.
8. Choose an optional LLM reference workflow.
9. Open the report folder created under `Output`.

### Hugging Face Model Download

In the interactive model menu, choose `D` to paste a Hugging Face model link or `owner/model` repo id. Repo links, `/tree/main` links, nested folder links, file links, and links with extra trailing slashes are accepted. Easy ASR Bench inspects the repo file list first and walks back to the nearest parent model package when a pasted folder is not itself a model folder.

It downloads the selected runnable package plus required metadata files such as config, tokenizer, processor, ONNX sidecars, Safetensors shard indexes, shard files for the chosen index, or matching GGUF `mmproj` files. It does not download every weight variant in the repo by default.

For large or unknown layouts, Easy ASR Bench shows the file count and asks before downloading. Unknown folders can be downloaded for inspection, but they are not treated as runnable unless the scanner recognizes them after download.

After a download, the app rescans the package. If it still looks incomplete and exact missing-file matches exist in the Hugging Face repo, it lists those files and asks before downloading them. Existing local files are skipped when the same package is downloaded again.

If the missing requirement is ambiguous, the app reports it instead of guessing. For tightly scoped same-package files, such as selected ONNX sidecars, parent metadata, or a matching ASR GGUF projector, it can offer a separate repair prompt without pulling alternate weight variants. It also writes `hf_missing_file_request.json` and `hf_missing_file_prompt.txt` beside the package. Those files can be pasted into a local or external LLM to get structured recommendations; the app accepts only exact filenames from the repo file list and asks again before downloading them.

The downloader has fixture coverage for representative Hugging Face layouts with GGUF quant folders, split GGUF parts, Safetensors shards, ONNX sidecars, and mixed ASR/LLM layouts. That is not a promise that every future repo layout is known; unknown layouts are handled conservatively and should be reviewed rather than silently treated as runnable. Live Hugging Face smoke results should be recorded in the release smoke artifact when they are actually run.

## Model Folder Examples

```text
Models/
  known-onnx-ar/
    int8/
      encoder.onnx
      encoder.onnx_data
      prompt_encode.onnx
      prompt_encode.onnx_data
      decode_step.onnx
      decode_step.onnx_data
      embed_tokens.onnx
      embed_tokens.onnx_data
    fp16w/
      ...
    tokenizer.json
    tokenizer_config.json
    preprocessor_config.json

  whisper-large-v3/
    config.json
    model.safetensors
    tokenizer.json
    preprocessor_config.json

  custom-ctc-onnx/
    model.onnx
    modelbench.json

  Reference LLMs/
    local-reference-llm.Q4_K_M.gguf
```

Easy ASR Bench scans `Models` recursively, so you can organize models in nested folders by family, app, format, or quality tier.

## Generic ONNX CTC Manifest v1

Generic ONNX CTC manifest v1 models need `modelbench.json` so Easy ASR Bench knows how to preprocess, feed inputs, select outputs, and decode them safely. This adapter supports CTC-style ASR ONNX only. Whisper encoder-decoder, transducer/RNNT, seq2seq, sherpa-onnx, Qwen split ONNX, and custom decoder-loop graphs require dedicated adapters.

Minimal CTC manifest:

```json
{
  "schema": "easy_asr_bench.model_manifest.v1",
  "display_name": "Custom ONNX CTC ASR",
  "task": "automatic-speech-recognition",
  "backend": "onnxruntime",
  "precision": "int8",
  "files": {"model": "model.onnx"},
  "audio": {
    "sample_rate": 16000,
    "channels": 1
  },
  "inputs": {"waveform": {"name": "input_values", "dtype": "float32"}},
  "outputs": {"logits": "logits"},
  "preprocessing": {"type": "raw_waveform", "normalize": true},
  "decoding": {"type": "ctc", "blank_token_id": 0, "vocab_file": "vocab.json"}
}
```

## LLM-Corrected Reference Workflow

Every `results.txt` includes an instruction block for creating an LLM-corrected reference transcript.

When you run the app interactively, the reference menu offers:

- use an auto-detected local GGUF LLM from `Models`
- paste a GGUF file path or folder from another app and save it for future runs
- use ChatGPT, Claude, or another external LLM manually
- skip LLM reference scoring

Workflow:

1. Run the benchmark.
2. Open `results.txt` or `results_llm_prompt_part_001.txt`.
3. Give the LLM-corrected reference instruction block to a local GGUF LLM or an external LLM.
4. The LLM returns JSON with schema `easy_asr_bench.llm_reference.v1`.
5. Open `compare.html`.
6. Paste that JSON into the reference box.
7. Click **Validate Reference and Score Models**.

The HTML report scores all models against that LLM-corrected reference and shows strict WER, normalized WER, CER, timing, memory, and pairwise differences.

This is a useful benchmark reference, not human ground truth.

## Output Report Folder

For each input file:

```text
Output/
  meeting__20260606_142231/
    results.txt
    results.json
    benchmark.csv
    compare.html
```

When a run processes multiple audio/video files, Easy ASR Bench also writes a batch dashboard:

```text
Output/
  batch__20260606_143012/
    index.html
    batch.json
```

The batch dashboard shows several files side by side, pages through large file sets, filters by path/status, summarizes every model per file with speed/RAM/VRAM/errors, and links to the full `compare.html` for detailed transcript review.

## Windows Launchers

- `setup.bat`: install or repair the app
- `setup.bat --dry-run`: verify setup command structure without changing files or network access
- `setup.bat --dry-run --verify-release`: download release assets to temp, verify hashes and ZIP layout, and exit without installing
- `setup.bat --doctor`: run environment checks
- `Run.bat`: scan models, choose models, process inputs
- `Drop_Audio_Or_Folders_Here.bat`: drag files/folders directly onto the app
- `Open_Latest_Report.bat`: open the newest `compare.html` report
- `Open_Models_Folder.bat`: open the model drop folder
- `Open_Input_Folder.bat`: open the input folder
- `Open_Output_Folder.bat`: open the report folder
- `Edit_Config.bat`: edit configuration

## Dependencies And GPU Support

Setup installs the core runtime first. Model-specific packages are installed only when a selected model needs them. Before installing an optional dependency group, the app shows the package names, requirement files, package indexes, install location, network destinations, PATH changes, size class, and fallback behavior. Press Enter once to install the disclosed group, or type `s` to skip only the affected models.

If an optional dependency install fails, Easy ASR Bench skips only the affected model and continues with any other runnable models. `setup.bat --doctor` lists each dependency group, what it enables, what is missing, and the manual repair command.

GPU acceleration is detected, not assumed. If `config.json` requests or prefers GPU but the selected runtime cannot use the requested provider, the console warns that the run may fall back to CPU. Reports also include provider diagnostics so a user can tell whether GPU support was actually available.

Easy ASR Bench is GPU-first. The default config prefers GPU and allows accelerator package installation when a supported provider is detected. CPU is a fallback, not the goal.

When GPU setup is possible, ONNX models use CUDA on NVIDIA, OpenVINO on Intel, or DirectML on Windows GPUs including AMD, Intel, and NVIDIA. Hugging Face/OpenAI Whisper models use the PyTorch CUDA helper on NVIDIA. AMD's native Windows PyTorch ROCm path is currently limited to AMD's supported Python/GPU matrix, so the packaged flow does not silently install it for every AMD system. faster-whisper installs CUDA cuBLAS/cuDNN Python runtime wheels on NVIDIA; AMD ROCm CTranslate2 requires a ROCm build path and is not treated as a simple Windows pip install. GGUF reference LLMs use llama-cpp-python CUDA 12.4 wheels on NVIDIA when the wheel index is reachable; if the prebuilt CUDA wheel index is unavailable, setup falls back to the CPU package instead of attempting a local source build. GGUF reference LLMs also expose a Vulkan build path for AMD/Intel/NVIDIA systems when the Vulkan runtime and Vulkan SDK build tools are detected.

`whisper.cpp` via `pywhispercpp` remains CPU-only in the packaged dependency flow because current GPU support requires source/build flags rather than a stable simple wheel install. Use faster-whisper or HF/OpenAI Whisper for GPU ASR.

## Safety

Easy ASR Bench does not execute arbitrary Python files from model folders. Safetensors are used for Hugging Face ASR folders because they avoid pickle-style checkpoint execution. Generic ONNX models run only through built-in manifest recipes. GGUF files are treated as local text LLMs for reference/correction unless a dedicated ASR adapter is added.

## Troubleshooting

- **No runnable ASR models:** put a complete supported model folder in `Models`.
- **Dependency missing:** accept the install prompt when a selected model needs an optional dependency group, or run `setup.bat --doctor` and use the printed repair command.
- **Standalone `.safetensors` file:** use the complete Hugging Face model folder, not only the weights file.
- **Generic `.onnx` file:** add `modelbench.json` with CTC decoding metadata and a vocab file. Non-CTC ONNX graphs require a dedicated adapter.
- **GPU unavailable:** run `setup.bat --doctor`. It reports NVIDIA, AMD, Intel, Vulkan, Torch CUDA, ONNX Runtime providers, and dependency repair commands. If setup cannot make GPU work, CPU fallback is reported explicitly.
- **Cannot find `compare.html`:** run a benchmark first, then open `Open_Latest_Report.bat` or go to the newest folder under `Output`.
- **Media conversion failed:** check that the file opens normally and that there is enough disk space in `Temp`.
- **GGUF dependency missing:** install the `llama_cpp` dependency group when prompted, or choose the manual ChatGPT/Claude workflow.
- **GGUF model lives in another app folder:** choose the paste-path option in the LLM reference menu. The path is saved in `config.json` and scanned again on the next run.
- **Whisper model not detected:** check `docs/whisper_models.md`.
- **Setup details:** see `docs/what_setup_installs.md`.
- **Release verification:** see `docs/release_verification.md`.
