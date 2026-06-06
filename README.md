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

It runs every selected ASR model on the same normalized audio and the same chunk boundaries, then compares transcript differences, speed, memory, and errors.

## Supported Model Types

### Runnable ASR Models

- Granite Speech ONNX AR folders with `int8`, `fp16w`, or `fp32` precision folders
- Granite Speech ONNX NAR folders with `int8`, `fp16w`, or `fp32` precision folders
- Hugging Face Transformers ASR folders using `.safetensors` weights
- Hugging Face Whisper Safetensors folders
- faster-whisper / CTranslate2 folders
- whisper.cpp GGML `.bin` models
- official-name OpenAI Whisper `.pt` files, with unsafe pickle restrictions
- Generic ONNX ASR models with a valid `modelbench.json` manifest using the built-in CTC recipe

### GGUF Reference Models

GGUF text LLMs are supported as local reference/correction models, not direct transcription models.

That means a GGUF LLM can be used to help create an LLM-corrected reference transcript from multiple ASR outputs. The product never labels that as human ground truth.

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

## Model Folder Examples

```text
Models/
  granite-ar/
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

## Generic ONNX Manifest

Generic ONNX ASR models need `modelbench.json` so Easy ASR Bench knows how to preprocess, feed inputs, select outputs, and decode them safely.

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
  "decoding": {"type": "ctc", "blank_token_id": 0}
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

## Windows Launchers

- `setup.bat`: install or repair the app
- `setup.bat --dry-run`: verify setup command structure without changing files
- `setup.bat --doctor`: run environment checks
- `Run.bat`: scan models, choose models, process inputs
- `Drop_Audio_Or_Folders_Here.bat`: drag files/folders directly onto the app
- `Open_Models_Folder.bat`: open the model drop folder
- `Open_Input_Folder.bat`: open the input folder
- `Open_Output_Folder.bat`: open the report folder
- `Edit_Config.bat`: edit configuration

## Safety

Easy ASR Bench does not execute arbitrary Python files from model folders. Safetensors are used for Hugging Face ASR folders because they avoid pickle-style checkpoint execution. Generic ONNX models run only through built-in manifest recipes. GGUF files are treated as local text LLMs for reference/correction unless a dedicated ASR adapter is added.

## Troubleshooting

- **No runnable ASR models:** put a complete supported model folder in `Models`.
- **Standalone `.safetensors` file:** use the complete Hugging Face model folder, not only the weights file.
- **Generic `.onnx` file:** add `modelbench.json`.
- **CUDA unavailable:** use CPU or install a compatible ONNX Runtime GPU stack.
- **Media conversion failed:** check that the file opens normally and that there is enough disk space in `Temp`.
- **GGUF dependency missing:** install the `llama_cpp` dependency group when prompted, or choose the manual ChatGPT/Claude workflow.
- **GGUF model lives in another app folder:** choose the paste-path option in the LLM reference menu. The path is saved in `config.json` and scanned again on the next run.
- **Whisper model not detected:** check `docs/whisper_models.md`.
- **Setup details:** see `docs/what_setup_installs.md`.
