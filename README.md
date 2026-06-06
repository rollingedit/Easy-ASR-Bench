# Easy ASR Bench

Easy ASR Bench is a Windows drag-and-drop tool for comparing local speech-to-text models on your own audio and video files.

The pitch is simple: drop ASR models into `Models/`, drop media into `Input/`, choose the models from a menu, and get a readable benchmark report showing transcripts, speed, memory use, errors, and model-to-model differences.

This first public review release focuses on the product shell and Granite Speech ONNX adapters. It also scans `.onnx`, `.safetensors`, and `.gguf` files and explains what is runnable, incomplete, or unsupported instead of pretending every model file can transcribe audio.

## Who it is for

- People testing local transcription models on Windows
- Reviewers comparing 8-bit, 16-bit-family, and other model precision choices
- Builders who want an adapter-based ASR benchmark app they can extend

## What v0.1 can do

- Scan `Models/` for local model folders and files
- Detect runnable Granite Speech ONNX AR/NAR model folders
- Recognize incomplete or unsupported ONNX, Safetensors, and GGUF candidates
- Normalize audio/video with FFmpeg through `imageio-ffmpeg`
- Reuse the same chunk plan for every selected model
- Write transcript and benchmark reports under `Output/`
- Run on CPU by default

## Current limits

- Hugging Face Safetensors ASR, generic ONNX manifests, and GGUF reference/correction adapters are planned but not runnable yet.
- Word-level timestamps are not produced. Report timestamps are chunk timestamps.
- You must provide model files locally in `Models/`; v0.1 does not auto-download models.

## Quick Start

1. Download `setup.bat` from the latest release.
2. Double-click `setup.bat`.
3. Drop ASR model folders into `Models/`.
4. Drop audio/video files into `Input/` or onto `Drop_Audio_Or_Folders_Here.bat`.
5. Run `Run.bat`.
6. Choose models and precision buckets from the menu.
7. Open the generated report in `Output/`.

Setup installs the app to `%LOCALAPPDATA%\Easy-ASR-Bench` when run as a standalone release BAT. When run inside a checkout, it sets up that checkout directly.

## Model Folder

Drop local model folders into `Models/`.

Runnable in this build:

- Granite Speech AR ONNX folders with `int8/`, `fp16w/`, or `fp32/`
- Granite Speech NAR ONNX folders with `int8/`, `fp16w/`, or `fp32/`

Here, "8-bit" means the repo's `int8` folder. "16-bit" means `fp16w`: FP16-stored weights with FP32 compute/I/O, not a pure FP16 runtime graph.

`Run.bat` scans `Models/` and lets you choose models from a numbered menu.

## Transcribe Files

Drag audio/video files or folders onto `Drop_Audio_Or_Folders_Here.bat`, or put files in `Input/` and run `Run.bat`.

`Run.bat` opens the model scanner, lets you choose runnable ASR models and precision buckets, then prompts for input files.

## Folder Input Mode

Put files in `Input/` and run `Run.bat`.

The app prompts for input after model selection. Press Enter at the input prompt to process supported files currently in `Input/`.

## Output

See `Output/`. Each input gets one consolidated `.txt` report by default, plus CSV benchmark rows in `Output/benchmark_results.csv`.

## Performance Comparison

The report shows model load time, inference time, total wall time, speed versus real time, generated token throughput, RAM, errors, and transcript difference metrics.

All selected variants use the same normalized WAV and identical chunk boundaries.

## Timestamp Note

Timestamps are chunk timestamps. They are not word-level timestamps.

## Troubleshooting

- Not enough disk space: remove unused local model folders from `Models/`.
- Model folder is incomplete: make sure the tokenizer/config files and model sidecar files are present.
- CUDA unavailable: set `runtime.provider` to `cpu` or leave `auto` to fall back to CPU.
- Unsupported/corrupt media file: the error is written to `Logs/` and the app continues with the next file.

## Repository Layout

```text
app/                         Python application
app/adapters/                ASR adapter interface and Granite ONNX adapters
Models/                      Drop model folders/files here
Input/                       Drop audio/video here
Output/                      Reports
Logs/                        Run logs
Temp/                        Temporary normalized audio
setup.bat                    Standalone installer or local setup
Run.bat                      Main app launcher
Drop_Audio_Or_Folders_Here.bat
```

## Safety Notes

Easy ASR Bench does not execute Python code from model folders by default. Standalone `.safetensors`, arbitrary `.onnx`, and `.gguf` files are scanned and reported, but only known ASR adapters are run.
