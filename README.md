# Easy ASR Bench

Windows app for answering one practical question:

```text
Which local speech-to-text model should I use for these files, on this machine?
```

Easy ASR Bench is built for people who want to compare ASR models without building a Python project, writing benchmark scripts, or guessing whether a model package is complete. Install it, add models, drop in audio or video, select the models to compare, and open one HTML report.

The app keeps the workflow deliberately simple:

```text
Download setup.bat
Double-click
Add or download models
Drop in audio or video
Open final_results.html
```

It runs each selected model against the same normalized audio and chunk boundaries, then reports transcript text, runtime, RAM, available VRAM/GPU-memory telemetry, failures, provider diagnostics, and optional quality scoring against an LLM-corrected reference.

Everything is local-first. Your media and local model files stay on your machine. Network access is used for setup, optional dependency installs, update checks, and Hugging Face model downloads that you choose.

<details>
<summary><strong>Contents</strong></summary>

- [How To Use](#how-to-use)
- [What You Get](#what-you-get)
- [Features](#features)
- [The Main Report](#the-main-report)
- [Hugging Face Model Downloader](#hugging-face-model-downloader)
- [Supported Model Packages](#supported-model-packages)
- [LLM-Corrected Reference Scoring](#llm-corrected-reference-scoring)
- [Batch Runs And Resume](#batch-runs-and-resume)
- [Installation, Repair, And Launchers](#installation-repair-and-launchers)
- [Dependencies And Acceleration](#dependencies-and-acceleration)
- [Output Files](#output-files)
- [Safety Model](#safety-model)
- [Release Verification](#release-verification)
- [Troubleshooting](#troubleshooting)
- [Support](#support)

</details>

## How To Use

1. Go to the [latest release](../../releases/latest).
2. Download `setup.bat`.
3. Double-click `setup.bat`.
4. When setup finishes, choose one of the normal actions:
   - `R`: run Easy ASR Bench
   - `P`: paste a Hugging Face model link
   - `M`: open the `Models` folder
   - `I`: open the `Input` folder
5. Add at least one ASR model:
   - paste a Hugging Face ASR model link in the downloader,
   - choose the small CPU first-run baseline for a quick sanity check, or
   - copy a supported model package into `Models`.
6. Add media:
   - put audio/video files in `Input`,
   - paste file or folder paths when prompted, or
   - drag files/folders onto `Drop_Audio_Or_Folders_Here.bat`.
7. Choose the models to compare.
8. Open `Open_Latest_Report.bat` or the newest `Output/.../final_results.html`.

The first-run baseline is intentionally small and CPU-safe. It is a sanity check that proves the app can run a model, not a benchmark pack that ranks several models.

Interactive model selections are saved in `config.json`. A later drag/drop run can reuse the saved selection without prompting. If the selected model IDs become stale because model folders moved or changed, the app stops and asks you to choose again instead of silently changing the benchmark set.

## What You Get

Easy ASR Bench produces a report folder for each run with:

- `final_results.html`: the main report entry point.
- `compare.html`: detailed per-file transcript comparison.
- `results.txt`: readable transcript and benchmark report.
- `results.json`: canonical machine-readable run data.
- `benchmark.csv`: spreadsheet-friendly performance rows.
- Optional `scored_report.json` and `compare_scored.html` when a corrected reference is imported or generated successfully.

The HTML report is offline and self-contained for normal review. It is designed to answer the real questions after a run:

- Which model was fastest?
- Which model used the most RAM or GPU memory?
- Which runs failed, and why?
- Which provider actually ran?
- Which transcript looks best?
- If a corrected reference is available, which model scored best?

## Features

- Windows-first app with a double-click setup and launcher flow.
- Drag-and-drop or paste-path media input for audio, video, files, and folders.
- Interactive model selection with saved repeat-last-run behavior.
- Built-in Hugging Face downloader for supported ASR packages and reference GGUF packages.
- Model scanner that recognizes runnable, incomplete, blocked, and unsupported packages without guessing.
- Same-media, same-chunk benchmarking across selected models.
- Lazy chunk materialization so long audio is not fully retained in memory as chunk arrays.
- Runtime-only ranking before a corrected reference is available.
- Accuracy-aware scoring after an LLM-corrected reference is imported or generated.
- Batch dashboard for many files, with filtering, paging, model summaries, per-file links, and resume support.
- Browser-local corrected-reference persistence in `final_results.html`.
- Export/import for edited corrected-reference data.
- Local GGUF reference/correction LLM workflow through llama.cpp when dependencies are available.
- Manual external LLM workflow through copy/paste prompts.
- Structured errors for model failures, media failures, and chunk failures.
- Dependency groups installed only when selected models need them.
- Doctor and repair commands for missing or broken runtime dependencies.
- Public release support matrix generated from release-smoke evidence.

## The Main Report

`final_results.html` is the main report entry point.

For a single file, it opens the single-file report wrapper and links to the detailed `compare.html`. For a batch, it opens the batch dashboard with every processed file and model summarized in one place.

The report can:

- show runtime, memory, provider, transcript, and error details for each model;
- compare transcripts side by side;
- show runtime-only ranking when no quality reference exists;
- accept pasted corrected-reference JSON;
- validate and score against that corrected reference;
- persist pasted reference edits in the browser;
- export edited reference data as JSON;
- import edited reference data again later;
- link from batch rows to per-file `compare.html` details.

This is the intended day-to-day workflow: run the benchmark, open `final_results.html`, inspect transcripts, paste or generate a corrected reference when you want quality scoring, and export edited reference data if you need to preserve browser edits outside local storage.

## Hugging Face Model Downloader

From the interactive model menu, choose `D` or choose `P` from the setup post-install menu to paste a Hugging Face link.

Accepted input includes:

- `owner/model` repo IDs;
- normal repo URLs;
- `/tree/main` URLs;
- nested folder URLs;
- direct file URLs;
- links with extra trailing slashes.

The downloader inspects the repo file list before downloading. If you paste a nested folder that is not itself a complete model package, it walks back to the nearest package parent when it can do so safely.

It downloads the selected package plus required companion files, such as:

- config, tokenizer, processor, and preprocessor files;
- selected ONNX sidecars;
- Safetensors shard indexes and selected shard files;
- matching GGUF `mmproj` projector files for ASR GGUF packages;
- split GGUF parts for a selected GGUF package.

It does not download every weight variant in a repo by default. For large or uncertain choices, it shows the file count and asks before downloading. When Hugging Face reports file sizes, the app checks space for both the Hub cache and the `Models` copy.

If a downloaded package is incomplete, the app rescans it. When exact missing-file matches exist in the same Hugging Face repo, it can ask before downloading those files. If the missing requirement is ambiguous, it reports the ambiguity instead of guessing. It can also write `hf_missing_file_request.json` and `hf_missing_file_prompt.txt` next to the package so a local or external LLM can recommend exact filenames from the repo list.

Before download, the app estimates projected local path lengths and warns or blocks when a package would exceed a conservative Windows path budget. Short install/model folders are still the lowest-risk path for very deep model repos.

## Supported Model Packages

Release verification is evidence-based. Code support means the scanner and adapter know the package family. A release claim means the release-smoke artifact proves that row for that release.

The generated release support matrix is in [docs/support_matrix.generated.md](docs/support_matrix.generated.md). Rows that have not been proven for the release stay marked `Not verified`.

Runnable ASR package families include:

- known multi-file ONNX ASR layouts, including AR and NAR exports with precision folders;
- Hugging Face Whisper and Transformers ASR folders using `.safetensors` weights;
- sharded Hugging Face Safetensors folders when the index and shards are complete;
- faster-whisper / CTranslate2 folders;
- whisper.cpp GGML `.bin` models;
- Generic ONNX CTC manifest v1 folders with `modelbench.json`;
- Audio/ASR GGUF packages with matching `mmproj` projectors when the packaged llama.cpp MTMD path is available.

Recognized but blocked, incomplete, or explained packages include:

- standalone `.safetensors` files without the rest of the Hugging Face model folder;
- incomplete Safetensors shard folders;
- OpenAI Whisper `.pt` checkpoints without explicit checksum allowlisting;
- NeMo `.nemo` archives;
- FunASR folders;
- sherpa-onnx Whisper packages;
- split Whisper/Transformers.js ONNX, Granite-style split ONNX, Qwen split ONNX, and ORT edge graph packages;
- Core ML / WhisperKit `.mlmodelc` packages;
- mismatched or incomplete ASR GGUF plus projector packages;
- Hugging Face text-generation folders that need a GGUF export before they can be used as local reference LLMs.

Generic ONNX CTC models need a manifest so the app knows how to preprocess audio, feed inputs, select outputs, and decode safely. See [docs/supported_models.md](docs/supported_models.md) for package details.

## LLM-Corrected Reference Scoring

Easy ASR Bench can score models against an LLM-corrected reference. This is useful for benchmarking, but it is not human ground truth.

You can use:

- an auto-detected local GGUF reference/correction LLM from `Models`;
- a pasted GGUF file or folder path from another app;
- ChatGPT, Claude, or another external LLM manually;
- no reference, leaving the report in runtime-only mode.

The normal manual workflow is:

1. Run the benchmark.
2. Open `final_results.html`.
3. Copy the report's LLM reference prompt or use the prompt text files.
4. Ask the LLM to return JSON with schema `easy_asr_bench.llm_reference.v1`.
5. Paste the JSON into the report.
6. Validate and score the models.
7. Export edited reference data if you want a portable copy of browser edits.

When a local GGUF reference LLM is selected, the app can generate a corrected reference, validate the returned JSON against source and chunk metadata, score the ASR outputs, and write `scored_report.json` plus `compare_scored.html`.

The score view includes strict WER, normalized WER, CER, balanced rank, timing, memory, pairwise differences, and readable validation errors when reference JSON does not match the run.

## Batch Runs And Resume

Batch mode processes many files while keeping each per-file report separate and publishing a batch dashboard:

```text
Output/
  batch__20260606_143012/
    final_results.html
    _data/
      batch.json
      batch-records.json
```

The batch dashboard shows files side by side, filters by status/path, pages through large sets, summarizes every model per file, stores corrected-reference edits in the browser, supports reference export/import, and links to each detailed `compare.html`.

Batch runs write `Logs/batch_resume_manifest.json`. If the same input file, selected model IDs, reference choice, and relevant runtime/transcription settings still match, reruns skip completed files and reuse the existing published report folder. Missing, corrupt, empty, or failed reports are not treated as complete.

If a batch is interrupted after at least one file completes, the app writes a partial `final_results.html` overview for completed rows.

## Installation, Repair, And Launchers

Normal users only need `setup.bat`. It downloads and verifies the matching installer script, manifest, checksums, and app ZIP for the release.

Common launchers:

- `setup.bat`: install or repair the app.
- `setup.bat --dry-run`: verify setup command structure without changing files or network access.
- `setup.bat --dry-run --verify-release`: download release assets to temp, verify hashes and ZIP layout, and exit without installing.
- `setup.bat --doctor`: run environment checks.
- `Run.bat`: scan models, choose models, and process inputs.
- `Drop_Audio_Or_Folders_Here.bat`: drag files/folders directly onto the app.
- `Open_Latest_Report.bat`: open the newest `final_results.html`.
- `Open_Models_Folder.bat`: open the model folder.
- `Open_Input_Folder.bat`: open the input folder.
- `Open_Output_Folder.bat`: open the report folder.
- `Edit_Config.bat`: edit configuration.

Installed releases also create Start Menu shortcuts for run, drag/drop, latest report, output folder, config editing, repair, and uninstall. Uninstall preserves user data by default.

Installer update and repair paths preserve `Models`, `Input`, `Output`, `Logs`, `Cache`, `Temp`, and `config.json` separately from the staged app until validation succeeds. If setup or installed validation fails, preserved data is restored to the previous install path.

## Dependencies And Acceleration

Setup installs the core runtime first. Model-specific packages are installed only when a selected model needs them. Before installing an optional dependency group, the app shows the package names, requirement files, package indexes, install location, network destinations, PATH changes, size class, and fallback behavior.

If an optional dependency install fails, Easy ASR Bench skips only the affected model and continues with any other runnable models. `setup.bat --doctor` lists dependency groups, what they enable, what is missing, and the repair command.

Acceleration is detected, not assumed. Reports include provider diagnostics so you can see whether a requested provider actually ran or fell back.

Provider paths include:

- ONNX Runtime CPU for Generic ONNX and known ONNX ASR layouts.
- ONNX Runtime CUDA on NVIDIA when installed and verified.
- OpenVINO on Intel when installed and verified.
- DirectML on Windows GPUs when installed and verified.
- PyTorch CUDA for Hugging Face/OpenAI Whisper when installed and verified.
- CTranslate2 CUDA for faster-whisper when installed and verified.
- llama.cpp CPU, CUDA, or Vulkan paths for GGUF reference/ASR flows when the matching dependency path is available and verified.

CPU fallback is explicit. If a requested accelerator is unavailable and a safe CPU path exists, the report records the request, fallback, and active provider instead of pretending the accelerator ran.

`whisper.cpp` through `pywhispercpp` remains CPU-only in the packaged dependency flow because GPU support currently requires source/build flags rather than a stable simple wheel install. Use faster-whisper or Hugging Face/OpenAI Whisper for common GPU ASR workflows.

## Output Files

Single-file reports look like:

```text
Output/
  meeting__20260606_142231_a1b2c3d4/
    results.txt
    results.json
    benchmark.csv
    final_results.html
    compare.html
```

Report directories are created through hidden staging folders and published only after required artifacts exist. IDs include time and source identity data to avoid same-name collisions.

Generated normalization WAVs under `Temp` are cleaned automatically after they are older than `advanced.stale_temp_wav_hours`. Set `advanced.keep_temp_wavs` to `true` only when you intentionally want to inspect those temporary files.

## Safety Model

Easy ASR Bench is conservative about model files and release claims:

- It does not execute arbitrary Python files from model folders.
- Hugging Face ASR folders use Safetensors instead of pickle-backed checkpoint loading.
- OpenAI Whisper `.pt` files are blocked by default unless the checksum is explicitly allowlisted or trusted-file loading is deliberately enabled.
- Generic ONNX models run only through built-in manifest recipes.
- Unsupported or incomplete packages are explained instead of treated as runnable by guesswork.
- Chunk failures are structured errors, not fake transcript text.
- Missing, corrupt, empty, or failed reports do not count as completed batch work.
- Config writes and report publishing use atomic replace/publish patterns.
- Release support claims are tied to smoke evidence, not hopeful documentation.

## Release Verification

A public release should include:

- `setup.bat`
- `install.ps1`
- `manifest.json`
- `checksums.json`
- the Windows ZIP
- `release-smoke-vX.Y.Z.json`
- release verification transcripts or notes

The release-smoke artifact is the authority for what was proven in that release. Automated package checks can pass while optional hardware/model/media rows remain `not_run`; those rows should not be described as verified until evidence exists.

See [docs/release_verification.md](docs/release_verification.md) and [docs/support_matrix.generated.md](docs/support_matrix.generated.md).

## Troubleshooting

- **No runnable ASR models:** add a complete supported model folder to `Models` or use the Hugging Face downloader.
- **Standalone `.safetensors` file:** download or copy the complete Hugging Face model folder, not only the weights file.
- **Generic `.onnx` file:** add `modelbench.json` with CTC decoding metadata and a vocab file. Non-CTC ONNX graphs need a dedicated adapter.
- **Dependency missing:** accept the install prompt for the selected model group, or run `setup.bat --doctor` and use the printed repair command.
- **Saved selection is stale:** run interactively once, choose the intended models again, and the saved drag/drop selection will be refreshed.
- **GPU unavailable:** run `setup.bat --doctor`. It reports provider visibility and repair commands. Reports also show actual provider/fallback metadata.
- **Cannot find a report:** run a benchmark first, then open `Open_Latest_Report.bat` or the newest folder under `Output`.
- **Media conversion failed:** check that the file opens normally and that there is enough disk space in `Temp`.
- **Output is getting large:** delete older folders under `Output` when you no longer need them. Temporary generated WAVs are swept automatically.
- **Install or model download says disk space is low:** free space on the install/model drive first. Model downloads still ask for typed confirmation before continuing in risky cases.
- **Doctor warns about install or profile paths:** a short ASCII local path outside synced folders is the lowest-risk install location for native Windows runtimes and deep model caches.
- **GGUF dependency missing:** install the `llama_cpp` dependency group when prompted, or use the manual external LLM workflow.
- **GGUF model lives in another app folder:** choose the paste-path option in the reference LLM menu. The path is saved in `config.json`.
- **Whisper model not detected:** see [docs/whisper_models.md](docs/whisper_models.md).
- **Unexpected app error:** check the printed `Logs/crash_*.log`, run `Run.bat --doctor --json`, and open a GitHub issue with the requested diagnostics.
- **SmartScreen, Defender, or antivirus warning:** see [docs/what_setup_installs.md](docs/what_setup_installs.md). The release uses unsigned batch/PowerShell launchers, verifies published setup assets by checksum, and keeps your media/model files local unless you choose a download.

## Support

For bugs, use the GitHub bug report template and include `Run.bat --doctor --json`, the latest run/setup/crash log, Windows version, install mode, and model/runtime type.

Do not upload private media or model files unless you intentionally choose to share them.
