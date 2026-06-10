# Easy ASR Bench

Transcribe audio and video with local speech-to-text models on Windows, then compare models on your own files and your own hardware.

It is the Open ASR Leaderboard for your audio, on your machine.

Public ASR leaderboards are useful, but they are not your files: fast speech, accents, casual slang, crosstalk, bad mics, lectures, meetings, songs, fast rap, noisy clips, language switches, and whatever else you actually need transcribed. They also are not your CPU, GPU, driver stack, or patience budget.

Give Easy ASR Bench a model link and media files. It handles the setup, downloads the package pieces it needs, installs only the runtime pieces required for that model, runs the transcription, and opens the results report.

That is the product promise:

```text
No Python project setup.
No hunting for sidecar files.
No guessing which optional runtime package to install.
No silent CPU/GPU fallback.
No batch ruined because one model failed.
```

Paste a link. Drop files. Get transcripts. Add more models when you want a real comparison.

The normal user flow is intentionally small:

```text
Download setup.bat
Double-click
Paste a Hugging Face model link or use a model in Models
Drop in audio/video
Open final_results.html
```

Use one model when you just want a transcript. Use several models when you want to compare quality, speed, RAM, GPU memory, provider behavior, and failures side by side.

Easy ASR Bench owns the ugly middle: setup, model detection, Hugging Face package selection, missing-file repair prompts, dependency planning, CPU/GPU provider probing, runtime repair, fallback reporting, batch continuation, and the final report.

<details>
<summary><strong>Contents</strong></summary>

- [Quick Start](#quick-start)
- [What It Does](#what-it-does)
- [Why It Exists](#why-it-exists)
- [The Smart Hugging Face Downloader](#the-smart-hugging-face-downloader)
- [The One Report To Open](#the-one-report-to-open)
- [Corrected Reference Scoring](#corrected-reference-scoring)
- [Model And Runtime Coverage](#model-and-runtime-coverage)
- [Doctor, Repair, And Recovery](#doctor-repair-and-recovery)
- [Batch Runs](#batch-runs)
- [Validation And Release Evidence](#validation-and-release-evidence)
- [Installed Files And Launchers](#installed-files-and-launchers)
- [Safety](#safety)
- [Troubleshooting](#troubleshooting)

</details>

## Quick Start

1. Go to the [latest release](../../releases/latest).
2. Download `setup.bat`.
3. Double-click it.
4. After setup, choose:
   - `R` to run now,
   - `P` to paste a Hugging Face model link,
   - `M` to open the model folder,
   - `I` to open the input folder.
5. Add a model by pasting a Hugging Face ASR link, using the first-run CPU baseline, or copying a supported model package into `Models`.
6. Add media by dropping files into `Input`, pasting paths when prompted, or dragging files/folders onto `Drop_Audio_Or_Folders_Here.bat`.
7. Choose one model for transcription or multiple models for comparison.
8. Open `Open_Latest_Report.bat` or the newest `final_results.html` under `Output`.

The first-run baseline is a small CPU-safe sanity check. It proves the app can run a model; it is not meant to be a full ranking pack.

## What It Does

Easy ASR Bench has two everyday modes:

- **Transcribe:** run one local ASR model on your files and get readable transcripts plus an offline HTML report.
- **Compare:** run multiple models on the same files, with the same normalized audio and chunk boundaries, then see which one is faster, lighter, more reliable, or better against a corrected reference.

It is not only a benchmark harness. The benchmark is the comparison layer on top of a local transcription workflow.

After a run, the report answers:

- What did each model transcribe?
- Which model was fastest?
- Which model used the most RAM or GPU memory?
- Did the requested provider actually run, or did it fall back?
- Which models failed, and at what stage?
- If a corrected reference is available, which model had the best WER/CER/rank?

## Why It Exists

Local ASR is messy in the real world.

A user may have:

- a Hugging Face repo link instead of a clean model folder;
- a subfolder or file URL instead of the package root;
- missing tokenizer, config, shard, ONNX sidecar, or projector files;
- a model that is not actually ASR;
- a broken Python environment;
- a missing media backend;
- a CPU-only machine;
- a GPU provider that installed but does not actually run;
- a batch where one model or one file fails but the rest should continue.

Easy ASR Bench treats that as the product problem, not the user's problem.

The app follows a control-plane loop:

```text
inspect environment, model, and media
classify package/runtime/input state
plan needed downloads, dependencies, providers, or fallbacks
probe real runtime behavior
repair what can be repaired safely
run transcription
continue past isolated failures
report what happened truthfully
```

That is the main moat: the app is a model-aware ASR runtime orchestrator, not just a launcher wrapped around one backend.

## The Smart Hugging Face Downloader

The downloader is built for the way people actually find models.

Paste:

- `owner/model`
- a normal Hugging Face repo URL
- a `/tree/main` URL
- a nested folder URL
- a direct file URL
- a link with trailing slashes

Easy ASR Bench inspects the repo file list first, identifies the nearest model package it can safely use, and downloads the selected runnable package plus required companion files.

It can pull the files that usually make or break local ASR packages:

- tokenizer/config/processor metadata;
- selected ONNX sidecars;
- Safetensors shard indexes and shard files;
- matching ASR GGUF projector files;
- split GGUF parts for a chosen package.

It does not blindly download every weight variant in a repo. For large or uncertain packages, it shows the file count and asks first. When file sizes are available, it checks disk space for both the Hugging Face cache and the final `Models` copy.

If a package lands incomplete, the app rescans it. When exact missing files exist in the same repo, it can offer a targeted repair download. If the missing requirement is ambiguous, it says so instead of guessing. For harder cases, it writes local missing-file request files that can be pasted into a local or external LLM; any returned recommendation still has to match exact repo filenames before the app will offer to download it.

It also preflights deep Windows paths before model downloads, because long Hugging Face cache/package paths are a real source of Windows failures.

## The One Report To Open

`final_results.html` is the report users should open.

For a single file, it links into the detailed comparison report. For a batch, it becomes the dashboard for all files.

The HTML report can show:

- transcripts for every selected model;
- runtime-only rankings before any reference exists;
- speed, RAM, and available GPU-memory telemetry;
- provider and fallback diagnostics;
- structured model/media/chunk failure messages;
- side-by-side transcript comparison;
- corrected-reference paste, validation, scoring, browser persistence, export, and import.

The other output files exist for automation and auditability, but the user-facing path is simple: open `final_results.html`.

## Corrected Reference Scoring

Benchmarks need a reference, but Easy ASR Bench does not pretend an LLM is human ground truth.

You can:

- use no reference and compare runtime/transcripts manually;
- paste corrected-reference JSON from ChatGPT, Claude, or another external LLM;
- use a local GGUF reference/correction LLM through llama.cpp when available;
- paste a GGUF path from another app and save it for future runs.

When reference JSON is pasted or generated, the app validates it against the run's source hash and chunk metadata before scoring. Valid references produce WER, normalized WER, CER, balanced rank, timing, memory, pairwise differences, `scored_report.json`, and `compare_scored.html`.

Invalid references are reported as invalid instead of silently producing misleading scores.

## Model And Runtime Coverage

Runnable ASR package families include:

### Runnable ASR Models

- known multi-file ONNX ASR layouts, including AR and NAR exports;
- Hugging Face Whisper and Transformers ASR folders using Safetensors;
- complete sharded Hugging Face Safetensors folders;
- faster-whisper / CTranslate2 folders;
- whisper.cpp GGML `.bin` models;
- Generic ONNX CTC manifest v1 folders with `modelbench.json`;
- Audio/ASR GGUF packages with matching `mmproj` projectors when the packaged llama.cpp MTMD path is available.

### Blocked By Default

OpenAI Whisper `.pt` checkpoint files are detected, but blocked by default unless the checkpoint checksum is allowlisted or trusted-file loading is explicitly enabled. They are not part of the default runnable model list because `.pt` loading is pickle-backed.

Reference/correction LLM support is GGUF-focused. Hugging Face text-generation Safetensors folders are recognized as needing a GGUF export for this app's local reference workflow.

The scanner also recognizes common packages that should not be misclassified as runnable ASR, including standalone Safetensors files, incomplete shard folders, unchecked OpenAI Whisper `.pt` checkpoints, NeMo, FunASR, sherpa-onnx, split ONNX app packages, Core ML / WhisperKit packages, and incomplete ASR GGUF projector pairs.

Provider paths are detected and reported rather than assumed:

- CPU paths for supported backends;
- ONNX Runtime CUDA where verified;
- OpenVINO where verified;
- DirectML where verified;
- PyTorch CUDA for Hugging Face/OpenAI Whisper where verified;
- CTranslate2 CUDA for faster-whisper where verified;
- llama.cpp CPU/CUDA/Vulkan paths where the matching dependency path is available and verified.

The generated release support matrix is in [docs/support_matrix.generated.md](docs/support_matrix.generated.md). It is built from release-smoke evidence, so unproven rows stay marked `Not verified`.

## Doctor, Repair, And Recovery

`setup.bat --doctor` and repair flows are part of the product, not an afterthought.

The app can diagnose and report:

- Python and virtual environment problems;
- missing or incompatible dependency groups;
- missing FFmpeg/media tooling;
- Hugging Face cache/path issues;
- ONNX provider visibility;
- Torch CUDA availability;
- CTranslate2/faster-whisper import problems;
- Transformers/OpenAI Whisper import problems;
- llama.cpp and native ASR GGUF runtime availability;
- stale saved runtime-resolution evidence.

Repair flows are conservative. They install or repair only the dependency groups needed by selected models, record what they did, probe the backend afterward, and skip only the affected models if a repair fails.

The goal is not "never fail." The goal is to avoid silent failure, avoid fake success, keep the rest of a batch alive when possible, and produce a report that tells the user what failed and what to try next.

## Batch Runs

Batch mode processes many files and writes one dashboard:

```text
Output/
  batch__...
    final_results.html
    _data/
      batch.json
      batch-records.json
```

The dashboard filters and pages through files, summarizes every model per file, saves pasted corrected references in the browser, supports reference export/import, and links to detailed per-file reports.

Batch runs also write `Logs/batch_resume_manifest.json`. If the same file, selected models, reference choice, and runtime settings still match, reruns skip completed files. Missing, corrupt, empty, or failed reports do not count as complete.

## Validation And Release Evidence

The source suite has hundreds of tests covering model scanning, downloader behavior, setup/repair flows, runtime contracts, reports, schemas, validators, and release tooling. The current local audit pass has run the full suite at more than 600 tests.

Release readiness is stricter than source tests. A release can have passing unit/package checks while still lacking manual runtime evidence for a provider, OS, or clean-machine row. Easy ASR Bench uses release-smoke JSON and generated support matrices so public docs do not imply unproven support.

A public release should include:

- `setup.bat`
- installer metadata and checksums
- the Windows ZIP
- release-smoke evidence
- release verification notes

See [docs/release_verification.md](docs/release_verification.md) for the release process.

## Installed Files And Launchers

Normal users only need `setup.bat`. It verifies the release assets and installs the app.

Common launchers:

- `Run.bat`: scan models, choose models, and process inputs.
- `Drop_Audio_Or_Folders_Here.bat`: drag files/folders directly onto the app.
- `Open_Latest_Report.bat`: open the newest `final_results.html`.
- `Open_Models_Folder.bat`: open the model folder.
- `Open_Input_Folder.bat`: open the input folder.
- `Open_Output_Folder.bat`: open the report folder.
- `Edit_Config.bat`: edit configuration.

Installed releases also create Start Menu shortcuts for the same actions. Uninstall preserves user data by default.

Update and repair paths preserve user data folders separately from the staged app until setup and installed validation succeed.

## Safety

Easy ASR Bench is conservative about external model artifacts:

- It does not execute arbitrary Python files from model folders.
- Hugging Face ASR uses Safetensors rather than pickle-backed checkpoint loading.
- OpenAI Whisper `.pt` checkpoints are blocked by default unless checksum allowlisted or explicitly trusted.
- Generic ONNX runs only through built-in manifest recipes.
- Unsupported and incomplete packages are explained rather than guessed.
- CPU/GPU fallback is reported explicitly.
- Chunk failures are structured errors, not fake transcript text.
- Config writes and report publishing use atomic write/publish patterns.
- Release support claims are tied to evidence.

## Troubleshooting

- **No runnable ASR models:** paste a Hugging Face ASR link or add a complete supported model folder to `Models`.
- **Standalone `.safetensors` file:** use the complete Hugging Face model folder, not only the weights file.
- **Generic `.onnx` file:** add `modelbench.json` with CTC decoding metadata and a vocab file. Non-CTC ONNX graphs need a dedicated adapter.
- **Dependency missing:** accept the install prompt for the selected model group, or run `setup.bat --doctor`.
- **Saved selection is stale:** run interactively once and choose the intended models again.
- **GPU unavailable:** run `setup.bat --doctor`; reports also show actual provider and fallback metadata.
- **Cannot find a report:** run a transcription/benchmark first, then open `Open_Latest_Report.bat`.
- **Media conversion failed:** check that the file opens normally and that there is enough disk space in `Temp`.
- **Model download says disk space or path length is risky:** use a short local install/model path and free space on the target drive.
- **Doctor warns about install/profile paths:** short ASCII local paths outside synced folders are the lowest-risk Windows setup.
- **GGUF dependency missing:** install the `llama_cpp` dependency group when prompted or use the manual external LLM workflow.
- **Unexpected app error:** check `Logs/crash_*.log`, run `Run.bat --doctor --json`, and open a GitHub issue with the GitHub bug report template and requested diagnostics.
- **SmartScreen, Defender, or antivirus warning:** see [docs/what_setup_installs.md](docs/what_setup_installs.md). The release uses unsigned batch/PowerShell launchers, verifies setup assets by checksum, and keeps media/model files local unless you choose a download.
