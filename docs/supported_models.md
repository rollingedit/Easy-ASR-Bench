# Supported Models

## ASR

- Known multi-file ONNX AR layouts: `int8`, `fp16w`, `fp32`, `f32`, `float32`
- Known multi-file ONNX NAR layouts: `int8`, `fp16w`, `fp32`, `f32`, `float32`
- Hugging Face Transformers ASR folders with `.safetensors`, including native FP32/float32 weights
- Hugging Face Whisper Safetensors folders, including native FP32/float32 weights
- faster-whisper / CTranslate2 folders
- whisper.cpp GGML `.bin` files
- Generic ONNX CTC ASR with `modelbench.json`

## Hugging Face Downloads

Choose `D` in the interactive model menu to paste a Hugging Face model URL or `owner/model` repo id. Repo-root links, `/tree/main` links, nested folder links, file links, and links with extra trailing slashes are accepted. The downloader lists available GGUF, Safetensors, and ONNX package choices, asks when multiple weights or quantizations exist, then downloads only the selected package plus required metadata/sidecar files.

Large selections and unknown folder layouts require confirmation before download. Unknown folder packages are saved for inspection and rescanned, but they are not presented as runnable unless the model scanner recognizes the downloaded files.

If the rescan reports exact missing files that are present in the same Hugging Face repo, the app can download those missing files after user confirmation. Existing local files are skipped on repeat downloads.

Ambiguous requirements are shown to the user instead of guessed. The app writes `hf_missing_file_request.json` and `hf_missing_file_prompt.txt` beside the incomplete package so a local or external LLM can return structured recommendations using only exact filenames from the repo file list.

The tested Hugging Face links are representative stress cases, not a complete map of every possible future repo layout. Unknown or mixed package structures should still be audited and expanded with new regression tests when found.

## Recognized, But Not Runnable Yet

These package types are recognized so the app can explain what was found, list missing sibling files, and recommend a runnable export or future adapter path:

- NeMo `.nemo`
- FunASR folders
- sherpa-onnx Whisper folders
- Split Whisper/Transformers.js ONNX, Granite-style split ONNX, Qwen split ONNX, and ORT edge graph packages
- Core ML / WhisperKit `.mlmodelc`
- Audio/ASR GGUF packages with matching `mmproj`
- Sharded Hugging Face Safetensors folders with `model.safetensors.index.json`

## Blocked by Default

- OpenAI Whisper `.pt` files are detected but not runnable by default unless a checksum is allowlisted by the app or unsafe trusted-file loading is explicitly enabled.

## Reference/Correction LLM

- GGUF text LLMs through llama.cpp dependencies

GGUF text LLMs are not shown as direct ASR models. They are used for LLM-corrected reference workflows.

Local reference/correction LLM loading is GGUF-only. Hugging Face `.safetensors` folders are supported for ASR adapters, not as local text LLMs. GPTQ/AWQ safetensors, EXL2, ONNX LLMs, TensorRT-LLM engines, raw PyTorch checkpoints, and similar text-generation formats are not loaded as local reference LLMs by this app. Use a GGUF export or the external/manual LLM workflow.

A `.gguf` file is treated as the complete local LLM artifact for llama.cpp because tokenizer/model metadata is normally embedded. If a Hugging Face text LLM safetensors folder is found, the scanner reports that a GGUF export is required for local reference/correction.

Reference LLM discovery:

- `.gguf` files under `Models` are scanned automatically.
- A user can paste a `.gguf` file path or a folder path from another app.
- Saved custom paths are stored in `config.json` under `llm_reference.custom_model_paths`.
- External LLMs such as ChatGPT or Claude are supported through the manual prompt workflow in `results.txt`.

Local and external LLM references are AI-assisted references, not human ground truth.

`.pt` checkpoints are blocked by default unless they match an allowlisted SHA256 or you explicitly enable unsafe pickle loading in `config.json`. A trusted-looking filename is not treated as proof that a checkpoint is safe.
