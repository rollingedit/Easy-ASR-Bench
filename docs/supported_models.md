# Supported Models

## ASR

- Known multi-file ONNX AR layouts: `int8`, `fp16w`, `fp32`, `f32`, `float32`
- Known multi-file ONNX NAR layouts: `int8`, `fp16w`, `fp32`, `f32`, `float32`
- Hugging Face Transformers ASR folders with `.safetensors`, including native FP32/float32 weights
- Hugging Face Whisper Safetensors folders, including native FP32/float32 weights
- faster-whisper / CTranslate2 folders
- whisper.cpp GGML `.bin` files
- Generic ONNX CTC ASR with `modelbench.json`

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
