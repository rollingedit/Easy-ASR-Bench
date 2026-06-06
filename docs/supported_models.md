# Supported Models

## ASR

- Granite Speech ONNX AR: `int8`, `fp16w`, `fp32`
- Granite Speech ONNX NAR: `int8`, `fp16w`, `fp32`
- Hugging Face Transformers ASR folders with `.safetensors`
- Hugging Face Whisper Safetensors folders
- faster-whisper / CTranslate2 folders
- whisper.cpp GGML `.bin` files
- official-name OpenAI Whisper `.pt` files, with unsafe pickle restrictions
- Generic ONNX CTC ASR with `modelbench.json`

## Reference/Correction LLM

- GGUF text LLMs through llama.cpp dependencies

GGUF text LLMs are not shown as direct ASR models. They are used for LLM-corrected reference workflows.

Reference LLM discovery:

- `.gguf` files under `Models` are scanned automatically.
- A user can paste a `.gguf` file path or a folder path from another app.
- Saved custom paths are stored in `config.json` under `llm_reference.custom_model_paths`.
- External LLMs such as ChatGPT or Claude are supported through the manual prompt workflow in `results.txt`.

Local and external LLM references are AI-assisted references, not human ground truth.
