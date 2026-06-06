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
