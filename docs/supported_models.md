# Supported Models

## ASR

- Granite Speech ONNX AR: `int8`, `fp16w`, `fp32`
- Granite Speech ONNX NAR: `int8`, `fp16w`, `fp32`
- Hugging Face Transformers ASR folders with `.safetensors`
- Generic ONNX CTC ASR with `modelbench.json`

## Reference/Correction LLM

- GGUF text LLMs through llama.cpp dependencies

GGUF text LLMs are not shown as direct ASR models. They are used for LLM-corrected reference workflows.
