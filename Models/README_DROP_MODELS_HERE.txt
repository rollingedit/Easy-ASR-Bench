Drop local ASR model folders or reference LLM files here.

Runnable ASR:
- Granite Speech AR ONNX folders containing int8/, fp16w/, or fp32/
- Granite Speech NAR ONNX folders containing int8/, fp16w/, or fp32/
- Hugging Face Transformers ASR folders with .safetensors weights
- Generic ONNX CTC ASR folders with modelbench.json

Reference/correction:
- GGUF text LLM files can be used for LLM-corrected reference workflows.
- GGUF files can also live outside this folder. Paste the GGUF file path or folder path in the LLM reference menu and Easy ASR Bench will save it for future runs.

Standalone weight files without config/tokenizer/processor metadata are scanned and explained, but they are not enough to run ASR.
