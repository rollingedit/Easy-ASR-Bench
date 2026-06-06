# Troubleshooting

## No Runnable ASR Models

Open `Models` and check that model folders are complete. Standalone weights are not enough for most ASR systems.

## Safetensors Not Runnable

Use a complete Hugging Face ASR folder with:

- `config.json`
- `.safetensors` weights
- tokenizer and processor files

## Generic ONNX Not Runnable

Add `modelbench.json`. Arbitrary ONNX files do not include enough information for preprocessing and decoding.

## GGUF Appears Under Reference LLMs

That is expected. GGUF text LLMs are used for transcript correction/reference generation, not direct speech-to-text.

## Whisper Model Not Detected

Check the format:

- HF Whisper folders need `config.json`, Safetensors weights, tokenizer files, and preprocessor/processor files.
- faster-whisper folders need `model.bin`, `config.json`, and tokenizer/vocabulary files.
- whisper.cpp files usually look like `ggml-base.bin`.
- OpenAI Whisper `.pt` files must use official model filenames unless unsafe loading is enabled.

## CUDA Fails

Use CPU mode or install a CUDA-compatible ONNX Runtime GPU stack. If provider is `auto`, Easy ASR Bench falls back to CPU.
