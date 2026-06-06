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

## CUDA Fails

Use CPU mode or install a CUDA-compatible ONNX Runtime GPU stack. If provider is `auto`, Easy ASR Bench falls back to CPU.
