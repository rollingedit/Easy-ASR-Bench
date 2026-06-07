# Troubleshooting

## No Runnable ASR Models

Open `Models` and check that model folders are complete. Standalone weights are not enough for most ASR systems.

## Safetensors Not Runnable

Use a complete Hugging Face ASR folder with:

- `config.json`
- `.safetensors` weights
- tokenizer and processor files

## Generic ONNX CTC Manifest Not Runnable

Add `modelbench.json` with CTC decoding metadata. Arbitrary ONNX files do not include enough information for preprocessing and decoding, and non-CTC graphs require dedicated adapters.

## Model Is Detected But Not Runnable

That means Easy ASR Bench recognized the package family but the folder is incomplete, platform-specific, safety-blocked, or outside the app's packaged runtime scope. Check the missing-file list shown in the menu/report. Keep split ONNX sidecars, safetensors shards, tokenizer files, `mmproj` files, and runtime-specific config files in the same model folder.

## GGUF Appears Under Reference LLMs

That is expected. GGUF text LLMs are used for transcript correction/reference generation, not direct speech-to-text.

## Whisper Model Not Detected

Check the format:

- HF Whisper folders need `config.json`, Safetensors weights, tokenizer files, and preprocessor/processor files.
- faster-whisper folders need `model.bin`, `config.json`, and tokenizer/vocabulary files.
- whisper.cpp files usually look like `ggml-base.bin`.
- OpenAI Whisper `.pt` files are blocked by default unless their SHA256 is allowlisted by the app or unsafe trusted-file loading is explicitly enabled. A trusted filename is not enough.

## CUDA Fails

Use CPU mode or install a CUDA-compatible ONNX Runtime GPU stack. If provider is `auto`, Easy ASR Bench falls back to CPU.
