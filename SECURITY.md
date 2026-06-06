# Security

Easy ASR Bench is designed to run local models without executing arbitrary model-folder code.

Defaults:

- `trust_remote_code` is disabled.
- Python files inside model folders are not executed.
- Pickle-style `.pt`, `.pth`, and `.bin` checkpoints are not loaded by default.
- Official-name OpenAI Whisper `.pt` files are detected separately; unknown `.pt` files are blocked unless unsafe loading is enabled.
- Safetensors are used for Hugging Face ASR model folders.
- Generic ONNX runs only through built-in `modelbench.json` recipes.
- GGUF files are treated as local text LLMs for reference/correction, not direct ASR.

Report security issues privately through the GitHub repository owner.
