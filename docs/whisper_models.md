# Whisper Models

Easy ASR Bench treats Whisper as a first-class baseline family.

Supported local formats:

- Hugging Face Whisper folders with Safetensors weights
- faster-whisper / CTranslate2 folders with `model.bin`
- whisper.cpp GGML files such as `ggml-base.bin`
- OpenAI Whisper `.pt` files with official model filenames

Security note: unknown `.pt` files are blocked by default because PyTorch pickle checkpoints can execute unsafe data. Keep `allow_pickle_or_pt_files` disabled unless you fully trust the file.

Dependency extras:

- HF Whisper: `requirements/transformers_cpu.txt`
- faster-whisper: `requirements/faster_whisper.txt`
- whisper.cpp: `requirements/whisper_cpp.txt`
- OpenAI Whisper `.pt`: `requirements/openai_whisper.txt`
