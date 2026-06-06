# What Setup Installs

`setup.bat` installs Easy ASR Bench into:

```text
%LOCALAPPDATA%\Easy-ASR-Bench
```

It creates a local Python virtual environment and these folders:

- `Models`
- `Input`
- `Output`
- `Logs`
- `Temp`
- `Cache`

Default setup installs only `requirements/core.txt`.

Optional runtimes are installed only when selected:

- ONNX: `requirements/onnx.txt`
- Hugging Face / HF Whisper: `requirements/transformers_cpu.txt`
- faster-whisper: `requirements/faster_whisper.txt`
- whisper.cpp: `requirements/whisper_cpp.txt`
- OpenAI Whisper `.pt`: `requirements/openai_whisper.txt`
- GGUF reference LLM: `requirements/llama_cpp.txt`

Setup modes:

```text
setup.bat
setup.bat --repair
setup.bat --update
setup.bat --uninstall
setup.bat --doctor
setup.bat --local
setup.bat --dry-run
```

Setup writes logs to `Logs/setup.log`.
