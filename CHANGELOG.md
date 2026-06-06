# Changelog

## v0.2.3

- Hardened installer updates to preserve `Models`, `Input`, `Output`, `Logs`, `Cache`, `Temp`, and `config.json` across installs and updates.
- Added installer rollback handling if the atomic directory swap fails.
- Removed bare `python` usage from the PowerShell installer staging validator.
- Added a repeatable release ZIP builder and CI ZIP validation.
- Blocked local OpenAI Whisper `.pt` files by default unless SHA256-allowlisted or explicitly trusted with unsafe pickle loading enabled.
- Passed HF Whisper long-form chunking options through the Transformers pipeline call.
- Required vocab metadata for generic ONNX CTC manifests so numeric token IDs are not emitted as transcripts.
- Made recommended model selection family-balanced instead of Granite-only.
- Improved Unicode-aware scoring normalization and multi-path paste parsing.
- Made dependency install failures skip only affected models instead of crashing the whole run.
- Made `benchmark.csv` writes atomic and watch queue skip completed files by fast size/mtime key before hashing.

## v0.2.2

- Added a complete LLM reference menu for detected GGUF models, saved external GGUF paths, manual ChatGPT/Claude workflows, and skipping reference scoring.
- Added `llm_reference.custom_model_paths` so GGUF files or folders used by another app can be saved and rescanned on future runs.
- Updated version metadata and docs for the v0.2.2 release pass.
- Kept GGUF wording explicit: GGUF text LLMs are reference/correction helpers, not direct ASR models.

## v0.2.0

- Added canonical `results.json`, `results.txt`, `benchmark.csv`, and offline `compare.html` output.
- Added LLM-corrected reference prompt and browser-side scoring workflow.
- Added Hugging Face Safetensors ASR adapter.
- Added generic ONNX CTC manifest adapter.
- Added GGUF local LLM reference/correction candidate handling.
- Added dependency group repair support and doctor command.
- Improved model scanner grouping and unsupported-model explanations.
- Updated all user-facing docs for product-ready wording.

## v0.1

- Initial Windows app shell, model scanner, Granite ONNX adapter foundation, and release BAT.
