# Hugging Face Downloader Validation

The smart downloader is validated with fixture file lists in `tests/fixtures/hf_filelists/`.

Current fixture classes:

- split GGUF reference LLM packages
- ASR GGUF packages with matching `mmproj` projectors
- sharded Safetensors indexes
- multi-variant ONNX folders
- unknown folders that must stay inspection-only

These fixtures prove package selection behavior, not live Hugging Face availability. A public release may add live-download smoke evidence in `release-smoke-vX.Y.Z.json`, but unrun live repos must not be described as passed.

The runtime-matrix row `hf_downloader_package_variant_taxonomy` exercises these fixture classes through the product downloader helpers. It verifies sharded Safetensors index expansion, ONNX variant isolation, and split GGUF grouping as executable release evidence.

Live-download evidence, when actually run, should record:

- `repo_id`
- `revision`
- pasted URL shape
- selected package label
- expected local files
- downloaded file count
- rescan result
- date tested
