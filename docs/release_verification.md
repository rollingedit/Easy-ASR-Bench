# Release Verification

Release verification is artifact-first. A release is not considered ready because source tests pass; the downloaded GitHub assets must also validate.

## Local Pre-Publish Gate

Run from a clean working tree after the version bump and final metadata regeneration:

```powershell
python scripts\validate_release_files.py
python scripts\validate_physical_files.py --repo .
python scripts\check_release_version_coherence.py --tag vX.Y.Z
python -m compileall app tests scripts
python -m pytest --basetemp .pytest_tmp -p no:cacheprovider
python scripts\build_release_zip.py --version vX.Y.Z --strict-checksums
python scripts\validate_physical_files.py --zip dist\Easy-ASR-Bench-vX.Y.Z-win.zip
cmd /c setup.bat --dry-run --local
cmd /c setup.bat --dry-run --verify-release
python -m app.doctor --config config.json --strict
```

## GitHub Release Asset Gate

After GitHub Actions publishes the release, verify the actual uploaded assets:

```powershell
python scripts\verify_github_release.py --repo rollingedit/Easy-ASR-Bench --tag vX.Y.Z --expected-commit <commit>
```

For pushed commits, CI also fetches selected files from `raw.githubusercontent.com` at the exact commit SHA:

```powershell
python scripts\validate_raw_github_files.py --repo rollingedit/Easy-ASR-Bench --commit <commit>
```

The release must upload:

- `setup.bat`
- `install.ps1`
- `manifest.json`
- `checksums.json`
- `Easy-ASR-Bench-vX.Y.Z-win.zip`

`setup.bat` must verify `install.ps1` before executing it. The app ZIP hash must match `checksums.json`.

## Manual Windows QA

These rows require real machines, VMs, hardware, or model files and cannot be proven by static CI alone:

- clean Windows 10/11 install with no Python
- Windows with existing Python 3.11/3.12
- install path with spaces
- repair, update, uninstall, and destructive uninstall confirmation
- empty `Models`
- complete HF ASR Safetensors, HF Whisper, faster-whisper, whisper.cpp, generic ONNX manifest, GGUF reference LLM
- unsupported standalone Safetensors, generic ONNX without manifest, HF text LLM Safetensors, unsafe `.pt`
- WAV, MP3, MP4 with audio, MP4 without audio, corrupt media, long audio
- CPU-only, NVIDIA CUDA, AMD/Intel DirectML, Intel OpenVINO, Vulkan runtime without SDK, Vulkan runtime with SDK

Record manual results separately from automated gates. Do not describe an unrun hardware row as verified.
