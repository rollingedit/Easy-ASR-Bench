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
python scripts\write_release_smoke.py --tag vX.Y.Z --commit <commit> --output release-smoke-vX.Y.Z.json
python scripts\validate_release_smoke.py --smoke release-smoke-vX.Y.Z.json --required tests\fixtures\release_required_rows_v2.json
cmd /c setup.bat --dry-run --local
Remove-Item -LiteralPath dist\release-assets -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path dist\release-assets
Copy-Item dist\Easy-ASR-Bench-vX.Y.Z-win.zip dist\release-assets\
Copy-Item setup.bat dist\release-assets\
Copy-Item installer\install.ps1 dist\release-assets\install.ps1
Copy-Item installer\manifest.json dist\release-assets\
Copy-Item installer\checksums.json dist\release-assets\
cmd /c setup.bat --dry-run --verify-release --asset-dir dist\release-assets
python -m app.doctor --config config.json --strict
```

The staged `--asset-dir` check must run before a draft release is published. It validates the same `setup.bat`, `install.ps1`, `manifest.json`, `checksums.json`, and ZIP bytes that will become release assets, without depending on public release URLs.
`validate_release_smoke.py` verifies that every required release QA row exists. Public release publication and release-gate verification use `--require-all-pass --require-log-hashes --require-environment-summary`; that means every required row must be `pass` and must include app version, release commit, log/result hash evidence, and environment summary. If real Windows/model/provider/media evidence has not been collected, the correct status is `not_run`, and the release must remain unpublished or draft.

## GitHub Release Asset Gate

After GitHub Actions publishes the release, verify the actual uploaded assets:

```powershell
python scripts\verify_github_release.py --repo rollingedit/Easy-ASR-Bench --tag vX.Y.Z --expected-commit <commit> --write-transcript release-verification-vX.Y.Z.txt
python scripts\write_release_verification_manifest.py --tag vX.Y.Z --transcript release-verification-vX.Y.Z.txt --assets-dir release-asset --output release-verification-manifest-vX.Y.Z.json
python scripts\verify_release_transcript.py --assets-dir release-asset --checksums release-asset\checksums.json --transcript release-asset\release-verification-vX.Y.Z.txt --detached-manifest release-verification-manifest-vX.Y.Z.json --strict
```

For pushed commits, CI also fetches selected files from `raw.githubusercontent.com` at the exact commit SHA:

```powershell
python scripts\validate_raw_github_files.py --repo rollingedit/Easy-ASR-Bench --ref vX.Y.Z
```

Raw validation prints byte counts, CRLF/LF/bare-CR counts, physical line counts, and first/last byte hex for each critical public file. Those diagnostics are required because raw GitHub bytes, release assets, and ZIP contents are the trust boundary.
GitHub raw URLs serve canonical Git blob bytes, which are normally LF-normalized even for files that check out as CRLF on Windows because of `.gitattributes`. For raw GitHub validation, `physical_line_count_universal` is the line-count gate. The CRLF diagnostic count is informational and must not be used to claim a file is collapsed when the universal physical line count is correct.
When a release ZIP is available, run raw validation with `--zip` so critical raw GitHub files are byte-compared against the matching ZIP copies:

```powershell
python scripts\validate_raw_github_files.py --repo rollingedit/Easy-ASR-Bench --ref vX.Y.Z --zip dist\Easy-ASR-Bench-vX.Y.Z-win.zip
```

The release must upload:

- `setup.bat`
- `install.ps1`
- `manifest.json`
- `checksums.json`
- `Easy-ASR-Bench-vX.Y.Z-win.zip`
- `release-smoke-vX.Y.Z.json`
- `release-verification-vX.Y.Z.txt`
- `release-verification-manifest-vX.Y.Z.json`

`setup.bat` must verify `install.ps1` before executing it. The app ZIP hash must match `checksums.json`.
The smoke JSON records automated release-candidate checks and leaves Windows VM, GPU/provider, model, and media rows as `not_run` until they are actually executed. Do not edit those rows to `pass` unless the matching downloaded-release test was run.
The verification transcript records actual downloaded release asset hashes and validation steps; it does not convert unrun manual matrix rows into passes.
The transcript must not contain a hash of itself. `release-verification-manifest-vX.Y.Z.json` is the detached metadata that records the transcript hash after the transcript exists.

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
