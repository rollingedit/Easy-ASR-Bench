param(
  [string]$Tag = "v0.3.4",
  [string]$Output = "qa\windows_matrix\evidence",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$rows = @(
  "win11_clean_no_python_setup",
  "win10_existing_python_setup",
  "install_path_with_spaces",
  "setup_verify_release_bad_checksum",
  "repair_broken_venv",
  "uninstall_preserve_user_data",
  "empty_models",
  "nested_models_scan",
  "hf_whisper_safetensors_cpu",
  "faster_whisper_cpu",
  "generic_onnx_manifest_cpu",
  "gguf_asr_mmproj_pair",
  "wav_mp3_mp4_media",
  "corrupt_media_readable_error",
  "no_audio_video_readable_error",
  "one_model_failure_continues",
  "one_chunk_failure_continues",
  "compare_html_offline",
  "dependency_install_declined"
)

New-Item -ItemType Directory -Force -Path $Output | Out-Null
foreach ($row in $rows) {
  if ($DryRun) {
    Write-Host "Would collect row: $row"
    continue
  }
  & powershell -ExecutionPolicy Bypass -File qa\windows_matrix\collect_release_evidence.ps1 `
    -Output $Output `
    -RowId $row `
    -Status "not_run" `
    -Commands @("manual row placeholder for $Tag")
}

Write-Host "Evidence directory: $Output"
