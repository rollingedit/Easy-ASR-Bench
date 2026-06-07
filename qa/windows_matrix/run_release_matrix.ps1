param(
  [string]$Tag = "v0.3.5",
  [string]$Output = "qa\windows_matrix\evidence",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$rows = @(
  "win11_clean_no_python_setup",
  "win10_existing_python_setup",
  "install_path_with_spaces",
  "setup_verify_release_bad_checksum",
  "setup_double_click_equivalent",
  "setup_dry_run_verify_release",
  "setup_doctor_strict",
  "update_preserves_user_data",
  "repair_broken_venv",
  "uninstall_preserve_user_data",
  "destructive_uninstall_requires_phrase",
  "bad_checksum_fails_before_execution",
  "tampered_installer_fails_before_execution",
  "interrupted_download_rollback",
  "broken_venv_repair",
  "empty_models_folder",
  "empty_models",
  "nested_models_folders",
  "nested_models_scan",
  "wav_mp3_mp4_no_audio_corrupt_media",
  "wav_mp3_mp4_media",
  "corrupt_media_readable_error",
  "no_audio_video_readable_error",
  "compare_html_offline_large_transcript",
  "compare_html_offline",
  "batch_continues_after_one_model_or_chunk_fails",
  "one_model_failure_continues",
  "one_chunk_failure_continues",
  "llm_reference_json_import",
  "dependency_install_declined",
  "cpu_model_smoke",
  "nvidia_cuda_torch_onnx_faster_whisper_llama",
  "amd_directml_onnx_smoke",
  "intel_directml_onnx_smoke",
  "intel_openvino_onnx_smoke",
  "vulkan_runtime_no_sdk",
  "vulkan_runtime_with_sdk",
  "hf_safetensors_asr",
  "hf_whisper_safetensors",
  "hf_whisper_safetensors_cpu",
  "sharded_safetensors_index",
  "faster_whisper_ctranslate2",
  "faster_whisper_cpu",
  "faster_whisper_cuda_unavailable_cpu_fallback",
  "whisper_cpp_ggml",
  "openai_whisper_pt_checksum_verified",
  "openai_whisper_pt_unknown_blocked",
  "openai_pt_unverified_blocked",
  "generic_onnx_ctc_manifest_v1",
  "generic_onnx_manifest_cpu",
  "generic_onnx_without_manifest_rejected",
  "multi_file_onnx_ar_nar",
  "audio_asr_gguf_mmproj",
  "gguf_asr_mmproj_pair",
  "incomplete_audio_asr_gguf_mmproj_rejected",
  "gguf_reference_llm",
  "gguf_text_llm_reference_only",
  "standalone_safetensors_incomplete",
  "hf_text_llm_safetensors_unsupported",
  "known_unsupported_asr_families_explained"
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
