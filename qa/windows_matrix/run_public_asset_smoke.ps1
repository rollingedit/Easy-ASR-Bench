param(
  [string]$Repo = "rollingedit/Easy-ASR-Bench",
  [string]$Tag = "v0.3.7",
  [string]$WorkDir = "qa\windows_matrix\public_asset_smoke",
  [string]$AssetDir = "",
  [string]$Output = "qa\windows_matrix\evidence",
  [string]$ReleaseCommit = "",
  [switch]$Install
)

$ErrorActionPreference = "Stop"

function Invoke-CapturedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$Command,
    [Parameter(Mandatory = $true)][string]$TranscriptPath,
    [string]$WorkingDirectory = ""
  )

  $parent = Split-Path -Parent $TranscriptPath
  if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
  $prefix = "COMMAND: $Command`r`nSTART_UTC: $((Get-Date).ToUniversalTime().ToString("o"))`r`n"
  Set-Content -LiteralPath $TranscriptPath -Value $prefix -Encoding UTF8
  $process = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/d", "/c", $Command `
    -WorkingDirectory ($(if ($WorkingDirectory) { $WorkingDirectory } else { (Get-Location).Path })) `
    -NoNewWindow `
    -Wait `
    -PassThru `
    -RedirectStandardOutput "$TranscriptPath.stdout" `
    -RedirectStandardError "$TranscriptPath.stderr"
  Add-Content -LiteralPath $TranscriptPath -Value "EXIT_CODE: $($process.ExitCode)"
  Add-Content -LiteralPath $TranscriptPath -Value "STDOUT:"
  if (Test-Path "$TranscriptPath.stdout") {
    Add-Content -LiteralPath $TranscriptPath -Value (Get-Content -Raw "$TranscriptPath.stdout")
    Remove-Item -LiteralPath "$TranscriptPath.stdout" -Force
  }
  Add-Content -LiteralPath $TranscriptPath -Value "STDERR:"
  if (Test-Path "$TranscriptPath.stderr") {
    Add-Content -LiteralPath $TranscriptPath -Value (Get-Content -Raw "$TranscriptPath.stderr")
    Remove-Item -LiteralPath "$TranscriptPath.stderr" -Force
  }
  if ($process.ExitCode -ne 0) {
    throw "Command failed with exit code $($process.ExitCode): $Command"
  }
}

function Write-AssetHashes {
  param(
    [Parameter(Mandatory = $true)][string]$SourceDir,
    [Parameter(Mandatory = $true)][string]$Destination
  )
  $hashes = @{}
  Get-ChildItem -LiteralPath $SourceDir -File | Sort-Object Name | ForEach-Object {
    $hashes[$_.Name] = "sha256:" + (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
  }
  $payload = @{
    schema = "easy_asr_bench.public_asset_hashes.v1"
    repo = $Repo
    tag = $Tag
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    files = $hashes
  }
  $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Destination -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$Output = (Resolve-Path -LiteralPath $Output).Path

if ([string]::IsNullOrWhiteSpace($ReleaseCommit)) {
  try {
    $remote = "https://github.com/$Repo.git"
    $ReleaseCommit = ((git ls-remote --tags $remote "refs/tags/$Tag") -split "\s+")[0]
  } catch {
    $ReleaseCommit = "unknown-public-release-$Tag"
  }
}

if ([string]::IsNullOrWhiteSpace($AssetDir)) {
  $AssetDir = Join-Path $WorkDir "assets-$Tag"
  Remove-Item -LiteralPath $AssetDir -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $AssetDir | Out-Null
  gh release download $Tag --repo $Repo --dir $AssetDir --clobber
} else {
  $AssetDir = (Resolve-Path -LiteralPath $AssetDir).Path
}

$required = @(
  "setup.bat",
  "install.ps1",
  "manifest.json",
  "checksums.json",
  "Easy-ASR-Bench-$Tag-win.zip"
)
foreach ($name in $required) {
  if (-not (Test-Path -LiteralPath (Join-Path $AssetDir $name))) {
    throw "Missing required release asset: $name"
  }
}

$assetHashPath = Join-Path $WorkDir "public-assets-$Tag.json"
Write-AssetHashes -SourceDir $AssetDir -Destination $assetHashPath

$setup = Join-Path $AssetDir "setup.bat"
$verifyTranscript = Join-Path $WorkDir "setup-verify-release-$Tag.txt"
Invoke-CapturedCommand `
  -Command "call `"$setup`" --dry-run --verify-release --asset-dir `"$AssetDir`"" `
  -TranscriptPath $verifyTranscript

$setupRowDir = Join-Path $Output "setup_dry_run_verify_release"
New-Item -ItemType Directory -Force -Path $setupRowDir | Out-Null
Copy-Item -LiteralPath $verifyTranscript -Destination (Join-Path $setupRowDir "setup-verify-release.log") -Force
Copy-Item -LiteralPath $assetHashPath -Destination (Join-Path $setupRowDir "public-assets.json") -Force

& powershell -ExecutionPolicy Bypass -File qa\windows_matrix\collect_release_evidence.ps1 `
  -Output $Output `
  -RowId "setup_dry_run_verify_release" `
  -Status "pass" `
  -AppVersion $Tag `
  -ReleaseCommit $ReleaseCommit `
  -Commands @("setup.bat --dry-run --verify-release --asset-dir <downloaded assets>")
if ($LASTEXITCODE -ne 0) {
  throw "collect_release_evidence failed for setup_dry_run_verify_release with exit code $LASTEXITCODE"
}

if ($Install) {
  $installTranscript = Join-Path $WorkDir "setup-install-$Tag.txt"
  Invoke-CapturedCommand `
    -Command "echo Q| call `"$setup`"" `
    -TranscriptPath $installTranscript `
    -WorkingDirectory $AssetDir

  $runBat = Join-Path $env:LOCALAPPDATA "Easy-ASR-Bench\Run.bat"
  if (-not (Test-Path -LiteralPath $runBat)) {
    throw "Installed Run.bat was not found: $runBat"
  }

  $doctorJson = Join-Path $WorkDir "doctor.json"
  $firstRunJson = Join-Path $WorkDir "first-run-smoke.json"
  cmd.exe /d /c "`"$runBat`" --doctor --json > `"$doctorJson`""
  if ($LASTEXITCODE -ne 0) {
    throw "Run.bat --doctor --json failed with exit code $LASTEXITCODE"
  }
  cmd.exe /d /c "`"$runBat`" --first-run-smoke > `"$firstRunJson`""
  if ($LASTEXITCODE -ne 0) {
    throw "Run.bat --first-run-smoke failed with exit code $LASTEXITCODE"
  }

  Get-Content -Raw -LiteralPath $doctorJson | ConvertFrom-Json | Out-Null
  Get-Content -Raw -LiteralPath $firstRunJson | ConvertFrom-Json | Out-Null

  $firstRunRowDir = Join-Path $Output "empty_models_guided_first_run"
  New-Item -ItemType Directory -Force -Path $firstRunRowDir | Out-Null
  Copy-Item -LiteralPath $doctorJson -Destination (Join-Path $firstRunRowDir "doctor.json") -Force
  Copy-Item -LiteralPath $firstRunJson -Destination (Join-Path $firstRunRowDir "first-run-smoke.json") -Force
  Copy-Item -LiteralPath $installTranscript -Destination (Join-Path $firstRunRowDir "setup-install.log") -Force

  & powershell -ExecutionPolicy Bypass -File qa\windows_matrix\collect_release_evidence.ps1 `
    -Output $Output `
    -RowId "empty_models_guided_first_run" `
    -Status "pass" `
    -AppVersion $Tag `
    -ReleaseCommit $ReleaseCommit `
    -Commands @("Run.bat --doctor --json", "Run.bat --first-run-smoke")
  if ($LASTEXITCODE -ne 0) {
    throw "collect_release_evidence failed for empty_models_guided_first_run with exit code $LASTEXITCODE"
  }
}

Write-Host "Public asset smoke work directory: $WorkDir"
Write-Host "Evidence directory: $Output"
