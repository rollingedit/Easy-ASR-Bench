param(
  [string]$InstallDir = "$env:LOCALAPPDATA\Easy-ASR-Bench",
  [string]$Version = "v0.2.1",
  [switch]$DryRun,
  [switch]$Repair,
  [switch]$Uninstall,
  [switch]$Doctor
)

$ErrorActionPreference = "Stop"
$Repo = "https://github.com/rollingedit/Easy-ASR-Bench"
$ReleaseBase = "$Repo/releases/download/$Version"
$LogDir = Join-Path $InstallDir "Logs"
$Log = Join-Path $LogDir "setup.log"

function Write-SetupLog($Message) {
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $line = "$(Get-Date -Format s) $Message"
  Write-Host $line
  Add-Content -LiteralPath $Log -Value $line
}

function Invoke-Step($Message, [scriptblock]$Block) {
  Write-SetupLog $Message
  if (-not $DryRun) { & $Block }
}

if ($Doctor) {
  if (Test-Path (Join-Path $InstallDir "app\doctor.py")) {
    & (Join-Path $InstallDir ".venv\Scripts\python.exe") -m app.doctor --config (Join-Path $InstallDir "config.json")
    exit $LASTEXITCODE
  }
  Write-SetupLog "Doctor requested but app is not installed."
  exit 1
}

if ($Uninstall) {
  Invoke-Step "Removing $InstallDir" { Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue }
  exit 0
}

Write-SetupLog "Installing Easy ASR Bench $Version to $InstallDir"
Write-SetupLog "Release source: $ReleaseBase"

$TempRoot = Join-Path $env:TEMP "Easy-ASR-Bench-install"
$Stage = Join-Path $TempRoot "stage"
$Zip = Join-Path $TempRoot "Easy-ASR-Bench-$Version-win.zip"
$Manifest = Join-Path $TempRoot "manifest.json"
$Checksums = Join-Path $TempRoot "checksums.json"

Invoke-Step "Preparing staging folder" {
  Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $Stage | Out-Null
}

Invoke-Step "Downloading release manifest and checksums" {
  Invoke-WebRequest -Uri "$ReleaseBase/manifest.json" -OutFile $Manifest
  Invoke-WebRequest -Uri "$ReleaseBase/checksums.json" -OutFile $Checksums
}

if (-not $DryRun) {
  $manifestJson = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
  $zipName = $manifestJson.app_zip
  $expected = (Get-Content -Raw -LiteralPath $Checksums | ConvertFrom-Json).files.$zipName
  Invoke-Step "Downloading app ZIP $zipName" {
    Invoke-WebRequest -Uri "$ReleaseBase/$zipName" -OutFile $Zip
  }
  $actual = "sha256:" + (Get-FileHash -Algorithm SHA256 -LiteralPath $Zip).Hash.ToLowerInvariant()
  if ($actual -ne $expected) {
    throw "Checksum mismatch for $zipName. Expected $expected, got $actual"
  }
}

Invoke-Step "Extracting and validating staging app" {
  Expand-Archive -LiteralPath $Zip -DestinationPath $Stage -Force
  $src = Get-ChildItem -LiteralPath $Stage -Directory | Select-Object -First 1
  if ($src) {
    Copy-Item -Path (Join-Path $src.FullName "*") -Destination $Stage -Recurse -Force
    Remove-Item -LiteralPath $src.FullName -Recurse -Force
  }
  python (Join-Path $Stage "scripts\validate_release_files.py")
}

Invoke-Step "Installing app atomically" {
  $Backup = "$InstallDir.backup"
  $New = "$InstallDir.new"
  Remove-Item -LiteralPath $New -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $New | Out-Null
  Copy-Item -Path (Join-Path $Stage "*") -Destination $New -Recurse -Force
  if (Test-Path $InstallDir) {
    Remove-Item -LiteralPath $Backup -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -LiteralPath $InstallDir -Destination $Backup -Force
  }
  Move-Item -LiteralPath $New -Destination $InstallDir -Force
}

Push-Location $InstallDir
try {
  Invoke-Step "Running local setup" {
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c setup.bat --local" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "local setup failed with exit code $($p.ExitCode)" }
  }
}
finally {
  Pop-Location
}

Write-SetupLog "Setup complete"
