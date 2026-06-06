param(
  [string]$InstallDir = "$env:LOCALAPPDATA\Easy-ASR-Bench",
  [string]$Version = "v0.2.3",
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
  $line = "$(Get-Date -Format s) $Message"
  Write-Host $line
  if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    Add-Content -LiteralPath $Log -Value $line
  }
}

function Invoke-Step($Message, [scriptblock]$Block) {
  Write-SetupLog $Message
  if (-not $DryRun) { & $Block }
}

function Resolve-Python {
  $commands = @(
    @{ File = "py"; Args = @("-3.11") },
    @{ File = "py"; Args = @("-3.12") },
    @{ File = "python"; Args = @() }
  )
  foreach ($command in $commands) {
    try {
      $probeArgs = @($command.Args) + @("-c", "import sys; print(sys.executable)")
      $result = & $command.File @probeArgs 2>$null
      if ($LASTEXITCODE -eq 0 -and $result) {
        return @{ File = $command.File; Args = $command.Args; Display = ($command.File + " " + ($command.Args -join " ")).Trim() }
      }
    }
    catch {
    }
  }
  return $null
}

function Invoke-PythonFile($Python, $ScriptPath) {
  & $Python.File @($Python.Args) $ScriptPath
  if ($LASTEXITCODE -ne 0) {
    throw "Python validation failed with exit code $LASTEXITCODE"
  }
}

function Copy-PreservedUserData($From, $To) {
  foreach ($name in @("Models", "Input", "Output", "Logs", "Cache", "Temp")) {
    $source = Join-Path $From $name
    $dest = Join-Path $To $name
    if (Test-Path $source) {
      New-Item -ItemType Directory -Force -Path $dest | Out-Null
      Copy-Item -LiteralPath (Join-Path $source "*") -Destination $dest -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
  $sourceConfig = Join-Path $From "config.json"
  $destConfig = Join-Path $To "config.json"
  if (Test-Path $sourceConfig) {
    Copy-Item -LiteralPath $sourceConfig -Destination $destConfig -Force
  }
}

if ($Doctor) {
  $python = Join-Path $InstallDir ".venv\Scripts\python.exe"
  if (Test-Path (Join-Path $InstallDir "app\doctor.py") -and Test-Path $python) {
    & $python -m app.doctor --config (Join-Path $InstallDir "config.json")
    exit $LASTEXITCODE
  }
  Write-SetupLog "Doctor requested but app is not installed."
  exit 1
}

if ($Uninstall) {
  Write-SetupLog "Uninstall requested for $InstallDir"
  if ($DryRun) {
    Write-SetupLog "Dry run: would remove app files and local user data under $InstallDir"
    exit 0
  }
  Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
  exit 0
}

Write-SetupLog "Installing Easy ASR Bench $Version to $InstallDir"
Write-SetupLog "Release source: $ReleaseBase"

$TempRoot = Join-Path $env:TEMP "Easy-ASR-Bench-install"
$Stage = Join-Path $TempRoot "stage"
$Zip = Join-Path $TempRoot "Easy-ASR-Bench-$Version-win.zip"
$Manifest = Join-Path $TempRoot "manifest.json"
$Checksums = Join-Path $TempRoot "checksums.json"
$Backup = "$InstallDir.backup"
$New = "$InstallDir.new"

$Python = Resolve-Python
if ($Python) {
  Write-SetupLog "Python command for validation: $($Python.Display)"
}
elseif ($DryRun) {
  Write-SetupLog "Python 3.11/3.12 was not found. Setup will install Python before local setup."
}
else {
  Write-SetupLog "Python 3.11/3.12 was not found yet. Staging validation will be skipped until local setup installs Python."
}

if ($DryRun) {
  Write-SetupLog "Dry run plan:"
  Write-SetupLog "  manifest: $ReleaseBase/manifest.json"
  Write-SetupLog "  checksums: $ReleaseBase/checksums.json"
  Write-SetupLog "  install dir: $InstallDir"
  Write-SetupLog "  preserved user data: Models, Input, Output, Logs, Cache, Temp, config.json"
  Write-SetupLog "  no files will be downloaded, moved, or deleted"
  exit 0
}

Invoke-Step "Preparing staging folder" {
  Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $Stage | Out-Null
}

Invoke-Step "Downloading release manifest and checksums" {
  Invoke-WebRequest -Uri "$ReleaseBase/manifest.json" -OutFile $Manifest
  Invoke-WebRequest -Uri "$ReleaseBase/checksums.json" -OutFile $Checksums
}

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

Invoke-Step "Extracting staging app" {
  Expand-Archive -LiteralPath $Zip -DestinationPath $Stage -Force
  $src = Get-ChildItem -LiteralPath $Stage -Directory | Select-Object -First 1
  if ($src) {
    Copy-Item -Path (Join-Path $src.FullName "*") -Destination $Stage -Recurse -Force
    Remove-Item -LiteralPath $src.FullName -Recurse -Force
  }
}

if ($Python) {
  Invoke-Step "Validating staging app" {
    Invoke-PythonFile $Python (Join-Path $Stage "scripts\validate_release_files.py")
  }
}
else {
  Write-SetupLog "Skipping Python staging validator because Python is not installed before bootstrap."
}

Invoke-Step "Installing app atomically with user-data preservation" {
  Remove-Item -LiteralPath $New -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $New | Out-Null
  Copy-Item -Path (Join-Path $Stage "*") -Destination $New -Recurse -Force
  if (Test-Path $InstallDir) {
    Copy-PreservedUserData $InstallDir $New
  }
  try {
    if (Test-Path $InstallDir) {
      Remove-Item -LiteralPath $Backup -Recurse -Force -ErrorAction SilentlyContinue
      Move-Item -LiteralPath $InstallDir -Destination $Backup -Force
    }
    Move-Item -LiteralPath $New -Destination $InstallDir -Force
  }
  catch {
    Write-SetupLog "Install swap failed; attempting rollback."
    if ((Test-Path $Backup) -and -not (Test-Path $InstallDir)) {
      Move-Item -LiteralPath $Backup -Destination $InstallDir -Force
    }
    throw
  }
}

Push-Location $InstallDir
try {
  Invoke-Step "Running local setup" {
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c setup.bat --local" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "local setup failed with exit code $($p.ExitCode)" }
  }
}
catch {
  Write-SetupLog "Local setup failed."
  if ((Test-Path $Backup) -and -not $Repair) {
    Write-SetupLog "Restoring previous install from backup."
    Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -LiteralPath $Backup -Destination $InstallDir -Force
  }
  throw
}
finally {
  Pop-Location
}

Write-SetupLog "Setup complete"
