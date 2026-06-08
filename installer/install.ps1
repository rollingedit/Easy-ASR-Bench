param(
  [string]$InstallDir = "$env:LOCALAPPDATA\Easy-ASR-Bench",
  [string]$Version = "v0.3.9",
  [switch]$DryRun,
  [switch]$VerifyRelease,
  [switch]$Repair,
  [switch]$Uninstall,
  [switch]$RemoveUserData,
  [string]$ConfirmRemoveUserData = "",
  [switch]$Doctor,
  [string]$AssetDir = ""
)

$ErrorActionPreference = "Stop"
trap {
  Write-Host $_
  [Environment]::Exit(1)
}
$Repo = "https://github.com/rollingedit/Easy-ASR-Bench"
$ReleaseBase = "$Repo/releases/download/$Version"
$LogDir = Join-Path $InstallDir "Logs"
$Log = Join-Path $LogDir "setup.log"

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

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

function Invoke-Download($Uri, $OutFile) {
  try {
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
  }
  catch {
    Write-SetupLog "Download failed: $Uri"
    Write-SetupLog "Check your internet connection, TLS support, antivirus/proxy settings, or GitHub availability. Log: $Log"
    throw
  }
}

function Copy-ReleaseAsset($Name, $OutFile) {
  if ($AssetDir) {
    $source = Join-Path $AssetDir $Name
    if (-not (Test-Path -LiteralPath $source)) {
      throw "Staged release asset is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $OutFile -Force
    return
  }
  Invoke-Download "$ReleaseBase/$Name" $OutFile
}

function Resolve-Python {
  $commands = @(
    @{ File = "py"; Args = @("-3.14") },
    @{ File = "py"; Args = @("-3.13") },
    @{ File = "py"; Args = @("-3.12") },
    @{ File = "py"; Args = @("-3.11") },
    @{ File = "py"; Args = @("-3.10") },
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

function Assert-Checksum($Path, $Expected, $Name) {
  if (-not $Expected) {
    throw "Missing expected SHA256 for $Name"
  }
  $actual = "sha256:" + (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
  if ($actual -ne $Expected) {
    throw "Checksum mismatch for $Name. Expected $Expected, got $actual"
  }
  Write-SetupLog "[OK] $Name SHA256 verified"
}

function Assert-TextLineEndings($Path, $Expected, $Name) {
  $bytes = [System.IO.File]::ReadAllBytes($Path)
  if ($bytes.Length -eq 0) { return }
  for ($i = 0; $i -lt $bytes.Length; $i++) {
    if ($bytes[$i] -eq 13) {
      if (($i + 1) -ge $bytes.Length -or $bytes[$i + 1] -ne 10) {
        throw "$Name contains CR-only line endings"
      }
    }
    if ($Expected -eq "CRLF" -and $bytes[$i] -eq 10) {
      if ($i -eq 0 -or $bytes[$i - 1] -ne 13) {
        throw "$Name must use CRLF line endings"
      }
    }
    if ($Expected -eq "LF" -and $i -gt 0 -and $bytes[$i - 1] -eq 13 -and $bytes[$i] -eq 10) {
      throw "$Name must use LF line endings"
    }
  }
}

function Assert-StagingPhysicalFiles($Root) {
  $markers = @{
    "setup.bat" = @("APP_VERSION=", "--dry-run", "--verify-release", "--local")
    "installer\install.ps1" = @("Move-PreservedUserData", "Restore-MovedUserData", "Assert-StagingPhysicalFiles")
    "scripts\validate_physical_files.py" = @("REQUIRED_TEXT_MARKERS", "def validate_root")
    "scripts\verify_github_release.py" = @("def verify_release", "release-smoke", "checksums.json")
    ".github\workflows\release-gate.yml" = @("Validate release files", "Run unit tests")
    ".github\workflows\publish-release.yml" = @("Publish verified release", "gh release upload")
    "app\model_scanner.py" = @("def scan_models", "ModelCandidate", "indexed_safetensor_missing_files")
    "app\results_writer.py" = @("def build_results", "def write_all_reports", "runtime_rankings")
    "app\scoring.py" = @("def edit_distance", "def balanced_score", "def score_against_reference")
  }
  foreach ($rel in $markers.Keys) {
    $path = Join-Path $Root $rel
    if (-not (Test-Path $path)) {
      throw "Staging validation failed: missing $rel"
    }
    $text = Get-Content -Raw -LiteralPath $path
    foreach ($marker in $markers[$rel]) {
      if (-not $text.Contains($marker)) {
        throw "Staging validation failed: $rel is missing required marker $marker"
      }
    }
  }
  $crlfFiles = @("setup.bat", "installer\install.ps1")
  foreach ($rel in $crlfFiles) {
    Assert-TextLineEndings (Join-Path $Root $rel) "CRLF" $rel
  }
  $lfFiles = @(
    "scripts\validate_physical_files.py",
    "scripts\verify_github_release.py",
    ".github\workflows\release-gate.yml",
    ".github\workflows\publish-release.yml",
    "app\model_scanner.py",
    "app\results_writer.py",
    "app\scoring.py"
  )
  foreach ($rel in $lfFiles) {
    Assert-TextLineEndings (Join-Path $Root $rel) "LF" $rel
  }
  Write-SetupLog "[OK] staging physical files passed PowerShell validation"
}

function Test-ReleaseAssets($Python) {
  Write-SetupLog "Verify-release dry run:"
  if ($AssetDir) {
    Write-SetupLog "  staged asset dir: $AssetDir"
  }
  else {
    Write-SetupLog "  release: $ReleaseBase"
  }
  Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $Stage | Out-Null
  Copy-ReleaseAsset "manifest.json" $Manifest
  Write-SetupLog "[OK] manifest downloaded"
  Copy-ReleaseAsset "checksums.json" $Checksums
  Write-SetupLog "[OK] checksums downloaded"
  $manifestJson = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
  $checksumsJson = Get-Content -Raw -LiteralPath $Checksums | ConvertFrom-Json
  if ($manifestJson.tag -ne $Version) {
    throw "Manifest tag mismatch. Expected $Version, got $($manifestJson.tag)"
  }
  Write-SetupLog "[OK] release tag pinned"
  if ($manifestJson.installer_asset) {
    if ($manifestJson.installer_asset -ne "install.ps1") {
      throw "Manifest installer asset mismatch. Expected install.ps1, got $($manifestJson.installer_asset)"
    }
    Assert-Checksum $Manifest $checksumsJson.files.'manifest.json' "manifest.json"
    $releaseSetup = Join-Path $TempRoot "release-setup.bat"
    $releaseInstaller = Join-Path $TempRoot "release-install.ps1"
    Copy-ReleaseAsset "setup.bat" $releaseSetup
    Write-SetupLog "[OK] setup.bat downloaded"
    Assert-Checksum $releaseSetup $checksumsJson.files.'setup.bat' "setup.bat"
    Copy-ReleaseAsset "install.ps1" $releaseInstaller
    Write-SetupLog "[OK] install.ps1 downloaded"
    Assert-Checksum $releaseInstaller $checksumsJson.files.'install.ps1' "install.ps1"
  }
  else {
    Write-SetupLog "Legacy manifest does not declare installer_asset; skipping standalone bootstrap asset verification."
  }
  $zipName = $manifestJson.app_zip
  Copy-ReleaseAsset $zipName $Zip
  Write-SetupLog "[OK] app ZIP downloaded"
  Assert-Checksum $Zip $checksumsJson.files.$zipName $zipName
  Expand-Archive -LiteralPath $Zip -DestinationPath $Stage -Force
  $root = Get-ChildItem -LiteralPath $Stage -Directory | Select-Object -First 1
  if (-not $root) {
    throw "Release ZIP did not contain an app root folder."
  }
  Write-SetupLog "[OK] ZIP layout valid"
  $validator = Join-Path $root.FullName "scripts\validate_physical_files.py"
  if (Test-Path $validator) {
    Assert-StagingPhysicalFiles $root.FullName
    if ($Python) {
      & $Python.File @($Python.Args) $validator --repo $root.FullName
      if ($LASTEXITCODE -ne 0) {
        throw "Physical release-file validation failed with exit code $LASTEXITCODE"
      }
      Write-SetupLog "[OK] release physical files valid"
    }
    else {
      Write-SetupLog "Python was not found; PowerShell physical-file validation passed and Python compile validation will run after Python is installed."
    }
  }
  else {
    Write-SetupLog "Release ZIP does not contain scripts\validate_physical_files.py; falling back to legacy validator if present."
    $legacy = Join-Path $root.FullName "scripts\validate_release_files.py"
    if ($Python -and (Test-Path $legacy)) {
      Invoke-PythonFile $Python $legacy
      Write-SetupLog "[OK] legacy release-file validator passed"
    }
  }
  Write-SetupLog "[OK] install would preserve Models, Input, Output, Logs, Cache, Temp, and config.json"
}

function Get-TreeStats($Path) {
  $files = @(Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction Stop)
  $bytes = 0
  foreach ($file in $files) { $bytes += $file.Length }
  return @{ file_count = $files.Count; byte_count = $bytes }
}

function Move-PreservedUserData($From, $To) {
  $report = @{
    schema = "easy_asr_bench.install_preservation_report.v1"
    created = (Get-Date -Format s)
    source = $From
    destination = $To
    method = "move_without_model_copy"
    items = @()
  }
  foreach ($name in @("Models", "Input", "Output", "Logs", "Cache", "Temp")) {
    $source = Join-Path $From $name
    $dest = Join-Path $To $name
    if (Test-Path $source) {
      if (Test-Path $dest) {
        Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction Stop
      }
      Move-Item -LiteralPath $source -Destination $dest -Force -ErrorAction Stop
      $stats = Get-TreeStats $dest
      $report.items += @{
        name = $name
        status = "moved"
        file_count = $stats.file_count
        byte_count = $stats.byte_count
      }
    }
    else {
      $report.items += @{
        name = $name
        status = "not_present"
        file_count = 0
        byte_count = 0
      }
    }
  }
  $sourceConfig = Join-Path $From "config.json"
  $destConfig = Join-Path $To "config.json"
  if (Test-Path $sourceConfig) {
    if (Test-Path $destConfig) {
      Remove-Item -LiteralPath $destConfig -Force -ErrorAction Stop
    }
    Move-Item -LiteralPath $sourceConfig -Destination $destConfig -Force -ErrorAction Stop
    $report.items += @{
      name = "config.json"
      status = "moved"
      file_count = 1
      byte_count = (Get-Item -LiteralPath $destConfig).Length
    }
  }
  else {
    $report.items += @{
      name = "config.json"
      status = "not_present"
      file_count = 0
      byte_count = 0
    }
  }
  $reportPath = Join-Path $To "Logs\install-preservation-report.json"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $reportPath) | Out-Null
  $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding UTF8
}

function Restore-MovedUserData($From, $To) {
  foreach ($name in @("Models", "Input", "Output", "Logs", "Cache", "Temp")) {
    $source = Join-Path $From $name
    $dest = Join-Path $To $name
    if (Test-Path $source) {
      if (Test-Path $dest) {
        Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction Stop
      }
      Move-Item -LiteralPath $source -Destination $dest -Force -ErrorAction Stop
    }
  }
  $sourceConfig = Join-Path $From "config.json"
  $destConfig = Join-Path $To "config.json"
  if (Test-Path $sourceConfig) {
    if (Test-Path $destConfig) {
      Remove-Item -LiteralPath $destConfig -Force -ErrorAction Stop
    }
    Move-Item -LiteralPath $sourceConfig -Destination $destConfig -Force -ErrorAction Stop
  }
}

if ($Doctor) {
  $python = Join-Path $InstallDir ".venv\Scripts\python.exe"
  if ((Test-Path (Join-Path $InstallDir "app\doctor.py")) -and (Test-Path $python)) {
    & $python -m app.doctor --config (Join-Path $InstallDir "config.json")
    exit $LASTEXITCODE
  }
  Write-SetupLog "Doctor requested but app is not installed."
  exit 1
}

if ($Uninstall) {
  Write-SetupLog "Uninstall requested for $InstallDir"
  if ($DryRun) {
    if ($RemoveUserData) {
      Write-SetupLog "Dry run: would remove app files and user data under $InstallDir"
    }
    else {
      Write-SetupLog "Dry run: would remove app/runtime files while preserving user data under $InstallDir"
    }
    exit 0
  }
  if ($RemoveUserData) {
    if ($ConfirmRemoveUserData -ne "DELETE EASY ASR BENCH USER DATA") {
      Write-SetupLog "Destructive uninstall refused. Re-run with -ConfirmRemoveUserData 'DELETE EASY ASR BENCH USER DATA' to delete Models, Input, Output, Logs, Cache, Temp, and config.json."
      exit 1
    }
    Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction Stop
    exit 0
  }
  foreach ($name in @("app", "requirements", "scripts", "installer", ".github", ".venv")) {
    $path = Join-Path $InstallDir $name
    if (Test-Path $path) {
      Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
    }
  }
  foreach ($name in @("Run.bat", "Drop_Audio_Or_Folders_Here.bat", "Open_Latest_Report.bat", "Open_Models_Folder.bat", "Open_Input_Folder.bat", "Open_Output_Folder.bat", "Edit_Config.bat", "setup.bat", "README.md", "CHANGELOG.md", "SECURITY.md", "SUPPORT.md", "pyproject.toml", "requirements.txt", "pytest.ini")) {
    $path = Join-Path $InstallDir $name
    if (Test-Path $path) {
      Remove-Item -LiteralPath $path -Force -ErrorAction Stop
    }
  }
  Write-SetupLog "Uninstall complete. User data was preserved. Use -RemoveUserData to delete it explicitly."
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
  Write-SetupLog "Python 3.10-3.14 was not found. Setup will install Python before local setup."
}
else {
  Write-SetupLog "Python 3.10-3.14 was not found yet. PowerShell staging validation will run before activation; Python validation will run after local setup installs Python."
}

if ($DryRun) {
  Write-SetupLog "Dry run plan:"
  Write-SetupLog "  manifest: $ReleaseBase/manifest.json"
  Write-SetupLog "  checksums: $ReleaseBase/checksums.json"
  Write-SetupLog "  install dir: $InstallDir"
  Write-SetupLog "  preserved user data: Models, Input, Output, Logs, Cache, Temp, config.json"
  if ($Repair) {
    Write-SetupLog "  repair mode: runtime files and .venv will be refreshed after release verification"
    $venvPython = Join-Path $InstallDir ".venv\Scripts\python.exe"
    if ((Test-Path $InstallDir) -and -not (Test-Path $venvPython)) {
      Write-SetupLog "  detected broken venv: .venv\Scripts\python.exe is missing and will be recreated by local setup"
    }
  }
  if ($VerifyRelease) {
    Test-ReleaseAssets $Python
  }
  else {
    Write-SetupLog "  no files will be downloaded, moved, or deleted"
    Write-SetupLog "  use setup.bat --dry-run --verify-release to validate public release assets"
  }
  exit 0
}

Invoke-Step "Preparing staging folder" {
  Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $Stage | Out-Null
}

Invoke-Step "Downloading release manifest and checksums" {
  Invoke-Download "$ReleaseBase/manifest.json" $Manifest
  Invoke-Download "$ReleaseBase/checksums.json" $Checksums
}

$manifestJson = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
$zipName = $manifestJson.app_zip
$expected = (Get-Content -Raw -LiteralPath $Checksums | ConvertFrom-Json).files.$zipName
Invoke-Step "Downloading app ZIP $zipName" {
  Invoke-Download "$ReleaseBase/$zipName" $Zip
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
    Assert-StagingPhysicalFiles $Stage
    Invoke-PythonFile $Python (Join-Path $Stage "scripts\validate_release_files.py")
  }
}
else {
  Invoke-Step "Validating staging app with PowerShell" {
    Assert-StagingPhysicalFiles $Stage
  }
}

Invoke-Step "Installing app atomically with user-data preservation" {
  Remove-Item -LiteralPath $New -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $New | Out-Null
  Copy-Item -Path (Join-Path $Stage "*") -Destination $New -Recurse -Force
  try {
    if (Test-Path $InstallDir) {
      Remove-Item -LiteralPath $Backup -Recurse -Force -ErrorAction SilentlyContinue
      Move-Item -LiteralPath $InstallDir -Destination $Backup -Force
      Move-PreservedUserData $Backup $New
    }
    Move-Item -LiteralPath $New -Destination $InstallDir -Force
  }
  catch {
    Write-SetupLog "Install swap failed; attempting rollback."
    if ((Test-Path $Backup) -and (Test-Path $New)) {
      Restore-MovedUserData $New $Backup
    }
    if ((Test-Path $Backup) -and -not (Test-Path $InstallDir)) {
      Move-Item -LiteralPath $Backup -Destination $InstallDir -Force
    }
    throw
  }
}

Push-Location $InstallDir
try {
  Invoke-Step "Running local setup" {
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c setup.bat --local --no-post-setup-menu" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "local setup failed with exit code $($p.ExitCode)" }
  }
  Invoke-Step "Validating installed app after local setup" {
    $installedPython = Join-Path $InstallDir ".venv\Scripts\python.exe"
    $installedValidator = Join-Path $InstallDir "scripts\validate_release_files.py"
    if (-not (Test-Path $installedPython)) {
      throw "Installed Python runtime was not created."
    }
    if (-not (Test-Path $installedValidator)) {
      throw "Installed release validator is missing."
    }
    & $installedPython $installedValidator
    if ($LASTEXITCODE -ne 0) {
      throw "Installed app validation failed with exit code $LASTEXITCODE"
    }
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
if (Test-Path $Backup) {
  Remove-Item -LiteralPath $Backup -Recurse -Force -ErrorAction SilentlyContinue
}
