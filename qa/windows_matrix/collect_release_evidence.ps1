param(
  [Parameter(Mandatory = $true)][string]$Output,
  [Parameter(Mandatory = $true)][string]$RowId,
  [Parameter(Mandatory = $true)][string]$Status,
  [string]$AppVersion = "",
  [string]$ReleaseCommit = "",
  [string[]]$Commands = @()
)

$ErrorActionPreference = "Stop"
if ($Status -eq "pass" -and ([string]::IsNullOrWhiteSpace($AppVersion) -or [string]::IsNullOrWhiteSpace($ReleaseCommit))) {
  throw "Rows marked pass must include -AppVersion and -ReleaseCommit so release smoke evidence is tied to exact artifacts."
}
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$rowDir = Join-Path $Output $RowId
New-Item -ItemType Directory -Force -Path $rowDir | Out-Null

$environment = @{
  computer_name = $env:COMPUTERNAME
  os = (Get-CimInstance Win32_OperatingSystem).Caption
  os_version = (Get-CimInstance Win32_OperatingSystem).Version
  powershell = $PSVersionTable.PSVersion.ToString()
  python = (& py -3 -c "import sys; print(sys.version.split()[0])" 2>$null)
  gpu = @(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)
}
$envPath = Join-Path $rowDir "environment.json"
$environment | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $envPath -Encoding UTF8

$hashes = @{}
Get-ChildItem -LiteralPath $rowDir -File -Recurse | ForEach-Object {
  $hashes[$_.FullName.Substring($rowDir.Length + 1)] = "sha256:" + (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
}
$logFiles = @(Get-ChildItem -LiteralPath $rowDir -File -Recurse | Where-Object { $_.Extension -eq ".log" })
$resultFiles = @(Get-ChildItem -LiteralPath $rowDir -File -Recurse | Where-Object { $_.Name -in @("results.json", "results.txt", "benchmark.csv", "compare.html", "batch.json", "index.html") })
$logsHash = ""
$resultsHash = ""
if ($logFiles.Count -gt 0) {
  $logDigest = [System.Security.Cryptography.SHA256]::Create()
  $logBytes = [System.Text.Encoding]::UTF8.GetBytes(($logFiles | Sort-Object FullName | ForEach-Object { "$($_.Name)=$((Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant())" }) -join "`n")
  $logsHash = "sha256:" + ([System.BitConverter]::ToString($logDigest.ComputeHash($logBytes)).Replace("-", "").ToLowerInvariant())
}
if ($resultFiles.Count -gt 0) {
  $resultDigest = [System.Security.Cryptography.SHA256]::Create()
  $resultBytes = [System.Text.Encoding]::UTF8.GetBytes(($resultFiles | Sort-Object FullName | ForEach-Object { "$($_.Name)=$((Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant())" }) -join "`n")
  $resultsHash = "sha256:" + ([System.BitConverter]::ToString($resultDigest.ComputeHash($resultBytes)).Replace("-", "").ToLowerInvariant())
}
if ($Status -eq "pass" -and [string]::IsNullOrWhiteSpace($logsHash) -and [string]::IsNullOrWhiteSpace($resultsHash)) {
  throw "Rows marked pass must include at least one log or result artifact in the row evidence folder."
}
$row = @{
  id = $RowId
  status = $Status
  app_version = $AppVersion
  release_commit = $ReleaseCommit
  machine = $environment.computer_name
  commands = $Commands
  environment_summary = $environment
  evidence_sha256 = $hashes
  logs_sha256 = $logsHash
  results_sha256 = $resultsHash
}
$row | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $rowDir "row.json") -Encoding UTF8
Write-Host (Join-Path $rowDir "row.json")
