param(
  [Parameter(Mandatory = $true)][string]$Output,
  [Parameter(Mandatory = $true)][string]$RowId,
  [Parameter(Mandatory = $true)][string]$Status,
  [string[]]$Commands = @()
)

$ErrorActionPreference = "Stop"
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
$row = @{
  id = $RowId
  status = $Status
  machine = $environment.computer_name
  commands = $Commands
  environment_summary = $environment
  evidence_sha256 = $hashes
}
$row | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $rowDir "row.json") -Encoding UTF8
Write-Host (Join-Path $rowDir "row.json")
