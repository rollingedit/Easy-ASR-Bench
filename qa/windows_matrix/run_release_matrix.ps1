param(
  [string]$Tag = "v0.3.6",
  [string]$Output = "qa\windows_matrix\evidence",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$manifest = Get-Content -Raw -LiteralPath "qa\release_manual_rows_v2.json" | ConvertFrom-Json
if ($manifest.schema -ne "easy_asr_bench.release_manual_rows.v2") {
  throw "Unexpected release manual row manifest schema."
}

function Get-ManualRows($Node) {
  foreach ($property in $Node.PSObject.Properties) {
    if ($property.Value -is [System.Management.Automation.PSCustomObject]) {
      Get-ManualRows $property.Value
    } else {
      $property.Name
    }
  }
}

$rows = @(Get-ManualRows $manifest.manual_matrix)

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
