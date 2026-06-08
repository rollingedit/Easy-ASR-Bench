param(
  [string]$WorkDir = "",
  [switch]$InstallDeps,
  [switch]$AllowDownloads,
  [switch]$IncludeNetwork,
  [switch]$IncludeExternal,
  [string[]]$Row = @(),
  [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ArgsList = @("qa\runtime_matrix\run_all_local.py")

if ($WorkDir) {
  $ArgsList += @("--workdir", $WorkDir)
}
if ($InstallDeps) {
  $ArgsList += "--install-deps"
}
if ($AllowDownloads) {
  $ArgsList += "--allow-downloads"
}
if ($IncludeNetwork) {
  $ArgsList += "--include-network"
}
if ($IncludeExternal) {
  $ArgsList += "--include-external"
}
foreach ($RowId in $Row) {
  $ArgsList += @("--row", $RowId)
}
if ($PlanOnly) {
  $ArgsList += "--plan-only"
}

Push-Location $RepoRoot
try {
  & python @ArgsList
  exit $LASTEXITCODE
}
finally {
  Pop-Location
}
