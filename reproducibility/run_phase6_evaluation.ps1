param(
    [Parameter(Mandatory = $true)][string]$OfficialRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [ValidateSet("validation", "test")][string]$Partition = "validation",
    [string]$Device = "cuda",
    [int]$BatchSize = 8,
    [string]$CacheDir = "",
    [string]$PythonExecutable = "python",
    [switch]$ConfirmRetrospectiveTest
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (($Partition -eq "test") -and (-not $ConfirmRetrospectiveTest)) {
    throw "Test evaluation is retrospective. Re-run with -ConfirmRetrospectiveTest to acknowledge this status."
}
$Arguments = @(
    (Join-Path $RepoRoot "src\evaluate_adaptation_phase6_v1_1.py"),
    "--phase6-freeze", (Join-Path $RepoRoot "configs\phase6_retrospective_evaluation_v1_1.freeze.json"),
    "--phase6a-acceptance", (Join-Path $RepoRoot "results\reports\phase6a_acceptance_v1_1.json"),
    "--official-root", $OfficialRoot,
    "--dataset", (Join-Path $RepoRoot "data\processed\model_dataset_v2.csv"),
    "--masked-table", (Join-Path $RepoRoot "data\processed\audit_area_masked_text_phase1.csv"),
    "--output-root", $OutputRoot,
    "--partition", $Partition,
    "--device", $Device,
    "--batch-size", $BatchSize
)
if ($CacheDir) { $Arguments += @("--cache-dir", $CacheDir) }
if ($ConfirmRetrospectiveTest) { $Arguments += "--confirm-retrospective-test" }
& $PythonExecutable @Arguments
exit $LASTEXITCODE

