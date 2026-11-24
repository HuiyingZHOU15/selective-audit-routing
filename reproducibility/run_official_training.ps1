param(
    [Parameter(Mandatory = $true)][string]$RunsRoot,
    [int]$Seed = 0,
    [string]$PythonExecutable = "python",
    [switch]$StopOnFailure
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $RepoRoot "configs\adaptation_phase5_official_matrix_v1_1.csv"
$Runner = Join-Path $RepoRoot "src\run_phase5_official.py"
$Arguments = @(
    $Runner,
    "--manifest", $Manifest,
    "--runs-root", $RunsRoot,
    "--project-root", $RepoRoot,
    "--python-executable", $PythonExecutable
)
if ($Seed -ne 0) { $Arguments += @("--seed", $Seed) }
if ($StopOnFailure) { $Arguments += "--stop-on-failure" }
& $PythonExecutable @Arguments
exit $LASTEXITCODE

