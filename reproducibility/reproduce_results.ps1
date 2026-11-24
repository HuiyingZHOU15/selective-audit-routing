param(
    [string]$PythonExecutable = "python",
    [string]$OutputRoot = "",
    [switch]$SkipPdf,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @((Join-Path $PSScriptRoot "reproduce_results.py"), "--project-root", $RepoRoot)
if ($OutputRoot) { $Arguments += @("--output-root", $OutputRoot) }
if ($SkipPdf) { $Arguments += "--skip-pdf" }
if ($VerifyOnly) { $Arguments += "--verify-only" }
& $PythonExecutable @Arguments
exit $LASTEXITCODE

