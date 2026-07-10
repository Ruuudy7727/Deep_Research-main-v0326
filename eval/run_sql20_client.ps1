param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("full", "wo_schema")]
    [string]$Variant,

    [string]$BaseUrl = "https://aiops-pre.szclou.com:50221",
    [string]$ExperimentId = "paper_v2_sql20",
    [string]$Python = "python",
    [int]$Limit = 0,
    [switch]$AllowCommitMismatch
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_experiment.py"
$dataset = Join-Path $PSScriptRoot "datasets\v2_sql_20_no_clarify.jsonl"

$status = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/api/eval/status" -TimeoutSec 30
if ($status.protocol_version -ne 2) {
    throw "Remote service does not expose ablation protocol V2."
}
if ($status.variant -ne $Variant) {
    throw "Remote variant mismatch: expected '$Variant', got '$($status.variant)'."
}

$arguments = @(
    $runner,
    "--base-url", $BaseUrl,
    "--variant", $Variant,
    "--experiment-id", $ExperimentId,
    "--dataset", $dataset,
    "--stream-timeout", "1200"
)
if ($Limit -gt 0) {
    $arguments += @("--limit", "$Limit")
}
if ($AllowCommitMismatch) {
    $arguments += "--allow-commit-mismatch"
}

Push-Location $repoRoot
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "SQL20 experiment failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
