# Run Forecaster locally using uv and python
param(
    [Parameter(Position=0)]
    [ValidateSet("run", "smoke", "test")]
    [string]$Command = "run",

    [Parameter(Position=1)]
    [int]$MaxQuestions = 20,

    [switch]$Publish
)

$ErrorActionPreference = "Stop"
$ForecasterDir = Join-Path $PSScriptRoot "..\forecaster"
Push-Location $ForecasterDir

$env:PYTHONIOENCODING = "utf-8"

try {
    switch ($Command) {
        "smoke" {
            Write-Host "Running Forecaster smoke test..." -ForegroundColor Cyan
            uv run python -m nfa_forecaster.cli smoke
        }
        "test" {
            Write-Host "Running Forecaster test on bot-testing-area (limit: $MaxQuestions, publish: $Publish)..." -ForegroundColor Cyan
            if ($Publish) {
                uv run python -m nfa_forecaster.cli test --limit $MaxQuestions --publish
            } else {
                uv run python -m nfa_forecaster.cli test --limit $MaxQuestions
            }
        }
        "run" {
            Write-Host "Running Forecaster tournament submissions (max: $MaxQuestions)..." -ForegroundColor Cyan
            uv run python -m nfa_forecaster.cli run --max-questions $MaxQuestions
        }
    }
}
finally {
    Pop-Location
}
