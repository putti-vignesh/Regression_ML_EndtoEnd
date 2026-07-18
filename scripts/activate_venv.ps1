$venv = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    & $venv
} else {
    Write-Host "No .venv found; create it with: python -m venv .venv"
}