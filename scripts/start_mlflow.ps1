Param(
    [int]$Port = 5001
)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
function Find-VenvActivate {
  $dir = $scriptDir
  for ($i = 0; $i -lt 4; $i++) {
    $candidate = Join-Path $dir ".venv\Scripts\Activate.ps1"
    if (Test-Path $candidate) { return $candidate }
    $dir = Split-Path -Parent $dir
    if ([string]::IsNullOrEmpty($dir)) { break }
  }
  return $null
}
$venvActivate = Find-VenvActivate
if ($venvActivate) {
    Write-Host "Activating venv..."
    & $venvActivate
} else {
    Write-Host "No virtual environment activation script found; continuing with system Python"
}
Write-Host "Starting MLflow UI on port $Port using sqlite backend..."
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root .\mlruns --port $Port --workers 1
