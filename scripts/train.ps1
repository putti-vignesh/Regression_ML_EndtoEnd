$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
function Find-VenvPython {
  $dir = $scriptDir
  for ($i = 0; $i -lt 4; $i++) {
    $candidate = Join-Path $dir ".venv\Scripts\python.exe"
    if (Test-Path $candidate) { return $candidate }
    $dir = Split-Path -Parent $dir
    if ([string]::IsNullOrEmpty($dir)) { break }
  }
  return $null
}
$venvPy = Find-VenvPython
$script = Join-Path $scriptDir "..\src\training_pipeline\train.py"
if ($venvPy) {
  Write-Host "Using venv python: $venvPy"
  & $venvPy $script
} else {
  Write-Host "No .venv found; falling back to system python"
  python $script
}