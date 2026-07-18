@echo off
if exist .\.venv\Scripts\activate.bat (
  call .\.venv\Scripts\activate.bat
)
set PORT=%1
if "%PORT%"=="" set PORT=5001
echo Starting MLflow UI on port %PORT% using sqlite backend...
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root .\mlruns --port %PORT% --workers 1
