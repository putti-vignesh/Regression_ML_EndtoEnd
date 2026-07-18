@echo off
if exist .\.venv\Scripts\python.exe (
  .\.venv\Scripts\python.exe src\training_pipeline\train.py
) else (
  python src\training_pipeline\train.py
)
