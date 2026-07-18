@echo off
if exist .\.venv\Scripts\activate.bat (
  call .\.venv\Scripts\activate.bat
) else (
  echo No .venv found; create with: python -m venv .venv
)
