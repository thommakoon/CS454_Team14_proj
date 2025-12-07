@echo off
REM Setup script for uv environment on Windows

echo Setting up project with uv...

REM Sync dependencies (creates virtual environment and installs packages)
uv sync

echo.
echo Setup complete!
echo.
echo To run the project, use:
echo   uv run python main.py
echo.
echo Or activate the virtual environment:
echo   .venv\Scripts\activate
echo   python main.py
echo.
pause
