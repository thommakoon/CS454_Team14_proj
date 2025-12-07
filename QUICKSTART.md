# Quick Start Guide with uv

## Setup (One-time)

### 1. Install uv (if not already installed)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup the project

**Windows:**
```cmd
setup_uv.bat
```

**Linux/Mac:**
```bash
chmod +x setup_uv.sh
./setup_uv.sh
```

**Or manually:**
```bash
uv sync
```

This will:
- Create a virtual environment (`.venv/`)
- Install all dependencies
- Make the package installable

## Running the Code

### Run main demo:
```bash
uv run python main.py
```

### Run examples:
```bash
# Test single gene evaluation
uv run python examples/test_single_gene.py

# Validate dataflow templates
uv run python examples/validate_dataflows.py

# Run NSGA-II optimization
uv run python examples/run_optimization.py
```

### Or activate the environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Then run normally
python main.py
```

## Troubleshooting

### If `uv` command not found:
- Make sure uv is installed and in your PATH
- Restart your terminal after installation

### If dependencies fail to install:
- Check your Python version: `uv python list`
- uv will automatically download the right Python version if needed

### If import errors:
- Make sure you ran `uv sync` first
- Try: `uv pip install -e .` to install the package in editable mode

## Next Steps

See `README.md` for full documentation and usage examples.

