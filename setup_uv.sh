#!/bin/bash
# Setup script for uv environment

echo "Setting up project with uv..."

# Sync dependencies (creates virtual environment and installs packages)
uv sync

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  uv run python main.py"
echo ""
echo "Or activate the shell:"
echo "  source .venv/bin/activate  # Linux/Mac"
echo "  .venv\\Scripts\\activate     # Windows"
