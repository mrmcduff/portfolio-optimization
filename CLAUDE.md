# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Lint/Test Commands
- Install dependencies: `uv pip install -e .` 
- Install development dependencies: `uv pip install -e ".[dev]"`
- Run the full pipeline: `./run.py all`
- Run linting: `ruff check src/`
- Format code: `black src/`
- Sort imports: `isort src/`

## Code Style Guidelines
- Line length: 88 characters (Black default)
- Use type hints for all function parameters and return values
- Imports: standard library first, then third-party, then local modules
- Error handling: use try/except blocks with specific exception types
- Naming: snake_case for variables/functions, PascalCase for classes
- Docstrings: use Google-style docstrings with Parameters and Returns sections
- Follow PEP 8 conventions where not overridden by Black
- Use f-strings for string formatting
- Organize imports with isort (black profile)
- Use meaningful variable names that reflect their purpose