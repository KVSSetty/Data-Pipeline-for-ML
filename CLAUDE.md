# Simple Data Preparation Pipeline Project

This is a Claude Code project for building a simple data preparation pipeline.

## Project Structure

- `src/` - Source code
- `data/` - Sample data files
- `tests/` - Unit tests
- `pyproject.toml` - Project configuration and dependencies
- `.venv/` - Virtual environment (created by UV)

## Setup

1. Activate virtual environment: `source .venv/bin/activate` (or `uv run` for commands)
2. Run tests: `uv run pytest tests/`
3. Run the pipeline: `uv run python src/main.py`

## Development

This project uses Python for data processing with pandas, numpy, and other common data science libraries.

**Visualization Priority**: This project prioritizes Plotly for all visualizations over matplotlib/seaborn due to its interactive capabilities and modern interface.