# Self-Improving Agents

A Python system for LLM prompt optimization.

> Warining: add `.sia` to `.gitignore` your gitignore as it will contain your `space_id` and `model_id`
## Project Overview

This project provides tools for evaluating and optimizing prompts for large language models (LLMs). It includes components for prompt evaluation, optimization, and testing.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/self_improving_agents.git
cd self_improving_agents

# Create and activate virtual environment using UV
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the package with development dependencies
uv pip install -e ".[dev]"
```

## Project Structure

```
self_improving_agents/
├── config/                # Configuration templates
├── examples/              # Usage examples
├── src/                   # Source code
│   └── self_improving_agents/
│       ├── evaluators/    # Components for evaluation
│       ├── instrumentation/ # Wrappers for tracking eval function usage
│       ├── models/        # Data structures and schemas
│       ├── optimizers/    # Components for improving prompts
│       ├── runners/       # Orchestration of evaluation/optimization loops
│       └── utils/         # Shared utilities
└── tests/                 # Test files
```

## Usage

```python
# Basic usage example will be provided here
```

## Development

This project uses:
- UV for dependency management
- Ruff for linting and formatting
- Black for code formatting
- MyPy for static type checking
- Pre-commit for git hooks

To set up the development environment:

```bash
# Install pre-commit hooks
pre-commit install
```

## License

MIT
