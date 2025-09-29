# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for cryptocurrency funding rate arbitrage trading strategies. The project is currently in its initial setup phase with a virtual environment configured at `.venv/`.

## Development Environment

- **Python Version**: 3.9 (based on virtual environment setup)
- **Virtual Environment**: Located in `.venv/` directory
- **IDE**: PyCharm project configuration present

## Common Commands

Since this is a new project, standard Python commands will apply:

```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt

# Run tests (once test framework is set up)
pytest

# Run the main application (once created)
python main.py
```

## Project Architecture

This project is designed for funding rate arbitrage in cryptocurrency markets. Key architectural considerations:

- **Data Collection**: Will likely need modules for collecting funding rates from multiple exchanges
- **Strategy Engine**: Core logic for identifying arbitrage opportunities
- **Risk Management**: Position sizing and risk controls
- **Execution**: Order management and trade execution
- **Monitoring**: Real-time monitoring and alerting systems

## Development Notes

- The project uses PyCharm as the primary IDE
- Virtual environment is pre-configured for Python 3.9
- Claude Code permissions are configured to allow file system operations