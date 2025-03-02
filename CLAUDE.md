# RAG-POC Project Guidelines

## Build & Run Commands
- **Recommended: Use the management script:**
  - Start all components: `./scripts/manage_system.sh start`
  - Start specific component: `./scripts/manage_system.sh start [api|ui|ingest]`
  - Start ingestion with file watching: `./scripts/manage_system.sh start ingest watch`
  - Stop all components: `./scripts/manage_system.sh stop`
  - Check system status: `./scripts/manage_system.sh status`
  - Restart system: `./scripts/manage_system.sh restart`

- Individual component commands:
  - Run main script: `python main.py [ingest|web|version]`
  - Process data: `python scripts/process_data.py`
  - Test ChromaDB: `python scripts/test.py`
  - Run tests: `python scripts/run_tests.py`
  - Run dashboard: `python scripts/run_dashboard.py`
  - Create virtualenv: `python -m venv venv && source venv/bin/activate`
  - Install deps: `uv pip install -r requirements.txt` (use uv instead of pip for faster installation)

## Code Style
- Use Python 3.12+
- Type hints for all function parameters and return values
- Docstrings for functions (""" Description of function """)
- Error handling with try/except blocks with specific exception types
- Organize imports: stdlib first, then third-party, then local
- Use semantic variable naming (snake_case for variables/functions)
- Log errors and important events (using the logging module)

## Project Structure
- `/data/raw`: Place new documents here for processing
- `/data/processed`: Processed documents are moved here
- `/data/failed`: Failed documents for debugging
- `/logs`: Log files for tracking processing
- `/scripts`: Processing and utility scripts
- `/ragstack/tests`: Test cases for the application
- `/ragstack/utils/monitoring`: Performance monitoring tools

## Package Management
- Always use `uv` instead of `pip` for all Python package operations
- Examples:
  - Install packages: `uv pip install package-name`
  - Install requirements: `uv pip install -r requirements.txt`
  - Add a new package to requirements: `uv pip install package-name && uv pip freeze > requirements.txt`