# Contributing Guidelines

Thank you for your interest in contributing to the Agent Platform project.

## Development Workflow

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/my-feature-name
   ```

2. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure local vector storage is populated if testing retrieval:
   ```bash
   python scripts/ingest.py
   ```

## Code Standards

- Follow PEP 8 guidelines for Python code style.
- Keep functions modular, type-annotated, and documented with docstrings adhering to standard conventions.
- Avoid committing personal machine paths, environment secrets, or compiled cache files.

## Running Tests

Run the unit and retrieval test suites locally:
```bash
pytest tests/test_tools.py tests/test_retrieval.py -v
```

If the local gateway service is running on `http://localhost:8000`:
```bash
pytest tests/ -v
```

## Pull Request Process

1. Verify that all tests pass locally before opening a pull request.
2. Submit a concise PR description outlining the motivation and changes made.
3. Ensure no unnecessary binary or runtime files (e.g., SQLite databases, `mlruns/`) are tracked.
