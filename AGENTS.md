# AGENTS.md

This file contains instructions for agentic coding agents working on the "LinReg from scratch" repository.

## Project Overview
This project focuses on implementing Linear Regression from scratch, primarily using NumPy and Python.

## Environment & Commands

### Dependencies
- Python 3.x
- Core Libraries: `numpy`, `pandas`, `matplotlib`
- Optional: `scikit-learn` (for performance comparison/verification)

### Build/Lint/Test
- **Setup:** Create a virtual environment, then `pip install -r requirements.txt`.
- **Linting:** We use `ruff`. Command: `ruff check .`
- **Formatting:** We use `black`. Command: `black .`
- **Testing:** We use `pytest`.
  - **Run all tests:** `pytest`
  - **Run a single test file:** `pytest path/to/test_file.py`
  - **Run a single test function:** `pytest path/to/test_file.py::test_function_name`
  - **Coverage:** `pytest --cov=src` (if configured)

## Code Style Guidelines

### Python
- **Imports:** Group imports: standard library, third-party libraries, local imports. Alphabetical order within groups.
- **Formatting:** Adhere strictly to [PEP 8](https://peps.python.org/pep-0008/). Use `black` for auto-formatting.
- **Naming Conventions:**
  - `snake_case`: functions, variables, methods, parameters.
  - `PascalCase`: classes.
  - `UPPER_CASE`: constants.
- **Types:** Use type hints for all public functions/methods (`def foo(x: int) -> float:`).
- **Error Handling:** 
  - Use specific exceptions (e.g., `ValueError`, `TypeError`).
  - Do not catch generic `Exception` unless necessary.
  - Log errors with meaningful context.

### Jupyter Notebooks
- **Structure:** Keep notebooks modular. Move reusable code (data loading, preprocessing, model logic) into the `src/` directory as Python modules.
- **Reproducibility:** Ensure notebooks are runnable top-to-bottom.
- **Output:** Avoid committing notebooks with large, transient outputs (e.g., extensive print statements or massive plot dumps) if possible; strip output using tools like `nbstripout` if needed.

### Documentation
- **Docstrings:** Use Google-style or NumPy-style docstrings for all classes and functions.
- **README:** Keep `README.md` updated with setup instructions and brief usage examples.

## Development Principles
- **Efficiency:** Prioritize vectorized NumPy operations over Python-level loops whenever possible in the linear regression implementation.
- **Safety:** Always verify existing code, run linting and tests before making significant changes.
- **Integration:** Prefer adding new functionality as testable Python modules in `src/` rather than only in notebooks.
- **Verification:** When implementing mathematical models from scratch, use `scikit-learn` as a ground-truth reference for validating your implementation's outputs on synthetic or small datasets.
- **Documentation:** When modifying or adding complex mathematical logic, clearly document the derivation or source in the code comments or docstrings.

## Agent Instructions
- **Proactiveness:** Fulfill the user's request thoroughly, including reasonable, directly implied follow-up actions.
- **Conciseness:** Be concise and direct in CLI interactions. Avoid unnecessary conversational filler.
- **Safety First:** If a command or change could be destructive or impact system state unexpectedly, warn the user first.
- **Tools:** Use tools for actions, text output *only* for communication.

## Troubleshooting
- If tests fail, investigate the stack trace first.
- If dependencies are missing, check `requirements.txt`.
- For performance issues, profile the code using `cProfile` or `timeit`.
