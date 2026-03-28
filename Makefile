PYTHON  = .venv/Scripts/python.exe
RUFF    = $(PYTHON) -m ruff
PYTEST  = $(PYTHON) -m pytest

SRC     = backend frontend

.PHONY: lint format test

## Run all lint checks (ruff check + format --check)
lint:
	$(RUFF) check $(SRC)
	$(RUFF) format --check $(SRC)

## Auto-fix lint issues and reformat code
lint-fix:
	$(RUFF) check --fix $(SRC)
	$(RUFF) format $(SRC)

## Run the full test suite
test:
	$(PYTEST) tests/ -v
