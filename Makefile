# Recipes call plain `python -m ...` so they run under cmd (Windows make) or sh.
# Activate the venv first, or override: make all PY=.venv/Scripts/python
PY ?= python

.PHONY: setup data clean analyze test all clobber

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.pipeline data

clean:
	$(PY) -m src.pipeline clean

analyze:
	$(PY) -m src.pipeline analyze

test:
	$(PY) -m pytest -q

all: data clean analyze

# Remove generated outputs (keeps downloaded raw CSVs).
clobber:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('data/processed', 'reports/figures', 'reports/tables')]"
