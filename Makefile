# DIX VISION v42.2 — Makefile
PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

.PHONY: help venv install install-dev install-windows smoke test verify run clean lint type package

help:
	@echo "Targets: venv install install-dev install-windows smoke test verify run lint type package clean"

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

install-windows: install
	$(PIP) install -r requirements-windows.txt

smoke:
	$(PYBIN) startup_test.py

test:
	$(PYBIN) tests/test_all.py

verify:
	$(PYBIN) dix.py verify

run:
	$(PYBIN) main.py --dev

lint:
	$(PYBIN) -m ruff check .

type:
	$(PYBIN) -m mypy .

package:
	$(PYBIN) -m build

clean:
	rm -rf $(VENV) build dist *.egg-info
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
