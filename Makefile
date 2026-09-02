PYTHON ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: help venv install models test lint format info doctor clean

help:
	@echo "make venv     create .venv and install dependencies"
	@echo "make models   download the pinned model weights"
	@echo "make install  install the package in editable mode"
	@echo "make test     run the pytest suite"
	@echo "make lint     run ruff and mypy if installed"
	@echo "make info     show configuration and registered backends"
	@echo "make doctor   report optional dependency availability"
	@echo "make clean    remove caches and build artefacts"

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install:
	$(PYTHON) -m pip install -e ".[dev,face,video,api,experiments]"

models:
	$(PYTHON) -m deepshield download-models

test:
	$(PYTHON) -m pytest

lint:
	-$(PYTHON) -m ruff check src tests
	-$(PYTHON) -m mypy src

info:
	$(PYTHON) -m deepshield info

doctor:
	$(PYTHON) -m deepshield doctor

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
