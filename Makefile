PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
export PYTHONPATH := $(CURDIR)

.PHONY: help test test-fast generative algorithm all clean

help:
	@echo "make test        run the full test pipeline"
	@echo "make test-fast   run the tests without the slow quadrature/CAVI checks"
	@echo "make generative  sample the generative model and write plots/spectrum_3d.png"
	@echo "make algorithm   run CAVI (not implemented yet)"
	@echo "make all         test, then generative"
	@echo "make clean       remove caches"

test:
	$(PYTHON) -m pytest tests -q

test-fast:
	$(PYTHON) -m pytest tests -q -k "not nig_matches_2d and not never_decrease and not monte_carlo"

generative:
	$(PYTHON) plots/plot_spectrum.py

# TODO: point this at the CAVI driver once it exists (e.g. cavi/main.py).
algorithm:
	@test -f cavi/main.py || { echo "cavi/main.py does not exist yet"; exit 1; }
	$(PYTHON) -m cavi.main

all: test generative

clean:
	find . -path ./venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache
