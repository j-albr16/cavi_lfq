PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
export PYTHONPATH := $(CURDIR)

# synthetic spectrum settings, override e.g. `make spectrum SEED=1 N_IONS=2000000`
DATA_DIR ?= data
# SPECTRUM_NAME ?= synthetic
# SPECTRUM_NAME ?= BSA1
SPECTRUM_NAME ?= FeatureFinderCentroided_1_input
SEED ?= 0
N_IONS ?= 1000000
SPECTRUM ?= $(DATA_DIR)/$(SPECTRUM_NAME).mzML
NUM_ITER ?= 40
# random candidate features scattered over the spectrum (no peak picker); INIT=pixels|uniform
NUM_FEATURES ?= 30
INIT ?= pixels
# optional answer key (truth.json or featureXML), only used to score the result
REFERENCE ?=

.PHONY: help test test-fast generative spectrum algorithm pdf readme-assets all clean

help:
	@echo "make test        run the full test pipeline"
	@echo "make test-fast   run the tests without the slow quadrature/CAVI checks"
	@echo "make generative  sample the generative model and write plots/spectrum_3d.png"
	@echo "make spectrum    write a synthetic LC-MS map for OpenMS to $(DATA_DIR)/$(SPECTRUM_NAME).mzML (+ .truth.json)"
	@echo "make algorithm   run CAVI from random features on $(SPECTRUM); plots + videos go to plots/cavi_real"
	@echo "make pdf          build lfq.pdf from lfq.typ"
	@echo "make readme-assets  graphical model image and preview gif for README.md"
	@echo "make all         test, then generative"
	@echo "make clean       remove caches"

test:
	$(PYTHON) -m pytest tests -q

test-fast:
	$(PYTHON) -m pytest tests -q -k "not nig_matches_2d and not never_decrease and not monte_carlo"

generative:
	$(PYTHON) plots/plot_spectrum.py

spectrum:
	$(PYTHON) -m cavi.generative --out $(DATA_DIR) --name $(SPECTRUM_NAME) --seed $(SEED) --n-ions $(N_IONS)

# run CAVI from randomly scattered features and write ELBO / metric plots and the fit videos to
# plots/cavi_real, e.g. `make algorithm SPECTRUM=data/other.mzML NUM_FEATURES=100 INIT=uniform`
algorithm:
	$(PYTHON) plots/plot_cavi.py --spectrum $(SPECTRUM) --num-iter $(NUM_ITER) --num-features $(NUM_FEATURES) --init $(INIT) $(if $(REFERENCE),--reference $(REFERENCE))

pdf:
	typst compile lfq.typ lfq.pdf

# images used by README.md; the GIF is a preview of plots/cavi_real/fit_full.mp4
readme-assets:
	typst compile --root . assets/graphical_model.typ assets/graphical_model.png --ppi 220
	ffmpeg -loglevel error -y -i plots/cavi_real/fit_full.mp4 -vf "fps=3,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" plots/cavi_real/fit_full.gif

all: test generative spectrum

clean:
	find . -path ./venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache
