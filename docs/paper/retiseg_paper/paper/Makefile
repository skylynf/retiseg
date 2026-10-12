# Build the paper. Run from paper/ or with: make -C paper
#   make            placeholder data (only if data/results.csv is missing), figures, tables, numbers, bibliography, PDF
#   make collect    replace placeholder rows with measured ones from ../runs, then rebuild
#   make final      rebuild with every number printed as measured (use when no placeholder row is left)
#   make clean
PY ?= python3
FIGS = fig1_design fig2_literature fig3_original fig4_retrained fig5_foundation fig6_sources fig7_scale

all: pdf

data/results.csv:
	$(PY) scripts/build_placeholder.py

figures: data/results.csv
	cd figures && for f in $(FIGS); do $(PY) $$f.py || exit 1; done && $(PY) ed_figs.py

tables: data/results.csv
	$(PY) scripts/make_tables.py

numbers: data/results.csv
	$(PY) scripts/make_numbers.py

refs.bib: scripts/make_bib.py
	$(PY) scripts/make_bib.py

pdf: figures tables numbers refs.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

collect:
	$(PY) scripts/collect_runs.py
	$(MAKE) pdf

final:
	RETISEG_FINAL=1 $(MAKE) figures tables numbers
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=main_final -usepretex='\def\retisegfinal{1}' main.tex

test:
	$(PY) scripts/test_collect_runs.py

clean:
	latexmk -C main.tex; rm -f main_final.*

.PHONY: all figures tables numbers pdf collect final test clean
