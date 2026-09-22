# TTB Label Verification — development tasks.
#
# `make test` is the one that must always work on a clean checkout: it exercises
# the rules engine and the field-assignment logic with no OCR engine installed,
# no model weights, and no network. That property is deliberate — see
# TRADEOFFS.md, "AI extracts; deterministic rules verify".

.PHONY: help install install-ocr dev test test-cov lint fmt audit fixtures evaluate evaluate-ocr run build clean

PYTHON ?= python3

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies (no OCR engine)
	$(PYTHON) -m pip install -c constraints.txt -r requirements-dev.txt

install-ocr:  ## Add the OCR engine (weights ship inside the wheel)
	$(PYTHON) -m pip install -c constraints.txt -r requirements-ocr.txt
	$(PYTHON) -m pip install --no-deps -c constraints.txt -r requirements-ocr-nodeps.txt

dev:  ## Run the API with autoreload
	$(PYTHON) -m uvicorn api.main:app --reload --port 8080

test:  ## Rules engine + assignment tests. No inference, no network.
	$(PYTHON) -m pytest tests/ -v

test-cov:  ## Tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=rules --cov=extraction --cov=api --cov-report=term-missing

lint:  ## Lint
	$(PYTHON) -m ruff check .

fmt:  ## Format
	$(PYTHON) -m ruff format .

audit:  ## Security scans: dependency CVEs, Python static analysis, npm. Needs network.
	@echo "== pip-audit (locked Python dependencies) =="
	@cat constraints.txt requirements.txt requirements-ocr.txt requirements-ocr-nodeps.txt \
		| grep -v '^\s*#' | grep '==' | sed 's/\s*#.*//' | sort -u > .audit-reqs.txt
	-$(PYTHON) -m pip_audit -r .audit-reqs.txt --no-deps --disable-pip
	@rm -f .audit-reqs.txt
	@echo "== bandit (Python static analysis) =="
	-$(PYTHON) -m bandit -q -r api extraction rules tools
	@echo "== npm audit (frontend) =="
	-cd web && npm audit

fixtures:  ## Regenerate the synthetic label corpus
	$(PYTHON) -m tools.generate_fixtures

evaluate:  ## Rules engine vs ground-truth fixtures (stub OCR). Fails on any false negative or ranking inversion.
	$(PYTHON) -m tools.evaluate --fail-on-false-negative --fail-on-ranking

evaluate-ocr:  ## Real OCR pipeline against fixtures — needs `make install-ocr`
	$(PYTHON) -m tools.evaluate --provider rapidocr

run:  ## Run the container
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache htmlcov .coverage
