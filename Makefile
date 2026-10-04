PYTHON ?= python
LINT_TARGETS := server.py EmotionDetection test_emotion_detection.py

# The five canonical emotion checks need a real model backend, so they are opt-in.
INTEGRATION_ENV := EMOTION_INTEGRATION_TESTS=1

.PHONY: help install install-dev test test-cov test-integration lint build run dev clean

help:
	@echo "install           Install runtime dependencies"
	@echo "install-dev       Install runtime and development dependencies"
	@echo "test              Run the unit test suite"
	@echo "test-cov          Run the unit test suite with a coverage report"
	@echo "test-integration  Run the canonical emotion checks against a real model"
	@echo "lint              Run static code analysis (pylint)"
	@echo "build             Build the wheel and the source distribution"
	@echo "run               Deploy the web application on localhost:5000"
	@echo "dev               Deploy the web application with the debug reloader"
	@echo "clean             Remove build artefacts and caches"
	@echo ""
	@echo "test-integration needs a POSIX shell. In PowerShell run instead:"
	@echo "  $$env:EMOTION_INTEGRATION_TESTS=1; $(PYTHON) -m pytest -q -m integration"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=EmotionDetection --cov=server --cov-report=term-missing

test-integration:
	$(INTEGRATION_ENV) $(PYTHON) -m pytest -q -m integration

lint:
	$(PYTHON) -m pylint $(LINT_TARGETS)

build: clean
	$(PYTHON) -m build

run:
	$(PYTHON) server.py

dev:
	$(PYTHON) -m flask --app server run --debug

# Implemented with Python rather than rm/find so the recipe also runs on Windows.
clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in (pathlib.Path(d) for d in ('build', 'dist', '.pytest_cache', 'htmlcov')) if p.is_dir()]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
	$(PYTHON) -c "import pathlib; [p.unlink(missing_ok=True) for p in (pathlib.Path(f) for f in ('.coverage', 'coverage.xml'))]"