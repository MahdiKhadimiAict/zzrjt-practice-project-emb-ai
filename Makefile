PYTHON ?= python
LINT_TARGETS := server.py EmotionDetection test_emotion_detection.py

.PHONY: help install install-dev test test-cov lint build run clean

help:
	@echo "install      Install runtime dependencies"
	@echo "install-dev  Install runtime and development dependencies"
	@echo "test         Run the unit test suite"
	@echo "test-cov     Run the unit test suite with a coverage report"
	@echo "lint         Run static code analysis (pylint)"
	@echo "build        Build the wheel and the source distribution"
	@echo "run          Deploy the web application on localhost:5000"
	@echo "clean        Remove build artefacts and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=EmotionDetection --cov=server --cov-report=term-missing

lint:
	$(PYTHON) -m pylint $(LINT_TARGETS)

build: clean
	$(PYTHON) -m build

run:
	$(PYTHON) server.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
