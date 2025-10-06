.PHONY: help install install-dev test test-coverage clean build build-release update-version upload-test upload docs format lint

help:
	@echo "DataStore Development Commands:"
	@echo ""
	@echo "  install         Install package in production mode"
	@echo "  install-dev     Install package in development mode with dev dependencies"
	@echo "  test            Run all tests"
	@echo "  test-coverage   Run tests with coverage report"
	@echo "  clean           Clean build artifacts"
	@echo "  build           Build distribution packages (dev build)"
	@echo "  build-release   Build distribution packages with version from git tag"
	@echo "  update-version  Update version from git tag (or pass VERSION=x.y.z)"
	@echo "  upload-test     Upload to TestPyPI"
	@echo "  upload          Upload to PyPI (production)"
	@echo "  format          Format code with black"
	@echo "  lint            Run linting checks"
	@echo ""

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	python -m unittest discover -s tests -v

test-coverage:
	pip install pytest pytest-cov
	pytest --cov=. --cov-report=html --cov-report=term tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean
	pip install build
	python -m build

build-release: clean update-version
	pip install build
	python -m build
	@echo "✓ Built package with version from git tag"

update-version:
	@if [ -n "$(VERSION)" ]; then \
		echo "Updating version to $(VERSION)..."; \
		python scripts/update_version.py v$(VERSION); \
	else \
		echo "Updating version from git tag..."; \
		python scripts/update_version.py; \
	fi

upload-test: build
	pip install twine
	python -m twine upload --repository testpypi dist/*

upload: build
	pip install twine
	python -m twine upload dist/*

format:
	pip install black
	black --line-length 120 .

lint:
	pip install flake8
	flake8 --max-line-length=120 --exclude=tests,build,dist .

