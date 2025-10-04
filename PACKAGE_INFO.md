# chdb-ds Package Information

## Package Details

- **Package Name**: `chdb-ds`
- **GitHub**: https://github.com/auxten/chdb-ds
- **PyPI**: https://pypi.org/project/chdb-ds/
- **Author**: auxten
- **License**: Apache License 2.0
- **Version**: 0.1.0

## Installation

```bash
pip install chdb-ds
```

## Quick Start

```python
from datastore import DataStore

# Create a DataStore
ds = DataStore(table="users")
ds.connect()

# Query with method chaining
result = ds.select("name", "age").filter(ds.age > 18).execute()
```

## Files Created for PyPI Packaging

### Core Configuration Files

1. **setup.py** - Traditional setup script
   - Package metadata
   - Dependencies
   - Build configuration

2. **setup.cfg** - Setup configuration
   - Metadata
   - Build options
   - Tool configurations

3. **pyproject.toml** - Modern Python project config
   - PEP 518 build system
   - Project metadata
   - Tool configurations (black, pytest, coverage)

4. **MANIFEST.in** - Package data inclusion rules
   - Includes README.md, LICENSE, pytest.ini
   - Includes test files

### Dependencies

5. **requirements.txt** - Production dependencies
   ```
   chdb>=2.0.0
   ```

6. **requirements-dev.txt** - Development dependencies
   ```
   pytest>=7.0.0
   pytest-cov>=4.0.0
   black>=23.0.0
   flake8>=6.0.0
   mypy>=1.0.0
   build>=0.10.0
   twine>=4.0.0
   ```

### Documentation

7. **README.md** - Project overview and usage guide
8. **CHANGELOG.md** - Version history and changes
9. **CONTRIBUTING.md** - Contribution guidelines
10. **RELEASING.md** - Release process documentation
11. **LICENSE** - Apache License 2.0

### Development Tools

12. **Makefile** - Common development tasks
    ```bash
    make install-dev    # Install in development mode
    make test          # Run tests
    make test-coverage # Run tests with coverage
    make build         # Build distribution packages
    make upload-test   # Upload to TestPyPI
    make upload        # Upload to PyPI
    make format        # Format code with black
    make lint          # Run linting
    ```

13. **.gitignore** - Git ignore rules
14. **pytest.ini** - Pytest configuration

## Building and Publishing

### Local Development

```bash
# Clone the repository
git clone https://github.com/auxten/chdb-ds.git
cd chdb-ds

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m unittest discover -s tests -v

# Or use make
make install-dev
make test
```

### Build Package

```bash
# Clean previous builds
make clean

# Build distribution packages
make build

# This creates:
# - dist/chdb-ds-0.1.0.tar.gz (source)
# - dist/chdb_ds-0.1.0-py3-none-any.whl (wheel)
```

### Publish to PyPI

```bash
# Test on TestPyPI first (recommended)
make upload-test

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple chdb-ds

# If everything works, publish to production PyPI
make upload
```

### Manual Publishing

```bash
# Install build tools
pip install build twine

# Build
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Version Management

Version is defined in `__init__.py`:

```python
__version__ = "0.1.0"
```

To release a new version:
1. Update `__version__` in `__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Commit changes
4. Build and upload to PyPI
5. Tag the release in git
6. Create GitHub release

## Package Structure

```
chdb-ds/
├── __init__.py          # Package entry point, version
├── core.py              # DataStore main class
├── expressions.py       # Expression system
├── conditions.py        # Condition system
├── functions.py         # SQL functions
├── connection.py        # Connection abstraction
├── executor.py          # Query executor
├── utils.py            # Utilities
├── enums.py            # Enumerations
├── exceptions.py       # Custom exceptions
├── tests/              # Test suite (530+ tests)
│   ├── test_*.py
│   └── ...
├── setup.py            # Setup script
├── setup.cfg           # Setup configuration
├── pyproject.toml      # Modern project config
├── MANIFEST.in         # Package data
├── requirements.txt    # Dependencies
├── requirements-dev.txt # Dev dependencies
├── README.md           # Documentation
├── CHANGELOG.md        # Version history
├── CONTRIBUTING.md     # Contribution guide
├── RELEASING.md        # Release guide
├── LICENSE             # Apache License 2.0
├── Makefile           # Development tasks
└── .gitignore         # Git ignore rules
```

## Testing

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_insert_update_delete -v

# Run with coverage
pip install pytest pytest-cov
pytest --cov=. --cov-report=html tests/

# Or use make
make test
make test-coverage
```

## Code Quality

```bash
# Format code
make format

# Check linting
make lint

# Type checking (optional)
pip install mypy
mypy . --ignore-missing-imports
```

## Features Implemented

### v0.1.0 - Initial Release
- ✅ Expression system (Field, Literal, Arithmetic)
- ✅ Condition system (Binary, Compound, Unary, IN, BETWEEN, LIKE)
- ✅ SQL functions (SUM, COUNT, AVG, MIN, MAX, UPPER, LOWER, CONCAT)
- ✅ DataStore query building (SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY)
- ✅ Subquery support (WHERE IN, INSERT SELECT)
- ✅ INSERT/UPDATE/DELETE operations (ClickHouse style)
- ✅ Immutable operations with @immutable decorator
- ✅ chdb integration
- ✅ Comprehensive test suite (530+ tests)

## Dependencies

### Production
- **chdb** >=2.0.0 - Embedded ClickHouse for query execution

### Development
- **pytest** >=7.0.0 - Testing framework
- **pytest-cov** >=4.0.0 - Coverage reporting
- **black** >=23.0.0 - Code formatting
- **flake8** >=6.0.0 - Linting
- **mypy** >=1.0.0 - Type checking
- **build** >=0.10.0 - Build tool
- **twine** >=4.0.0 - PyPI upload tool

## Support

- **Issues**: https://github.com/auxten/chdb-ds/issues
- **Discussions**: https://github.com/auxten/chdb-ds/discussions
- **Email**: auxtenwpc@gmail.com

## License

Apache License 2.0 - See LICENSE file for details.

