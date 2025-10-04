# Contributing to DataStore

Thank you for your interest in contributing to DataStore! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/auxten/chdb-ds.git
   cd chdb-ds
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_expressions -v

# Run with coverage
pip install pytest pytest-cov
pytest --cov=. tests/
```

## Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

- Maximum line length: 120 characters
- Use double quotes for strings (unless single quotes avoid escaping)
- Use type hints for function signatures

**Format your code with Black:**
```bash
black --line-length 120 .
```

**Check with flake8:**
```bash
flake8 --max-line-length=120 .
```

### Design Principles

1. **Immutability**: All query-building methods should be immutable using the `@immutable` decorator
2. **Fluent API**: Support method chaining for a natural workflow
3. **Type Safety**: Use type hints consistently
4. **Clear Errors**: Provide helpful error messages
5. **Test Coverage**: Add tests for all new features

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `style`: Code style changes (formatting, etc.)
- `perf`: Performance improvements

Examples:
```
feat(expressions): add support for CASE expressions

Implements CASE WHEN ... THEN ... ELSE ... END SQL expressions
with fluent API support.

Closes #123
```

```
fix(conditions): handle NULL in IN conditions correctly

Previously NULL values in IN conditions were not handled properly.
This fix ensures they are converted to SQL NULL.
```

## Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add awesome feature"
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request:**
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template with details

### PR Requirements

- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated if needed
- [ ] Code follows style guidelines
- [ ] Commit messages follow convention
- [ ] No unnecessary dependencies added

## Testing Guidelines

### Writing Tests

- Use `unittest` framework
- One test class per module being tested
- Clear, descriptive test names
- Test both success and failure cases
- Include edge cases

**Example:**
```python
class TestExpressions(unittest.TestCase):
    """Tests for Expression class."""
    
    def test_field_to_sql_basic(self):
        """Test basic field SQL generation."""
        field = Field('name')
        self.assertEqual('"name"', field.to_sql())
    
    def test_field_to_sql_with_table(self):
        """Test field SQL with table prefix."""
        field = Field('name', table='users')
        self.assertEqual('"users"."name"', field.to_sql())
```

### Test Organization

```
tests/
├── __init__.py
├── test_core.py          # Core DataStore tests
├── test_expressions.py   # Expression system tests
├── test_conditions.py    # Condition system tests
├── test_functions.py     # SQL functions tests
└── test_execution.py     # Query execution tests
```

## Documentation

### Docstring Style

Use Google-style docstrings:

```python
def select(self, *fields: Union[str, Expression]) -> 'DataStore':
    """
    Select specific columns.
    
    Args:
        *fields: Column names (strings) or Expression objects
        
    Returns:
        DataStore for chaining
        
    Example:
        >>> ds.select("name", "age")
        >>> ds.select(ds.name, ds.age + 1)
    """
    ...
```

### README Updates

If your change affects usage:
- Update examples in README.md
- Add new features to the feature list
- Update the roadmap if needed

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and discussions first

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

