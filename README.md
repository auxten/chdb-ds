# DataStore

[![CI/CD](https://github.com/auxten/chdb-ds/actions/workflows/datastore-ci.yml/badge.svg)](https://github.com/auxten/chdb-ds/actions/workflows/datastore-ci.yml)
[![codecov](https://codecov.io/gh/auxten/chdb-ds/branch/main/graph/badge.svg)](https://codecov.io/gh/auxten/chdb-ds)
[![PyPI version](https://badge.fury.io/py/chdb-ds.svg)](https://badge.fury.io/py/chdb-ds)
[![Python versions](https://img.shields.io/pypi/pyversions/chdb-ds.svg)](https://pypi.org/project/chdb-ds/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> ⚠️ **EXPERIMENTAL**: This project is currently in experimental stage. APIs may change without notice. Not recommended for production use yet.

A Pandas-like data manipulation framework with automatic SQL generation and execution capabilities.

## Features

- **Fluent API**: Pandas-like interface for data manipulation
- **Immutable Operations**: Thread-safe method chaining
- **SQL Generation**: Automatic conversion to optimized SQL queries
- **Multiple Backends**: Support for ClickHouse, PostgreSQL, Parquet files, etc.
- **Type-Safe**: Comprehensive type hints and validation
- **Extensible**: Easy to add custom functions and data sources

## Quick Start

### Installation

```bash
pip install chdb-ds
```

### Basic Usage

```python
from datastore import DataStore

# Create a DataStore
ds = DataStore(table="customers")

# Build a query with method chaining
sql = (ds
    .select("name", "age", "city")
    .filter(ds.age > 18)
    .filter(ds.city == "NYC")
    .sort("name")
    .limit(10)
    .to_sql())

print(sql)
# Output: SELECT "name", "age", "city" FROM "customers" 
#         WHERE ("age" > 18 AND "city" = 'NYC') 
#         ORDER BY "name" ASC LIMIT 10
```

### Working with Expressions

```python
from datastore import Field, Sum, Count

# Arithmetic operations
ds.select(
    ds.price * 1.1,  # 10% price increase
    (ds.revenue - ds.cost).as_("profit")
)

# Aggregate functions
ds.groupby("category").select(
    Field("category"),
    Sum(Field("amount"), alias="total"),
    Count("*", alias="count")
)
```

### Conditions

```python
# Simple conditions
ds.filter(ds.age > 18)
ds.filter(ds.status == "active")

# Complex conditions
ds.filter(
    ((ds.age > 18) & (ds.age < 65)) | 
    (ds.status == "premium")
)

# Negation
ds.filter(~(ds.deleted == True))
```

## Design Philosophy

DataStore is inspired by pypika's excellent query builder design but focuses on:

1. **High-level API**: Pandas-like interface for data scientists
2. **Query Execution**: Built-in execution capabilities (not just SQL generation)
3. **Data Source Abstraction**: Unified interface across different backends
4. **Modern Python**: Type hints, dataclasses, and Python 3.7+ features

## Architecture

```
datastore/
├── core.py           # DataStore main class
├── expressions.py    # Expression system (Field, Literal, Arithmetic)
├── conditions.py     # Condition system (WHERE clause)
├── functions.py      # SQL functions (SUM, COUNT, UPPER, etc.)
├── connections.py    # Data source connections
├── executors.py      # Query execution
└── utils.py          # Utilities (@immutable decorator, etc.)
```

### Key Design Patterns

#### 1. Immutability via @immutable Decorator

```python
from datastore.utils import immutable

class DataStore:
    @immutable
    def select(self, *fields):
        self._select_fields.extend(fields)
        # Decorator handles copying and returning new instance
```

#### 2. Operator Overloading

```python
# Natural Python syntax
ds.age > 18          # BinaryCondition('>', Field('age'), Literal(18))
ds.price * 1.1       # ArithmeticExpression('*', Field('price'), Literal(1.1))
(cond1) & (cond2)    # CompoundCondition('AND', cond1, cond2)
```

#### 3. Smart Value Wrapping

```python
Expression.wrap(42)        # Literal(42)
Expression.wrap("hello")   # Literal("hello")
Expression.wrap(None)      # Literal(None)
Expression.wrap(Field('x'))# Field('x') (unchanged)
```


## Development

### Running Tests

```bash
# Run all tests
python -m pytest datastore/tests/

# Run specific test file
python -m pytest datastore/tests/test_expressions.py

# Run with coverage
python -m pytest --cov=datastore datastore/tests/

# Generate HTML coverage report
python -m pytest --cov=datastore --cov-report=html datastore/tests/
# Open htmlcov/index.html in browser to view detailed coverage
```

### Running Individual Test Modules

```bash
# Test expressions
python -m unittest datastore.tests.test_expressions

# Test conditions
python -m unittest datastore.tests.test_conditions

# Test functions
python -m unittest datastore.tests.test_functions

# Test core DataStore
python -m unittest datastore.tests.test_datastore_core
```

## Roadmap

- [x] Core expression system
- [x] Condition system
- [x] Function system
- [x] Basic DataStore operations
- [x] Immutability support
- [ ] ClickHouse table engines support
- [ ] DataFrame operations (drop, assign, fillna, etc.)
- [ ] Query executors
- [ ] Multiple backend support
- [ ] Mock data support
- [ ] Schema management(infer or set manually)
- [ ] ClickHouse functions support
- [ ] Connection managers
- [ ] Image, Video, Audio data support
- [ ] PyTorch Dataloader support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0

## Credits

Inspired by:
- [SQLAlchemy](https://www.sqlalchemy.org/) - Excellent SQL query builder
- [PyPika](https://github.com/kayak/pypika) - Simple SQL query builder
- [Pandas](https://pandas.pydata.org/) - DataFrame API

