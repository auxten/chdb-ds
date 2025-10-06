# DataStore

[![CI/CD](https://github.com/auxten/chdb-ds/actions/workflows/datastore-ci.yml/badge.svg)](https://github.com/auxten/chdb-ds/actions/workflows/datastore-ci.yml)
[![codecov](https://codecov.io/gh/auxten/chdb-ds/branch/main/graph/badge.svg)](https://codecov.io/gh/auxten/chdb-ds)
[![PyPI version](https://badge.fury.io/py/chdb-ds.svg)](https://badge.fury.io/py/chdb-ds)
[![Python versions](https://img.shields.io/pypi/pyversions/chdb-ds.svg)](https://pypi.org/project/chdb-ds/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> ⚠️ **EXPERIMENTAL**: This project is currently in experimental stage. APIs may change without notice. Not recommended for production use yet.

A Pandas-like data manipulation framework powered by chDB (ClickHouse) with automatic SQL generation and execution capabilities. Query files, databases, and cloud storage with a unified interface.

## Features

- **Fluent API**: Pandas-like interface for data manipulation
- **Immutable Operations**: Thread-safe method chaining
- **Unified Interface**: Query files, databases, and cloud storage with the same API
- **20+ Data Sources**: Local files, S3, Azure, GCS, HDFS, MySQL, PostgreSQL, MongoDB, Redis, SQLite, ClickHouse, and more
- **Data Lake Support**: Iceberg, Delta Lake, Hudi table formats
- **Format Auto-Detection**: Automatically detect file formats from extensions
- **SQL Generation**: Automatic conversion to optimized SQL queries
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

# Query local files
ds = DataStore.from_file("data.parquet")
result = ds.select("*").filter(ds.age > 18).execute()

# Query S3
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)
result = ds.select("name", "age").limit(10).execute()

# Query MySQL
ds = DataStore.from_mysql(
    host="localhost:3306",
    database="mydb",
    table="users",
    user="root",
    password="pass"
)
result = ds.select("*").filter(ds.active == True).execute()

# Build complex queries with method chaining
query = (ds
    .select("name", "age", "city")
    .filter(ds.age > 18)
    .filter(ds.city == "NYC")
    .sort("name")
    .limit(10))

# Generate SQL
print(query.to_sql())
# Output: SELECT "name", "age", "city" FROM mysql(...) 
#         WHERE ("age" > 18 AND "city" = 'NYC') 
#         ORDER BY "name" ASC LIMIT 10

# Execute query
result = query.execute()
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

## Supported Data Sources

DataStore provides factory methods for easy data source creation:

### Local Files
```python
# Automatically detect format from extension
ds = DataStore.from_file("data.parquet")
ds = DataStore.from_file("data.csv", format="CSV")
ds = DataStore.from_file("data.json", format="JSONEachRow")
```

### Cloud Storage
```python
# Amazon S3
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)
ds = DataStore.from_s3("s3://bucket/*.csv", 
                       access_key_id="KEY",
                       secret_access_key="SECRET")

# Azure Blob Storage
ds = DataStore.from_azure(
    connection_string="DefaultEndpointsProtocol=https;...",
    container="mycontainer",
    path="data/*.parquet"
)

# Google Cloud Storage
ds = DataStore.from_gcs("gs://bucket/data.parquet",
                        hmac_key="KEY",
                        hmac_secret="SECRET")

# HDFS
ds = DataStore.from_hdfs("hdfs://namenode:9000/data/*.parquet")
```

### Databases
```python
# MySQL
ds = DataStore.from_mysql("localhost:3306", "mydb", "users",
                          user="root", password="pass")

# PostgreSQL
ds = DataStore.from_postgresql("localhost:5432", "mydb", "users",
                               user="postgres", password="pass")

# ClickHouse (remote)
ds = DataStore.from_clickhouse("localhost:9000", "default", "events")

# MongoDB (read-only)
ds = DataStore.from_mongodb("localhost:27017", "mydb", "users",
                            user="admin", password="pass")

# SQLite (read-only)
ds = DataStore.from_sqlite("/path/to/database.db", "users")

# Redis
ds = DataStore.from_redis("localhost:6379", 
                          key="key",
                          structure="key String, value String")
```

### Data Lakes
```python
# Apache Iceberg (read-only)
ds = DataStore.from_iceberg("s3://warehouse/my_table",
                            access_key_id="KEY",
                            secret_access_key="SECRET")

# Delta Lake (read-only)
ds = DataStore.from_delta("s3://bucket/delta_table",
                          access_key_id="KEY",
                          secret_access_key="SECRET")

# Apache Hudi (read-only)
ds = DataStore.from_hudi("s3://bucket/hudi_table",
                         access_key_id="KEY",
                         secret_access_key="SECRET")
```

### Data Generation
```python
# Generate number sequences
ds = DataStore.from_numbers(100)  # 0 to 99
ds = DataStore.from_numbers(10, start=10)  # 10 to 19
ds = DataStore.from_numbers(10, start=0, step=2)  # Even numbers

# Generate random data for testing
ds = DataStore.from_random(
    structure="id UInt32, name String, value Float64",
    random_seed=42,
    max_string_length=20
)
```

### URL/HTTP
```python
ds = DataStore.from_url("https://example.com/data.json",
                        format="JSONEachRow")
```

### Multi-Source Queries
```python
# Join data from different sources
csv_data = DataStore.from_file("sales.csv", format="CSV")
mysql_data = DataStore.from_mysql("localhost:3306", "mydb", "customers",
                                  user="root", password="pass")

result = (mysql_data
    .join(csv_data, left_on="id", right_on="customer_id")
    .select("name", "product", "revenue")
    .filter(csv_data.date >= '2024-01-01')
    .execute())
```

### Format Settings

Optimize performance with format-specific settings:

```python
# CSV settings
ds = DataStore.from_file("data.csv", format="CSV")
ds = ds.with_format_settings(
    format_csv_delimiter=',',
    input_format_csv_skip_first_lines=1,
    input_format_csv_trim_whitespaces=1
)

# Parquet optimization
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)
ds = ds.with_format_settings(
    input_format_parquet_filter_push_down=1,
    input_format_parquet_bloom_filter_push_down=1
)

# JSON settings
ds = DataStore.from_file("data.json", format="JSONEachRow")
ds = ds.with_format_settings(
    input_format_json_validate_types_from_metadata=1,
    input_format_json_ignore_unnecessary_fields=1
)
```

## Design Philosophy

DataStore is inspired by pypika's excellent query builder design but focuses on:

1. **High-level API**: Pandas-like interface for data scientists
2. **Query Execution**: Built-in execution capabilities (not just SQL generation)
3. **Data Source Abstraction**: Unified interface across different backends
4. **Modern Python**: Type hints, dataclasses, and Python 3.7+ features


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
- [x] ClickHouse table functions and formats support
- [ ] DataFrame operations (drop, assign, fillna, etc.)
- [ ] Query executors
- [ ] Multiple backend support
- [ ] Mock data support
- [ ] Schema management(infer or set manually)
- [ ] ClickHouse functions support
- [ ] Connection managers
- [ ] Image, Video, Audio data support
- [ ] PyTorch DataLoader integration

## Examples

For more comprehensive examples, see:

- **[examples/examples_table_functions.py](examples/examples_table_functions.py)** - Complete examples for all data sources including:
  - Local files (CSV, Parquet, JSON, ORC, Avro and [80+ formats](https://clickhouse.com/docs/interfaces/formats))
  - Cloud storage (S3, Azure, GCS, HDFS, HTTP and [20+ protocols](https://clickhouse.com/docs/integrations/data-sources/index))
  - Databases (MySQL, PostgreSQL, MongoDB, Redis, SQLite, ClickHouse)
  - Data lakes (Iceberg, Delta Lake, Hudi)
  - Data generation (numbers, random data)
  - Multi-source joins
  - Format-specific optimization settings

## License

Apache License 2.0

## Credits

Built with and inspired by:
- [chDB](https://github.com/chdb-io/chdb) - Embedded ClickHouse engine for Python
- [ClickHouse](https://clickhouse.com/) - Fast open-source OLAP database
- [Pandas](https://pandas.pydata.org/) - DataFrame API design
- [PyPika](https://github.com/kayak/pypika) - Query builder patterns
- [SQLAlchemy](https://www.sqlalchemy.org/) - ORM and query builder concepts

