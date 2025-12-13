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
- **Wide Pandas Compatibility**: 180+ pandas DataFrame methods and properties
- **Mixed Execution Engine**: Arbitrary mixing of SQL(chDB) and pandas operations
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

### Simplest Way: URI-based Creation (Recommended)

The easiest way to create a DataStore is using a URI string. The source type and format are automatically inferred:

```python
from datastore import DataStore

# Local files - format auto-detected from extension
ds = DataStore.uri("/path/to/data.csv")
ds.connect()
result = ds.select("*").filter(ds.age > 18).execute()

# S3 with anonymous access
ds = DataStore.uri("s3://bucket/data.parquet?nosign=true")
result = ds.select("*").limit(10).execute()

# MySQL with connection string
ds = DataStore.uri("mysql://root:pass@localhost:3306/mydb/users")
result = ds.select("*").filter(ds.active == True).execute()

# PostgreSQL
ds = DataStore.uri("postgresql://user:pass@localhost:5432/mydb/products")
result = ds.select("*").execute()

# Google Cloud Storage
ds = DataStore.uri("gs://bucket/data.parquet")

# Azure Blob Storage
ds = DataStore.uri("az://container/blob.csv?account_name=NAME&account_key=KEY")
```

**Supported URI formats:**
- Local files: `file:///path/to/data.csv` or `/path/to/data.csv`
- S3: `s3://bucket/key`
- Google Cloud Storage: `gs://bucket/path`
- Azure Blob Storage: `az://container/blob`
- HDFS: `hdfs://namenode:port/path`
- HTTP/HTTPS: `https://example.com/data.json`
- MySQL: `mysql://user:pass@host:port/database/table`
- PostgreSQL: `postgresql://user:pass@host:port/database/table`
- MongoDB: `mongodb://user:pass@host:port/database.collection`
- SQLite: `sqlite:///path/to/db.db?table=tablename`
- ClickHouse: `clickhouse://host:port/database/table`
- Delta Lake: `deltalake:///path/to/table`
- Apache Iceberg: `iceberg://catalog/namespace/table`
- Apache Hudi: `hudi:///path/to/table`

### Traditional Way: Factory Methods

You can also use dedicated factory methods for more control:

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

# exec() is an alias for execute() - use whichever you prefer
result = query.exec()  # Same as execute()
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

### ClickHouse SQL Functions

DataStore provides 100+ ClickHouse SQL functions through Pandas-like accessors:

```python
# String functions via .str accessor
ds['name'].str.upper()              # upper(name)
ds['name'].str.length()             # length(name)
ds['text'].str.replace('old', 'new') # replace(text, 'old', 'new')
ds['email'].str.contains('@')       # position(email, '@') > 0

# DateTime functions via .dt accessor
ds['date'].dt.year                  # toYear(date)
ds['date'].dt.month                 # toMonth(date)
ds['date'].dt.add_days(7)           # addDays(date, 7)
ds['start'].dt.days_diff(ds['end']) # dateDiff('day', start, end)

# Math functions as expression methods
ds['value'].abs()                   # abs(value)
ds['price'].round(2)                # round(price, 2)
ds['value'].sqrt()                  # sqrt(value)

# Type conversion
ds['value'].cast('Float64')         # CAST(value AS Float64)
ds['id'].to_string()                # toString(id)

# Aggregate functions
ds['amount'].sum()                  # sum(amount)
ds['price'].avg()                   # avg(price)
ds['user_id'].count_distinct()      # uniq(user_id)

# Column assignment with functions
ds['upper_name'] = ds['name'].str.upper()
ds['age_group'] = ds['age'] // 10 * 10
```

**See [Function Reference](docs/FUNCTIONS.md) for the complete list of 100+ functions.**

### Working with Results

DataStore provides convenient methods to get results as pandas DataFrames or dictionaries:

```python
# Get results as DataFrame (simplified)
df = ds.select("*").filter(ds.age > 18).to_df()

# Get results as list of dictionaries (simplified)
records = ds.select("*").filter(ds.age > 18).to_dict()

# Traditional way (also supported)
result = ds.select("*").execute()
df = result.to_df()
records = result.to_dict()

# Access raw result metadata
result = ds.select("*").execute()
print(result.column_names)  # ['id', 'name', 'age']
print(result.row_count)     # 42
print(result.rows)          # List of tuples
```

### Pandas DataFrame Compatibility

DataStore now includes **wide pandas DataFrame API compatibility**, allowing you to use all pandas methods directly:

```python
# All pandas properties work
print(ds.shape)        # (rows, columns)
print(ds.columns)      # Column names
print(ds.dtypes)       # Data types
print(ds.values)       # NumPy array

# All pandas statistical methods
ds.mean()              # Mean values
ds.median()            # Median values
ds.std()               # Standard deviation
ds.corr()              # Correlation matrix
ds.describe()          # Statistical summary

# All pandas data manipulation methods
ds.drop(columns=['col1'])
ds.rename(columns={'old': 'new'})
ds.sort_values('column', ascending=False)
ds.fillna(0)
ds.dropna()
ds.drop_duplicates()
ds.assign(new_col=lambda x: x['col1'] * 2)

# Advanced operations
ds.pivot_table(values='sales', index='region', columns='product')
ds.melt(id_vars=['id'], value_vars=['col1', 'col2'])
ds.merge(other_ds, on='id', how='left')
ds.groupby('category').agg({'amount': 'sum', 'count': 'count'})

# Column selection (pandas style)
ds['column']           # Single column
ds[['col1', 'col2']]   # Multiple columns

# Convenience methods
first_5 = ds.head()      # First 5 rows
last_5 = ds.tail()       # Last 5 rows
sample = ds.sample(n=100, random_state=42)

# Mix SQL-style and pandas operations - arbitrary order!
result = (ds
    .select('*')
    .filter(ds.price > 100)              # SQL-style filtering
    .assign(revenue=lambda x: x['price'] * x['quantity'])  # Pandas assign
    .filter(ds.revenue > 1000)           # SQL on DataFrame!
    .add_prefix('sales_')                # Pandas transform
    .query('sales_revenue > 5000')       # Pandas query
    .select('sales_id', 'sales_revenue'))  # SQL on DataFrame again!

# Export to various formats
ds.to_csv('output.csv')
ds.to_json('output.json')
ds.to_parquet('output.parquet')
ds.to_excel('output.xlsx')
```

**See [Pandas Compatibility Guide](docs/PANDAS_COMPATIBILITY.md) for the complete feature checklist and examples.**

### Conditions

```python
# Simple conditions
ds.filter(ds.age > 18)
ds.filter(ds.status == "active")

# where() is an alias for filter() - use whichever you prefer
ds.where(ds.age > 18)  # Same as filter()

# Complex conditions
ds.filter(
    ((ds.age > 18) & (ds.age < 65)) | 
    (ds.status == "premium")
)

# Negation
ds.filter(~(ds.deleted == True))
```

## Supported Data Sources

DataStore supports 20+ data sources through a unified interface:

| Category | Sources | Quick Example |
|----------|---------|---------------|
| **Local Files** | CSV, Parquet, JSON, ORC, Avro<br/>[+ 80 more formats](https://clickhouse.com/docs/interfaces/formats) | `DataStore.from_file("data.csv")` |
| **Cloud Storage** | S3, GCS, Azure Blob, HDFS | `DataStore.from_s3("s3://bucket/data.parquet")` |
| **Databases** | MySQL, PostgreSQL, ClickHouse,<br/>MongoDB, SQLite, Redis | `DataStore.from_mysql(host, db, table)` |
| **Data Lakes** | Apache Iceberg, Delta Lake, Hudi | `DataStore.from_delta("s3://bucket/table")` |
| **Other** | HTTP/HTTPS, Number generation,<br/>Random data | `DataStore.from_url("https://...")` |

### Quick Examples

**Local Files** (auto-detects format):
```python
ds = DataStore.from_file("data.parquet")
ds = DataStore.from_file("data.csv")
ds = DataStore.from_file("data.json")
```

**Cloud Storage**:
```python
# S3 with public access
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)

# S3 with credentials
ds = DataStore.from_s3("s3://bucket/*.csv",
                       access_key_id="KEY",
                       secret_access_key="SECRET")

# Google Cloud Storage
ds = DataStore.from_gcs("gs://bucket/data.parquet")

# Azure Blob Storage
ds = DataStore.from_azure(container="mycontainer",
                          path="data/*.parquet",
                          connection_string="...")
```

**Databases**:
```python
# MySQL
ds = DataStore.from_mysql("localhost:3306", "mydb", "users",
                          user="root", password="pass")

# PostgreSQL
ds = DataStore.from_postgresql("localhost:5432", "mydb", "users",
                               user="postgres", password="pass")

# ClickHouse (remote)
ds = DataStore.from_clickhouse("localhost:9000", "default", "events")
```

**Data Generation** (for testing):
```python
# Number sequence
ds = DataStore.from_numbers(100)  # 0-99

# Random data
ds = DataStore.from_random(
    structure="id UInt32, name String, value Float64",
    random_seed=42
)
```

📖 **For comprehensive examples of all data sources, see [examples/examples_table_functions.py](examples/examples_table_functions.py)**

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

# Simplified join syntax with USING (when column names match)
users = DataStore.from_file("users.csv")
orders = DataStore.from_file("orders.csv")
products = DataStore.from_file("products.csv")

# Chain multiple joins easily - no table prefix needed!
result = (users
    .join(orders, on="user_id")           # USING (user_id)
    .join(products, on="product_id")      # USING (product_id)
    .select("name", "amount", "product_name")
    .to_df())

# Also supports multiple columns
ds.join(other, on=["user_id", "country"])  # USING (user_id, country)
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

### Alpha release
- [x] Core expression system
- [x] Condition system
- [x] Function system
- [x] Basic DataStore operations
- [x] Immutability support
- [x] ClickHouse table functions and formats support
- [x] DataFrame operations (drop, assign, fillna, etc.) see [Pandas Compatibility Guide](docs/PANDAS_COMPATIBILITY.md)
- [x] Query executors
- [x] ClickHouse SQL functions support (100+ functions via `.str`, `.dt` accessors) see [Function Reference](docs/FUNCTIONS.md)
- [x] Hybrid execution engine (configurable chDB/Pandas execution)
- [ ] Function args completion
- [ ] Update and Save back data
- [ ] Chart support
- [ ] More data exploration functions, faster describe()
- [ ] Multiple backend support
- [ ] Mock data support
- [ ] Schema management(infer or set manually)
- [ ] Connection managers

### Beta release
- [ ] Unstructured data support(Images, Audios as a column)
- [ ] Arrow Table support (read/write directly)
- [ ] Embedding Generation support
- [ ] PyTorch DataLoader integration
- [ ] Python native UDFs support
- [ ] Hybrid Execution (Local and Remote)

## Documentation

- **[Function Reference](docs/FUNCTIONS.md)** - Complete list of 100+ ClickHouse SQL functions with examples
- **[Pandas Compatibility Guide](docs/PANDAS_COMPATIBILITY.md)** - 180+ pandas DataFrame methods and properties

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

