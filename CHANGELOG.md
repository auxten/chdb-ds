# Changelog

All notable changes to DataStore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Extended Pandas DataFrame API Compatibility**: DataStore now supports 180+ pandas DataFrame methods and properties (increased from ~80)
  - All statistical methods: `mean()`, `median()`, `std()`, `var()`, `min()`, `max()`, `sum()`, `count()`, `corr()`, `cov()`, etc.
  - Data manipulation: `drop()`, `rename()`, `sort_values()`, `fillna()`, `dropna()`, `drop_duplicates()`, `assign()`, etc.
  - Indexing and selection: `loc`, `iloc`, `at`, `iat`, column selection with `[]`
  - Reshaping: `pivot()`, `pivot_table()`, `melt()`, `stack()`, `unstack()`, `transpose()`
  - Combining: `merge()`, `concat()`, `append()`
  - Time series: `resample()`, `rolling()`, `expanding()`, `ewm()`
  - Export: `to_csv()`, `to_json()`, `to_excel()`, `to_parquet()`, `to_html()`, etc.
  - Properties: `shape`, `columns`, `dtypes`, `values`, `size`, `empty`, `ndim`, `T`
- Comprehensive pandas compatibility layer (`PandasCompatMixin`) for seamless integration
- Internal DataFrame caching for improved performance
- Support for mixing SQL-style query building with pandas operations
- **New Method Categories**:
  - 26+ binary operator methods (add, sub, mul, div + all reverse versions + comparison operators)
  - 15+ reindexing methods (add_prefix, add_suffix, align, reindex, take, truncate, etc.)
  - 10+ time series methods (asfreq, asof, shift, tz_convert, tz_localize, etc.)
  - 8+ reshaping methods (droplevel, swaplevel, swapaxes, squeeze, etc.)
  - 10+ advanced selection methods (query, where, mask, isin, xs, get, select_dtypes, etc.)
  - 4+ missing data methods (bfill, ffill, backfill, pad)
  - 4+ additional IO methods (to_hdf, to_stata, to_gbq, to_orc)
- Subquery support in WHERE, FROM, and SELECT clauses
- INSERT statement support (ClickHouse style)
  - INSERT INTO ... VALUES
  - INSERT INTO ... SELECT
- UPDATE statement support (ClickHouse style: ALTER TABLE ... UPDATE)
- DELETE statement support (ClickHouse style: ALTER TABLE ... DELETE)
- Comprehensive test coverage for new features
- Execution tests with chdb backend

### Changed
- **Breaking Change**: Renamed `values()` method to `insert_values()` to avoid conflict with pandas `values` property
  - Old: `ds.insert_into('id', 'name').values(1, 'Alice')`
  - New: `ds.insert_into('id', 'name').insert_values(1, 'Alice')`
- Enhanced `__getitem__` to support pandas-style column selection: `ds['column']` and `ds[['col1', 'col2']]`
- **Improved `_wrap_result()`**: Now correctly handles Series by returning them as-is (not wrapping)
  - DataFrame methods return DataStore (wrapped)
  - Series methods return Series (pandas semantics preserved)
  - This maintains pandas API expectations (e.g., `df['column']` returns Series)
- **Mixed Execution Engine**: Revolutionary execution model supporting **arbitrary mixing** of SQL and pandas operations
  - SQL operations build queries lazily (no execution)
  - First pandas operation triggers SQL execution and executes result
  - **SQL operations after execution use chDB's `Python()` table function** to execute SQL on cached DataFrame
  - Subsequent pandas operations work on cached DataFrame
  - Enables patterns like: SQL → Pandas → SQL → Pandas → SQL (any order!)
  - `to_df()` respects execution state (returns cached df when appropriate)
  - Fixes critical issue where `ds.add_prefix("x_").to_df()` now works correctly
  - See [Mixed Execution Engine Guide](docs/MIXED_EXECUTION_ENGINE.md) for details

### Documentation
- Added comprehensive [Pandas Compatibility Guide](docs/PANDAS_COMPATIBILITY.md) with 180+ method checklist
- Added [Mixed Execution Engine Guide](docs/MIXED_EXECUTION_ENGINE.md) explaining the execution model
- Updated README with extended pandas compatibility examples
- Added `example_pandas_compat.py` demonstrating common pandas methods
- Added `example_pandas_extended.py` showcasing binary operators, time series, and advanced methods
- Added extensive test coverage:
  - 53 tests for pandas compatibility
  - 12 tests for basic mixed operations
  - 29 tests for complex mixed scenarios
  - Total: **94 tests** for pandas/mixed operations (all passing)

## [0.1.0] - 2025-01-XX

### Added
- Initial release of DataStore
- Core expression system
  - Field expressions
  - Literal values
  - Arithmetic operations (+, -, *, /, %)
- Condition system
  - Binary conditions (=, !=, >, >=, <, <=)
  - Compound conditions (AND, OR, XOR)
  - Unary conditions (IS NULL, IS NOT NULL)
  - IN/NOT IN conditions
  - BETWEEN conditions
  - LIKE/ILIKE conditions
- SQL function support
  - Aggregate functions (SUM, COUNT, AVG, MIN, MAX)
  - String functions (UPPER, LOWER, CONCAT)
  - Custom function support
- DataStore query building
  - SELECT with field selection
  - WHERE clause with conditions
  - JOIN operations (INNER, LEFT, RIGHT, OUTER, CROSS)
  - GROUP BY and HAVING
  - ORDER BY with ASC/DESC
  - LIMIT and OFFSET
  - DISTINCT
- Immutable query operations using @immutable decorator
- chdb integration for ClickHouse backend
- Connection and executor abstractions
- Comprehensive test suite (500+ tests)
- Documentation and examples

### Core Features
- Fluent, Pandas-like API
- Automatic SQL generation
- Method chaining with immutability
- Type hints throughout
- Extensible architecture

[Unreleased]: https://github.com/auxten/chdb-ds/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/auxten/chdb-ds/releases/tag/v0.1.0

