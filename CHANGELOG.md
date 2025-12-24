# Changelog

All notable changes to DataStore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Design Principles
- **API Style Independence**: Both pandas-style and fluent SQL-style APIs should compile to the same optimized SQL. API style must not determine the execution engine.
- **Fully Lazy Execution**: All operations return lazy objects; execution is triggered naturally through `.values`, `.index`, `repr()`, etc.
- **Automatic Engine Selection**: The execution engine (chDB vs pandas) is determined at execution time based on operation type and configuration, not API style.

### Added
- **Full Pandas DataFrame API Compatibility**: DataStore now supports **209 pandas DataFrame methods**, **56 str accessor methods**, and **42+ dt accessor methods**
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
- **Profiling Capabilities**: Built-in performance profiling with `Profiler` and `ProfileStep` classes
  - `enable_profiling()` / `disable_profiling()` to control profiling
  - `get_profiler()` to retrieve profiling data and generate reports
  - Track execution timing across SQL execution, cache operations, and DataFrame operations
- **LazySeries**: New lazy evaluation wrapper for Series method calls
  - Replaces `LazySlice` with unified `LazySeries` for all deferred Series operations
  - Enables lazy evaluation for `head()`, `tail()`, comparison operators, and more
- **LazyCondition**: Dual SQL and pandas support for conditions
  - `isin()`, `between()` now work in both SQL filtering and pandas boolean Series contexts
- **LazyGroupBy**: Improved groupby implementation
  - Returns `LazySeries` for `size()` method
  - SQL pushdown optimizations for groupby aggregations
  - Proper `ORDER BY` / `sort_index()` for pandas-compatible ordering
- **SQL Pushdown Optimizations**: Enhanced query optimization
  - Groupby operations pushed to SQL when possible
  - Introduced `Star` expression for `COUNT(*)` operations
- **DataFrame Interchange Protocol**: `__dataframe__` method for library interoperability
  - Direct use with seaborn, plotly, and other visualization libraries
- **to_pandas() Method**: Added to `DataStore`, `ColumnExpr`, and `LazySeries`
  - Explicit conversion to pandas DataFrame/Series for API consistency
- **Pandas Module-Level Functions**: `pandas_api` module with pandas-compatible functions
  - `read_csv`, `read_json`, `read_excel`, `read_parquet`, etc.
  - `concat`, `merge`, `isna`, `notna`, `to_datetime`, `date_range`, etc.
- **Cumulative and Window Functions**: Added to `ColumnExpr`
  - `cummax()`, `cummin()`, `rolling()`, `expanding()`, `ewm()`
  - `shift()`, `diff()`, `pct_change()`, `rank()`
- **String Accessor Enhancements**
  - Regex support in `str.replace()` and `str.split()`
  - `str[index]` accessor for split results
- **DateTime Enhancements**
  - `DateTimePropertyExpr` and `DateTimeMethodExpr` for lazy datetime operations
  - Automatic string to datetime conversion
- **In-Place Column Operations**
  - `__delitem__()` for column deletion: `del ds['column']`
  - `pop()` method for removing and returning columns
  - `update()` method for in-place modifications
- **NumPy Compatibility Enhancements**
  - Improved `__array__` method with `dtype` and `copy` parameters (NumPy 2.0+)
  - Better handling of categorical and extension dtypes
- **Kaggle Pandas Compatibility Test Suite**: Comprehensive tests based on common Kaggle notebook patterns

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
  - See [Explain Method](docs/EXPLAIN_METHOD.md) for execution plan visualization
- **Terminology Update**: Changed "materialization" to "execution" across the codebase for clarity
- **Execution Engine Renamed**: `ExecutionEngine.CLICKHOUSE` → `ExecutionEngine.CHDB`, `use_clickhouse()` → `use_chdb()`
- **ColumnExpr Unified**: Comparison operators now return `ColumnExpr` wrapping `Condition` instead of separate `BoolColumnExpr`
- **Nested Subquery Support**: Enhanced SQL query building for complex LIMIT-FILTER patterns

### Documentation
- Added comprehensive [Pandas Compatibility Guide](docs/PANDAS_COMPATIBILITY.md) with 209 method checklist (100% coverage)
- Added [Profiling Guide](docs/PROFILING.md) for performance analysis
- Added [Explain Method](docs/EXPLAIN_METHOD.md) for execution plan visualization
- Updated README with extended pandas compatibility examples
- Added `example_pandas_compat.py` demonstrating common pandas methods
- Added `example_pandas_extended.py` showcasing binary operators, time series, and advanced methods
- Added `example_profiling.py` demonstrating profiling usage
- Added Kaggle pandas compatibility test suite (`kaggle_pandas_compat_test.py`)
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

