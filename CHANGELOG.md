# Changelog

All notable changes to DataStore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Subquery support in WHERE, FROM, and SELECT clauses
- INSERT statement support (ClickHouse style)
  - INSERT INTO ... VALUES
  - INSERT INTO ... SELECT
- UPDATE statement support (ClickHouse style: ALTER TABLE ... UPDATE)
- DELETE statement support (ClickHouse style: ALTER TABLE ... DELETE)
- Comprehensive test coverage for new features
- Execution tests with chdb backend

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

