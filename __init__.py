"""
DataStore - A Pandas-like Data Manipulation Framework
======================================================

DataStore provides a high-level API for data manipulation with automatic
query generation and execution capabilities.

Key Features:
- Fluent API similar to Pandas/Polars
- Automatic SQL generation
- Multiple data source support (File, ClickHouse, PostgreSQL, etc.)
- Mock data support for testing
- Immutable operations for thread safety

Example:
    >>> from datastore import DataStore
    >>> ds = DataStore("file", path="data.parquet")
    >>> ds.connect()
    >>> result = ds.select("name", "age").filter(ds.age > 18).execute()

Core Classes:
- DataStore: Main entry point for data operations
- Expression: Base class for all expressions
- Function: SQL function wrapper
- Connection: Database connection abstraction
"""

from .core import DataStore
from .expressions import Expression, Field, Literal
from .functions import (
    Function,
    AggregateFunction,
    CustomFunction,
    # Common functions
    Sum,
    Count,
    Avg,
    Min,
    Max,
    Upper,
    Lower,
    Concat,
)
from .conditions import Condition, BinaryCondition
from .connection import Connection, QueryResult
from .executor import Executor
from .exceptions import (
    DataStoreError,
    ConnectionError,
    SchemaError,
    QueryError,
    ExecutionError,
)
from .enums import JoinType

__version__ = "0.1.0"
__author__ = "DataStore Contributors"

__all__ = [
    # Core
    'DataStore',
    # Expressions
    'Expression',
    'Field',
    'Literal',
    # Functions
    'Function',
    'AggregateFunction',
    'CustomFunction',
    'Sum',
    'Count',
    'Avg',
    'Min',
    'Max',
    'Upper',
    'Lower',
    'Concat',
    # Conditions
    'Condition',
    'BinaryCondition',
    # Enums
    'JoinType',
    # Connection and Execution
    'Connection',
    'QueryResult',
    'Executor',
    # Exceptions
    'DataStoreError',
    'ConnectionError',
    'SchemaError',
    'QueryError',
    'ExecutionError',
]
