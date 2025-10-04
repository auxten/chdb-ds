"""
Core DataStore class - main entry point for data operations
"""

from typing import Any, Optional, List, Dict, Union
from copy import copy

from .expressions import Field, Expression, Literal
from .conditions import Condition
from .utils import immutable, ignore_copy, format_identifier
from .exceptions import DataStoreError, QueryError, ConnectionError, ExecutionError
from .connection import Connection, QueryResult
from .executor import Executor

__all__ = ['DataStore']


class DataStore:
    """
    DataStore - Pandas-like data manipulation with SQL generation.

    Example:
        >>> ds = DataStore("file", path="data.parquet")
        >>> ds.connect()
        >>> result = ds.select("name", "age").filter(ds.age > 18).execute()
    """

    def __init__(
        self,
        source_type: str = None,
        table: str = None,
        database: str = ":memory:",
        connection: Connection = None,
        **kwargs,
    ):
        """
        Initialize DataStore.

        Args:
            source_type: Type of data source ('chdb', 'clickhouse', etc.)
            table: Table name
            database: Database path (":memory:" for in-memory, or file path)
            connection: Existing Connection object (creates new if None)
            **kwargs: Additional connection parameters
        """
        self.source_type = source_type or 'chdb'
        self.table_name = table
        self.database = database
        self.connection_params = kwargs

        # Query state
        self._select_fields: List[Expression] = []
        self._where_condition: Optional[Condition] = None
        self._joins: List[tuple] = []  # [(table/datastore, join_type, on_condition), ...]
        self._groupby_fields: List[Expression] = []
        self._having_condition: Optional[Condition] = None
        self._orderby_fields: List[tuple] = []  # [(field, ascending), ...]
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._distinct: bool = False

        # INSERT/UPDATE/DELETE state
        self._insert_columns: List[str] = []
        self._insert_values: List[List[Any]] = []
        self._insert_select: Optional['DataStore'] = None
        self._update_fields: List[tuple] = []  # [(field, value), ...]
        self._delete_flag: bool = False

        # Subquery support
        self._alias: Optional[str] = None
        self._is_subquery: bool = False

        # Connection and execution
        self._connection: Optional[Connection] = connection
        self._executor: Optional[Executor] = None
        self._schema: Optional[Dict[str, str]] = None

        # Configuration
        self.is_immutable = True
        self.quote_char = '"'

    # ========== Data Source Operations ==========

    def connect(self) -> 'DataStore':
        """
        Connect to the data source using chdb.

        Returns:
            self for chaining
        """
        if self._connection is None:
            self._connection = Connection(self.database, **self.connection_params)

        try:
            self._connection.connect()
            self._executor = Executor(self._connection)

            # Try to get schema if table exists
            if self.table_name:
                self._discover_schema()

            return self
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")

    def _discover_schema(self):
        """Discover table schema from chdb."""
        if not self._executor or not self.table_name:
            return

        try:
            # Query system tables for schema info
            sql = f"DESCRIBE TABLE {format_identifier(self.table_name, self.quote_char)}"
            result = self._executor.execute(sql)

            # Build schema dictionary
            self._schema = {}
            for row in result.rows:
                # ClickHouse DESCRIBE returns: (name, type, default_type, default_expression, comment, ...)
                col_name = row[0]
                col_type = row[1]
                self._schema[col_name] = col_type
        except:
            # Table might not exist yet, that's ok
            self._schema = {}

    def execute(self) -> QueryResult:
        """
        Execute the query and return results.

        Returns:
            QueryResult object with data and metadata
        """
        # Ensure we're connected
        if self._executor is None:
            self.connect()

        # Generate SQL
        sql = self.to_sql()

        try:
            return self._executor.execute(sql)
        except Exception as e:
            raise ExecutionError(f"Query execution failed: {e}")

    def create_table(self, schema: Dict[str, str], engine: str = "Memory") -> 'DataStore':
        """
        Create a table in chdb.

        Args:
            schema: Dictionary of column_name -> column_type
            engine: ClickHouse table engine (default: Memory for in-memory)

        Example:
            >>> ds = DataStore(table="users")
            >>> ds.connect()
            >>> ds.create_table({"id": "UInt64", "name": "String", "age": "UInt8"})

        Returns:
            self for chaining
        """
        if not self.table_name:
            raise ValueError("Table name required to create table")

        if self._executor is None:
            self.connect()

        # Build CREATE TABLE statement
        columns = ", ".join([f"{format_identifier(name, self.quote_char)} {dtype}" for name, dtype in schema.items()])

        sql = f"CREATE TABLE IF NOT EXISTS {format_identifier(self.table_name, self.quote_char)} ({columns}) ENGINE = {engine}"

        self._executor.execute(sql)
        self._schema = schema

        return self

    def insert(self, data: List[Dict[str, Any]] = None, **columns) -> 'DataStore':
        """
        Insert data into the table (executes immediately).

        Args:
            data: List of dictionaries with column_name -> value
            **columns: Alternative way to specify columns (for single row)

        Example:
            >>> ds.insert([{"id": 1, "name": "Alice", "age": 25}])
            >>> ds.insert(id=1, name="Alice", age=25)

        Returns:
            self for chaining
        """
        if not self.table_name:
            raise ValueError("Table name required to insert data")

        # Handle single row via keyword arguments
        if columns and not data:
            data = [columns]

        if not data:
            return self

        if self._executor is None:
            self.connect()

        # Get column names from first row
        columns = list(data[0].keys())
        columns_sql = ", ".join([format_identifier(col, self.quote_char) for col in columns])

        # Build values
        values_list = []
        for row in data:
            values = []
            for col in columns:
                val = row.get(col)
                if val is None:
                    values.append("NULL")
                elif isinstance(val, str):
                    # Escape single quotes
                    escaped = val.replace("'", "''")
                    values.append(f"'{escaped}'")
                elif isinstance(val, bool):
                    values.append("1" if val else "0")
                else:
                    values.append(str(val))
            values_list.append(f"({', '.join(values)})")

        values_sql = ", ".join(values_list)
        sql = f"INSERT INTO {format_identifier(self.table_name, self.quote_char)} ({columns_sql}) VALUES {values_sql}"

        self._executor.execute(sql)

        return self

    def close(self):
        """Close the connection."""
        if self._executor:
            self._executor.close()
            self._executor = None
        if self._connection:
            self._connection.close()
            self._connection = None

    # ========== INSERT/UPDATE/DELETE Query Building ==========

    @immutable
    def insert_into(self, *columns: str) -> 'DataStore':
        """
        Start building an INSERT query (ClickHouse style).

        Args:
            *columns: Column names to insert into

        Example:
            >>> ds.insert_into('id', 'name', 'age').values(1, 'Alice', 25)
            >>> ds.insert_into('id', 'name').select_from(other_ds.select('id', 'name'))

        Returns:
            DataStore with INSERT query state
        """
        self._insert_columns = list(columns)

    @immutable
    def values(self, *rows) -> 'DataStore':
        """
        Add VALUES clause to INSERT query.

        Args:
            *rows: Each row can be a tuple/list or individual values

        Example:
            >>> ds.insert_into('id', 'name').values((1, 'Alice'), (2, 'Bob'))
            >>> ds.insert_into('id', 'name').values(1, 'Alice').values(2, 'Bob')

        Returns:
            DataStore for chaining
        """
        if not self._insert_columns:
            raise QueryError("Must call insert_into() before values()")

        # Handle different input formats
        if len(rows) == 1 and isinstance(rows[0], (list, tuple)):
            # Single row as tuple: values((1, 'Alice'))
            self._insert_values.append(list(rows[0]))
        elif len(rows) > 1 and all(isinstance(r, (list, tuple)) for r in rows):
            # Multiple rows: values((1, 'Alice'), (2, 'Bob'))
            for row in rows:
                self._insert_values.append(list(row))
        else:
            # Individual values: values(1, 'Alice')
            self._insert_values.append(list(rows))

    @immutable
    def select_from(self, subquery: 'DataStore') -> 'DataStore':
        """
        Add SELECT subquery to INSERT query (INSERT INTO ... SELECT ...).

        Args:
            subquery: DataStore representing the SELECT query

        Example:
            >>> ds.insert_into('id', 'name').select_from(
            ...     other_ds.select('user_id', 'user_name').filter(other_ds.active == True)
            ... )

        Returns:
            DataStore for chaining
        """
        if not self._insert_columns:
            raise QueryError("Must call insert_into() before select_from()")

        self._insert_select = subquery

    @immutable
    def update_set(self, **fields) -> 'DataStore':
        """
        Build an UPDATE query (ClickHouse style: ALTER TABLE ... UPDATE ...).

        Args:
            **fields: Field-value pairs to update

        Example:
            >>> ds.update_set(age=26, city='NYC').filter(ds.id == 1)
            >>> # Generates: ALTER TABLE table UPDATE age=26, city='NYC' WHERE id=1

        Returns:
            DataStore for chaining
        """
        for field_name, value in fields.items():
            self._update_fields.append((field_name, value))

    @immutable
    def delete_rows(self) -> 'DataStore':
        """
        Build a DELETE query (ClickHouse style: ALTER TABLE ... DELETE).

        Example:
            >>> ds.delete_rows().filter(ds.age < 18)
            >>> # Generates: ALTER TABLE table DELETE WHERE age < 18

        Returns:
            DataStore for chaining
        """
        self._delete_flag = True

    # ========== Query Building Methods ==========

    @immutable
    def select(self, *fields: Union[str, Expression]) -> 'DataStore':
        """
        Select specific columns.

        Args:
            *fields: Column names (strings) or Expression objects

        Example:
            >>> ds.select("name", "age")
            >>> ds.select(ds.name, ds.age + 1)
        """
        for field in fields:
            if isinstance(field, str):
                # Special case: "*" means SELECT *
                if field == "*":
                    # Clear existing fields and set to empty (will render as *)
                    self._select_fields = []
                    return
                # Don't add table prefix for string fields - user's explicit choice
                field = Field(field)
            self._select_fields.append(field)

    @immutable
    def filter(self, condition: Union[Condition, str]) -> 'DataStore':
        """
        Filter rows (WHERE clause).

        Args:
            condition: Condition object or SQL string

        Example:
            >>> ds.filter(ds.age > 18)
            >>> ds.filter((ds.age > 18) & (ds.city == 'NYC'))
        """
        if isinstance(condition, str):
            # TODO: Parse string conditions
            raise NotImplementedError("String conditions not yet implemented")

        if self._where_condition is None:
            self._where_condition = condition
        else:
            # Combine with existing condition using AND
            self._where_condition = self._where_condition & condition

    @immutable
    def join(
        self, other: 'DataStore', on: Condition = None, how: str = 'inner', left_on: str = None, right_on: str = None
    ) -> 'DataStore':
        """
        Join with another DataStore.

        Args:
            other: Another DataStore to join with
            on: Join condition (e.g., ds1.id == ds2.user_id)
            how: Join type ('inner', 'left', 'right', 'outer', 'cross')
            left_on: Column name from left table (alternative to on)
            right_on: Column name from right table (alternative to on)

        Example:
            >>> ds1.join(ds2, on=ds1.id == ds2.user_id)
            >>> ds1.join(ds2, left_on='id', right_on='user_id', how='left')
        """
        from .enums import JoinType

        # Convert how string to JoinType
        join_type_map = {
            'inner': JoinType.inner,
            'left': JoinType.left,
            'right': JoinType.right,
            'outer': JoinType.outer,
            'full': JoinType.full_outer,
            'cross': JoinType.cross,
        }

        if how.lower() not in join_type_map:
            raise QueryError(f"Invalid join type: {how}")

        join_type = join_type_map[how.lower()]

        # Build join condition
        if on is not None:
            join_condition = on
        elif left_on and right_on:
            # Create condition from column names
            left_field = Field(left_on, table=self.table_name)
            right_field = Field(right_on, table=other.table_name)
            join_condition = left_field == right_field
        else:
            raise QueryError("Either 'on' or both 'left_on' and 'right_on' must be specified")

        self._joins.append((other, join_type, join_condition))

    @immutable
    def groupby(self, *fields: Union[str, Expression]) -> 'DataStore':
        """
        Group by columns.

        Args:
            *fields: Column names (strings) or Expression objects

        Example:
            >>> ds.groupby("category")
            >>> ds.groupby(ds.category, ds.region)
        """
        for field in fields:
            if isinstance(field, str):
                # Don't add table prefix for string fields
                field = Field(field)
            self._groupby_fields.append(field)

    @immutable
    def sort(self, *fields: Union[str, Expression], ascending: bool = True) -> 'DataStore':
        """
        Sort results (ORDER BY clause).

        Args:
            *fields: Column names (strings) or Expression objects
            ascending: Sort direction (default: True)

        Example:
            >>> ds.sort("name")
            >>> ds.sort("price", ascending=False)
            >>> ds.sort(ds.date, ds.amount, ascending=False)
        """
        for field in fields:
            if isinstance(field, str):
                # Don't add table prefix for string fields
                field = Field(field)
            self._orderby_fields.append((field, ascending))

    @immutable
    def limit(self, n: int) -> 'DataStore':
        """Limit number of results."""
        self._limit_value = n

    @immutable
    def offset(self, n: int) -> 'DataStore':
        """Skip first n results."""
        self._offset_value = n

    @immutable
    def distinct(self) -> 'DataStore':
        """
        Add DISTINCT to query.

        Example:
            >>> ds.select("city").distinct()
        """
        self._distinct = True

    @immutable
    def having(self, condition: Union[Condition, str]) -> 'DataStore':
        """
        Add HAVING clause for filtering aggregated results.

        Args:
            condition: Condition object or SQL string

        Example:
            >>> ds.groupby("city").having(Count("*") > 10)
        """
        if isinstance(condition, str):
            raise NotImplementedError("String conditions not yet implemented")

        if self._having_condition is None:
            self._having_condition = condition
        else:
            # Combine with existing condition using AND
            self._having_condition = self._having_condition & condition

    @immutable
    def as_(self, alias: str) -> 'DataStore':
        """
        Set an alias for this DataStore (for use as subquery).

        Args:
            alias: Alias name

        Example:
            >>> subquery = ds.select('id', 'name').as_('sub')
            >>> main_ds.select('*').from_subquery(subquery)

        Returns:
            DataStore for chaining
        """
        self._alias = alias
        self._is_subquery = True

    def __getitem__(self, key: Union[int, slice]) -> 'DataStore':
        """
        Support slice notation for LIMIT and OFFSET.

        Examples:
            >>> ds[:10]          # LIMIT 10
            >>> ds[10:]          # OFFSET 10
            >>> ds[10:20]        # LIMIT 10 OFFSET 10
        """
        new_ds = copy(self)

        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step

            if step is not None:
                raise ValueError("Step not supported in slice notation")

            if stop is not None:
                if start is not None:
                    # ds[start:stop] -> LIMIT (stop-start) OFFSET start
                    new_ds._limit_value = stop - start if stop > start else stop
                    new_ds._offset_value = start
                else:
                    # ds[:stop] -> LIMIT stop
                    new_ds._limit_value = stop
            elif start is not None:
                # ds[start:] -> OFFSET start
                new_ds._offset_value = start
        else:
            raise TypeError("DataStore indices must be slices, not integers")

        return new_ds

    # ========== SQL Generation ==========

    def to_sql(self, quote_char: str = None, as_subquery: bool = False) -> str:
        """
        Generate SQL query.

        Args:
            quote_char: Quote character for identifiers
            as_subquery: Whether to format as subquery with parentheses

        Returns:
            SQL query string
        """
        if quote_char is None:
            quote_char = self.quote_char

        # Handle different query types
        if self._delete_flag:
            sql = self._generate_delete_sql(quote_char)
        elif self._update_fields:
            sql = self._generate_update_sql(quote_char)
        elif self._insert_columns:
            sql = self._generate_insert_sql(quote_char)
        else:
            sql = self._generate_select_sql(quote_char)

        # Add subquery formatting
        if as_subquery or self._is_subquery:
            sql = f"({sql})"
            if self._alias:
                sql = f"{sql} AS {format_identifier(self._alias, quote_char)}"

        return sql

    def _generate_select_sql(self, quote_char: str) -> str:
        """Generate SELECT SQL."""
        parts = []

        # SELECT clause
        if self._select_fields:
            fields_sql = ', '.join(
                field.to_sql(quote_char=quote_char, with_alias=True) for field in self._select_fields
            )
        else:
            fields_sql = '*'

        distinct_keyword = 'DISTINCT ' if self._distinct else ''
        parts.append(f"SELECT {distinct_keyword}{fields_sql}")

        # FROM clause
        if self.table_name:
            parts.append(f"FROM {format_identifier(self.table_name, quote_char)}")

        # JOIN clauses
        if self._joins:
            for other_ds, join_type, join_condition in self._joins:
                # Generate JOIN clause
                join_keyword = join_type.value if join_type.value else ''
                if join_keyword:
                    join_clause = f"{join_keyword} JOIN"
                else:
                    join_clause = "JOIN"

                # Handle subquery joins
                if isinstance(other_ds, DataStore) and other_ds._is_subquery:
                    other_table = other_ds.to_sql(quote_char=quote_char, as_subquery=True)
                else:
                    other_table = format_identifier(other_ds.table_name, quote_char)

                condition_sql = join_condition.to_sql(quote_char=quote_char)

                parts.append(f"{join_clause} {other_table} ON {condition_sql}")

        # WHERE clause
        if self._where_condition:
            where_sql = self._where_condition.to_sql(quote_char=quote_char)
            parts.append(f"WHERE {where_sql}")

        # GROUP BY clause
        if self._groupby_fields:
            groupby_sql = ', '.join(field.to_sql(quote_char=quote_char) for field in self._groupby_fields)
            parts.append(f"GROUP BY {groupby_sql}")

        # HAVING clause
        if self._having_condition:
            having_sql = self._having_condition.to_sql(quote_char=quote_char)
            parts.append(f"HAVING {having_sql}")

        # ORDER BY clause
        if self._orderby_fields:
            orderby_parts = []
            for field, ascending in self._orderby_fields:
                field_sql = field.to_sql(quote_char=quote_char)
                direction = 'ASC' if ascending else 'DESC'
                orderby_parts.append(f"{field_sql} {direction}")
            parts.append(f"ORDER BY {', '.join(orderby_parts)}")

        # LIMIT clause
        if self._limit_value is not None:
            parts.append(f"LIMIT {self._limit_value}")

        # OFFSET clause
        if self._offset_value is not None:
            parts.append(f"OFFSET {self._offset_value}")

        return ' '.join(parts)

    def _generate_insert_sql(self, quote_char: str) -> str:
        """Generate INSERT SQL (ClickHouse style)."""
        if not self.table_name:
            raise QueryError("Table name required for INSERT")

        parts = [f"INSERT INTO {format_identifier(self.table_name, quote_char)}"]

        # Columns
        if self._insert_columns:
            columns_sql = ', '.join(format_identifier(col, quote_char) for col in self._insert_columns)
            parts.append(f"({columns_sql})")

        # VALUES or SELECT
        if self._insert_select:
            # INSERT INTO ... SELECT ...
            select_sql = self._insert_select.to_sql(quote_char=quote_char)
            parts.append(select_sql)
        elif self._insert_values:
            # INSERT INTO ... VALUES ...
            values_parts = []
            for row in self._insert_values:
                row_values = []
                for value in row:
                    if value is None:
                        row_values.append('NULL')
                    elif isinstance(value, bool):
                        row_values.append('1' if value else '0')
                    elif isinstance(value, str):
                        escaped = value.replace("'", "''")
                        row_values.append(f"'{escaped}'")
                    elif isinstance(value, Expression):
                        row_values.append(value.to_sql(quote_char=quote_char))
                    else:
                        row_values.append(str(value))
                values_parts.append(f"({', '.join(row_values)})")
            parts.append(f"VALUES {', '.join(values_parts)}")
        else:
            raise QueryError("INSERT query requires either VALUES or SELECT")

        return ' '.join(parts)

    def _generate_update_sql(self, quote_char: str) -> str:
        """Generate UPDATE SQL (ClickHouse style: ALTER TABLE ... UPDATE ...)."""
        if not self.table_name:
            raise QueryError("Table name required for UPDATE")

        if not self._update_fields:
            raise QueryError("UPDATE query requires at least one field to update")

        parts = [f"ALTER TABLE {format_identifier(self.table_name, quote_char)}"]

        # UPDATE clause
        update_parts = []
        for field_name, value in self._update_fields:
            field_sql = format_identifier(field_name, quote_char)

            if value is None:
                value_sql = 'NULL'
            elif isinstance(value, bool):
                value_sql = '1' if value else '0'
            elif isinstance(value, str):
                escaped = value.replace("'", "''")
                value_sql = f"'{escaped}'"
            elif isinstance(value, Expression):
                value_sql = value.to_sql(quote_char=quote_char)
            else:
                value_sql = str(value)

            update_parts.append(f"{field_sql}={value_sql}")

        parts.append(f"UPDATE {', '.join(update_parts)}")

        # WHERE clause
        if self._where_condition:
            where_sql = self._where_condition.to_sql(quote_char=quote_char)
            parts.append(f"WHERE {where_sql}")

        return ' '.join(parts)

    def _generate_delete_sql(self, quote_char: str) -> str:
        """Generate DELETE SQL (ClickHouse style: ALTER TABLE ... DELETE WHERE ...)."""
        if not self.table_name:
            raise QueryError("Table name required for DELETE")

        parts = [f"ALTER TABLE {format_identifier(self.table_name, quote_char)}"]
        parts.append("DELETE")

        # WHERE clause (required for ClickHouse DELETE)
        if self._where_condition:
            where_sql = self._where_condition.to_sql(quote_char=quote_char)
            parts.append(f"WHERE {where_sql}")
        else:
            raise QueryError("ClickHouse DELETE requires WHERE clause. Use WHERE 1=1 to delete all rows.")

        return ' '.join(parts)

    # ========== Dynamic Field Access ==========

    @ignore_copy
    def __getattr__(self, name: str) -> Field:
        """
        Support dynamic field access: ds.column_name

        Example:
            >>> ds.age > 18  # Same as Field('age') > 18

        Note:
            Dynamic field access does NOT add table prefix by default.
            For single-table queries, this keeps SQL clean.
            For multi-table queries, use Field('name', table='table') explicitly.
        """
        # Avoid infinite recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Don't add table prefix - keep it simple
        return Field(name)

    # ========== Copy Support ==========

    def __copy__(self) -> 'DataStore':
        """Create a shallow copy for immutability."""
        new_ds = type(self).__new__(type(self))
        new_ds.__dict__.update(self.__dict__)

        # Copy mutable collections
        new_ds._select_fields = self._select_fields.copy()
        new_ds._joins = self._joins.copy()
        new_ds._groupby_fields = self._groupby_fields.copy()
        new_ds._orderby_fields = self._orderby_fields.copy()
        new_ds._insert_columns = self._insert_columns.copy()
        new_ds._insert_values = self._insert_values.copy()
        new_ds._update_fields = self._update_fields.copy()

        # Share connection and executor (not copied)
        # Each copy can share the same connection

        return new_ds

    # ========== String Representation ==========

    def __str__(self) -> str:
        return self.to_sql()

    def __repr__(self) -> str:
        return f"DataStore(source_type={self.source_type!r}, table={self.table_name!r})"
