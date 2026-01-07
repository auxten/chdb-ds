"""
Centralized xfail markers for DataStore tests.

This module provides a single source of truth for all expected test failures
due to known limitations in chDB, DataStore, or design differences from pandas.

Usage:
    from tests.xfail_markers import chdb_category_type, bug_groupby_first_last

    @chdb_category_type
    def test_something():
        ...

Marker Naming Conventions:
    - chdb_*     : chDB/ClickHouse engine limitations (cannot fix in DataStore)
    - bug_*      : DataStore bugs to be fixed (should match pandas behavior)
    - limit_*    : DataStore limitations (features not yet implemented)
    - design_*   : Intentional behavioral differences from pandas
    - deprecated_*: Deprecated pandas features

When a bug is fixed or limitation is resolved, remove the marker from this file
and update all tests that use it.
"""

from typing import List

import pytest


# =============================================================================
# chDB Engine Limitations (chdb_*)
# Cannot be fixed in DataStore - inherent to chDB/ClickHouse
# =============================================================================

# Type Support
# NOTE: Categorical/Timedelta work for read-only access but fail during SQL execution
chdb_category_type = pytest.mark.xfail(
    reason="chDB does not support CATEGORY numpy type in SQL operations",
    strict=True,
)

chdb_timedelta_type = pytest.mark.xfail(
    reason="chDB does not support TIMEDELTA numpy type in SQL operations",
    strict=True,
)

chdb_array_nullable = pytest.mark.xfail(
    reason="chDB: Array type cannot be inside Nullable type",
    strict=True,
)

# NOTE: numpy arrays work for read-only access but may have issues in SQL operations
chdb_array_string_conversion = pytest.mark.xfail(
    reason="chDB converts numpy arrays to strings via Python() table function",
    strict=True,
)

# FIXED: chDB now handles Nullable Int64 comparison correctly (resolved in recent chDB version)
# chdb_nullable_int64_comparison = pytest.mark.xfail(
#     reason="chDB does not handle Nullable Int64 comparison correctly - returns raw bytes",
#     strict=True,
# )

# Function Limitations
chdb_no_product_function = pytest.mark.xfail(
    reason="chDB does not support product() aggregate function",
    strict=True,
)

chdb_no_normalize_utf8 = pytest.mark.xfail(
    reason="chDB: normalizeUTF8NFD function does not exist",
    strict=True,
)

chdb_no_quantile_array = pytest.mark.xfail(
    reason="chDB does not support quantile with array parameter",
    strict=True,
)

chdb_median_in_where = pytest.mark.xfail(
    reason="Aggregate function median() in WHERE clause requires subquery - not supported",
    strict=True,
)

# NULL/NaN Handling
# NOTE: chdb_null_in_groupby REMOVED - Fixed by implementing dropna parameter support
# in groupby operations. DataStore now properly excludes NULL groups by default (dropna=True)
# and includes them when dropna=False, matching pandas behavior.
# See: tracking/discoveries/2026-01-06_groupby_dropna_alignment_research.md

# NOTE: chdb_nan_sum_behavior REMOVED
# Fixed in column_expr.py _execute_groupby_aggregation() by adding fillna(0)
# for sum aggregation results to match pandas behavior.
# chdb_nan_sum_behavior = pytest.mark.xfail(
#     reason="chDB returns NA for sum of all-NaN, pandas returns 0 (SQL standard behavior, may add workaround in DataStore)",
#     strict=True,
# )

# NOTE: chdb_null_comparison_semantics and chdb_null_string_comparison REMOVED
# Fixed in conditions.py using ifNull() wrapping for pandas NULL semantics

# String/Unicode
chdb_unicode_filter = pytest.mark.xfail(
    reason="Unicode string equality in SQL filter has encoding issues",
    strict=True,
)

chdb_strip_whitespace = pytest.mark.xfail(
    reason="str.strip() doesn't handle all whitespace types correctly in chDB",
    strict=True,
)

# FIXED: String concatenation now auto-converts '+' to concat() in ArithmeticExpression
# chdb_string_plus_operator = pytest.mark.xfail(
#     reason="chDB/ClickHouse does not support '+' operator for string concatenation, must use concat() function",
#     strict=True,
# )

# Datetime
chdb_datetime_timezone = pytest.mark.xfail(
    reason="chDB adds timezone to datetime, causing boundary comparison differences",
    strict=False,  # behavior varies by Python/chDB version
)

chdb_datetime_extraction_conflict = pytest.mark.xfail(
    reason="chDB column name conflict with multiple datetime extractions",
    strict=True,
)

chdb_dt_month_type = pytest.mark.xfail(
    reason="chDB type mismatch: dt.month returns different types in SQL vs DataFrame",
    strict=True,
)

# SQL
chdb_duplicate_column_rename = pytest.mark.xfail(
    reason="SQL automatically renames duplicate columns - known limitation",
    strict=True,
)

chdb_case_bool_conversion = pytest.mark.xfail(
    reason="SQL CASE WHEN cannot convert Bool to Int64/String",
    strict=True,
)

# Dtype Differences - values are CORRECT, only dtype differs from pandas
# These are acceptable differences where DataStore returns more semantically correct types
chdb_nat_returns_nullable_int = pytest.mark.xfail(
    reason="chDB datetime accessor with NaT returns nullable Int (Int32), pandas returns float64 - VALUES ARE CORRECT",
    strict=True,
)

chdb_replace_none_dtype = pytest.mark.xfail(
    reason="chDB replace with None returns nullable Int64, pandas returns object dtype - VALUES ARE CORRECT",
    strict=True,
)

chdb_mask_dtype_nullable = pytest.mark.xfail(
    reason="chDB mask/where on int returns nullable Int64, pandas returns float64 (due to NaN) - VALUES ARE CORRECT",
    strict=True,
)


# =============================================================================
# DataStore Bugs (bug_*)
# Should be fixed to match pandas behavior
# =============================================================================

# FIXED: chDB any()/anyLast() now correctly returns row-order based first/last (2026-01-06)
# bug_groupby_first_last = pytest.mark.xfail(
#     reason="""DataStore first()/last() not pandas-compatible.
#     Uses chDB any()/anyLast() which don't guarantee row-order.
#     Fix: use argMin/argMax with row_number or pandas fallback.""",
#     strict=False,  # behavior varies by chDB version
# )

# FIXED: DataStore groupby aggregation now preserves index correctly (2026-01-06)
# bug_groupby_index = pytest.mark.xfail(
#     reason="""DataStore groupby aggregation doesn't preserve index correctly.
#     Pandas groupby().agg() returns Series with groupby column as index.
#     DataStore should match this behavior.""",
#     strict=False,  # behavior varies by chDB version
# )

# FIXED (2026-01-06): Index info is now preserved through lazy SQL execution
# The fix tracks index info in _index_info during _ensure_sql_source() and
# restores the index after SQL execution in _execute().
# bug_index_not_preserved = pytest.mark.xfail(
#     reason="Index info not preserved through lazy SQL execution",
#     strict=True,
# )


# No-op decorator for import compatibility
def bug_index_not_preserved(func):
    """FIXED: Index info is now preserved through lazy SQL execution."""
    return func


bug_extractall_multiindex = pytest.mark.xfail(
    reason="extractall returns MultiIndex DataFrame, index info lost through lazy execution",
    strict=True,
)

bug_null_string_comparison = pytest.mark.xfail(
    reason="BUG: ds[ds['col'] != None] returns 0 rows, should return non-None rows. "
    "Fix: convert != None to IS NOT NULL in DataStore layer",
    strict=True,
)

bug_where_computed_column = pytest.mark.xfail(
    reason="BUG: where() with lazy assigned column fails with 'Unknown expression identifier'. "
    "Fix: resolve computed columns before where execution",
    strict=True,
)

bug_groupby_apply_method_call = pytest.mark.xfail(
    reason="BUG: groupby.apply(lambda x: x.sum()) fails. " "Fix: ensure apply passes Series not scalar to lambda",
    strict=True,
)


# =============================================================================
# DataStore Limitations (limit_*)
# Features not yet implemented
# =============================================================================

limit_callable_index = pytest.mark.xfail(
    reason="DataStore does not support callable as index",
    strict=True,
)

limit_query_variable_scope = pytest.mark.xfail(
    reason="query() with @variable requires local variable scope, not available after _get_df()",
    strict=True,
)

limit_loc_conditional_assignment = pytest.mark.xfail(
    reason="loc conditional assignment with ColumnExpr not fully supported",
    strict=True,
)


# FIXED 2026-01-06: where() with DataStore condition now works
# limit_where_condition was an xfail marker for a bug that has been fixed
def limit_where_condition(func):
    """No-op decorator - bug has been fixed."""
    return func


# NOTE: limit_unstack_column_expr moved to design_* - this is an intentional design decision
# unstack() requires MultiIndex which is only available after execution.
# Use pivot_table() instead for the same functionality.

limit_str_join_array = pytest.mark.xfail(
    reason="str.join() requires Array type column, not string column",
    strict=True,
)

# =============================================================================
# Design Differences (design_*)
# Intentional behavioral differences from pandas
# These are conscious decisions, not bugs to be fixed.
# =============================================================================

design_datetime_fillna_nat = pytest.mark.xfail(
    reason="Design decision: Pandas where/mask replaces datetime with 0/-1, DataStore uses NaT (semantically clearer)",
    strict=True,
)

design_unstack_column_expr = pytest.mark.xfail(
    reason="Design decision: ColumnExpr doesn't support unstack() (requires MultiIndex only available after execution). Use pivot_table() instead.",
    strict=True,
)

# Alias for backward compatibility
limit_unstack_column_expr = design_unstack_column_expr

# FIXED: DataStore now restricts column access after select() to match pandas behavior
# design_sql_select_column_access = pytest.mark.xfail(
#     reason="Design difference: SQL pushdown allows accessing original columns after select(), pandas restricts to selected columns only",
#     strict=True,
# )

# FIXED: SQL builder now properly layers computed columns between LIMIT and WHERE
# limit_sql_column_dependency_after_limit = pytest.mark.xfail(
#     reason="SQL pushdown limitation: FILTER referencing computed column created after LIMIT requires complex subquery nesting not yet implemented",
#     strict=True,
# )


# =============================================================================
# Deprecated Features (deprecated_*)
# Deprecated pandas features
# =============================================================================

deprecated_fillna_downcast = pytest.mark.xfail(
    reason="fillna downcast parameter is deprecated in pandas 2.x",
    strict=True,
)


# =============================================================================
# Legacy Aliases (for backward compatibility during migration)
# TODO: Remove after updating all test files
# =============================================================================

# bug_* aliases
# datastore_groupby_first_last_order = bug_groupby_first_last  # FIXED
# datastore_groupby_index_preservation = bug_groupby_index  # FIXED
# lazy_index_not_preserved = bug_index_not_preserved  # FIXED - now a no-op
lazy_index_not_preserved = bug_index_not_preserved  # Now a no-op function
lazy_extractall_multiindex = bug_extractall_multiindex

# limit_* aliases
datastore_callable_index = limit_callable_index
datastore_query_variable_scope = limit_query_variable_scope
datastore_loc_conditional_assignment = limit_loc_conditional_assignment
datastore_where_condition = limit_where_condition
datastore_unstack_column_expr = design_unstack_column_expr  # Reclassified as design decision
datastore_str_join_array = limit_str_join_array

# deprecated_* aliases
pandas_deprecated_fillna_downcast = deprecated_fillna_downcast


# =============================================================================
# Marker Registry
#
# This registry provides metadata for tracking and reporting.
# Format: marker_name -> (category, issue_url, notes)
# =============================================================================

MARKER_REGISTRY = {
    # =========================================================================
    # chDB Engine Limitations (cannot fix in DataStore)
    # =========================================================================
    # Type Support
    # NOTE: These types work for read-only access but fail during SQL execution
    "chdb_category_type": ("chdb", None, "CATEGORY type fails in SQL operations (read-only works)"),
    "chdb_timedelta_type": ("chdb", None, "TIMEDELTA type fails in SQL operations (read-only works)"),
    "chdb_array_nullable": ("chdb", None, "Array cannot be inside Nullable type"),
    "chdb_array_string_conversion": ("chdb", None, "numpy arrays may be converted to strings in SQL operations"),
    # Functions
    "chdb_no_product_function": ("chdb", None, "product() aggregate not available"),
    "chdb_no_normalize_utf8": ("chdb", None, "normalizeUTF8NFD function not available"),
    "chdb_no_quantile_array": ("chdb", None, "quantile with array parameter not supported"),
    "chdb_median_in_where": ("chdb", None, "Aggregate in WHERE requires subquery"),
    # NULL/NaN
    # "chdb_nan_sum_behavior": ("chdb", None, "Sum of all-NaN returns NA (SQL standard)"),  # FIXED
    # String/Unicode
    "chdb_unicode_filter": ("chdb", None, "Unicode in SQL filter has encoding issues"),
    "chdb_strip_whitespace": ("chdb", None, "strip() doesn't handle all whitespace types"),
    # Datetime
    "chdb_datetime_timezone": ("chdb", None, "Timezone handling differs by version"),
    "chdb_datetime_extraction_conflict": ("chdb", None, "Multiple datetime extractions cause column name conflict"),
    "chdb_dt_month_type": ("chdb", None, "dt.month type inconsistency between SQL and DataFrame"),
    # SQL Behavior
    "chdb_duplicate_column_rename": ("chdb", None, "SQL auto-renames duplicate columns"),
    "chdb_case_bool_conversion": ("chdb", None, "CASE WHEN cannot mix Bool with other types"),
    # Dtype Differences (values correct, only dtype differs)
    "chdb_nat_returns_nullable_int": ("chdb", None, "dt accessor with NaT returns Nullable Int (values correct)"),
    "chdb_replace_none_dtype": ("chdb", None, "replace with None returns Nullable Int (values correct)"),
    "chdb_mask_dtype_nullable": ("chdb", None, "mask/where returns Nullable Int64 (values correct)"),
    # =========================================================================
    # DataStore Bugs (should be fixed)
    # =========================================================================
    "bug_index_not_preserved": ("bug", None, "Index info lost through lazy SQL execution"),
    "bug_extractall_multiindex": ("bug", None, "MultiIndex lost in extractall due to lazy execution"),
    # =========================================================================
    # DataStore Limitations (not yet implemented)
    # =========================================================================
    "limit_callable_index": ("limit", None, "Callable as index not implemented"),
    "limit_query_variable_scope": ("limit", None, "@variable scope in query() not available after _get_df()"),
    "limit_loc_conditional_assignment": ("limit", None, "loc conditional assignment with ColumnExpr incomplete"),
    "limit_where_condition": ("fixed", "2026-01-06", "where() with DataStore condition - FIXED"),
    "limit_str_join_array": ("limit", None, "str.join() needs Array type column"),
    # =========================================================================
    # Design Decisions (intentional differences)
    # =========================================================================
    "design_datetime_fillna_nat": ("design", None, "Use NaT instead of 0/-1 for datetime where/mask"),
    "design_unstack_column_expr": ("design", None, "unstack not supported on ColumnExpr, use pivot_table()"),
    # =========================================================================
    # Deprecated Features (pandas deprecated)
    # =========================================================================
    "deprecated_fillna_downcast": ("deprecated", None, "fillna downcast deprecated in pandas 2.x"),
    # =========================================================================
    # FIXED (kept for reference)
    # =========================================================================
    # "chdb_nullable_int64_comparison": FIXED in chDB 4.0.0b3
    # "chdb_null_in_groupby": FIXED by dropna parameter implementation
    # "chdb_empty_df_str_dtype": FIXED in core.py empty DataFrame handling
    # "chdb_integer_column_names": FIXED in connection.py via string conversion
    # "bug_groupby_first_last": FIXED - chDB any()/anyLast() now row-order based
    # "bug_groupby_index": FIXED - groupby now preserves index correctly
    # "bug_setitem_computed_column_groupby": FIXED - setitem updates _computed_columns
    # "bug_groupby_column_selection_extra_columns": FIXED - column selection filters correctly
    # =========================================================================
    # FIXED 2026-01-07
    # =========================================================================
    "chdb_alias_shadows_column_in_where": ("fixed", "2026-01-07", "Alias no longer shadows column in WHERE - FIXED"),
    "limit_datastore_no_invert": ("fixed", "2026-01-07", "__invert__ (~) operator - FIXED"),
}


def get_markers_by_category(category: str) -> List[str]:
    """Get all marker names in a specific category."""
    return [name for name, (cat, _, _) in MARKER_REGISTRY.items() if cat == category]


def get_all_categories() -> List[str]:
    """Get all unique categories."""
    return list(set(cat for cat, _, _ in MARKER_REGISTRY.values()))


# =============================================================================
# FIXED Bug markers - kept for import compatibility
# =============================================================================


# FIXED (2026-01-06): setitem now correctly updates _computed_columns
def bug_setitem_computed_column_groupby(func):
    """FIXED: ds['col'] = expr now correctly populates _computed_columns."""
    return func


# =============================================================================
# FIXED markers - kept as no-op functions for import compatibility
# =============================================================================


# FIXED (2026-01-06): Empty DataFrame now executes SQL to get correct dtypes
def chdb_empty_df_str_dtype(func):
    """FIXED: Empty DataFrame str accessor now returns correct dtype."""
    return func


# FIXED (2026-01-06): Integer column names now work via string conversion in connection.py
def chdb_integer_column_names(func):
    """FIXED: Integer column names now work via string conversion."""
    return func


# =============================================================================
# Bug: groupby column selection includes extra columns - FIXED (2026-01-06)
# =============================================================================

# FIXED: groupby column selection now correctly filters columns
# bug_groupby_column_selection_extra_columns = pytest.mark.xfail(
#     reason="Bug: After assign() + groupby(), selecting specific columns [['a', 'b']] includes extra columns in result",
#     strict=True,
# )


# =============================================================================
# chDB limitation: datetime method not implemented
# =============================================================================

chdb_no_day_month_name = pytest.mark.xfail(
    reason="chDB limitation: day_name/month_name methods not implemented in SQL mapping",
    strict=True,
)


# =============================================================================
# chDB limitation: strftime format codes differ from pandas
# =============================================================================

chdb_strftime_format_difference = pytest.mark.xfail(
    reason="chDB limitation: strftime %M format returns month name instead of minutes",
    strict=True,
)


# =============================================================================
# chDB limitation: str.pad doesn't support 'side' parameter
# =============================================================================

chdb_pad_no_side_param = pytest.mark.xfail(
    reason="chDB limitation: str.pad() only supports left padding, 'side' parameter not implemented",
    strict=True,
)


# =============================================================================
# chDB limitation: str.center implementation differs
# =============================================================================

chdb_center_implementation = pytest.mark.xfail(
    reason="chDB limitation: str.center() implementation uses rightPad instead of proper centering",
    strict=True,
)


# =============================================================================
# chDB limitation: startswith/endswith don't support tuple argument
# =============================================================================

chdb_startswith_no_tuple = pytest.mark.xfail(
    reason="chDB limitation: startswith/endswith don't support tuple of prefixes/suffixes",
    strict=True,
)


# =============================================================================
# DataStore limitation: index property has no setter
# =============================================================================

limit_datastore_index_setter = pytest.mark.xfail(
    reason="DataStore limitation: index property does not have a setter",
    strict=True,
)

# DataStore Limitations: groupby does not support Series/ColumnExpr as parameter
limit_groupby_series_param = pytest.mark.xfail(
    reason="DataStore groupby does not support ColumnExpr/Series as groupby parameter. "
    "Use column name after assigning the expression to a column instead: "
    "ds['col'] = ds['date'].dt.year; ds.groupby('col')...",
    strict=True,
)

# NOTE: Simple alias cases work but complex chains with groupby still have issues
chdb_alias_shadows_column_in_where = pytest.mark.xfail(
    reason="chDB: In complex chains with groupby, SELECT alias may still shadow original column"
)

# NOTE: ~column works but ~DataStore (entire DataFrame invert) does not
limit_datastore_no_invert = pytest.mark.xfail(
    reason="DataStore does not implement __invert__ (~) operator for entire DataFrame (column invert ~ds['col'] works)"
)

# =============================================================================
# chDB Non-deterministic Behavior
# =============================================================================

chdb_any_anylast_nondeterministic = pytest.mark.xfail(
    reason="chDB any()/anyLast() is non-deterministic - may return arbitrary row's value instead of first/last. "
    "ClickHouse docs explicitly state any() returns an 'arbitrary' value. "
    "See: https://clickhouse.com/docs/en/sql-reference/aggregate-functions/reference/any",
    strict=False,  # behavior varies by environment
)
