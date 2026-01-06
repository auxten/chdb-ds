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
chdb_category_type = pytest.mark.xfail(
    reason="chDB does not support CATEGORY numpy type",
    strict=True,
)

chdb_timedelta_type = pytest.mark.xfail(
    reason="chDB does not support TIMEDELTA numpy type",
    strict=True,
)

chdb_array_nullable = pytest.mark.xfail(
    reason="chDB: Array type cannot be inside Nullable type",
    strict=True,
)

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
chdb_null_in_groupby = pytest.mark.xfail(
    reason="chDB treats None/NaN as empty string in groupby - pandas excludes NULL by default",
    strict=True,
)

chdb_nan_sum_behavior = pytest.mark.xfail(
    reason="chDB returns NA for sum of all-NaN, pandas returns 0",
    strict=True,
)

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

bug_index_not_preserved = pytest.mark.xfail(
    reason="Index info not preserved through lazy SQL execution",
    strict=True,
)

bug_extractall_multiindex = pytest.mark.xfail(
    reason="extractall returns MultiIndex DataFrame, index info lost through lazy execution",
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

limit_where_condition = pytest.mark.xfail(
    reason="DataFrame.where with DataStore condition has SQL execution bug",
    strict=True,
)

limit_unstack_column_expr = pytest.mark.xfail(
    reason="ColumnExpr doesn't support unstack - Series method on MultiIndex",
    strict=True,
)

limit_str_join_array = pytest.mark.xfail(
    reason="str.join() requires Array type column, not string column",
    strict=True,
)


# =============================================================================
# Design Differences (design_*)
# Intentional behavioral differences from pandas
# =============================================================================

design_datetime_fillna_nat = pytest.mark.xfail(
    reason="Design difference: Pandas replaces datetime with 0/-1, DataStore uses NaT",
    strict=True,
)

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
lazy_index_not_preserved = bug_index_not_preserved
lazy_extractall_multiindex = bug_extractall_multiindex

# limit_* aliases
datastore_callable_index = limit_callable_index
datastore_query_variable_scope = limit_query_variable_scope
datastore_loc_conditional_assignment = limit_loc_conditional_assignment
datastore_where_condition = limit_where_condition
datastore_unstack_column_expr = limit_unstack_column_expr
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
    # chDB Engine Limitations
    "chdb_category_type": ("chdb", None, "CATEGORY type not supported"),
    "chdb_timedelta_type": ("chdb", None, "TIMEDELTA type not supported"),
    "chdb_array_nullable": ("chdb", None, "Array cannot be inside Nullable"),
    "chdb_array_string_conversion": ("chdb", None, "Python() converts arrays to strings"),
    "chdb_nullable_int64_comparison": ("chdb", None, "Nullable Int64 comparison issue"),
    "chdb_no_product_function": ("chdb", None, "product() not available"),
    "chdb_no_normalize_utf8": ("chdb", None, "normalizeUTF8NFD not available"),
    "chdb_no_quantile_array": ("chdb", None, "quantile with array not supported"),
    "chdb_median_in_where": ("chdb", None, "Aggregate in WHERE requires subquery"),
    "chdb_null_in_groupby": ("chdb", None, "NULL handling differs in groupby"),
    "chdb_nan_sum_behavior": ("chdb", None, "Sum of all-NaN returns NA"),
    # NOTE: chdb_null_comparison_semantics and chdb_null_string_comparison FIXED
    "chdb_unicode_filter": ("chdb", None, "Unicode in SQL filter issues"),
    "chdb_strip_whitespace": ("chdb", None, "strip() whitespace handling"),
    "chdb_datetime_timezone": ("chdb", None, "Timezone handling differs"),
    "chdb_datetime_extraction_conflict": ("chdb", None, "Datetime extraction column conflict"),
    "chdb_dt_month_type": ("chdb", None, "dt.month type inconsistency"),
    "chdb_duplicate_column_rename": ("chdb", None, "SQL auto-renames duplicate columns"),
    "chdb_case_bool_conversion": ("chdb", None, "CASE WHEN Bool conversion"),
    # DataStore Bugs
    "bug_groupby_first_last": ("bug", None, "first()/last() not order-based"),
    "bug_groupby_index": ("bug", None, "groupby doesn't preserve index"),
    "bug_index_not_preserved": ("bug", None, "Index lost in lazy execution"),
    "bug_extractall_multiindex": ("bug", None, "MultiIndex lost in extractall"),
    # DataStore Limitations
    "limit_callable_index": ("limit", None, "Callable index not implemented"),
    "limit_query_variable_scope": ("limit", None, "@variable scope in query()"),
    "limit_loc_conditional_assignment": ("limit", None, "loc conditional assignment"),
    "limit_where_condition": ("limit", None, "where() with DataStore condition"),
    "limit_unstack_column_expr": ("limit", None, "unstack not supported"),
    "limit_str_join_array": ("limit", None, "str.join needs Array type"),
    # Design Differences
    "design_datetime_fillna_nat": ("design", None, "Use NaT instead of 0/-1"),
    # Deprecated
    "deprecated_fillna_downcast": ("deprecated", None, "fillna downcast deprecated"),
    # Bugs discovered in exploratory batch 38
    "bug_setitem_computed_column_groupby": ("bug", None, "setitem computed column not tracked for groupby"),
    "chdb_empty_df_str_dtype": ("chdb", None, "Empty df str accessor dtype issue"),
    # chDB integer column names
    "chdb_integer_column_names": ("chdb", None, "Integer column names cause errors"),
    # Bug: groupby column selection
    # "bug_groupby_column_selection_extra_columns": ("bug", None, "FIXED - groupby column selection includes extra columns"),
}


def get_markers_by_category(category: str) -> List[str]:
    """Get all marker names in a specific category."""
    return [name for name, (cat, _, _) in MARKER_REGISTRY.items() if cat == category]


def get_all_categories() -> List[str]:
    """Get all unique categories."""
    return list(set(cat for cat, _, _ in MARKER_REGISTRY.values()))


# =============================================================================
# Bug: setitem computed column not tracked in _computed_columns
# =============================================================================

bug_setitem_computed_column_groupby = pytest.mark.xfail(
    reason="Bug: ds['col'] = expr does not populate _computed_columns, causing groupby on computed column to fail with SQL UNKNOWN_IDENTIFIER error",
    strict=True,
)

# Add to MARKER_REGISTRY at the end

chdb_empty_df_str_dtype = pytest.mark.xfail(
    reason="chDB: str accessor on empty DataFrame returns float64 instead of object dtype",
    strict=True,
)


# =============================================================================
# chDB: Integer column names (from transpose) cause errors
# =============================================================================

chdb_integer_column_names = pytest.mark.xfail(
    reason="chDB: Python() table function cannot handle integer column names (e.g., 0, 1, 2 from transpose), causes KeyError",
    strict=True,
)

# =============================================================================
# Bug: groupby column selection includes extra columns - FIXED (2026-01-06)
# =============================================================================

# FIXED: groupby column selection now correctly filters columns
# bug_groupby_column_selection_extra_columns = pytest.mark.xfail(
#     reason="Bug: After assign() + groupby(), selecting specific columns [['a', 'b']] includes extra columns in result",
#     strict=True,
# )
