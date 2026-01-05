"""
Centralized xfail markers for DataStore tests.

This module provides a single source of truth for all expected test failures
due to known limitations in chDB, DataStore, or design differences from pandas.

Usage:
    from tests.xfail_markers import chdb_category_type, chdb_array_nullable

    @chdb_category_type
    def test_something():
        ...

Categories:
    - chDB Engine Limitations: Limitations in the underlying chDB/ClickHouse engine
    - DataStore Limitations: Features not yet implemented in DataStore
    - Design Differences: Intentional behavioral differences from pandas

When a limitation is resolved, remove the marker from this file and update
all tests that use it.
"""

import pytest


# =============================================================================
# chDB Engine Limitations
# =============================================================================

# Type Support Issues
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

chdb_nullable_int64_comparison = pytest.mark.xfail(
    reason="chDB does not handle Nullable Int64 comparison correctly - returns raw bytes",
    strict=True,
)

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

# String/Unicode Issues
chdb_unicode_filter = pytest.mark.xfail(
    reason="Unicode string equality in SQL filter has encoding issues",
    strict=True,
)

chdb_strip_whitespace = pytest.mark.xfail(
    reason="str.strip() doesn't handle all whitespace types correctly in chDB",
    strict=True,
)

# Datetime Issues
chdb_datetime_timezone = pytest.mark.xfail(
    reason="chDB adds timezone to datetime, causing boundary comparison differences",
    strict=True,
)

chdb_datetime_extraction_conflict = pytest.mark.xfail(
    reason="chDB column name conflict with multiple datetime extractions",
    strict=True,
)

chdb_dt_month_type = pytest.mark.xfail(
    reason="chDB type mismatch: dt.month returns different types in SQL vs DataFrame",
    strict=True,
)

# SQL Limitations
chdb_duplicate_column_rename = pytest.mark.xfail(
    reason="SQL automatically renames duplicate columns - known limitation",
    strict=True,
)

chdb_case_bool_conversion = pytest.mark.xfail(
    reason="SQL CASE WHEN cannot convert Bool to Int64/String",
    strict=True,
)


# =============================================================================
# DataStore Limitations
# =============================================================================

datastore_callable_index = pytest.mark.xfail(
    reason="DataStore does not support callable as index",
    strict=True,
)

datastore_query_variable_scope = pytest.mark.xfail(
    reason="query() with @variable requires local variable scope, not available after _get_df()",
    strict=True,
)

datastore_loc_conditional_assignment = pytest.mark.xfail(
    reason="loc conditional assignment with ColumnExpr not fully supported",
    strict=True,
)

datastore_where_condition = pytest.mark.xfail(
    reason="DataFrame.where with DataStore condition has SQL execution bug",
    strict=True,
)

datastore_unstack_column_expr = pytest.mark.xfail(
    reason="ColumnExpr doesn't support unstack - Series method on MultiIndex",
    strict=True,
)

datastore_str_join_array = pytest.mark.xfail(
    reason="str.join() requires Array type column, not string column",
    strict=True,
)


# =============================================================================
# Index Preservation Limitations
# =============================================================================

lazy_index_not_preserved = pytest.mark.xfail(
    reason="Index info not preserved through lazy SQL execution",
    strict=True,
)

lazy_extractall_multiindex = pytest.mark.xfail(
    reason="extractall returns MultiIndex DataFrame, index info lost through lazy execution",
    strict=True,
)


# =============================================================================
# Design Differences (Intentional)
# =============================================================================

design_datetime_fillna_nat = pytest.mark.xfail(
    reason="Design difference: Pandas replaces datetime with 0/-1, DataStore uses NaT",
    strict=True,
)


# =============================================================================
# Deprecated Features
# =============================================================================

pandas_deprecated_fillna_downcast = pytest.mark.xfail(
    reason="fillna downcast parameter is deprecated in pandas 2.x",
    strict=True,
)


# =============================================================================
# Marker Registry
#
# This registry provides metadata for tracking and reporting.
# Format: marker_name -> (category, issue_url, notes)
# =============================================================================

MARKER_REGISTRY = {
    # chDB Type Issues
    "chdb_category_type": ("chdb_engine", None, "CATEGORY type not supported in ClickHouse"),
    "chdb_timedelta_type": ("chdb_engine", None, "TIMEDELTA type not supported in ClickHouse"),
    "chdb_array_nullable": ("chdb_engine", None, "Array cannot be inside Nullable in ClickHouse"),
    "chdb_array_string_conversion": ("chdb_engine", None, "Python() table function converts arrays to strings"),
    "chdb_nullable_int64_comparison": ("chdb_engine", None, "Nullable Int64 comparison returns raw bytes"),

    # chDB Function Issues
    "chdb_no_product_function": ("chdb_engine", None, "product() not available in ClickHouse"),
    "chdb_no_normalize_utf8": ("chdb_engine", None, "normalizeUTF8NFD not available"),
    "chdb_no_quantile_array": ("chdb_engine", None, "quantile with array param not supported"),
    "chdb_median_in_where": ("chdb_engine", None, "Aggregate in WHERE requires subquery"),

    # chDB NULL Handling
    "chdb_null_in_groupby": ("chdb_engine", None, "NULL handling differs from pandas in groupby"),
    "chdb_nan_sum_behavior": ("chdb_engine", None, "Sum of all-NaN returns NA, not 0"),

    # chDB String Issues
    "chdb_unicode_filter": ("chdb_engine", None, "Unicode in SQL filter has issues"),
    "chdb_strip_whitespace": ("chdb_engine", None, "strip() whitespace handling incomplete"),

    # chDB Datetime Issues
    "chdb_datetime_timezone": ("chdb_engine", None, "Timezone handling differs"),
    "chdb_datetime_extraction_conflict": ("chdb_engine", None, "Column naming conflict in datetime extraction"),
    "chdb_dt_month_type": ("chdb_engine", None, "dt.month type inconsistency"),

    # chDB SQL Issues
    "chdb_duplicate_column_rename": ("chdb_engine", None, "SQL auto-renames duplicate columns"),
    "chdb_case_bool_conversion": ("chdb_engine", None, "CASE WHEN Bool conversion issue"),

    # DataStore Limitations
    "datastore_callable_index": ("datastore", None, "Callable index not implemented"),
    "datastore_query_variable_scope": ("datastore", None, "@variable scope issue in query()"),
    "datastore_loc_conditional_assignment": ("datastore", None, "loc conditional assignment incomplete"),
    "datastore_where_condition": ("datastore", None, "where() with DataStore condition bug"),
    "datastore_unstack_column_expr": ("datastore", None, "unstack not supported on ColumnExpr"),
    "datastore_str_join_array": ("datastore", None, "str.join needs Array type column"),

    # Index Issues
    "lazy_index_not_preserved": ("index", None, "Index lost in lazy execution"),
    "lazy_extractall_multiindex": ("index", None, "MultiIndex lost in extractall"),

    # Design Differences
    "design_datetime_fillna_nat": ("design", None, "Intentional: use NaT instead of 0/-1"),

    # Deprecated
    "pandas_deprecated_fillna_downcast": ("deprecated", None, "fillna downcast deprecated in pandas 2.x"),
}


def get_markers_by_category(category: str) -> list[str]:
    """Get all marker names in a specific category."""
    return [name for name, (cat, _, _) in MARKER_REGISTRY.items() if cat == category]


def get_all_categories() -> list[str]:
    """Get all unique categories."""
    return list(set(cat for cat, _, _ in MARKER_REGISTRY.values()))
