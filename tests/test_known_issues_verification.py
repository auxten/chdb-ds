"""
Tests for known issues documented in unresolved_issues_summary.md

This file verifies the current status of all known issues and uses
appropriate xfail markers from tests/xfail_markers.py.

Run with: pytest tests/test_known_issues_verification.py -v
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# Import xfail markers
# =============================================================================
from tests.xfail_markers import (
    chdb_category_type,
    chdb_timedelta_type,
    chdb_array_string_conversion,
    design_unstack_column_expr,
    limit_callable_index,
    limit_query_variable_scope,
    bug_extractall_multiindex,
    chdb_nat_returns_nullable_int,
    chdb_replace_none_dtype,
    limit_datastore_no_invert,
    chdb_alias_shadows_column_in_where,
)


# =============================================================================
# SECTION 1: Data Types - Clarify what works and what doesn't
# =============================================================================

class TestDataTypesReadOnly:
    """Tests for data type issues - READ-ONLY access works, SQL operations fail."""

    def test_categorical_type_read_only_works(self):
        """Categorical type works for read-only access (no SQL execution)."""
        df = pd.DataFrame({
            'cat_col': pd.Categorical(['a', 'b', 'c']),
            'value': [1, 2, 3]
        })
        ds = DataStore(df)

        # Read-only access works (no SQL execution triggered)
        assert list(ds['cat_col']) == ['a', 'b', 'c']

    @chdb_category_type
    def test_categorical_type_fails_in_sql(self):
        """Categorical type fails when SQL operations are needed."""
        df = pd.DataFrame({
            'cat_col': pd.Categorical(['a', 'b', 'c']),
            'value': [1, 2, 3]
        })
        ds = DataStore(df)

        # SQL operation (filter) fails
        result = ds[ds['value'] > 1]
        len(result)  # Triggers SQL execution

    def test_timedelta_type_read_only_works(self):
        """Timedelta type works for read-only access."""
        df = pd.DataFrame({
            'td_col': pd.to_timedelta(['1 day', '2 days', '3 days']),
            'value': [1, 2, 3]
        })
        ds = DataStore(df)

        # Read-only access works
        pd_values = df['td_col'].tolist()
        ds_values = list(ds['td_col'])
        assert pd_values == ds_values

    @chdb_timedelta_type
    def test_timedelta_type_fails_in_sql(self):
        """Timedelta type fails when SQL operations are needed."""
        df = pd.DataFrame({
            'td_col': pd.to_timedelta(['1 day', '2 days', '3 days']),
            'value': [1, 2, 3]
        })
        ds = DataStore(df)

        # SQL operation (filter) fails
        result = ds[ds['value'] > 1]
        len(result)

    def test_numpy_array_columns_read_only_works(self):
        """Numpy array columns are preserved for read-only access."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        df = pd.DataFrame({'arr_col': [arr1, arr2], 'id': [1, 2]})
        ds = DataStore(df)

        ds_values = list(ds['arr_col'])
        # Arrays should be preserved for read-only access
        assert isinstance(ds_values[0], np.ndarray), f"Expected ndarray, got {type(ds_values[0])}"
        np.testing.assert_array_equal(ds_values[0], arr1)
        np.testing.assert_array_equal(ds_values[1], arr2)


class TestDataTypesFixed:
    """Tests for data type issues that are truly FIXED."""

    def test_invert_operator_now_works(self):
        """~ invert operator now works (was raising TypeError)."""
        df = pd.DataFrame({'a': [True, False, True]})
        ds = DataStore(df)

        pd_result = ~df['a']
        ds_result = ~ds['a']

        assert pd_result.tolist() == list(ds_result)

    def test_sum_of_all_nan_returns_zero(self):
        """Sum of all NaN now returns 0.0 same as pandas (was returning NA)."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        ds = DataStore(df)

        pd_sum = df['a'].sum()
        ds_sum = ds['a'].sum()

        # Both should return 0.0
        assert pd_sum == 0.0
        # DataStore returns ColumnExpr, use repr to get value
        assert repr(ds_sum) == 'np.float64(0.0)'

    def test_index_preserved_after_filter(self):
        """Index is now preserved after SQL execution (was lost)."""
        df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds = DataStore(df)

        pd_filtered = df[df['a'] > 1]
        ds_filtered = ds[ds['a'] > 1]

        assert pd_filtered.index.tolist() == ds_filtered.index.tolist()

    def test_index_preserved_after_sort(self):
        """Index is preserved after sort operation."""
        df = pd.DataFrame({'a': [3, 1, 2]}, index=['x', 'y', 'z'])
        ds = DataStore(df)

        pd_sorted = df.sort_values('a')
        ds_sorted = ds.sort_values('a')

        assert pd_sorted.index.tolist() == ds_sorted.index.tolist()

    def test_filter_with_lazy_assigned_column(self):
        """Filter with lazy assigned column now works."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df.copy())

        # Assign computed column
        ds['b'] = ds['a'] * 2

        # Filter on computed column should work
        ds_filtered = ds[ds['b'] > 5]

        # Verify
        assert len(ds_filtered) == 3
        assert list(ds_filtered['b']) == [6, 8, 10]

    def test_alias_no_longer_shadows_column(self):
        """SELECT alias no longer shadows column in filter (was broken)."""
        df = pd.DataFrame({'value': [10, 20, 30]})
        ds = DataStore(df.copy())

        # Assign same column name
        ds['value'] = ds['value'] * 2  # 20, 40, 60

        # Filter should work on the new values
        ds_filtered = ds[ds['value'] > 25]

        # Compare with pandas
        pdf = df.copy()
        pdf['value'] = pdf['value'] * 2
        pdf_filtered = pdf[pdf['value'] > 25]

        assert len(ds_filtered) == len(pdf_filtered)
        assert list(ds_filtered['value']) == pdf_filtered['value'].tolist()


# =============================================================================
# SECTION 2: Still Existing Issues
# =============================================================================

class TestStillExistingIssues:
    """Tests for issues that still exist."""

    @design_unstack_column_expr
    def test_unstack_not_available(self):
        """unstack() is not available on ColumnExpr (use pivot_table instead)."""
        df = pd.DataFrame({
            'cat1': ['A', 'A', 'B', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4]
        })
        ds = DataStore(df)

        # This should fail - unstack is not available
        grouped = ds.groupby(['cat1', 'cat2'])['value'].sum()
        result = grouped.unstack()  # Should raise AttributeError
        len(result)  # Trigger execution

    @limit_callable_index
    def test_callable_as_index_not_supported(self):
        """Callable as index is not supported."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        # This should fail
        result = ds[lambda x: x['a'] > 2]
        len(result)  # Trigger execution

    @limit_query_variable_scope
    def test_query_at_variable_scope_issue(self):
        """query() with @variable has scope issues."""
        df = pd.DataFrame({'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        threshold = 25
        # This should fail with UndefinedVariableError
        result = ds.query('value > @threshold')
        len(result)

    def test_query_at_variable_workaround(self):
        """Workaround: use f-string instead of @variable."""
        df = pd.DataFrame({'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        threshold = 25
        # F-string works
        result = ds.query(f'value > {threshold}')
        assert len(result) == 2
        assert list(result['value']) == [30, 40]

    @bug_extractall_multiindex
    def test_extractall_multiindex_lost(self):
        """extractall MultiIndex is lost (becomes RangeIndex)."""
        df = pd.DataFrame({'text': ['a1b2', 'c3d4']})
        ds = DataStore(df)

        pd_result = df['text'].str.extractall(r'(\d)')
        ds_result = ds['text'].str.extractall(r'(\d)')

        # pandas returns MultiIndex
        assert isinstance(pd_result.index, pd.MultiIndex)
        # DataStore should also return MultiIndex (but currently doesn't)
        assert isinstance(ds_result.index, pd.MultiIndex)

    @chdb_nat_returns_nullable_int
    def test_dtype_difference_dt_with_nat(self):
        """dt accessor with NaT returns different dtype (values are correct)."""
        df = pd.DataFrame({'dt': pd.to_datetime(['2020-01-01', None, '2020-03-01'])})
        ds = DataStore(df)

        pd_year = df['dt'].dt.year
        ds_year = ds['dt'].dt.year

        # Values should match (ignoring NA representation)
        pd_vals = [v if pd.notna(v) else None for v in pd_year.tolist()]
        ds_vals = [v if pd.notna(v) else None for v in list(ds_year)]
        # Filter out None for comparison
        pd_non_null = [v for v in pd_vals if v is not None]
        ds_non_null = [int(v) for v in ds_vals if v is not None]
        assert pd_non_null == ds_non_null

        # But dtypes differ
        assert pd_year.dtype == ds_year.dtype  # This will fail


class TestNullComparisonSemantics:
    """Tests for NULL comparison semantic differences between pandas and SQL."""

    def test_null_equals_none_same_behavior(self):
        """== None behaves the same in pandas and DataStore (both return 0 rows)."""
        df = pd.DataFrame({'s': ['abc', 'def', None]})
        ds = DataStore(df)

        pd_result = df[df['s'] == None]  # noqa: E711
        ds_result = ds[ds['s'] == None]  # noqa: E711

        # Both should return 0 rows
        assert len(pd_result) == 0
        assert len(ds_result) == 0

    def test_null_not_equals_none_differs(self):
        """!= None behaves differently (pandas: all rows, DataStore: 0 rows).

        This is a SQL semantic difference - use .notna() instead.
        """
        df = pd.DataFrame({'s': ['abc', 'def', None]})
        ds = DataStore(df)

        pd_result = df[df['s'] != None]  # noqa: E711
        ds_result = ds[ds['s'] != None]  # noqa: E711

        # pandas returns all 3 rows (including None!)
        # DataStore returns 0 rows (SQL: != NULL -> NULL -> falsy)
        assert len(pd_result) == 3  # pandas behavior
        assert len(ds_result) == 0  # DataStore/SQL behavior

    def test_null_comparison_workaround_notna(self):
        """Workaround: use .notna() instead of != None."""
        df = pd.DataFrame({'s': ['abc', 'def', None]})
        ds = DataStore(df)

        pd_result = df[df['s'].notna()]
        ds_result = ds[ds['s'].notna()]

        # Both return 2 rows (non-null values)
        assert len(pd_result) == 2
        assert len(ds_result) == 2
        assert pd_result['s'].tolist() == list(ds_result['s'])


class TestWhereWithComputedColumn:
    """Tests for where() with computed column issue."""

    def test_where_with_computed_column_fails(self):
        """where() with lazy assigned column fails (use filter instead)."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df.copy())

        # Assign computed column
        ds['b'] = ds['a'] * 2

        # where() with computed column fails
        with pytest.raises(Exception) as exc_info:
            result = ds.where(ds['b'] > 5)
            len(result)  # Trigger execution

        assert "Unknown expression" in str(exc_info.value) or "identifier" in str(exc_info.value)

    def test_where_with_computed_column_workaround(self):
        """Workaround: use filter [] instead of where()."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df.copy())

        # Assign computed column
        ds['b'] = ds['a'] * 2

        # Filter works
        ds_filtered = ds[ds['b'] > 5]

        assert len(ds_filtered) == 3
        assert list(ds_filtered['b']) == [6, 8, 10]


class TestGroupbyApply:
    """Tests for groupby.apply() limitations."""

    def test_groupby_apply_with_method_call_fails(self):
        """groupby.apply() with lambda calling methods fails."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds = DataStore(df)

        # Lambda that calls .sum() method fails
        with pytest.raises((AttributeError, TypeError)):
            result = ds.groupby('group')['value'].apply(lambda x: x.sum())
            list(result)

    def test_groupby_apply_workaround_agg(self):
        """Workaround: use agg() instead of apply()."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds = DataStore(df)

        # agg() with built-in functions works
        pd_result = df.groupby('group')['value'].agg('sum')
        ds_result = ds.groupby('group')['value'].sum()  # Use .sum() directly

        assert pd_result.tolist() == list(ds_result)


class TestGroupbyNull:
    """Tests for GROUP BY NULL behavior."""

    def test_groupby_null_matches_pandas_dropna_true(self):
        """DataStore groupby matches pandas dropna=True (default) behavior."""
        df = pd.DataFrame({
            'group': ['A', 'A', None, None],
            'value': [1, 2, 3, 4]
        })
        ds = DataStore(df)

        # pandas with dropna=True (default)
        pd_result = df.groupby('group', dropna=True)['value'].sum()

        # DataStore
        ds_result = ds.groupby('group')['value'].sum()

        # Both should have 1 group (A)
        assert len(pd_result) == 1
        assert len(list(ds_result)) == 1


# =============================================================================
# SECTION 3: Verify xfail markers that may be outdated
# =============================================================================

class TestXfailMarkersStatus:
    """Tests to clarify what's fixed and what still needs xfail markers."""

    def test_invert_operator_column_works(self):
        """~ds['col'] (column invert) works, but ~ds (DataFrame invert) doesn't."""
        df = pd.DataFrame({'a': [True, False, True]})
        ds = DataStore(df)

        # Column invert works!
        result = ~ds['a']
        assert list(result) == [False, True, False]

    @limit_datastore_no_invert
    def test_invert_operator_dataframe_fails(self):
        """~ds (entire DataFrame invert) fails - still needs xfail."""
        df = pd.DataFrame({'a': [True, False, True]})
        ds = DataStore(df)

        # Entire DataFrame invert fails
        result = ~ds
        len(result)

    def test_simple_alias_chain_works(self):
        """Simple assign + filter chain works now."""
        df = pd.DataFrame({'value': [10, 20, 30]})
        ds = DataStore(df.copy())

        ds['value'] = ds['value'] * 2
        ds_filtered = ds[ds['value'] > 25]

        # Compare with pandas
        pdf = df.copy()
        pdf['value'] = pdf['value'] * 2
        pdf_filtered = pdf[pdf['value'] > 25]

        # Should match!
        assert len(ds_filtered) == len(pdf_filtered)


# =============================================================================
# Run summary at end of module
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

