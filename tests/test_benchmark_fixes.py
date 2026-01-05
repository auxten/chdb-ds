"""
Tests for benchmark-identified issues fixed in this PR.

These tests verify the fixes for:
1. Row order preservation - using subquery with rowNumberInAllBlocks() BEFORE filtering
   to correctly preserve original row order when chDB processes Parquet files
2. where/mask Variant type - uses Variant(String, Int/Float) for mixed-type columns
3. GroupBy sort parameter - sort=True by default (pandas compatibility)
4. sort_values stability - default kind='stable' with rowNumberInAllBlocks() tie-breaker

Key insight: rowNumberInAllBlocks() must be computed BEFORE WHERE clause,
otherwise it returns row numbers after filtering which doesn't preserve original order.

SQL pattern used:
    SELECT * EXCEPT(__orig_row_num__)
    FROM (SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM source)
    WHERE conditions
    ORDER BY __orig_row_num__

Reference: benchmark_datastore_vs_pandas.py verification failures
"""

from tests.test_utils import get_dataframe
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore


class TestWhereMaskVariantFix:
    """
    Test that where/mask works correctly with Variant type and row order preservation.

    The fix uses a two-part approach:
    1. Variant type for string columns with numeric other values - e.g., Variant(String, Int64)
       This preserves the original int type rather than converting to string '0'.
    2. Subquery with rowNumberInAllBlocks() to preserve original row order
       when WHERE clause filters rows (otherwise chDB may return rows in wrong order)
    """

    def test_where_large_dataset_row_order_preserved(self):
        """Verify where() on large dataset preserves row order correctly.

        This was the main symptom of the Variant bug: rows were in wrong order.
        """
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.where(df['int_col'] > 500, 0)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['int_col'] > 500, 0).to_df()

            # Row 2 has int_col=860 > 500, should preserve original values
            assert pd_result['id'].iloc[2] == ds_result['id'].iloc[2] == 2
            assert pd_result['int_col'].iloc[2] == ds_result['int_col'].iloc[2] == 860

            # Row 0 has int_col < 500, should be replaced
            assert pd_result['id'].iloc[0] == ds_result['id'].iloc[0] == 0
            assert pd_result['int_col'].iloc[0] == ds_result['int_col'].iloc[0] == 0

    def test_where_numeric_columns_match_pandas(self):
        """where() on numeric-only DataFrame should match pandas exactly."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.where(df['int_col'] > 500, 0)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['int_col'] > 500, 0).to_df()

            # Numeric columns should match exactly
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)
            np.testing.assert_array_equal(ds_result['int_col'].values, pd_result['int_col'].values)
            np.testing.assert_array_almost_equal(ds_result['float_col'].values, pd_result['float_col'].values)

    def test_where_string_column_numeric_other_preserves_type(self):
        """String columns preserve int type with Variant (pandas alignment)."""
        df = pd.DataFrame(
            {
                'id': [0, 1, 2],
                'str_col': ['A', 'B', 'C'],
                'value': [100, 600, 300],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 500, 0).to_df()
            pd_result = df.where(df['value'] > 500, 0)

            # Should match pandas exactly (int 0, not string '0')
            # Row 0: value=100 <= 500, str_col replaced with 0 (int)
            assert ds_result['str_col'].iloc[0] == 0
            assert ds_result['str_col'].iloc[0] == pd_result['str_col'].iloc[0]
            # Row 1: value=600 > 500, str_col preserved
            assert ds_result['str_col'].iloc[1] == 'B'
            # Row 2: value=300 <= 500, str_col replaced with 0 (int)
            assert ds_result['str_col'].iloc[2] == 0

    def test_mask_large_dataset_correct_values(self):
        """mask() on large dataset should have correct values."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.mask(df['int_col'] > 500, -1)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['int_col'] > 500, -1).to_df()

            # Should match exactly for numeric columns
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)
            np.testing.assert_array_equal(ds_result['int_col'].values, pd_result['int_col'].values)


class TestWhereMaskRowOrderPreservation:
    """
    Test that where/mask SQL pushdown preserves original row order.

    chDB may process row groups in parallel, which can cause rows to be
    returned in different order. We fix this by always adding
    ORDER BY rowNumberInAllBlocks() when no explicit ORDER BY is specified.
    """

    def test_where_preserves_row_order_numeric_only(self):
        """where() on numeric-only DataFrame preserves row order."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.where(df['value'] > 500, 0)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 500, 0).to_df()

            # Row order must be preserved
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)

    def test_mask_preserves_row_order_numeric_only(self):
        """mask() on numeric-only DataFrame preserves row order."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.mask(df['value'] > 500, -1)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['value'] > 500, -1).to_df()

            # Row order must be preserved
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)

    def test_where_with_explicit_sort_uses_user_order(self):
        """where() + sort_values() uses user-specified order, not original."""
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.where(df['value'] > 500, 0).sort_values('value', ascending=False, kind='stable').head(100)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 500, 0).sort_values('value', ascending=False).head(100).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Should match pandas exactly
            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_where_variant_column_sort_raises_error(self):
        """Sorting by Variant column raises error (same as pandas with mixed types)."""
        df = pd.DataFrame(
            {
                'id': range(10),
                'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'str_col': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas also raises error when sorting mixed types
            pd_result = df.where(df['value'] > 50, 0)
            with pytest.raises(TypeError):
                pd_result.sort_values('str_col')

            # DataStore should also raise error
            ds = DataStore.from_file(path)
            with pytest.raises(Exception):  # ClickHouse raises ILLEGAL_COLUMN
                ds.where(ds['value'] > 50, 0).sort_values('str_col').to_df()

    def test_where_preserves_order_with_string_columns(self):
        """where() with Variant type preserves row order."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.where(df['value'] > 500, 0)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 500, 0).to_df()

            # Row order and values must match
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)
            np.testing.assert_array_equal(ds_result['value'].values, pd_result['value'].values)
            # str_col uses Variant type - values should also match
            np.testing.assert_array_equal(ds_result['str_col'].values, pd_result['str_col'].values)


class TestGroupBySortParameter:
    """
    Test GroupBy sort parameter for pandas compatibility.

    Pandas default: groupby(sort=True) - results sorted by group keys
    DataStore now matches this behavior.
    """

    def test_groupby_size_sorted_by_default(self):
        """groupby().size() should be sorted by group keys by default."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'str_col': np.random.choice(['E', 'D', 'C', 'B', 'A'], n),  # Unsorted keys
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas (default sort=True)
            pd_result = df.groupby('str_col').size().reset_index(name='count')

            # DataStore (now also default sort=True)
            ds = DataStore.from_file(path)
            ds_result = ds.groupby('str_col').size().reset_index(name='count')
            ds_result = get_dataframe(ds_result)

            # Order should match (A, B, C, D, E)
            assert pd_result['str_col'].tolist() == ['A', 'B', 'C', 'D', 'E']
            assert ds_result['str_col'].tolist() == ['A', 'B', 'C', 'D', 'E']

    def test_groupby_sort_false_unsorted(self):
        """groupby(sort=False) should not guarantee order."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'str_col': np.random.choice(['E', 'D', 'C', 'B', 'A'], n),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.groupby('str_col', sort=False).size().reset_index(name='count')
            ds_result = get_dataframe(ds_result)

            # Should have all groups (order not guaranteed)
            assert set(ds_result['str_col'].tolist()) == {'A', 'B', 'C', 'D', 'E'}

    def test_groupby_agg_sorted_by_default(self):
        """groupby().agg() should be sorted by group keys by default."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'category': np.random.choice(['cat3', 'cat1', 'cat5', 'cat2', 'cat4'], n),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.groupby('category').agg({'value': 'sum'}).reset_index()

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.groupby('category').agg({'value': 'sum'}).reset_index()
            ds_result = get_dataframe(ds_result)

            # Order should match (cat1, cat2, cat3, cat4, cat5)
            assert pd_result['category'].tolist() == ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']
            assert ds_result['category'].tolist() == ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']

    def test_groupby_count_matches_pandas(self):
        """GroupBy count should match pandas exactly (was failing in benchmark)."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.groupby('str_col').size().reset_index(name='count')

            ds = DataStore.from_file(path)
            ds_result = ds.groupby('str_col').size().reset_index(name='count')
            ds_result = get_dataframe(ds_result)

            # Should be exactly equal
            pd.testing.assert_frame_equal(
                ds_result.reset_index(drop=True), pd_result.reset_index(drop=True), check_dtype=False
            )


class TestSortValuesStability:
    """
    Test sort_values stability with rowNumberInAllBlocks() tie-breaker.

    Default kind='stable' ensures consistent ordering when sort keys have duplicates.
    This matches pandas sort_values(kind='stable') behavior.
    """

    def test_sort_values_stable_by_default(self):
        """sort_values should be stable by default (matching pandas kind='stable')."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),  # Many duplicates
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas with kind='stable'
            pd_result = df.sort_values('int_col', ascending=False, kind='stable').head(100)

            # DataStore (default kind='stable')
            ds = DataStore.from_file(path)
            ds_result = ds.sort_values('int_col', ascending=False).head(100).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Should be exactly equal with stable sort
            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_combined_ops_filter_sort_head_matches_pandas(self):
        """Combined ops (filter+sort+head) should match pandas (was failing in benchmark)."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['int_col'] > 200]
            pd_result = pd_result[['id', 'int_col', 'str_col', 'float_col']]
            pd_result = pd_result.sort_values('int_col', ascending=False, kind='stable')
            pd_result = pd_result.head(100)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 200]
            ds_result = ds_result[['id', 'int_col', 'str_col', 'float_col']]
            ds_result = ds_result.sort_values('int_col', ascending=False)
            ds_result = ds_result.head(100).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Should be exactly equal
            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_sort_with_duplicates_preserves_original_order(self):
        """When sort keys have duplicates, original row order should be preserved."""
        df = pd.DataFrame(
            {
                'id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'value': [100, 100, 100, 200, 200, 200, 300, 300, 300, 400],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas stable sort
            pd_result = df.sort_values('value', ascending=False, kind='stable')

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.sort_values('value', ascending=False).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # For value=300, should be in original order: [6, 7, 8]
            value_300_ids = ds_result[ds_result['value'] == 300]['id'].tolist()
            assert value_300_ids == [6, 7, 8], f"Expected [6, 7, 8], got {value_300_ids}"

            # For value=200, should be in original order: [3, 4, 5]
            value_200_ids = ds_result[ds_result['value'] == 200]['id'].tolist()
            assert value_200_ids == [3, 4, 5], f"Expected [3, 4, 5], got {value_200_ids}"

    def test_filter_select_sort_matches_pandas(self):
        """Filter+Select+Sort should match pandas (was passing in benchmark)."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['int_col'] > 300]
            pd_result = pd_result[['id', 'int_col', 'str_col', 'float_col']]
            pd_result = pd_result.sort_values('int_col', ascending=False, kind='stable')

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 300]
            ds_result = ds_result[['id', 'int_col', 'str_col', 'float_col']]
            ds_result = ds_result.sort_values('int_col', ascending=False).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)


class TestBenchmarkScenarios:
    """
    Reproduce exact benchmark scenarios that were failing.

    These tests verify the fixes work for the specific operations
    that were reported as "Results mismatch" in the benchmark.
    """

    def test_groupby_agg_benchmark_scenario(self):
        """GroupBy agg scenario from benchmark."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
                'category': np.random.choice(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'})
            pd_result = pd_result.reset_index()
            pd_result.columns = ['category', 'int_sum', 'float_avg']

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'})
            ds_result = ds_result.reset_index()
            ds_result = get_dataframe(ds_result)
            ds_result.columns = ['category', 'int_sum', 'float_avg']

            # Sort for comparison
            pd_result = pd_result.sort_values('category').reset_index(drop=True)
            ds_result = ds_result.sort_values('category').reset_index(drop=True)

            # Should match
            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False, rtol=1e-5)

    def test_complex_pipeline_benchmark_scenario(self):
        """Complex pipeline scenario from benchmark."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas - use secondary sort key for deterministic order
            # (SQL sort doesn't guarantee stability, so we need deterministic order)
            pd_result = df.copy()
            pd_result['computed'] = pd_result['int_col'] * 2
            pd_result = pd_result[pd_result['computed'] > 500]
            pd_result = pd_result[['id', 'int_col', 'str_col', 'computed']]
            pd_result = pd_result.sort_values(['computed', 'id'], ascending=[False, True])
            pd_result = pd_result.head(500)

            # DataStore - use same secondary sort key
            ds = DataStore.from_file(path)
            ds['computed'] = ds['int_col'] * 2
            ds_result = ds[ds['computed'] > 500]
            ds_result = ds_result[['id', 'int_col', 'str_col', 'computed']]
            ds_result = ds_result.sort_values(['computed', 'id'], ascending=[False, True])
            ds_result = ds_result.head(500).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_filter_groupby_sort_benchmark_scenario(self):
        """Filter+GroupBy+Sort scenario from benchmark."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
                'category': np.random.choice(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['int_col'] > 200]
            pd_result = pd_result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
            pd_result.columns = ['category', 'int_sum', 'float_avg']
            pd_result = pd_result.sort_values('int_sum', ascending=False, kind='stable')

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 200]
            ds_result = ds_result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
            ds_result = get_dataframe(ds_result)
            ds_result.columns = ['category', 'int_sum', 'float_avg']
            ds_result = ds_result.sort_values('int_sum', ascending=False)

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False, rtol=1e-5)

    def test_multi_filter_benchmark_scenario(self):
        """Multi-filter (4x) scenario from benchmark - was failing due to row order."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas - multi filter
            pd_result = df[
                (df['int_col'] > 200)
                & (df['int_col'] < 800)
                & (df['float_col'] > 100)
                & (df['str_col'].isin(['A', 'B', 'C']))
            ]

            # DataStore - multi filter
            ds = DataStore.from_file(path)
            ds_result = ds[
                (ds['int_col'] > 200)
                & (ds['int_col'] < 800)
                & (ds['float_col'] > 100)
                & (ds['str_col'].isin(['A', 'B', 'C']))
            ].to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Row order and values must match
            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_chain_five_filters_benchmark_scenario(self):
        """Chain 5 filters scenario from benchmark - was failing due to row order."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas - chain filters
            pd_result = df[df['int_col'] > 200]
            pd_result = pd_result[pd_result['int_col'] < 800]
            pd_result = pd_result[pd_result['float_col'] > 100]
            pd_result = pd_result[pd_result['float_col'] < 900]
            pd_result = pd_result[pd_result['str_col'].isin(['A', 'B', 'C'])]

            # DataStore - chain filters
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 200]
            ds_result = ds_result[ds_result['int_col'] < 800]
            ds_result = ds_result[ds_result['float_col'] > 100]
            ds_result = ds_result[ds_result['float_col'] < 900]
            ds_result = ds_result[ds_result['str_col'].isin(['A', 'B', 'C'])].to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_pandas_style_multifilter_sort_head_benchmark_scenario(self):
        """Pandas-style: MultiFilter+Sort+Head scenario from benchmark."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[(df['int_col'] > 200) & (df['int_col'] < 800)]
            pd_result = pd_result.sort_values('int_col', ascending=False, kind='stable')
            pd_result = pd_result.head(100)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[(ds['int_col'] > 200) & (ds['int_col'] < 800)]
            ds_result = ds_result.sort_values('int_col', ascending=False)
            ds_result = ds_result.head(100).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_where_value_replace_benchmark_scenario(self):
        """Where (value replace) scenario from benchmark."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.where(df['int_col'] > 500, 0)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['int_col'] > 500, 0).to_df()

            # Row order and values must match
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)
            np.testing.assert_array_equal(ds_result['int_col'].values, pd_result['int_col'].values)
            np.testing.assert_array_almost_equal(ds_result['float_col'].values, pd_result['float_col'].values)

    def test_filter_where_benchmark_scenario(self):
        """Filter+Where scenario from benchmark."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['float_col'] > 300]
            pd_result = pd_result.where(pd_result['int_col'] > 500, 0)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['float_col'] > 300]
            ds_result = ds_result.where(ds_result['int_col'] > 500, 0).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Row order and values must match
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)
            np.testing.assert_array_equal(ds_result['int_col'].values, pd_result['int_col'].values)
            np.testing.assert_array_almost_equal(ds_result['float_col'].values, pd_result['float_col'].values)

    def test_mask_value_replace_benchmark_scenario(self):
        """Mask (value replace) scenario from benchmark."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.mask(df['int_col'] > 500, -1)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['int_col'] > 500, -1).to_df()

            # Row order and values must match
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)
            np.testing.assert_array_equal(ds_result['int_col'].values, pd_result['int_col'].values)
            np.testing.assert_array_almost_equal(ds_result['float_col'].values, pd_result['float_col'].values)


class TestBoolColumnBehavior:
    """
    Tests for bool column type behavior differences between Pandas and DataStore.

    Note: There are subtle differences in how Pandas and DataStore handle
    bool columns with numeric replacement values:
    - Pandas: promotes to object type to hold mixed int/bool values
    - DataStore: keeps bool type (numeric values are coerced to bool)

    Most tests pass because the values are semantically equivalent.
    """

    def test_where_bool_column_type_behavior(self):
        """
        Test where() on DataFrame with bool column.

        Note: Type behavior differs between Pandas and DataStore:
        - Pandas: converts bool column to object type, replaces with int 0
        - DataStore: keeps bool type, replaces with False

        This test verifies the values match (0 == False in boolean context).
        """
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'bool_col': np.random.choice([True, False], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.where(df['int_col'] > 500, 0)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['int_col'] > 500, 0).to_df()

            # Pandas converts bool to object, DataStore keeps bool
            # But values should be semantically equivalent (0 == False)
            assert pd_result['bool_col'].dtype == 'object'  # Pandas: mixed int/bool
            # Compare as bool (0 and False are equivalent)
            pd_bool = pd_result['bool_col'].astype(bool)
            ds_bool = ds_result['bool_col'].astype(bool)
            np.testing.assert_array_equal(ds_bool.values, pd_bool.values)

    def test_mask_bool_column_type_mismatch(self):
        """
        Test: mask() on DataFrame with bool column matches Pandas.

        When using mask(cond, -1) on a DataFrame with bool column:
        - Pandas: converts bool column to object type, replaces with int -1
        - DataStore: now falls back to Pandas execution for bool columns with numeric other,
          ensuring the same behavior and object dtype conversion.
        """
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'bool_col': np.random.choice([True, False], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            pd_result = df.mask(df['int_col'] > 500, -1)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['int_col'] > 500, -1).to_df()

            # This test documents the mismatch
            np.testing.assert_array_equal(ds_result['bool_col'].values, pd_result['bool_col'].values)

    def test_ultra_complex_operations_row_order(self):
        """
        Test ultra-complex operations (10+ chained ops) row order.

        This tests very complex operation chains that include:
        - Multiple filters
        - Column assignments
        - Sort operations
        - Head/limit operations

        The subquery-based row order fix handles these cases correctly.
        """
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'float_col': np.random.uniform(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas - ultra-complex pipeline
            # Use secondary sort key for deterministic order (SQL sort isn't stable)
            pd_result = df.copy()
            pd_result['computed'] = pd_result['int_col'] * 2
            pd_result['computed2'] = pd_result['float_col'] / 2
            pd_result = pd_result[pd_result['int_col'] > 100]
            pd_result = pd_result[pd_result['int_col'] < 900]
            pd_result = pd_result[pd_result['float_col'] > 100]
            pd_result = pd_result[pd_result['float_col'] < 900]
            pd_result = pd_result[pd_result['str_col'].isin(['A', 'B', 'C'])]
            pd_result = pd_result[pd_result['computed'] > 300]
            pd_result = pd_result[pd_result['computed2'] < 400]
            pd_result = pd_result.sort_values(['int_col', 'id'], ascending=[False, True])
            pd_result = pd_result.head(100)

            # DataStore - use same secondary sort key
            ds = DataStore.from_file(path)
            ds['computed'] = ds['int_col'] * 2
            ds['computed2'] = ds['float_col'] / 2
            ds_result = ds[ds['int_col'] > 100]
            ds_result = ds_result[ds_result['int_col'] < 900]
            ds_result = ds_result[ds_result['float_col'] > 100]
            ds_result = ds_result[ds_result['float_col'] < 900]
            ds_result = ds_result[ds_result['str_col'].isin(['A', 'B', 'C'])]
            ds_result = ds_result[ds_result['computed'] > 300]
            ds_result = ds_result[ds_result['computed2'] < 400]
            ds_result = ds_result.sort_values(['int_col', 'id'], ascending=[False, True])
            ds_result = ds_result.head(100).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)


class TestMultiColumnSortWithDifferentAscending:
    """Tests for multi-column sort_values with different ascending per column."""

    def test_sort_two_columns_different_ascending(self):
        """Test sort_values with two columns and different ascending values."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                'id': range(100),
                'category': np.random.choice(['A', 'B', 'C'], 100),
                'value': np.random.randint(0, 50, 100),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.sort_values(['category', 'value'], ascending=[True, False], kind='stable')

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.sort_values(['category', 'value'], ascending=[True, False]).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_sort_three_columns_mixed_ascending(self):
        """Test sort_values with three columns and mixed ascending values."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                'id': range(200),
                'cat1': np.random.choice(['X', 'Y', 'Z'], 200),
                'cat2': np.random.choice(['P', 'Q'], 200),
                'value': np.random.randint(0, 100, 200),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas: cat1 ASC, cat2 DESC, value ASC
            pd_result = df.sort_values(['cat1', 'cat2', 'value'], ascending=[True, False, True], kind='stable')

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.sort_values(['cat1', 'cat2', 'value'], ascending=[True, False, True]).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_sort_with_filter_and_different_ascending(self):
        """Test sort_values with filter and different ascending per column."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                'id': range(500),
                'int_col': np.random.randint(0, 1000, 500),
                'category': np.random.choice(['cat1', 'cat2', 'cat3'], 500),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['int_col'] > 300]
            pd_result = pd_result.sort_values(['category', 'int_col'], ascending=[True, False], kind='stable')
            pd_result = pd_result.head(50)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 300]
            ds_result = ds_result.sort_values(['category', 'int_col'], ascending=[True, False])
            ds_result = ds_result.head(50).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)


class TestStableSortWithFilter:
    """Tests for stable sort behavior when combined with filter operations."""

    def test_stable_sort_preserves_original_order_after_filter(self):
        """Test that stable sort preserves original row order as tie-breaker after filtering."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'category': np.repeat(['A', 'B', 'C', 'D', 'E'], n // 5),  # Many duplicates
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['int_col'] > 500]
            pd_result = pd_result.sort_values('category', ascending=True, kind='stable')
            pd_result = pd_result.head(100)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 500]
            ds_result = ds_result.sort_values('category', ascending=True)
            ds_result = ds_result.head(100).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # IDs should match exactly (stable sort preserves original order)
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)

    def test_stable_sort_single_column_with_multiple_filters(self):
        """Test stable sort with multiple filter conditions."""
        np.random.seed(42)
        n = 2000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['int_col'] > 100]
            pd_result = pd_result[pd_result['int_col'] < 900]
            pd_result = pd_result[pd_result['float_col'] > 50]
            pd_result = pd_result[pd_result['str_col'].isin(['A', 'B', 'C', 'D'])]
            pd_result = pd_result.sort_values('int_col', ascending=False, kind='stable')
            pd_result = pd_result.head(200)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 100]
            ds_result = ds_result[ds_result['int_col'] < 900]
            ds_result = ds_result[ds_result['float_col'] > 50]
            ds_result = ds_result[ds_result['str_col'].isin(['A', 'B', 'C', 'D'])]
            ds_result = ds_result.sort_values('int_col', ascending=False)
            ds_result = ds_result.head(200).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)


class TestBoolColumnWhereMaskFallback:
    """Tests for bool column where/mask falling back to Pandas for type alignment."""

    def test_where_bool_column_falls_back_to_pandas(self):
        """Test that where() on bool column with numeric other falls back to Pandas."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                'id': range(50),
                'int_col': np.random.randint(0, 100, 50),
                'bool_col': np.random.choice([True, False], 50),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.where(df['int_col'] > 50, 0)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['int_col'] > 50, 0).to_df()

            # bool_col should be object type (like Pandas) not bool
            assert pd_result['bool_col'].dtype == object
            assert ds_result['bool_col'].dtype == object

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=True)

    def test_mask_bool_column_falls_back_to_pandas(self):
        """Test that mask() on bool column with numeric other falls back to Pandas."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                'id': range(50),
                'int_col': np.random.randint(0, 100, 50),
                'bool_col': np.random.choice([True, False], 50),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.mask(df['int_col'] > 50, -1)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['int_col'] > 50, -1).to_df()

            # bool_col should be object type (like Pandas)
            assert pd_result['bool_col'].dtype == object
            assert ds_result['bool_col'].dtype == object

            pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=True)

    def test_where_no_bool_column_uses_sql(self):
        """Test that where() without bool column can still use SQL pushdown."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                'id': range(50),
                'int_col': np.random.randint(0, 100, 50),
                'str_col': np.random.choice(['A', 'B', 'C'], 50),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.where(df['int_col'] > 50, 0)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['int_col'] > 50, 0).to_df()

            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Values should match (int columns replaced with 0)
            np.testing.assert_array_equal(ds_result['int_col'].values, pd_result['int_col'].values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
