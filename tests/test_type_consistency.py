"""
Test type consistency between DataStore and pandas.

This module tests that DataStore operations return the correct types:
- DataFrame operations return DataStore (equivalent to pd.DataFrame)
- Series operations return ColumnExpr (equivalent to pd.Series)
- Aggregations return ColumnExpr (for SQL building compatibility)

Design Decision: ColumnExpr for Aggregations
============================================
Unlike pandas where df['col'].sum() returns scalar, DataStore returns ColumnExpr.
This is by design for SQL building compatibility:
  ds.groupby('x').agg(avg=ds['value'].mean())

ColumnExpr behaves like scalar in most operations:
- Arithmetic: ds['a'].sum() + 10 works
- Coercion: int(ds['a'].sum()) works
- Display: print(ds['a'].sum()) shows value
- Comparison: ds['a'].sum() > 5 works
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from datastore.column_expr import ColumnExpr


class TestIndexingTypeConsistency:
    """Test that indexing operations return correct types."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': ['x', 'y', 'z']})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': ['x', 'y', 'z']})

    def test_single_column_returns_columnexpr(self, ds, df):
        """ds['col'] should return ColumnExpr (like pd.Series)."""
        ds_result = ds['a']
        pd_result = df['a']

        assert isinstance(ds_result, ColumnExpr), f"Expected ColumnExpr, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)

    def test_multi_column_returns_datastore(self, ds, df):
        """ds[['col1', 'col2']] should return DataStore (like pd.DataFrame)."""
        ds_result = ds[['a', 'b']]
        pd_result = df[['a', 'b']]

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_boolean_indexing_returns_datastore(self, ds, df):
        """ds[condition] should return DataStore (like pd.DataFrame)."""
        ds_result = ds[ds['a'] > 1]
        pd_result = df[df['a'] > 1]

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_slice_returns_datastore(self, ds, df):
        """ds[:n] should return DataStore (like pd.DataFrame)."""
        ds_result = ds[:2]
        pd_result = df[:2]

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_integer_column_name_returns_columnexpr(self):
        """ds[0] should return ColumnExpr when column name is integer."""
        ds = DataStore({0: [1, 2, 3], 1: [4, 5, 6]})
        df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})

        ds_result = ds[0]
        pd_result = df[0]

        assert isinstance(ds_result, ColumnExpr), f"Expected ColumnExpr, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)


class TestAggregationReturnsColumnExpr:
    """Test that aggregations return ColumnExpr (for SQL building)."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0], 'cat': ['A', 'A', 'B', 'B', 'B']})

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0], 'cat': ['A', 'A', 'B', 'B', 'B']}
        )

    # ========== No groupby - returns ColumnExpr ==========

    def test_sum_returns_columnexpr(self, ds):
        """ds['col'].sum() returns ColumnExpr for SQL building compatibility."""
        result = ds['a'].sum()
        assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result).__name__}"

    def test_mean_returns_columnexpr(self, ds):
        """ds['col'].mean() returns ColumnExpr for SQL building compatibility."""
        result = ds['a'].mean()
        assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result).__name__}"

    def test_min_returns_columnexpr(self, ds):
        """ds['col'].min() returns ColumnExpr."""
        result = ds['a'].min()
        assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result).__name__}"

    def test_max_returns_columnexpr(self, ds):
        """ds['col'].max() returns ColumnExpr."""
        result = ds['a'].max()
        assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result).__name__}"

    def test_count_returns_columnexpr(self, ds):
        """ds['col'].count() returns ColumnExpr."""
        result = ds['a'].count()
        assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result).__name__}"

    # ========== With groupby - also returns ColumnExpr ==========

    def test_groupby_sum_returns_columnexpr(self, ds):
        """ds.groupby('cat')['col'].sum() returns ColumnExpr."""
        result = ds.groupby('cat')['a'].sum()
        assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result).__name__}"


class TestColumnExprBehavesLikeScalar:
    """Test that ColumnExpr aggregations behave like scalars."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3]})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3]})

    def test_add_to_aggregation(self, ds, df):
        """ds['col'].sum() + 10 should work and give correct result."""
        ds_result = ds['a'].sum() + 10
        pd_result = df['a'].sum() + 10
        assert ds_result == pd_result == 16

    def test_subtract_from_aggregation(self, ds, df):
        """ds['col'].sum() - 1 should work."""
        ds_result = ds['a'].sum() - 1
        pd_result = df['a'].sum() - 1
        assert ds_result == pd_result == 5

    def test_multiply_aggregation(self, ds, df):
        """ds['col'].sum() * 2 should work."""
        ds_result = ds['a'].sum() * 2
        pd_result = df['a'].sum() * 2
        assert ds_result == pd_result == 12

    def test_divide_aggregation(self, ds, df):
        """ds['col'].sum() / 2 should work."""
        ds_result = ds['a'].sum() / 2
        pd_result = df['a'].sum() / 2
        assert ds_result == pd_result == 3.0

    def test_int_coercion(self, ds, df):
        """int(ds['col'].sum()) should work."""
        ds_result = int(ds['a'].sum())
        pd_result = int(df['a'].sum())
        assert ds_result == pd_result == 6

    def test_float_coercion(self, ds, df):
        """float(ds['col'].mean()) should work."""
        ds_result = float(ds['a'].mean())
        pd_result = float(df['a'].mean())
        assert ds_result == pd_result == 2.0

    def test_compare_to_scalar(self, ds, df):
        """ds['col'].sum() > 5 should work."""
        ds_result = ds['a'].sum() > 5
        pd_result = df['a'].sum() > 5
        assert ds_result == pd_result == True

    def test_compare_less_than(self, ds, df):
        """ds['col'].sum() < 10 should work."""
        ds_result = ds['a'].sum() < 10
        pd_result = df['a'].sum() < 10
        assert ds_result == pd_result == True

    def test_repr_shows_value(self, ds, df):
        """repr(ds['col'].sum()) should show the value."""
        ds_repr = repr(ds['a'].sum())
        pd_repr = repr(df['a'].sum())
        # Both should contain '6'
        assert '6' in ds_repr
        assert '6' in pd_repr


class TestAggregationValueCorrectness:
    """Test that aggregation values match pandas."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0]})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0]})

    def test_sum_value(self, ds, df):
        """Sum values should match."""
        ds_val = int(ds['a'].sum())
        pd_val = int(df['a'].sum())
        assert ds_val == pd_val == 15

    def test_mean_value(self, ds, df):
        """Mean values should match."""
        ds_val = float(ds['a'].mean())
        pd_val = float(df['a'].mean())
        assert ds_val == pd_val == 3.0

    def test_min_value(self, ds, df):
        """Min values should match."""
        ds_val = int(ds['a'].min())
        pd_val = int(df['a'].min())
        assert ds_val == pd_val == 1

    def test_max_value(self, ds, df):
        """Max values should match."""
        ds_val = int(ds['a'].max())
        pd_val = int(df['a'].max())
        assert ds_val == pd_val == 5

    def test_count_value(self, ds, df):
        """Count values should match."""
        ds_val = int(ds['a'].count())
        pd_val = int(df['a'].count())
        assert ds_val == pd_val == 5

    def test_std_value(self, ds, df):
        """Std values should match."""
        ds_val = float(ds['a'].std())
        pd_val = float(df['a'].std())
        assert np.isclose(ds_val, pd_val)

    def test_var_value(self, ds, df):
        """Var values should match."""
        ds_val = float(ds['a'].var())
        pd_val = float(df['a'].var())
        assert np.isclose(ds_val, pd_val)

    def test_median_value(self, ds, df):
        """Median values should match."""
        ds_val = float(ds['a'].median())
        pd_val = float(df['a'].median())
        assert ds_val == pd_val == 3.0

    def test_prod_value(self, ds, df):
        """Prod values should match."""
        ds_val = int(ds['a'].prod())
        pd_val = int(df['a'].prod())
        assert ds_val == pd_val == 120


class TestDataFrameOperationTypes:
    """Test that DataFrame-level operations return correct types."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def test_head_returns_datastore(self, ds, df):
        """ds.head() should return DataStore."""
        ds_result = ds.head()
        pd_result = df.head()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_tail_returns_datastore(self, ds, df):
        """ds.tail() should return DataStore."""
        ds_result = ds.tail()
        pd_result = df.tail()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_dropna_returns_datastore(self, ds, df):
        """ds.dropna() should return DataStore."""
        ds_result = ds.dropna()
        pd_result = df.dropna()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_assign_returns_datastore(self, ds, df):
        """ds.assign(c=1) should return DataStore."""
        ds_result = ds.assign(c=1)
        pd_result = df.assign(c=1)

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_sample_returns_datastore(self, ds, df):
        """ds.sample(2) should return DataStore."""
        ds_result = ds.sample(2)
        pd_result = df.sample(2)

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_describe_returns_datastore(self, ds, df):
        """ds.describe() should return DataStore."""
        ds_result = ds.describe()
        pd_result = df.describe()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_dataframe_sum_returns_series(self, ds, df):
        """ds.sum() (DataFrame-level) should return Series."""
        ds_result = ds.sum()
        pd_result = df.sum()

        # DataStore.sum() returns Series (not DataStore)
        assert isinstance(ds_result, pd.Series), f"Expected Series, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)

    def test_dataframe_mean_returns_series(self, ds, df):
        """ds.mean() (DataFrame-level) should return Series."""
        ds_result = ds.mean()
        pd_result = df.mean()

        assert isinstance(ds_result, pd.Series), f"Expected Series, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)


class TestSQLBuildingCompatibility:
    """Test that aggregations work in SQL building context."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3, 4, 5], 'cat': ['A', 'A', 'B', 'B', 'B']})

    def test_aggregation_has_expr_for_sql(self, ds):
        """Aggregation ColumnExpr should have _expr for SQL building."""
        result = ds['a'].sum()
        assert hasattr(result, '_expr'), "Aggregation ColumnExpr should have _expr"

    def test_aggregation_can_use_as_alias(self, ds):
        """Aggregation ColumnExpr should support .as_() for SQL aliases."""
        result = ds['a'].sum()
        # Should have as_ method for aliasing
        assert hasattr(result, 'as_'), "Aggregation ColumnExpr should have as_() method"
