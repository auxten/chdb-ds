"""
Tests for pandas DataFrame compatibility layer in DataStore.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np

from datastore import DataStore


class TestPandasCompatibility(unittest.TestCase):
    """Test pandas DataFrame compatibility methods."""

    @classmethod
    def setUpClass(cls):
        """Create test data files."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        # Create sample CSV
        with open(cls.csv_file, "w") as f:
            f.write("id,name,age,salary,department,active\n")
            f.write("1,Alice,25,50000,Engineering,1\n")
            f.write("2,Bob,30,60000,Sales,1\n")
            f.write("3,Charlie,35,70000,Engineering,1\n")
            f.write("4,David,28,55000,Marketing,0\n")
            f.write("5,Eve,32,65000,Sales,1\n")
            f.write("6,Frank,29,58000,Engineering,1\n")
            f.write("7,Grace,31,62000,Marketing,1\n")
            f.write("8,Henry,27,52000,Sales,0\n")
            f.write("9,Iris,33,68000,Engineering,1\n")
            f.write("10,Jack,26,51000,Marketing,1\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def setUp(self):
        """Set up test DataStore and reference DataFrame."""
        self.ds = DataStore.from_file(self.csv_file)
        self.df = pd.read_csv(self.csv_file)  # Reference pandas DataFrame

    # ========== Properties Tests ==========

    def test_dtypes(self):
        """Test dtypes property matches pandas."""
        ds_dtypes = self.ds.dtypes
        pd_dtypes = self.df.dtypes
        # Column names should match
        self.assertEqual(list(ds_dtypes.index), list(pd_dtypes.index))

    def test_shape(self):
        """Test shape property matches pandas."""
        self.assertEqual(self.ds.shape, self.df.shape)

    def test_columns(self):
        """Test columns property matches pandas."""
        pd.testing.assert_index_equal(self.ds.columns, self.df.columns)

    def test_values(self):
        """Test values property matches pandas."""
        np.testing.assert_array_equal(self.ds.values, self.df.values)

    def test_empty(self):
        """Test empty property matches pandas."""
        self.assertEqual(self.ds.empty, self.df.empty)

    def test_size(self):
        """Test size property matches pandas."""
        self.assertEqual(self.ds.size, self.df.size)

    # ========== Statistical Methods Tests ==========

    def test_mean(self):
        """Test mean method matches pandas."""
        ds_mean = self.ds.mean(numeric_only=True)
        pd_mean = self.df.mean(numeric_only=True)
        pd.testing.assert_series_equal(ds_mean, pd_mean, check_names=False)

    def test_median(self):
        """Test median method matches pandas."""
        ds_median = self.ds.median(numeric_only=True)
        pd_median = self.df.median(numeric_only=True)
        pd.testing.assert_series_equal(ds_median, pd_median, check_names=False)

    def test_std(self):
        """Test std method matches pandas."""
        ds_std = self.ds.std(numeric_only=True)
        pd_std = self.df.std(numeric_only=True)
        pd.testing.assert_series_equal(ds_std, pd_std, check_names=False, rtol=1e-5)

    def test_min_max(self):
        """Test min and max methods match pandas."""
        ds_min = self.ds.min(numeric_only=True)
        pd_min = self.df.min(numeric_only=True)
        pd.testing.assert_series_equal(ds_min, pd_min, check_names=False)

        ds_max = self.ds.max(numeric_only=True)
        pd_max = self.df.max(numeric_only=True)
        pd.testing.assert_series_equal(ds_max, pd_max, check_names=False)

    def test_sum(self):
        """Test sum method matches pandas."""
        ds_sum = self.ds.sum(numeric_only=True)
        pd_sum = self.df.sum(numeric_only=True)
        pd.testing.assert_series_equal(ds_sum, pd_sum, check_names=False)

    def test_corr(self):
        """Test correlation method matches pandas."""
        ds_corr = self.ds.corr(numeric_only=True)
        pd_corr = self.df.corr(numeric_only=True)
        pd.testing.assert_frame_equal(ds_corr, pd_corr, rtol=1e-5)

    def test_quantile(self):
        """Test quantile method matches pandas."""
        ds_q50 = self.ds.quantile(0.5, numeric_only=True)
        pd_q50 = self.df.quantile(0.5, numeric_only=True)
        pd.testing.assert_series_equal(ds_q50, pd_q50, check_names=False)

    def test_nunique(self):
        """Test nunique method matches pandas."""
        ds_nunique = self.ds.nunique()
        pd_nunique = self.df.nunique()
        pd.testing.assert_series_equal(ds_nunique, pd_nunique)

    # ========== Data Manipulation Tests ==========

    def test_drop_columns(self):
        """Test drop method matches pandas."""
        ds_result = self.ds.drop(columns=['active'])
        pd_result = self.df.drop(columns=['active'])
        self.assertTrue(ds_result.equals(pd_result))

    def test_drop_duplicates(self):
        """Test drop_duplicates matches pandas."""
        ds_result = self.ds.drop_duplicates(subset=['department'])
        pd_result = self.df.drop_duplicates(subset=['department'])
        # Compare length (order may differ due to implementation)
        self.assertEqual(len(ds_result), len(pd_result))

    def test_dropna(self):
        """Test dropna matches pandas."""
        ds_result = self.ds.dropna()
        pd_result = self.df.dropna()
        self.assertTrue(ds_result.equals(pd_result))

    def test_fillna(self):
        """Test fillna matches pandas."""
        ds_result = self.ds.fillna(0)
        pd_result = self.df.fillna(0)
        self.assertTrue(ds_result.equals(pd_result))

    def test_rename(self):
        """Test rename matches pandas."""
        ds_result = self.ds.rename(columns={'name': 'employee_name'})
        pd_result = self.df.rename(columns={'name': 'employee_name'})
        self.assertTrue(ds_result.equals(pd_result))

    def test_sort_values(self):
        """Test sort_values matches pandas."""
        ds_result = self.ds.sort_values('age').reset_index(drop=True)
        pd_result = self.df.sort_values('age').reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    def test_reset_index(self):
        """Test reset_index matches pandas."""
        ds_result = self.ds.reset_index(drop=True)
        pd_result = self.df.reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    def test_assign(self):
        """Test assign matches pandas."""
        ds_result = self.ds.assign(bonus=lambda x: x['salary'] * 0.1)
        pd_result = self.df.assign(bonus=lambda x: x['salary'] * 0.1)
        self.assertTrue(ds_result.equals(pd_result))

    def test_nlargest(self):
        """Test nlargest matches pandas."""
        ds_result = self.ds.nlargest(3, 'salary').reset_index(drop=True)
        pd_result = self.df.nlargest(3, 'salary').reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    def test_nsmallest(self):
        """Test nsmallest matches pandas."""
        ds_result = self.ds.nsmallest(3, 'age').reset_index(drop=True)
        pd_result = self.df.nsmallest(3, 'age').reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    # ========== Function Application Tests ==========

    def test_apply(self):
        """Test apply method matches pandas."""
        func = lambda x: x * 2 if x.dtype in ['int64', 'float64'] else x
        ds_result = self.ds.apply(func, axis=0)
        pd_result = self.df.apply(func, axis=0)
        self.assertTrue(ds_result.equals(pd_result))

    def test_agg(self):
        """Test aggregate method matches pandas."""
        ds_result = self.ds.agg({'age': 'mean', 'salary': 'sum'})
        pd_result = self.df.agg({'age': 'mean', 'salary': 'sum'})
        pd.testing.assert_series_equal(ds_result, pd_result)

    # ========== Indexing Tests ==========

    def test_loc(self):
        """Test loc indexer."""
        loc_indexer = self.ds.loc
        # Just verify it returns the pandas loc indexer
        self.assertIsNotNone(loc_indexer)

    def test_iloc(self):
        """Test iloc indexer."""
        iloc_indexer = self.ds.iloc
        # Just verify it returns the pandas iloc indexer
        self.assertIsNotNone(iloc_indexer)

    def test_getitem_column(self):
        """Test column selection with [] - returns ColumnExpr that displays like Series."""
        from datastore.expressions import Field
        from datastore.column_expr import ColumnExpr

        # ds['col'] returns ColumnExpr that wraps a Field
        result = self.ds['name']
        self.assertIsInstance(result, ColumnExpr)  # Returns ColumnExpr
        self.assertIsInstance(result._expr, Field)  # Wrapping a Field

        # ColumnExpr materializes and displays actual values like Series
        self.assertIsInstance(result._materialize(), pd.Series)

        # To get actual Series, use to_df() first (still works)
        series = self.ds.to_df()['name']
        self.assertIsInstance(series, pd.Series)

    def test_getitem_columns(self):
        """Test multiple column selection with [] matches pandas."""
        ds_result = self.ds[['name', 'age']]
        pd_result = self.df[['name', 'age']]
        self.assertTrue(ds_result.equals(pd_result))

    # ========== Transformation Tests ==========

    def test_abs(self):
        """Test abs method matches pandas."""
        ds_result = self.ds.abs()
        pd_result = self.df.select_dtypes(include=[np.number]).abs()
        # Compare only numeric columns using values
        np.testing.assert_array_equal(ds_result.select_dtypes(include=[np.number]).values, pd_result.values)

    def test_round(self):
        """Test round method matches pandas."""
        ds_result = self.ds.round(decimals=2)
        pd_result = self.df.round(decimals=2)
        self.assertTrue(ds_result.equals(pd_result))

    def test_transpose(self):
        """Test transpose method matches pandas."""
        ds_result = self.ds.transpose()
        pd_result = self.df.transpose()
        # Check shape matches
        self.assertEqual(ds_result.shape, pd_result.shape)

    # ========== Reshaping Tests ==========

    def test_melt(self):
        """Test melt method matches pandas."""
        ds_result = self.ds.melt(id_vars=['id'], value_vars=['age', 'salary'])
        pd_result = self.df.melt(id_vars=['id'], value_vars=['age', 'salary'])
        # Sort and compare using equals
        ds_sorted = ds_result.sort_values(['id', 'variable']).reset_index(drop=True)
        pd_sorted = pd_result.sort_values(['id', 'variable']).reset_index(drop=True)
        self.assertTrue(ds_sorted.equals(pd_sorted))

    # ========== Boolean Methods Tests ==========

    def test_isna(self):
        """Test isna method matches pandas."""
        ds_result = self.ds.isna()
        pd_result = self.df.isna()
        self.assertTrue(ds_result.equals(pd_result))

    def test_isna_sum(self):
        """Test isna().sum() matches pandas."""
        ds_result = self.ds.isna().sum()
        pd_result = self.df.isna().sum()
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_notna(self):
        """Test notna method matches pandas."""
        ds_result = self.ds.notna()
        pd_result = self.df.notna()
        self.assertTrue(ds_result.equals(pd_result))

    # ========== Conversion Tests ==========

    def test_astype(self):
        """Test astype method matches pandas."""
        ds_result = self.ds.astype({'age': 'float64'})
        pd_result = self.df.astype({'age': 'float64'})
        self.assertTrue(ds_result.equals(pd_result))

    def test_copy(self):
        """Test copy method."""
        result = self.ds.copy()
        self.assertIsInstance(result, DataStore)
        # Verify it's a different object
        self.assertIsNot(result, self.ds)

    # ========== IO Tests ==========

    def test_to_csv(self):
        """Test to_csv method."""
        output_file = os.path.join(self.temp_dir, "output.csv")
        try:
            self.ds.to_csv(output_file, index=False)
            self.assertTrue(os.path.exists(output_file))
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_to_json(self):
        """Test to_json method."""
        output_file = os.path.join(self.temp_dir, "output.json")
        try:
            self.ds.to_json(output_file, orient='records')
            self.assertTrue(os.path.exists(output_file))
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_to_dict(self):
        """Test to_dict method (from existing API)."""
        result = self.ds.to_dict()
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, dict) for item in result))

    def test_to_numpy(self):
        """Test to_numpy method matches pandas."""
        ds_arr = self.ds.to_numpy()
        pd_arr = self.df.to_numpy()
        np.testing.assert_array_equal(ds_arr, pd_arr)

    # ========== Iteration Tests ==========

    def test_iterrows(self):
        """Test iterrows method matches pandas."""
        ds_rows = list(self.ds.iterrows())
        pd_rows = list(self.df.iterrows())
        self.assertEqual(len(ds_rows), len(pd_rows))
        for (ds_idx, ds_row), (pd_idx, pd_row) in zip(ds_rows, pd_rows):
            pd.testing.assert_series_equal(ds_row, pd_row)

    def test_itertuples(self):
        """Test itertuples method matches pandas."""
        tuples = list(self.ds.itertuples())
        self.assertEqual(len(tuples), 10)

    # ========== Merge Tests ==========

    def test_merge(self):
        """Test merge method with another DataStore."""
        # Create second dataset
        csv_file2 = os.path.join(self.temp_dir, "test_data2.csv")
        with open(csv_file2, "w") as f:
            f.write("id,bonus\n")
            f.write("1,5000\n")
            f.write("2,6000\n")
            f.write("3,7000\n")

        try:
            ds2 = DataStore.from_file(csv_file2)
            result = self.ds.merge(ds2, on='id', how='inner')
            self.assertIsInstance(result, DataStore)
            df = result.to_df()
            self.assertIn('bonus', df.columns)
        finally:
            if os.path.exists(csv_file2):
                os.unlink(csv_file2)

    # ========== Comparison Tests ==========

    def test_equals(self):
        """Test equals method."""
        ds2 = DataStore.from_file(self.csv_file)
        self.assertTrue(self.ds.equals(ds2))

    # ========== Inplace Parameter Tests ==========

    def test_inplace_not_supported(self):
        """Test that inplace=True raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.ds.drop(columns=['age'], inplace=True)
        self.assertIn("immutable", str(cm.exception).lower())

    def test_fillna_inplace_not_supported(self):
        """Test that fillna with inplace=True raises ValueError."""
        with self.assertRaises(ValueError):
            self.ds.fillna(0, inplace=True)

    def test_rename_inplace_not_supported(self):
        """Test that rename with inplace=True raises ValueError."""
        with self.assertRaises(ValueError):
            self.ds.rename(columns={'name': 'new_name'}, inplace=True)


class TestPandasCompatChaining(unittest.TestCase):
    """Test chaining of pandas compatibility methods with DataStore methods."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "chain_test.csv")

        with open(cls.csv_file, "w") as f:
            f.write("id,value,category\n")
            f.write("1,100,A\n")
            f.write("2,200,B\n")
            f.write("3,150,A\n")
            f.write("4,250,B\n")
            f.write("5,180,A\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_chaining_pandas_with_datastore(self):
        """Test chaining pandas methods with DataStore methods."""
        ds = DataStore.from_file(self.csv_file)

        # Chain: select -> filter -> sort_values
        # Note: head() returns DataStore for method chaining
        result = ds.select('id', 'value', 'category').filter(ds.value > 100).sort_values('value', ascending=False)

        self.assertIsInstance(result, DataStore)

        # Now apply head() which returns DataStore for chaining
        df_head = result.head(3)
        self.assertIsInstance(df_head, DataStore)

    def test_pandas_methods_return_datastore(self):
        """Test that pandas methods return DataStore for chaining."""
        ds = DataStore.from_file(self.csv_file)

        # Apply pandas operations
        result = ds.fillna(0).drop_duplicates().sort_values('value')

        self.assertIsInstance(result, DataStore)


if __name__ == '__main__':
    unittest.main()
