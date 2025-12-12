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
        """Set up test DataStore."""
        self.ds = DataStore.from_file(self.csv_file)

    # ========== Properties Tests ==========

    def test_dtypes(self):
        """Test dtypes property."""
        dtypes = self.ds.dtypes
        self.assertIsInstance(dtypes, pd.Series)
        self.assertIn('id', dtypes.index)
        self.assertIn('name', dtypes.index)

    def test_shape(self):
        """Test shape property."""
        shape = self.ds.shape
        self.assertEqual(shape, (10, 6))  # 10 rows, 6 columns

    def test_columns(self):
        """Test columns property."""
        cols = self.ds.columns
        self.assertIsInstance(cols, pd.Index)
        self.assertEqual(len(cols), 6)
        self.assertIn('id', cols)
        self.assertIn('name', cols)

    def test_values(self):
        """Test values property (pandas compatible)."""
        values = self.ds.values
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.shape, (10, 6))

    def test_empty(self):
        """Test empty property."""
        self.assertFalse(self.ds.empty)

    def test_size(self):
        """Test size property."""
        size = self.ds.size
        self.assertEqual(size, 60)  # 10 rows * 6 columns

    # ========== Statistical Methods Tests ==========

    def test_mean(self):
        """Test mean method."""
        mean_vals = self.ds.mean(numeric_only=True)
        self.assertIsInstance(mean_vals, pd.Series)
        self.assertIn('age', mean_vals.index)
        self.assertIn('salary', mean_vals.index)

    def test_median(self):
        """Test median method."""
        median_vals = self.ds.median(numeric_only=True)
        self.assertIsInstance(median_vals, pd.Series)

    def test_std(self):
        """Test std method."""
        std_vals = self.ds.std(numeric_only=True)
        self.assertIsInstance(std_vals, pd.Series)

    def test_min_max(self):
        """Test min and max methods."""
        min_vals = self.ds.min(numeric_only=True)
        max_vals = self.ds.max(numeric_only=True)
        self.assertIsInstance(min_vals, pd.Series)
        self.assertIsInstance(max_vals, pd.Series)

    def test_sum(self):
        """Test sum method."""
        sum_vals = self.ds.sum(numeric_only=True)
        self.assertIsInstance(sum_vals, pd.Series)

    def test_corr(self):
        """Test correlation method."""
        corr_matrix = self.ds.corr(numeric_only=True)
        self.assertIsInstance(corr_matrix, pd.DataFrame)

    def test_quantile(self):
        """Test quantile method."""
        q50 = self.ds.quantile(0.5, numeric_only=True)
        self.assertIsInstance(q50, pd.Series)

    def test_nunique(self):
        """Test nunique method."""
        unique_counts = self.ds.nunique()
        self.assertIsInstance(unique_counts, pd.Series)

    # ========== Data Manipulation Tests ==========

    def test_drop_columns(self):
        """Test drop method for columns."""
        result = self.ds.drop(columns=['active'])
        self.assertIsInstance(result, DataStore)
        df = result._get_df()
        self.assertNotIn('active', df.columns)
        self.assertIn('name', df.columns)

    def test_drop_duplicates(self):
        """Test drop_duplicates method."""
        result = self.ds.drop_duplicates(subset=['department'])
        self.assertIsInstance(result, DataStore)

    def test_dropna(self):
        """Test dropna method."""
        result = self.ds.dropna()
        self.assertIsInstance(result, DataStore)

    def test_fillna(self):
        """Test fillna method."""
        result = self.ds.fillna(0)
        self.assertIsInstance(result, DataStore)

    def test_rename(self):
        """Test rename method."""
        result = self.ds.rename(columns={'name': 'employee_name'})
        self.assertIsInstance(result, DataStore)
        df = result._get_df()
        self.assertIn('employee_name', df.columns)
        self.assertNotIn('name', df.columns)

    def test_sort_values(self):
        """Test sort_values method."""
        result = self.ds.sort_values('age')
        self.assertIsInstance(result, DataStore)
        df = result._get_df()
        # Check if sorted
        ages = df['age'].tolist()
        self.assertEqual(ages, sorted(ages))

    def test_reset_index(self):
        """Test reset_index method."""
        result = self.ds.reset_index(drop=True)
        self.assertIsInstance(result, DataStore)

    def test_assign(self):
        """Test assign method."""
        result = self.ds.assign(bonus=lambda x: x['salary'] * 0.1)
        self.assertIsInstance(result, DataStore)
        df = result._get_df()
        self.assertIn('bonus', df.columns)

    def test_nlargest(self):
        """Test nlargest method."""
        result = self.ds.nlargest(3, 'salary')
        self.assertIsInstance(result, DataStore)
        df = result._get_df()
        self.assertEqual(len(df), 3)

    def test_nsmallest(self):
        """Test nsmallest method."""
        result = self.ds.nsmallest(3, 'age')
        self.assertIsInstance(result, DataStore)
        df = result._get_df()
        self.assertEqual(len(df), 3)

    # ========== Function Application Tests ==========

    def test_apply(self):
        """Test apply method."""
        # Apply on axis=0 (columns) - double numeric values
        result = self.ds.apply(lambda x: x * 2 if x.dtype in ['int64', 'float64'] else x, axis=0)
        # Apply returns DataStore
        self.assertIsInstance(result, DataStore)

    def test_agg(self):
        """Test aggregate method."""
        result = self.ds.agg({'age': 'mean', 'salary': 'sum'})
        # agg with dict returns Series
        self.assertIsInstance(result, pd.Series)

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
        """Test multiple column selection with [] - should return DataStore."""
        result = self.ds[['name', 'age']]
        self.assertIsInstance(result, DataStore)  # Multiple columns return DataStore
        df = result._get_df()
        self.assertEqual(len(df.columns), 2)

    # ========== Transformation Tests ==========

    def test_abs(self):
        """Test abs method (only numeric columns)."""
        result = self.ds.abs()
        # abs() returns DataFrame (numeric columns only)
        self.assertIsInstance(result, DataStore)

    def test_round(self):
        """Test round method."""
        result = self.ds.round(decimals=2)
        self.assertIsInstance(result, DataStore)

    def test_transpose(self):
        """Test transpose method."""
        result = self.ds.transpose()
        self.assertIsInstance(result, DataStore)

    # ========== Reshaping Tests ==========

    def test_melt(self):
        """Test melt method."""
        result = self.ds.melt(id_vars=['id'], value_vars=['age', 'salary'])
        self.assertIsInstance(result, DataStore)

    # ========== Boolean Methods Tests ==========

    def test_isna(self):
        """Test isna method."""
        result = self.ds.isna()
        self.assertIsInstance(result, DataStore)

    def test_notna(self):
        """Test notna method."""
        result = self.ds.notna()
        self.assertIsInstance(result, DataStore)

    # ========== Conversion Tests ==========

    def test_astype(self):
        """Test astype method."""
        result = self.ds.astype({'age': 'float64'})
        self.assertIsInstance(result, DataStore)
        df = result._get_df()
        self.assertEqual(df['age'].dtype, np.float64)

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
        """Test to_numpy method."""
        arr = self.ds.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (10, 6))

    # ========== Iteration Tests ==========

    def test_iterrows(self):
        """Test iterrows method."""
        rows = list(self.ds.iterrows())
        self.assertEqual(len(rows), 10)
        for idx, row in rows:
            self.assertIsInstance(row, pd.Series)

    def test_itertuples(self):
        """Test itertuples method."""
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
            df = result._get_df()
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
