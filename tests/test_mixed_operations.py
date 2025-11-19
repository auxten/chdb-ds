"""
Tests for mixed SQL and pandas operations.

This module tests the critical behavior of mixing DataStore SQL operations
with pandas DataFrame operations in a chain.
"""

import unittest
import tempfile
import os

from datastore import DataStore


class TestMixedOperations(unittest.TestCase):
    """Test mixing SQL queries with pandas operations."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        with open(cls.csv_file, "w") as f:
            f.write("id,name,age,salary\n")
            f.write("1,Alice,25,50000\n")
            f.write("2,Bob,30,60000\n")
            f.write("3,Charlie,35,70000\n")
            f.write("4,David,28,55000\n")
            f.write("5,Eve,32,65000\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_sql_then_pandas_prefix(self):
        """Test SQL operations followed by pandas add_prefix."""
        ds = DataStore.from_file(self.csv_file)
        
        # SQL operations
        ds_filtered = ds.select('*').filter(ds.age > 25)
        
        # Pandas operation
        ds_prefixed = ds_filtered.add_prefix('col_')
        
        # Get DataFrame
        df = ds_prefixed.to_df()
        
        # Verify columns have prefix
        self.assertIn('col_id', df.columns)
        self.assertIn('col_name', df.columns)
        self.assertIn('col_age', df.columns)
        self.assertNotIn('id', df.columns)
        
        # Verify data is filtered (age > 25)
        self.assertTrue(all(df['col_age'] > 25))
        
    def test_sql_then_pandas_suffix(self):
        """Test SQL operations followed by pandas add_suffix."""
        ds = DataStore.from_file(self.csv_file)
        
        # SQL filter and pandas suffix
        result = ds.filter(ds.salary > 55000).add_suffix('_info')
        df = result.to_df()
        
        # Verify columns have suffix
        self.assertIn('id_info', df.columns)
        self.assertIn('name_info', df.columns)
        self.assertNotIn('id', df.columns)
        
        # Verify data is filtered
        self.assertTrue(all(df['salary_info'] > 55000))
        
    def test_sql_then_pandas_rename(self):
        """Test SQL operations followed by pandas rename."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'name', 'age')
            .filter(ds.age < 35)
            .rename(columns={'id': 'employee_id', 'name': 'employee_name'}))
        
        df = result.to_df()
        
        # Verify renamed columns
        self.assertIn('employee_id', df.columns)
        self.assertIn('employee_name', df.columns)
        self.assertNotIn('id', df.columns)
        self.assertNotIn('name', df.columns)
        
        # Verify filtering worked
        self.assertTrue(all(df['age'] < 35))
        
    def test_sql_then_pandas_drop(self):
        """Test SQL select followed by pandas drop."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('*')
            .filter(ds.age > 25)
            .drop(columns=['salary']))
        
        df = result.to_df()
        
        # Verify column dropped
        self.assertNotIn('salary', df.columns)
        self.assertIn('id', df.columns)
        self.assertIn('name', df.columns)
        
        # Verify filtering
        self.assertTrue(all(df['age'] > 25))
        
    def test_pandas_then_pandas(self):
        """Test chaining multiple pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .add_prefix('p1_')
            .add_suffix('_p2')
            .rename(columns={'p1_id_p2': 'final_id'}))
        
        df = result.to_df()
        
        # Verify all transformations applied
        self.assertIn('final_id', df.columns)
        self.assertIn('p1_name_p2', df.columns)
        self.assertNotIn('id', df.columns)
        self.assertNotIn('p1_id_p2', df.columns)
        
    def test_sql_pandas_sql_warning(self):
        """Test that materialized DataStore behavior is correct."""
        ds = DataStore.from_file(self.csv_file)
        
        # SQL operation
        ds1 = ds.select('*').filter(ds.age > 25)
        
        # Pandas operation (materializes)
        ds2 = ds1.add_prefix('col_')
        
        # Verify ds2 is materialized
        self.assertTrue(ds2._materialized)
        
        # Verify cached DataFrame has correct columns
        self.assertIsNotNone(ds2._cached_df)
        self.assertIn('col_id', ds2._cached_df.columns)
        
        # to_df() should return cached DataFrame
        df = ds2.to_df()
        self.assertIn('col_id', df.columns)
        
    def test_materialized_datastore_properties(self):
        """Test that materialized DataStore properties work correctly."""
        ds = DataStore.from_file(self.csv_file)
        
        # Apply pandas operation
        ds_transformed = ds.add_prefix('x_')
        
        # These should all use the cached DataFrame
        shape = ds_transformed.shape
        cols = ds_transformed.columns
        dtypes = ds_transformed.dtypes
        
        # Verify they reflect the pandas transformation
        self.assertIn('x_id', cols)
        self.assertIn('x_name', cols)
        self.assertNotIn('id', cols)
        
    def test_multiple_to_df_calls(self):
        """Test that multiple to_df() calls on materialized DataStore return same data."""
        ds = DataStore.from_file(self.csv_file)
        
        # Apply pandas transformation
        ds_renamed = ds.rename(columns={'id': 'ID', 'name': 'NAME'})
        
        # Multiple to_df() calls should return consistent results
        df1 = ds_renamed.to_df()
        df2 = ds_renamed.to_df()
        df3 = ds_renamed.to_df()
        
        # All should have renamed columns
        self.assertIn('ID', df1.columns)
        self.assertIn('ID', df2.columns)
        self.assertIn('ID', df3.columns)
        
        # Should be the same cached object
        self.assertIs(df1, df2)
        self.assertIs(df2, df3)
        
    def test_complex_mixed_chain(self):
        """Test complex chain with SQL, pandas, SQL-style, pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'name', 'age', 'salary')  # SQL: SELECT
            .filter(ds.age > 25)                    # SQL: WHERE
            .assign(bonus=lambda x: x['salary'] * 0.1)  # Pandas: add column
            .add_prefix('emp_')                     # Pandas: rename
            .query('emp_salary > 55000')            # Pandas: filter
            .sort_values('emp_salary', ascending=False)  # Pandas: sort
            .head(3))                               # Pandas: limit
        
        # Verify final result has all transformations
        self.assertIn('emp_id', result.columns)
        self.assertIn('emp_name', result.columns)
        self.assertIn('emp_bonus', result.columns)
        self.assertNotIn('id', result.columns)
        
        # Verify filtering worked (age > 25 AND salary > 55000)
        self.assertTrue(all(result['emp_age'] > 25))
        self.assertTrue(all(result['emp_salary'] > 55000))
        
        # Verify only 3 rows
        self.assertEqual(len(result), 3)


class TestExecutionModel(unittest.TestCase):
    """Test the execution model for mixed operations."""
    
    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "exec_test.csv")

        with open(cls.csv_file, "w") as f:
            f.write("a,b,c\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            f.write("7,8,9\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)
    
    def test_sql_only_not_materialized(self):
        """Test that SQL-only operations don't materialize."""
        ds = DataStore.from_file(self.csv_file)
        
        # Only SQL operations
        ds2 = ds.select('*').filter(ds.a > 1).limit(10)
        
        # Should not be materialized yet
        self.assertFalse(ds2._materialized)
        self.assertIsNone(ds2._cached_df)
        
    def test_first_pandas_operation_materializes(self):
        """Test that first pandas operation triggers materialization."""
        ds = DataStore.from_file(self.csv_file)
        
        # SQL operations don't materialize
        ds1 = ds.select('*').filter(ds.a > 0)
        self.assertFalse(ds1._materialized)
        
        # First pandas operation materializes
        ds2 = ds1.add_prefix('x_')
        self.assertTrue(ds2._materialized)
        self.assertIsNotNone(ds2._cached_df)
        
    def test_subsequent_pandas_use_cache(self):
        """Test that subsequent pandas operations use cached DataFrame."""
        ds = DataStore.from_file(self.csv_file)
        
        # First pandas operation
        ds1 = ds.add_prefix('p1_')
        cache_id_1 = id(ds1._cached_df)
        
        # Second pandas operation on ds1
        ds2 = ds1.add_suffix('_p2')
        
        # Verify ds2 used ds1's cached df as input (not re-executed SQL)
        # The cache should be different (new DataFrame), but based on ds1's cache
        self.assertTrue(ds2._materialized)
        self.assertIn('p1_a_p2', ds2._cached_df.columns)


if __name__ == '__main__':
    unittest.main()

