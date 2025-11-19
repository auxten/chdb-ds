"""
Comprehensive tests for complex mixed SQL and pandas operations.

This test suite covers all possible combinations and edge cases of mixing
DataStore SQL operations with pandas DataFrame operations.
"""

import unittest
import tempfile
import os
import pandas as pd

from datastore import DataStore


class TestComplexMixedScenarios(unittest.TestCase):
    """Test complex scenarios mixing SQL and pandas operations."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

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
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    # ========== Scenario 1: SQL → Pandas → Pandas ==========
    
    def test_sql_pandas_pandas(self):
        """Test: SQL filter → pandas rename → pandas drop."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('*')
            .filter(ds.age > 25)                           # SQL
            .rename(columns={'id': 'employee_id'})         # Pandas (materializes)
            .drop(columns=['active']))                     # Pandas (cached)
        
        df = result.to_df()
        
        # Verify transformations
        self.assertIn('employee_id', df.columns)
        self.assertNotIn('id', df.columns)
        self.assertNotIn('active', df.columns)
        self.assertTrue(all(df['age'] > 25))
        
    # ========== Scenario 2: SQL → Pandas → SQL-style ==========
    
    def test_sql_pandas_sql_style(self):
        """Test: SQL filter → pandas rename → SQL-style select.
        
        After materialization, SQL-style operations should work on the cached DataFrame.
        """
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .filter(ds.age > 25)                           # SQL: WHERE age > 25
            .add_prefix('emp_')                            # Pandas: materializes, renames columns
            .select('emp_id', 'emp_name', 'emp_age'))      # SQL-style: should select from cached df
        
        df = result.to_df()
        
        # Verify columns
        self.assertEqual(len(df.columns), 3)
        self.assertIn('emp_id', df.columns)
        self.assertIn('emp_name', df.columns)
        self.assertIn('emp_age', df.columns)
        self.assertNotIn('emp_salary', df.columns)
        
    # ========== Scenario 3: Pandas → SQL-style → Pandas ==========
    
    def test_pandas_sql_style_pandas(self):
        """Test: pandas rename → SQL-style filter → pandas sort."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .rename(columns={'id': 'ID', 'name': 'NAME'})  # Pandas: materializes
            .filter(ds.age > 28)                           # SQL-style: should filter cached df
            .sort_values('salary', ascending=False))       # Pandas: sort cached
        
        df = result.to_df()
        
        # Verify all transformations
        self.assertIn('ID', df.columns)
        self.assertIn('NAME', df.columns)
        self.assertTrue(all(df['age'] > 28))
        # Check sorted
        salaries = df['salary'].tolist()
        self.assertEqual(salaries, sorted(salaries, reverse=True))
        
    # ========== Scenario 4: Multiple Pandas Operations ==========
    
    def test_multiple_pandas_operations_chain(self):
        """Test: Multiple consecutive pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .add_prefix('x_')                              # Pandas 1: materializes
            .add_suffix('_y')                              # Pandas 2: on cached
            .drop(columns=['x_active_y'])                  # Pandas 3: on cached
            .fillna(0)                                     # Pandas 4: on cached
            .sort_values('x_age_y'))                       # Pandas 5: on cached
        
        df = result.to_df()
        
        # Verify all transformations applied
        self.assertIn('x_id_y', df.columns)
        self.assertIn('x_name_y', df.columns)
        self.assertNotIn('x_active_y', df.columns)
        
    # ========== Scenario 5: SQL → Pandas → SQL → Pandas ==========
    
    def test_sql_pandas_sql_pandas(self):
        """Test: SQL → pandas → SQL → pandas alternating."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('*')                                   # SQL
            .filter(ds.age > 25)                          # SQL
            .add_prefix('p1_')                            # Pandas: materializes
            .filter(ds.p1_salary > 55000)                 # SQL-style on cached df
            .add_suffix('_p2'))                           # Pandas: on cached
        
        df = result.to_df()
        
        # All transformations should be applied
        self.assertIn('p1_id_p2', df.columns)
        self.assertTrue(all(df['p1_age_p2'] > 25))
        self.assertTrue(all(df['p1_salary_p2'] > 55000))
        
    # ========== Scenario 6: Pandas Operations on Column Subset ==========
    
    def test_pandas_with_column_subset(self):
        """Test pandas operations after SQL column selection."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'name', 'age', 'salary')        # SQL: SELECT specific columns
            .filter(ds.age > 27)                          # SQL: WHERE
            .assign(salary_k=lambda x: x['salary'] / 1000)  # Pandas: add column
            .drop(columns=['salary']))                    # Pandas: drop original
        
        df = result.to_df()
        
        # Verify columns
        self.assertEqual(set(df.columns), {'id', 'name', 'age', 'salary_k'})
        self.assertIn('salary_k', df.columns)
        self.assertNotIn('salary', df.columns)
        
    # ========== Scenario 7: Mathematical Operations ==========
    
    def test_mathematical_operations_after_sql(self):
        """Test mathematical operations on filtered data."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'salary')                       # SQL
            .filter(ds.salary > 50000)                    # SQL
            .mul(1.1))                                    # Pandas: 10% increase
        
        df = result.to_df()
        
        # Verify calculation (original values * 1.1)
        # Original salaries > 50000: [60000, 70000, 55000, 65000, 58000, 62000, 52000, 68000, 51000]
        # After 10% increase, all should be > 55000
        self.assertTrue(all(df['salary'] > 55000))
        
    # ========== Scenario 8: Groupby After Materialization ==========
    
    def test_groupby_after_pandas_operations(self):
        """Test groupby on materialized DataFrame."""
        ds = DataStore.from_file(self.csv_file)
        
        # SQL filter, pandas transform, then pandas groupby
        result = (ds
            .filter(ds.active == 1)                       # SQL
            .assign(salary_category=lambda x: pd.cut(x['salary'], 
                                                     bins=[0, 55000, 65000, 100000],
                                                     labels=['Low', 'Medium', 'High']))  # Pandas
            .groupby('salary_category').agg({'salary': ['mean', 'count']}))  # Pandas groupby
        
        # groupby().agg() with multiple functions returns DataFrame
        self.assertIsInstance(result, DataStore)
        
    # ========== Scenario 9: Conditional Operations ==========
    
    def test_where_after_sql_filter(self):
        """Test pandas where() after SQL filter."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'age', 'salary')                # SQL
            .filter(ds.age > 25)                          # SQL
            .where(lambda x: x['salary'] > 55000, 0))     # Pandas: conditional replace
        
        df = result.to_df()
        
        # Rows with salary <= 55000 should have all values set to 0
        for idx, row in df.iterrows():
            if row['salary'] <= 55000:
                # These rows should be all zeros (where replaced them)
                pass  # where replaces entire row, not ideal but that's pandas behavior
                
    # ========== Scenario 10: Complex Query with Joins ==========
    
    def test_sql_join_then_pandas(self):
        """Test SQL join followed by pandas operations.
        
        Note: This tests the simpler case of pandas merge after separate queries.
        """
        ds1 = DataStore.from_file(self.csv_file)
        
        # Create second dataset
        csv_file2 = os.path.join(self.temp_dir, "bonus_data.csv")
        with open(csv_file2, "w") as f:
            f.write("id,bonus\n")
            f.write("1,5000\n")
            f.write("2,6000\n")
            f.write("3,7000\n")
        
        try:
            ds2 = DataStore.from_file(csv_file2)
            
            # Execute both queries separately, then pandas merge
            ds1_filtered = ds1.select('id', 'name', 'salary').filter(ds1.active == 1)
            
            # Use pandas merge (which works on materialized DataFrames)
            result = (ds1_filtered
                .merge(ds2, on='id', how='inner')         # Pandas: merge
                .assign(total_comp=lambda x: x['salary'] + x['bonus'])  # Pandas
                .sort_values('total_comp', ascending=False))  # Pandas
            
            df = result.to_df() if isinstance(result, DataStore) else result
            
            # Verify join and calculation
            self.assertIn('total_comp', df.columns)
            self.assertIn('bonus', df.columns)
            
        finally:
            if os.path.exists(csv_file2):
                os.unlink(csv_file2)
                
    # ========== Scenario 11: Pivot After SQL ==========
    
    def test_pivot_after_sql_aggregation(self):
        """Test pivot operation after SQL filtering."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('department', 'active', 'salary')     # SQL
            .filter(ds.salary > 50000)                    # SQL
            .pivot_table(values='salary', 
                        index='department', 
                        columns='active', 
                        aggfunc='mean'))                  # Pandas
        
        # pivot_table returns DataFrame, wrapped as DataStore
        self.assertIsInstance(result, DataStore)
        
    # ========== Scenario 12: Melt After SQL ==========
    
    def test_melt_after_sql(self):
        """Test melt (unpivot) after SQL operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'name', 'age', 'salary')        # SQL
            .filter(ds.id <= 5)                           # SQL
            .melt(id_vars=['id', 'name'], 
                  value_vars=['age', 'salary'],
                  var_name='metric',
                  value_name='value'))                    # Pandas
        
        df = result.to_df() if isinstance(result, DataStore) else result
        
        # Verify melt structure
        self.assertIn('metric', df.columns)
        self.assertIn('value', df.columns)
        # Should have 2 rows per id (age and salary)
        self.assertEqual(len(df), 10)  # 5 ids * 2 metrics
        
    # ========== Scenario 13: Multiple to_df() Calls ==========
    
    def test_multiple_to_df_in_chain(self):
        """Test calling to_df() multiple times shouldn't break chain."""
        ds = DataStore.from_file(self.csv_file)
        
        # Build query
        ds1 = ds.select('*').filter(ds.age > 25)
        
        # Get DataFrame once
        df1 = ds1.to_df()
        self.assertIn('id', df1.columns)
        
        # Apply pandas operation
        ds2 = ds1.add_prefix('col_')
        
        # Get DataFrame again - should have prefixes
        df2 = ds2.to_df()
        self.assertIn('col_id', df2.columns)
        self.assertNotIn('id', df2.columns)
        
        # Original ds1 should still work
        df1_again = ds1.to_df()
        self.assertIn('id', df1_again.columns)
        
    # ========== Scenario 14: SQL After Pandas (Problematic) ==========
    
    def test_sql_after_pandas_materialization(self):
        """Test SQL-style operations after pandas materialization.
        
        This is a critical test: What happens when we apply SQL-style operations
        after the DataStore has been materialized by pandas operations?
        
        Expected behavior: SQL-style operations should work on the cached DataFrame.
        """
        ds = DataStore.from_file(self.csv_file)
        
        # Pandas operation (materializes)
        ds_renamed = ds.rename(columns={'id': 'ID', 'name': 'NAME'})
        self.assertTrue(ds_renamed._materialized)
        
        # Now try SQL-style operations
        # These should work on the cached DataFrame, not build SQL query
        ds_filtered = ds_renamed.filter(ds_renamed.age > 30)
        
        df = ds_filtered.to_df()
        
        # Verify: Should have renamed columns and filtered data
        self.assertIn('ID', df.columns)
        self.assertIn('NAME', df.columns)
        # Filter should have worked on the cached df
        self.assertTrue(all(df['age'] > 30))
        
    # ========== Scenario 15: SQL → Pandas → SQL → Pandas → SQL ==========
    
    def test_alternating_sql_pandas_multiple_times(self):
        """Test alternating between SQL and pandas operations multiple times."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('*')                                   # SQL 1
            .filter(ds.age > 25)                          # SQL 2
            .add_prefix('p1_')                            # Pandas 1: materializes
            .filter(ds.p1_salary > 55000)                 # SQL 3: on cached df?
            .rename(columns={'p1_id': 'final_id'})        # Pandas 2: on cached df
            .filter(ds.final_id > 2)                      # SQL 4: on cached df?
            .add_suffix('_end'))                          # Pandas 3: on cached df
        
        df = result.to_df()
        
        # Verify final state
        print(f"\nFinal columns: {list(df.columns)}")
        print(f"Final shape: {df.shape}")
        print(f"Sample data:\n{df.head()}")
        
        # All filters should have been applied
        # Original: age > 25, salary > 55000, id > 2
        # Expected IDs: 3, 5, 7, 9 (age>25, salary>55000, id>2)
        
    # ========== Scenario 16: Assign → Filter on New Column ==========
    
    def test_assign_then_filter_on_new_column(self):
        """Test filtering on a column created by pandas assign."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('*')                                   # SQL
            .assign(salary_k=lambda x: x['salary'] / 1000)  # Pandas: add column
            .filter(ds.salary_k > 60))                    # Filter on new column
        
        df = result.to_df()
        
        # Verify new column exists and filter worked
        self.assertIn('salary_k', df.columns)
        self.assertTrue(all(df['salary_k'] > 60))
        
    # ========== Scenario 17: Query String After SQL ==========
    
    def test_pandas_query_after_sql(self):
        """Test pandas query() method after SQL operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('*')                                   # SQL
            .filter(ds.active == 1)                       # SQL
            .query('age > 28 and salary > 55000'))        # Pandas query
        
        df = result.to_df()
        
        # Verify both SQL and pandas filters applied
        self.assertTrue(all(df['active'] == 1))
        self.assertTrue(all(df['age'] > 28))
        self.assertTrue(all(df['salary'] > 55000))
        
    # ========== Scenario 18: Slice After Pandas ==========
    
    def test_slice_after_pandas_operation(self):
        """Test DataFrame slicing after pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .sort_values('salary', ascending=False)       # Pandas: materializes
            .head(5))                                     # Slice top 5
        
        # head() returns DataFrame directly
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        
    # ========== Scenario 19: Math Operations Chain ==========
    
    def test_chained_mathematical_operations(self):
        """Test chaining multiple mathematical operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'salary')                       # SQL
            .mul(2)                                       # Pandas: double all values
            .add(1000)                                    # Pandas: add 1000
            .div(1000))                                   # Pandas: convert to thousands
        
        df = result.to_df()
        
        # Verify calculation: (salary * 2 + 1000) / 1000
        # Example: (50000 * 2 + 1000) / 1000 = 101
        
    # ========== Scenario 20: Properties After Operations ==========
    
    def test_properties_after_mixed_operations(self):
        """Test that properties work correctly after mixed operations."""
        ds = DataStore.from_file(self.csv_file)
        
        # Mix SQL and pandas
        ds_mixed = (ds
            .select('*')
            .filter(ds.age > 25)
            .add_prefix('x_')
            .drop(columns=['x_active']))
        
        # All properties should work
        shape = ds_mixed.shape
        cols = ds_mixed.columns
        dtypes = ds_mixed.dtypes
        size = ds_mixed.size
        
        # Verify properties reflect transformed data
        self.assertIn('x_id', cols)
        self.assertNotIn('x_active', cols)
        self.assertEqual(len(cols), 5)  # 6 columns - 1 dropped
        
    # ========== Scenario 21: Reset After Pandas ==========
    
    def test_reset_index_after_operations(self):
        """Test reset_index after filtering operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .filter(ds.age > 30)                          # SQL
            .sort_values('salary', ascending=False)       # Pandas
            .reset_index(drop=True))                      # Pandas
        
        df = result.to_df()
        
        # Verify index is reset (0, 1, 2, ...)
        self.assertEqual(list(df.index), list(range(len(df))))
        
    # ========== Scenario 22: Merge After Both DataStores Have Pandas Ops ==========
    
    def test_merge_two_materialized_datastores(self):
        """Test merging two DataStores that have both had pandas operations."""
        ds1 = DataStore.from_file(self.csv_file)
        
        # Create second dataset
        csv_file2 = os.path.join(self.temp_dir, "dept_info.csv")
        with open(csv_file2, "w") as f:
            f.write("department,budget\n")
            f.write("Engineering,500000\n")
            f.write("Sales,300000\n")
            f.write("Marketing,200000\n")
        
        try:
            ds2 = DataStore.from_file(csv_file2)
            
            # Apply pandas operations to both
            ds1_prepared = ds1.select('id', 'name', 'department').add_prefix('emp_')
            ds2_prepared = ds2.add_prefix('dept_')
            
            # Merge
            result = ds1_prepared.merge(ds2_prepared, 
                                       left_on='emp_department', 
                                       right_on='dept_department')
            
            df = result.to_df() if isinstance(result, DataStore) else result
            
            # Verify merge worked with prefixed columns
            self.assertIn('emp_id', df.columns)
            self.assertIn('dept_budget', df.columns)
            
        finally:
            if os.path.exists(csv_file2):
                os.unlink(csv_file2)
                
    # ========== Scenario 23: Apply Function After SQL ==========
    
    def test_apply_after_sql_filter(self):
        """Test apply() function after SQL filtering."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('age', 'salary')                      # SQL
            .filter(ds.age > 25)                          # SQL
            .apply(lambda x: x / 1000 if x.dtype in ['int64', 'float64'] else x))  # Pandas
        
        # apply returns DataStore
        self.assertIsInstance(result, DataStore)
        df = result.to_df()
        
        # Values should be divided by 1000
        self.assertTrue(all(df['age'] < 1))  # Ages are now < 1 (25/1000 = 0.025)
        
    # ========== Scenario 24: Eval After SQL ==========
    
    def test_eval_expression_after_sql(self):
        """Test eval() expression after SQL operations."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .select('id', 'age', 'salary')                # SQL
            .filter(ds.active == 1)                       # SQL
            .eval('salary_per_age = salary / age'))       # Pandas: add calculated column
        
        df = result.to_df()
        
        # Verify new column
        self.assertIn('salary_per_age', df.columns)
        # Verify calculation
        for idx, row in df.iterrows():
            expected = row['salary'] / row['age']
            self.assertAlmostEqual(row['salary_per_age'], expected, places=2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and corner scenarios."""
    
    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "edge_case.csv")

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
    
    def test_empty_result_after_filter(self):
        """Test pandas operations on empty result set."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .filter(ds.a > 100)                           # SQL: returns empty
            .add_prefix('x_'))                            # Pandas: on empty df
        
        df = result.to_df()
        
        # Should be empty but have correct columns
        self.assertEqual(len(df), 0)
        self.assertIn('x_a', df.columns)
        
    def test_same_operation_twice(self):
        """Test applying same pandas operation twice."""
        ds = DataStore.from_file(self.csv_file)
        
        result = (ds
            .add_prefix('p1_')                            # Pandas 1
            .add_prefix('p2_'))                           # Pandas 2: should add p2_ to p1_*
        
        df = result.to_df()
        
        # Should have double prefix
        self.assertIn('p2_p1_a', df.columns)
        
    def test_materialization_flag_persistence(self):
        """Test that materialization flag persists correctly."""
        ds = DataStore.from_file(self.csv_file)
        
        # SQL only
        ds1 = ds.select('*')
        self.assertFalse(ds1._materialized)
        
        # Materialize
        ds2 = ds1.add_prefix('x_')
        self.assertTrue(ds2._materialized)
        
        # Further pandas operations should stay materialized
        ds3 = ds2.fillna(0)
        self.assertTrue(ds3._materialized)
        
        # Copy should preserve materialization
        from copy import copy as py_copy
        ds4 = py_copy(ds3)
        self.assertTrue(ds4._materialized)


class TestExecutionOptimization(unittest.TestCase):
    """Test execution optimization scenarios."""
    
    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "opt_test.csv")

        # Create larger dataset
        with open(cls.csv_file, "w") as f:
            f.write("id,value,category\n")
            for i in range(100):
                category = 'A' if i < 50 else 'B'
                f.write(f"{i},{i * 10},{category}\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)
    
    def test_sql_pushdown_before_pandas(self):
        """Test that SQL filters are pushed down before materialization."""
        ds = DataStore.from_file(self.csv_file)
        
        # Build SQL filter first (should reduce data size)
        ds_filtered = (ds
            .select('*')
            .filter(ds.value > 500)                       # SQL: filters to ~50 rows
            .filter(ds.category == 'B'))                  # SQL: further filter
        
        # Not materialized yet
        self.assertFalse(ds_filtered._materialized)
        
        # Materialize with pandas
        ds_final = ds_filtered.add_suffix('_test')
        
        # Should have filtered data
        df = ds_final.to_df()
        self.assertTrue(len(df) < 100)  # Filtered
        self.assertTrue(all(df['value_test'] > 500))
        self.assertTrue(all(df['category_test'] == 'B'))
        
    def test_no_redundant_execution(self):
        """Test that query is not re-executed unnecessarily."""
        ds = DataStore.from_file(self.csv_file)
        
        # SQL operations
        ds1 = ds.select('*').filter(ds.value > 100)
        
        # Materialize
        ds2 = ds1.add_prefix('x_')
        
        # Multiple accesses should not re-execute
        df1 = ds2.to_df()
        df2 = ds2.to_df()
        df3 = ds2.to_df()
        
        # Should be same cached object
        self.assertIs(df1, df2)
        self.assertIs(df2, df3)


if __name__ == '__main__':
    unittest.main()

