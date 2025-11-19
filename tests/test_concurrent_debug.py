"""
Debug test for concurrent execution issues.
"""

import unittest
import tempfile
import os
import pandas as pd
import concurrent.futures
import time

from datastore import DataStore


class TestConcurrentDebug(unittest.TestCase):
    """Debug concurrent execution issues."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        with open(cls.csv_file, "w") as f:
            f.write("id,value,category\n")
            for j in range(10):
                f.write(f"{j},{j * 10},cat_{j % 3}\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)
    
    def test_simple_materialization(self):
        """Test simple materialization without concurrency."""
        ds = DataStore.from_file(self.csv_file)
        
        print(f"\n1. Created DataStore, var_name: {ds._df_var_name}")
        
        # SQL operation
        ds1 = ds.select('*').filter(ds.value > 3)
        print(f"2. After SQL filter, var_name: {ds1._df_var_name}")
        print(f"   materialized: {ds1._materialized}")
        
        # Pandas operation (materializes)
        print("3. Calling add_prefix (should materialize)...")
        ds2 = ds1.add_prefix('col_')
        print(f"4. After add_prefix, var_name: {ds2._df_var_name}")
        print(f"   materialized: {ds2._materialized}")
        print(f"   cached_df shape: {ds2._cached_df.shape if ds2._cached_df is not None else None}")
        
        # SQL on DataFrame
        print("5. Calling filter on materialized DataFrame...")
        ds3 = ds2.filter(ds.col_value > 5)
        print(f"6. After filter on DataFrame, var_name: {ds3._df_var_name}")
        print(f"   materialized: {ds3._materialized}")
        
        # Get result
        print("7. Getting final DataFrame...")
        df = ds3.to_df()
        print(f"8. Final result shape: {df.shape}")
        print(f"   columns: {list(df.columns)}")
        
        self.assertIn('col_id', df.columns)
    
    def test_sequential_operations(self):
        """Test sequential operations on same file."""
        results = []
        
        for i in range(3):
            print(f"\n--- Iteration {i} ---")
            ds = DataStore.from_file(self.csv_file)
            print(f"Created ds, var_name: {ds._df_var_name}")
            
            result = (ds
                .select('*')
                .filter(ds.value > 3)
                .add_prefix('col_')
                .filter(ds.col_value > 5)
                .to_df())
            
            print(f"Got result, shape: {result.shape}")
            results.append(result)
        
        self.assertEqual(len(results), 3)
    
    # TODO: Re-enable this test when chDB concurrency issues are resolved
    # def test_two_thread_simple(self):
    #     """Test with just 2 threads to isolate issue."""
    #     def process_with_debug(task_id):
    #         print(f"\n[Thread {task_id}] Starting...")
    #         ds = DataStore.from_file(self.csv_file)
    #         print(f"[Thread {task_id}] Created DataStore, var: {ds._df_var_name}")
            
    #         # SQL
    #         print(f"[Thread {task_id}] Applying SQL filter...")
    #         ds1 = ds.select('*').filter(ds.value > 3)
    #         print(f"[Thread {task_id}] SQL done")
            
    #         # Pandas (materializes)
    #         print(f"[Thread {task_id}] Calling add_prefix...")
    #         ds2 = ds1.add_prefix(f'col{task_id}_')
    #         print(f"[Thread {task_id}] add_prefix done, new var: {ds2._df_var_name}")
            
    #         # SQL on DataFrame
    #         print(f"[Thread {task_id}] Calling filter on DataFrame...")
    #         ds3 = ds2.filter(getattr(ds2, f'col{task_id}_value') > 5)
    #         print(f"[Thread {task_id}] filter done")
            
    #         # Get result
    #         print(f"[Thread {task_id}] Getting to_df...")
    #         result = ds3.to_df()
    #         print(f"[Thread {task_id}] Complete! Shape: {result.shape}")
            
    #         return result
        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #         future1 = executor.submit(process_with_debug, 1)
    #         future2 = executor.submit(process_with_debug, 2)
            
    #         print("\nWaiting for results...")
    #         result1 = future1.result(timeout=10)
    #         print(f"Got result1: {result1.shape}")
            
    #         result2 = future2.result(timeout=10)
    #         print(f"Got result2: {result2.shape}")
        
    #     self.assertEqual(len(result1), len(result2))


if __name__ == '__main__':
    unittest.main()

