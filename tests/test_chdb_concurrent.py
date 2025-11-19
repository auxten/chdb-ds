"""
Independent test for chDB query on DataFrame concurrency.

This test verifies that chDB can safely execute SQL on pandas DataFrames
concurrently when using unique variable names, without relying on DataStore.
"""

import unittest
import pandas as pd
import concurrent.futures
import uuid
import time


class TestChDBConcurrency(unittest.TestCase):
    """Test chDB concurrent execution on pandas DataFrames."""

    def test_chdb_basic_query_on_dataframe(self):
        """Test basic chDB query on DataFrame."""
        import chdb

        # Create test DataFrame
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['tom', 'jerry', 'auxten', 'alice', 'bob']})

        # Register DataFrame in global namespace
        var_name = '__test_df__'
        globals()[var_name] = df

        try:
            # Execute SQL using Python() table function
            result = chdb.query(f"SELECT b, sum(a) as total FROM Python({var_name}) GROUP BY b ORDER BY b", 'DataFrame')

            # Verify result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertIn('b', result.columns)
            self.assertIn('total', result.columns)

        finally:
            # Clean up
            if var_name in globals():
                del globals()[var_name]

    def test_unique_variable_names_prevent_collision(self):
        """Test that unique variable names prevent collisions."""
        import chdb

        # Create two different DataFrames
        df1 = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        df2 = pd.DataFrame({'x': [4, 5, 6], 'y': [40, 50, 60]})

        # Use unique variable names
        var1 = f"__df_{uuid.uuid4().hex}__"
        var2 = f"__df_{uuid.uuid4().hex}__"

        # Register both
        globals()[var1] = df1
        globals()[var2] = df2

        try:
            # Query both using their unique names
            result1 = chdb.query(f"SELECT sum(x) as total FROM Python({var1})", 'DataFrame')
            result2 = chdb.query(f"SELECT sum(x) as total FROM Python({var2})", 'DataFrame')

            # Verify different results
            self.assertEqual(result1['total'].iloc[0], 6)  # 1+2+3
            self.assertEqual(result2['total'].iloc[0], 15)  # 4+5+6

        finally:
            if var1 in globals():
                del globals()[var1]
            if var2 in globals():
                del globals()[var2]

    def test_concurrent_queries_with_unique_names(self):
        """Test concurrent SQL queries on different DataFrames with unique names."""
        import chdb

        def execute_query(data_id, values):
            """Execute a query on a DataFrame with unique variable name."""
            # Create DataFrame
            df = pd.DataFrame(
                {'id': range(len(values)), 'value': values, 'category': [f'cat_{i % 3}' for i in range(len(values))]}
            )

            # Generate unique variable name
            var_name = f"__df_{uuid.uuid4().hex}__"

            # Register in global namespace
            globals()[var_name] = df

            try:
                # Execute SQL query
                sql = f"""
                    SELECT category, 
                           sum(value) as total, 
                           count(*) as count
                    FROM Python({var_name})
                    WHERE value > 10
                    GROUP BY category
                    ORDER BY category
                """
                result = chdb.query(sql, 'DataFrame')

                # Add data_id for verification
                result['data_id'] = data_id
                return result

            finally:
                # Clean up
                if var_name in globals():
                    del globals()[var_name]

        # Execute multiple queries concurrently
        datasets = [
            (0, list(range(0, 100, 5))),  # [0, 5, 10, 15, ...]
            (1, list(range(10, 110, 5))),  # [10, 15, 20, ...]
            (2, list(range(20, 120, 5))),  # [20, 25, 30, ...]
            (3, list(range(5, 105, 5))),  # [5, 10, 15, ...]
            (4, list(range(15, 115, 5))),  # [15, 20, 25, ...]
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_query, data_id, values) for data_id, values in datasets]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify all queries succeeded
        self.assertEqual(len(results), 5)

        # Verify each has correct structure
        for result in results:
            self.assertIn('category', result.columns)
            self.assertIn('total', result.columns)
            self.assertIn('count', result.columns)
            self.assertIn('data_id', result.columns)

    def test_high_concurrency_stress(self):
        """Stress test with many concurrent queries."""
        import chdb

        def quick_query(iteration):
            """Execute a quick query with unique variable name."""
            df = pd.DataFrame({'iter': [iteration] * 5, 'val': list(range(5))})

            var_name = f"__stress_df_{uuid.uuid4().hex}__"
            globals()[var_name] = df

            try:
                result = chdb.query(
                    f"SELECT iter, sum(val) as total FROM Python({var_name}) GROUP BY iter", 'DataFrame'
                )
                return result['total'].iloc[0]

            finally:
                if var_name in globals():
                    del globals()[var_name]

        # Execute many queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(quick_query, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed and return same result (0+1+2+3+4 = 10)
        self.assertEqual(len(results), 50)
        self.assertTrue(all(r == 10 for r in results))

    def test_variable_name_cleanup(self):
        """Test that variable names are properly cleaned up."""
        import chdb

        # Track variable names
        created_vars = []

        def query_with_tracking():
            df = pd.DataFrame({'x': [1, 2, 3]})
            var_name = f"__cleanup_df_{uuid.uuid4().hex}__"
            created_vars.append(var_name)

            globals()[var_name] = df

            try:
                result = chdb.query(f"SELECT sum(x) as s FROM Python({var_name})", 'DataFrame')
                return result
            finally:
                if var_name in globals():
                    del globals()[var_name]

        # Execute multiple times
        for _ in range(10):
            query_with_tracking()

        # Verify all variables were cleaned up
        for var_name in created_vars:
            self.assertNotIn(var_name, globals(), f"Variable {var_name} was not cleaned up!")

    def test_nested_concurrent_queries(self):
        """Test nested concurrent queries (query in query)."""
        import chdb

        def outer_query(outer_id):
            """Outer query that spawns inner queries."""
            df_outer = pd.DataFrame({'id': [outer_id], 'value': [outer_id * 10]})

            var_outer = f"__outer_{uuid.uuid4().hex}__"
            globals()[var_outer] = df_outer

            try:
                # Execute outer query
                result_outer = chdb.query(f"SELECT id, value FROM Python({var_outer})", 'DataFrame')

                # Now execute inner query
                def inner_query(inner_id):
                    df_inner = pd.DataFrame({'inner_id': [inner_id], 'inner_value': [inner_id * 100]})

                    var_inner = f"__inner_{uuid.uuid4().hex}__"
                    globals()[var_inner] = df_inner

                    try:
                        return chdb.query(f"SELECT sum(inner_value) as s FROM Python({var_inner})", 'DataFrame')
                    finally:
                        if var_inner in globals():
                            del globals()[var_inner]

                # Execute inner queries
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    inner_futures = [executor.submit(inner_query, i) for i in range(3)]
                    inner_results = [f.result() for f in inner_futures]

                return (result_outer, inner_results)

            finally:
                if var_outer in globals():
                    del globals()[var_outer]

        # Execute outer queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(outer_query, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        self.assertEqual(len(results), 3)

        # Each result should have outer result and 3 inner results
        for outer_result, inner_results in results:
            self.assertIsInstance(outer_result, pd.DataFrame)
            self.assertEqual(len(inner_results), 3)

    def test_variable_name_format(self):
        """Test that generated variable names are valid Python identifiers."""
        # Generate many variable names
        var_names = [f"__ds_df_{uuid.uuid4().hex}__" for _ in range(100)]

        # All should be valid identifiers
        for var_name in var_names:
            self.assertTrue(var_name.isidentifier(), f"{var_name} is not a valid Python identifier")

        # All should be unique
        self.assertEqual(len(var_names), len(set(var_names)))

        # All should follow expected format
        for var_name in var_names:
            self.assertTrue(var_name.startswith('__ds_df_'))
            self.assertTrue(var_name.endswith('__'))
            # UUID hex is 32 characters
            self.assertEqual(len(var_name), 8 + 32 + 2)  # __ds_df_ + hex + __


class TestChDBPerformance(unittest.TestCase):
    """Performance tests for chDB on DataFrames."""

    def test_multiple_sequential_queries(self):
        """Test multiple sequential queries with cleanup."""
        import chdb

        results = []

        for i in range(10):
            df = pd.DataFrame({'iteration': [i] * 10, 'value': range(10)})

            var_name = f"__perf_df_{uuid.uuid4().hex}__"
            globals()[var_name] = df

            try:
                result = chdb.query(
                    f"SELECT iteration, sum(value) as total FROM Python({var_name}) GROUP BY iteration", 'DataFrame'
                )
                results.append(result)
            finally:
                if var_name in globals():
                    del globals()[var_name]

        # All queries should succeed
        self.assertEqual(len(results), 10)

        # Each should have correct sum (0+1+2+...+9 = 45)
        for result in results:
            self.assertEqual(result['total'].iloc[0], 45)


if __name__ == '__main__':
    unittest.main()
