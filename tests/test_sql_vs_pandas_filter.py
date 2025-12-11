"""
Test to understand how SQL engine vs Pandas engine filtering works.

This test creates independent data and verifies both execution paths.
"""

import unittest
import tempfile
import os
import pandas as pd

from datastore import DataStore


class TestSQLvsPandasFilter(unittest.TestCase):
    """Test SQL engine vs Pandas engine for filtering."""

    @classmethod
    def setUpClass(cls):
        """Create test data file."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        # Create simple test data
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.csv_file, index=False)

    def test_1_pure_sql_filter(self):
        """
        Test 1: Pure SQL filter - uses SQL engine only.

        When we only use SQL operations (select, filter), the entire query
        is compiled to SQL and executed by chdb/ClickHouse.
        """
        ds = DataStore.from_file(self.csv_file)

        # Pure SQL operations
        result = ds.select('name', 'age').filter(ds.age > 30)

        # Check what SQL is generated
        sql = result.to_sql()
        print(f"\n=== Test 1: Pure SQL Filter ===")
        print(f"Generated SQL: {sql}")

        # Execute and get results
        df = result.to_df()
        print(f"Result:\n{df}")

        # Verify: should have 3 rows (age > 30: Charlie=35, David=40, Eve=45)
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df['name']), ['Charlie', 'David', 'Eve'])
        self.assertListEqual(list(df['age']), [35, 40, 45])

    def test_2_pandas_operation_then_sql_filter(self):
        """
        Test 2: Pandas operation followed by SQL-like filter.

        When we mix pandas operations (like column assignment) with SQL operations,
        the execution becomes more complex:
        1. First, SQL operations before pandas ops are executed via SQL engine
        2. Then pandas operations are applied to the DataFrame
        3. Then SQL-like operations after pandas ops must be applied via pandas
           (because the data is already in DataFrame form)
        """
        ds = DataStore.from_file(self.csv_file)

        # SQL operation first
        result = ds.select('name', 'age', 'salary')

        # Pandas operation - this forces materialization
        result['age_doubled'] = result['age'] * 2

        # Now filter AFTER pandas operation
        # This filter must be applied via pandas, not SQL!
        result = result.filter(result.age > 30)

        print(f"\n=== Test 2: Pandas Op then SQL-like Filter ===")
        print(f"Explain plan:")
        result.explain()

        # Execute and get results
        df = result.to_df()
        print(f"Result:\n{df}")

        # Verify: should have 3 rows with age_doubled column
        self.assertEqual(len(df), 3)
        self.assertIn('age_doubled', df.columns)
        # age_doubled should be 70, 80, 90 for ages 35, 40, 45
        self.assertListEqual(list(df['age_doubled']), [70, 80, 90])

    def test_3_filter_before_and_after_pandas(self):
        """
        Test 3: Filter before AND after pandas operation.

        This shows both execution paths:
        - First filter (age > 25) is compiled into SQL
        - Pandas operation (age_doubled) is applied to DataFrame
        - Second filter (age_doubled > 70) must use pandas engine
        """
        ds = DataStore.from_file(self.csv_file)

        # First filter - will be in SQL
        result = ds.select('name', 'age').filter(ds.age > 25)

        # Pandas operation
        result['age_doubled'] = result['age'] * 2

        # Second filter - must use pandas since data is already materialized
        result = result.filter(result.age_doubled > 70)

        print(f"\n=== Test 3: Filter Before and After Pandas ===")
        print(f"Explain plan:")
        result.explain()

        df = result.to_df()
        print(f"Result:\n{df}")

        # First filter: age > 25 -> Bob(30), Charlie(35), David(40), Eve(45)
        # age_doubled: 60, 70, 80, 90
        # Second filter: age_doubled > 70 -> David(80), Eve(90)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df['name']), ['David', 'Eve'])

    def test_4_verify_lazy_sql_snapshot_execute(self):
        """
        Test 4: Directly verify that LazyRelationalOp.execute() works on DataFrames.

        This test shows what LazyRelationalOp.execute() is actually doing:
        It's the pandas-based execution of SQL-like operations on a DataFrame.
        """
        from datastore.lazy_ops import LazyRelationalOp
        from datastore.conditions import BinaryCondition
        from datastore.expressions import Field, Literal

        # Create a test DataFrame
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})

        # Create a condition: age > 28
        # BinaryCondition signature: (operator, left, right)
        condition = BinaryCondition('>', Field('age'), Literal(28))

        # Create a LazyRelationalOp for WHERE operation
        lazy_op = LazyRelationalOp('WHERE', 'age > 28', condition=condition)

        # Execute on DataFrame - this is the pandas execution path!
        result_df = lazy_op.execute(df, context=None)

        print(f"\n=== Test 4: LazyRelationalOp.execute() on DataFrame ===")
        print(f"Input DataFrame:\n{df}")
        print(f"After WHERE age > 28:\n{result_df}")

        # Should have 2 rows: Bob(30) and Charlie(35)
        self.assertEqual(len(result_df), 2)
        self.assertListEqual(list(result_df['name']), ['Bob', 'Charlie'])

    def test_5_execution_path_comparison(self):
        """
        Test 5: Compare execution paths and verify both give same results.
        """
        ds1 = DataStore.from_file(self.csv_file)
        ds2 = DataStore.from_file(self.csv_file)

        # Path 1: Pure SQL filter
        result1 = ds1.select('name', 'age').filter(ds1.age > 30)
        df1 = result1.to_df()

        # Path 2: Force pandas execution by adding a no-op pandas operation
        result2 = ds2.select('name', 'age')
        result2['temp'] = result2['age']  # Force materialization
        result2 = result2.filter(result2.age > 30)  # This filter uses pandas
        result2 = result2[['name', 'age']]  # Remove temp column
        df2 = result2.to_df()

        print(f"\n=== Test 5: Execution Path Comparison ===")
        print(f"Pure SQL result:\n{df1}")
        print(f"Pandas execution result:\n{df2}")

        # Both should give same results
        self.assertEqual(len(df1), len(df2))
        self.assertListEqual(list(df1['name']), list(df2['name']))
        self.assertListEqual(list(df1['age']), list(df2['age']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
