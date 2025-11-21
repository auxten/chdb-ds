"""
Tests for the explain() method.
"""

import unittest
import tempfile
import os
from datastore import DataStore


class TestExplainMethod(unittest.TestCase):
    """Test the explain() method."""

    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.temp_dir, "test.csv")

        with open(self.csv_file, "w") as f:
            f.write("id,name,age,salary\n")
            f.write("1,Alice,28,65000\n")
            f.write("2,Bob,32,70000\n")
            f.write("3,Charlie,26,55000\n")
            f.write("4,David,35,80000\n")

    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.csv_file):
            os.unlink(self.csv_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_explain_returns_string(self):
        """Test that explain() returns a string."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*')

        output = result.explain()
        self.assertIsInstance(output, str)
        self.assertIn("Execution Plan", output)

    def test_explain_pure_sql(self):
        """Test explain() with pure SQL operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25)

        output = result.explain()
        self.assertIn("Phase 1", output)
        self.assertIn("SQL Query Building", output)
        self.assertIn("SELECT *", output)
        self.assertIn("WHERE", output)

    def test_explain_mixed_operations(self):
        """Test explain() with mixed SQL and Pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_').filter(ds.p1_salary > 55000)

        output = result.explain()
        self.assertIn("Phase 1", output)
        self.assertIn("Phase 2", output)
        self.assertIn("Phase 3", output)
        self.assertIn("Materialization", output)

    def test_explain_verbose_mode(self):
        """Test explain() with verbose=True."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_')

        normal_output = result.explain()
        verbose_output = result.explain(verbose=True)

        # Verbose should have more content
        self.assertGreater(len(verbose_output), len(normal_output))
        self.assertIn("shape", verbose_output)

    def test_explain_does_not_execute(self):
        """Test that explain() does not execute the query."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25)

        # Should not be materialized before explain
        self.assertFalse(result._materialized)

        # Call explain
        result.explain()

        # Should still not be materialized
        self.assertFalse(result._materialized)
        self.assertIsNone(result._cached_df)

    def test_explain_materialized_dataframe(self):
        """Test explain() on a materialized DataFrame."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').add_prefix('p1_')

        output = result.explain()
        self.assertIn("Materialized DataFrame", output)
        self.assertIn("cached", output)

    def test_explain_shows_sql_query(self):
        """Test that explain() shows the SQL query for unmaterialized queries."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).filter(ds.salary > 60000)

        output = result.explain()
        self.assertIn("Generated SQL Query", output)
        self.assertIn("SELECT", output)
        self.assertIn("FROM", output)
        self.assertIn("WHERE", output)

    def test_explain_tracks_operations(self):
        """Test that explain() tracks operations correctly."""
        ds = DataStore.from_file(self.csv_file)
        result = (
            ds.select('*')  # Op 1
            .filter(ds.age > 25)  # Op 2
            .add_prefix('p1_')  # Op 3 (materialization)
            .filter(ds.p1_salary > 55000)  # Op 4
            .rename(columns={'p1_id': 'id2'})  # Op 5
        )

        output = result.explain()

        # Should have numbered operations
        self.assertIn("[1]", output)
        self.assertIn("[2]", output)
        self.assertIn("[3]", output)
        self.assertIn("[4]", output)
        self.assertIn("[5]", output)

    def test_explain_phase_separation(self):
        """Test that explain() correctly separates phases."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_').filter(ds.p1_salary > 55000)

        output = result.explain()

        # Check phase order
        phase1_idx = output.index("Phase 1")
        phase2_idx = output.index("Phase 2")
        phase3_idx = output.index("Phase 3")

        self.assertLess(phase1_idx, phase2_idx)
        self.assertLess(phase2_idx, phase3_idx)

    def test_explain_with_no_operations(self):
        """Test explain() with a DataStore that has no operations."""
        ds = DataStore.from_file(self.csv_file)

        output = ds.explain()
        self.assertIn("Execution Plan", output)

    def test_explain_pandas_first(self):
        """Test explain() when pandas operation comes first."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').add_prefix('emp_').filter(ds.emp_age > 28)

        output = result.explain()
        self.assertIn("Phase 2", output)
        self.assertIn("Materialization", output)

    def test_explain_extreme_many_operations(self):
        """Test explain() with 100+ mixed operations (only tests explain, not execution)."""
        ds = DataStore.from_file(self.csv_file)

        # Build a chain with 100 operations - but don't execute it!
        # Just track the operations for explain()
        result = ds.select('*')

        # 10 SQL filter operations (different filters to avoid redundancy)
        for i in range(10):
            result = result.filter(ds.age > 20)  # Same filter is OK for explain testing

        # Trigger materialization
        result = result.add_prefix('p1_')

        # 40 mixed operations (SQL + Pandas)
        for i in range(20):
            # Track these operations without executing
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "p1_age" > {20 + i}',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"rename(id_{i})",
                    'details': {'shape': (2, 4)},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )

        # 50 more Pandas operations
        for i in range(25):
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"add_suffix('_s{i}')",
                    'details': {'shape': (2, 4)},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"add_prefix('p{i}_')",
                    'details': {'shape': (2, 4)},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )

        # explain should handle many operations without crashing
        output = result.explain()

        # Verify basic structure exists
        self.assertIn("Execution Plan", output)
        self.assertIn("Final State", output)

        # Should have correct phases
        self.assertIn("Phase 1", output)  # Data source + 10 SQL operations
        self.assertIn("Phase 2", output)  # Materialization point
        self.assertIn("Phase 3", output)  # 90 operations after materialization

        # Verify operation count roughly correct (100+ operations)
        # Numbers should go from [1] to [100+]
        self.assertIn("[1]", output)
        self.assertIn("[50]", output)
        self.assertIn("[100]", output)

    def test_explain_extreme_deep_nesting(self):
        """Test explain() with deeply nested operations (explain only, not execution)."""
        ds = DataStore.from_file(self.csv_file)

        # Simulate deeply nested operations by directly adding to history
        result = ds.select('*')

        # 25 lazy SQL operations
        for i in range(25):
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "age" > {20 + i}',
                    'details': {'lazy': True},
                    'is_on_dataframe': False,
                    'materialized_at_call': False,
                }
            )

        # Materialization
        result = result.add_prefix('mid_')

        # 25 more operations after materialization
        for i in range(25):
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "mid_age" > {20 + i}',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )

        output = result.explain()
        self.assertIn("Execution Plan", output)
        self.assertIn("[50]", output)  # Should have operation #50

    def test_explain_extreme_alternating_sql_pandas(self):
        """Test explain() with alternating SQL and Pandas operations (explain only)."""
        ds = DataStore.from_file(self.csv_file)

        # Start with SQL
        result = ds.select('*').filter(ds.age > 25)

        # Simulate alternating operations without executing
        # First pandas triggers materialization
        result = result.add_prefix('p0_')

        # Add 24 more alternating operations to history (48 ops total)
        for i in range(1, 25):
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"add_prefix('p{i}_')",
                    'details': {'shape': (2, 4), 'on_cached_df': True},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "p{i}_age" > 20',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )

        output = result.explain()

        # Should show correct materialization and subsequent operations
        self.assertIn("Phase 2: Materialization Point", output)
        self.assertIn("Phase 3: Operations on Materialized DataFrame", output)

        # Verify has 50+ operations
        self.assertIn("[50]", output)

    def test_explain_extreme_performance(self):
        """Test that explain() performs well with many operations (200+ ops)."""
        import time

        ds = DataStore.from_file(self.csv_file)

        # Build a chain with 200 operations by directly manipulating history
        result = ds.select('*')

        # 100 lazy SQL operations (add to history without executing)
        for i in range(100):
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "age" > {20 + (i % 10)}',
                    'details': {'lazy': True},
                    'is_on_dataframe': False,
                    'materialized_at_call': False,
                }
            )

        # Trigger materialization
        result = result.add_prefix('p1_')

        # 100 more operations after materialization
        for i in range(100):
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "p1_age" > {20 + (i % 10)}',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'materialized_at_call': True,
                }
            )

        # explain() should complete quickly (<2 seconds)
        start = time.time()
        output = result.explain()
        duration = time.time() - start

        self.assertLess(duration, 2.0, "explain() should complete in less than 2 seconds")
        self.assertIn("[200]", output)

    def test_explain_only_pandas_operations(self):
        """Test explain() with only pandas operations (no explicit SQL)."""
        ds = DataStore.from_file(self.csv_file)

        # Only Pandas operations
        result = ds.add_prefix('p1_').add_suffix('_s1').rename(columns={'p1_id_s1': 'new_id'}).add_prefix('p2_')

        output = result.explain()

        # Should have Phase 1 (data source)
        self.assertIn("Phase 1", output)
        self.assertIn("Data Source", output)

        # Should have Phase 2 (materialization point)
        self.assertIn("Phase 2", output)
        self.assertIn("Materialization", output)

        # Should have Phase 3 (subsequent pandas operations)
        self.assertIn("Phase 3", output)


if __name__ == '__main__':
    unittest.main()
