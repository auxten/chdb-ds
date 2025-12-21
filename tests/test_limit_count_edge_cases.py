"""
Tests for limit() and count_rows() edge cases.

This module tests various scenarios including:
- Data size less than limit
- Data size equal to limit
- Data size greater than limit
- Limit of 0
- Limit with filter
- Chained limits
- Count with limit
- Offset and limit combinations
"""

import unittest
import tempfile
import os
import pandas as pd

from datastore import DataStore


class TestLimitCountEdgeCases(unittest.TestCase):
    """Test edge cases for limit() and count_rows()."""

    @classmethod
    def setUpClass(cls):
        """Create test data files with known row counts."""
        cls.temp_dir = tempfile.mkdtemp()

        # Small dataset (4 rows)
        cls.small_csv = os.path.join(cls.temp_dir, "small.csv")
        pd.DataFrame(
            {
                'id': [1, 2, 3, 4],
                'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                'age': [25, 30, 35, 28],
                'score': [85.5, 90.0, 78.5, 92.0],
            }
        ).to_csv(cls.small_csv, index=False)

        # Medium dataset (10 rows)
        cls.medium_csv = os.path.join(cls.temp_dir, "medium.csv")
        pd.DataFrame({'id': list(range(1, 11)), 'value': [i * 10 for i in range(1, 11)]}).to_csv(
            cls.medium_csv, index=False
        )

        # Single row dataset
        cls.single_csv = os.path.join(cls.temp_dir, "single.csv")
        pd.DataFrame({'id': [1], 'name': ['Only']}).to_csv(cls.single_csv, index=False)

        # Empty dataset - use parquet which handles empty data correctly
        cls.empty_parquet = os.path.join(cls.temp_dir, "empty.parquet")
        pd.DataFrame(columns=['id', 'name']).to_parquet(cls.empty_parquet)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # ==================== Limit Less Than Data Size ====================

    def test_limit_less_than_data_size(self):
        """Test limit when n < total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df['id']), [1, 2])

    def test_head_less_than_data_size(self):
        """Test head when n < total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df['name']), ['Alice', 'Bob'])

    # ==================== Limit Equal To Data Size ====================

    def test_limit_equal_to_data_size(self):
        """Test limit when n == total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(4)
        self.assertEqual(len(result), 4)

    def test_head_equal_to_data_size(self):
        """Test head when n == total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(4)
        self.assertEqual(len(result), 4)

    # ==================== Limit Greater Than Data Size ====================

    def test_limit_greater_than_data_size(self):
        """Test limit when n > total rows - should return all rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(100)
        self.assertEqual(len(result), 4)  # Only 4 rows exist

    def test_head_greater_than_data_size(self):
        """Test head when n > total rows - should return all rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(100)
        self.assertEqual(len(result), 4)  # Only 4 rows exist

    def test_limit_much_greater_than_data_size(self):
        """Test limit when n >> total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(1000000)
        self.assertEqual(len(result), 4)

    # ==================== Limit of Zero ====================

    def test_limit_zero(self):
        """Test limit(0) returns no rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(0)
        self.assertEqual(len(result), 0)
        df = result.to_df()
        self.assertEqual(len(df), 0)

    def test_head_zero(self):
        """Test head(0) returns no rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(0)
        self.assertEqual(len(result), 0)

    # ==================== Limit of One ====================

    def test_limit_one(self):
        """Test limit(1) returns exactly one row."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(1)
        self.assertEqual(len(result), 1)
        df = result.to_df()
        self.assertEqual(df['id'].iloc[0], 1)

    def test_head_one(self):
        """Test head(1) returns exactly one row."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(1)
        self.assertEqual(len(result), 1)

    # ==================== Single Row Dataset ====================

    def test_limit_on_single_row_less(self):
        """Test limit(0) on single row dataset."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.limit(0)
        self.assertEqual(len(result), 0)

    def test_limit_on_single_row_equal(self):
        """Test limit(1) on single row dataset."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.limit(1)
        self.assertEqual(len(result), 1)

    def test_limit_on_single_row_greater(self):
        """Test limit(10) on single row dataset."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.limit(10)
        self.assertEqual(len(result), 1)

    def test_head_on_single_row(self):
        """Test head() on single row dataset with default n=5."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.head()  # default n=5
        self.assertEqual(len(result), 1)

    # ==================== Empty Dataset ====================

    def test_limit_on_empty_dataset(self):
        """Test limit on empty dataset."""
        ds = DataStore.from_file(self.empty_parquet)
        result = ds.limit(10)
        self.assertEqual(len(result), 0)

    def test_head_on_empty_dataset(self):
        """Test head on empty dataset."""
        ds = DataStore.from_file(self.empty_parquet)
        result = ds.head(10)
        self.assertEqual(len(result), 0)

    def test_count_rows_on_empty_dataset(self):
        """Test count_rows on empty dataset."""
        ds = DataStore.from_file(self.empty_parquet)
        self.assertEqual(ds.count_rows(), 0)

    # ==================== Limit With Filter ====================

    def test_limit_with_filter_less_than_filtered_rows(self):
        """Test limit < filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches: Bob(30), Charlie(35), Diana(28) = 3 rows
        result = ds.filter(ds.age > 25).limit(2)
        self.assertEqual(len(result), 2)

    def test_limit_with_filter_equal_to_filtered_rows(self):
        """Test limit == filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches: Bob(30), Charlie(35), Diana(28) = 3 rows
        result = ds.filter(ds.age > 25).limit(3)
        self.assertEqual(len(result), 3)

    def test_limit_with_filter_greater_than_filtered_rows(self):
        """Test limit > filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches: Bob(30), Charlie(35), Diana(28) = 3 rows
        result = ds.filter(ds.age > 25).limit(100)
        self.assertEqual(len(result), 3)

    def test_head_with_filter_greater_than_filtered_rows(self):
        """Test head > filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 30 matches only Charlie(35) = 1 row
        result = ds.filter(ds.age > 30).head(10)
        self.assertEqual(len(result), 1)

    def test_limit_with_filter_returns_no_rows(self):
        """Test limit when filter matches no rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.filter(ds.age > 100).limit(10)
        self.assertEqual(len(result), 0)

    # ==================== Chained Limits ====================
    # Note: In SQL semantics, later LIMIT overwrites earlier LIMIT.
    # The final LIMIT in the query determines the result.

    def test_chained_limits_decreasing(self):
        """Test limit(10).limit(5) - later limit overwrites (SQL semantics)."""
        ds = DataStore.from_file(self.medium_csv)
        result = ds.limit(10).limit(5)
        self.assertEqual(len(result), 5)

    def test_chained_limits_increasing(self):
        """Test limit(5).limit(10) - later limit overwrites (SQL semantics).

        Note: In SQL, later LIMIT replaces earlier LIMIT, so limit(5).limit(10)
        results in LIMIT 10, returning all 10 rows (not 5).
        """
        ds = DataStore.from_file(self.medium_csv)
        result = ds.limit(5).limit(10)
        # SQL semantics: later LIMIT overwrites, so LIMIT 10 returns all 10 rows
        self.assertEqual(len(result), 10)

    def test_chained_head(self):
        """Test head(5).head(3) - later head overwrites (SQL semantics)."""
        ds = DataStore.from_file(self.medium_csv)
        result = ds.head(5).head(3)
        self.assertEqual(len(result), 3)

    def test_chained_head_increasing(self):
        """Test head(3).head(5) - later head overwrites (SQL semantics)."""
        ds = DataStore.from_file(self.medium_csv)
        result = ds.head(3).head(5)
        # SQL semantics: later LIMIT overwrites, so LIMIT 5 returns 5 rows
        self.assertEqual(len(result), 5)

    # ==================== Offset and Limit Combinations ====================

    def test_offset_and_limit(self):
        """Test offset + limit combination."""
        ds = DataStore.from_file(self.medium_csv)
        result = ds.offset(3).limit(4)
        self.assertEqual(len(result), 4)
        df = result.to_df()
        # Should skip first 3, get rows 4-7 (id: 4, 5, 6, 7)
        self.assertEqual(list(df['id']), [4, 5, 6, 7])

    def test_offset_greater_than_data_size(self):
        """Test offset > total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.offset(100).limit(10)
        self.assertEqual(len(result), 0)

    def test_offset_with_limit_exceeding_remaining(self):
        """Test offset + limit when limit > remaining rows."""
        ds = DataStore.from_file(self.medium_csv)
        # 10 rows, offset 8 leaves 2 rows
        result = ds.offset(8).limit(100)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df['id']), [9, 10])

    # ==================== Count Rows with Limit ====================

    def test_count_rows_with_limit(self):
        """Test count_rows() respects limit."""
        ds = DataStore.from_file(self.medium_csv)
        self.assertEqual(ds.limit(5).count_rows(), 5)

    def test_count_rows_with_limit_greater_than_data(self):
        """Test count_rows() with limit > data size."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(ds.limit(100).count_rows(), 4)

    def test_count_rows_with_limit_zero(self):
        """Test count_rows() with limit(0)."""
        ds = DataStore.from_file(self.medium_csv)
        self.assertEqual(ds.limit(0).count_rows(), 0)

    def test_len_with_limit(self):
        """Test len() respects limit."""
        ds = DataStore.from_file(self.medium_csv)
        self.assertEqual(len(ds.limit(3)), 3)
        self.assertEqual(len(ds.head(7)), 7)

    def test_len_with_limit_greater_than_data(self):
        """Test len() with limit > data size."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(len(ds.limit(1000)), 4)
        self.assertEqual(len(ds.head(1000)), 4)

    # ==================== Count Rows with Filter and Limit ====================

    def test_count_rows_filter_then_limit(self):
        """Test count_rows() with filter + limit."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches 3 rows, limit to 2
        result = ds.filter(ds.age > 25).limit(2)
        self.assertEqual(result.count_rows(), 2)

    def test_count_rows_filter_reduces_below_limit(self):
        """Test count_rows() when filter result < limit."""
        ds = DataStore.from_file(self.small_csv)
        # age > 32 matches only Charlie(35) = 1 row
        result = ds.filter(ds.age > 32).limit(100)
        self.assertEqual(result.count_rows(), 1)

    # ==================== Sort with Limit ====================

    def test_sort_then_limit(self):
        """Test sort + limit returns correct top N."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.sort_values('age', ascending=False).head(2)
        df = result.to_df()
        # Should get Charlie(35) and Bob(30)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(df['name']), ['Charlie', 'Bob'])

    def test_sort_then_limit_greater_than_data(self):
        """Test sort + limit > data size."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.sort_values('age', ascending=True).head(100)
        self.assertEqual(len(result), 4)

    # ==================== Count Without Limit (baseline) ====================

    def test_count_rows_without_limit(self):
        """Test count_rows() without limit - baseline."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(ds.count_rows(), 4)

    def test_count_rows_with_filter_no_limit(self):
        """Test count_rows() with filter but no limit."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(ds.filter(ds.age > 25).count_rows(), 3)

    def test_len_without_limit(self):
        """Test len() without limit - baseline."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(len(ds), 4)

    # ==================== Complex Combinations ====================

    def test_filter_sort_limit_count(self):
        """Test filter + sort + limit + count."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.filter(ds.age > 25).sort_values('score', ascending=False).head(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        # age > 25: Bob(90), Charlie(78.5), Diana(92)
        # sorted by score desc: Diana(92), Bob(90), Charlie(78.5)
        # top 2: Diana, Bob
        self.assertEqual(list(df['name']), ['Diana', 'Bob'])

    def test_select_filter_limit(self):
        """Test select + filter + limit."""
        ds = DataStore.from_file(self.small_csv)
        result = ds[['name', 'age']][ds['age'] > 25].head(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df.columns), ['name', 'age'])


class TestLimitCountWithDataStore(unittest.TestCase):
    """Test limit and count with DataStore created from DataFrame."""

    def test_from_dataframe_limit(self):
        """Test limit on DataStore created from DataFrame."""
        df = pd.DataFrame({'x': range(100)})
        ds = DataStore.from_dataframe(df)
        result = ds.limit(10)
        self.assertEqual(len(result), 10)

    def test_from_dataframe_head_greater_than_data(self):
        """Test head > data size on DataFrame-backed DataStore."""
        df = pd.DataFrame({'x': range(5)})
        ds = DataStore.from_dataframe(df)
        result = ds.head(100)
        self.assertEqual(len(result), 5)

    def test_from_dataframe_count_rows(self):
        """Test count_rows on DataFrame-backed DataStore."""
        df = pd.DataFrame({'x': range(50)})
        ds = DataStore.from_dataframe(df)
        self.assertEqual(ds.count_rows(), 50)

    def test_from_dataframe_count_rows_with_limit(self):
        """Test count_rows with limit on DataFrame-backed DataStore."""
        df = pd.DataFrame({'x': range(50)})
        ds = DataStore.from_dataframe(df)
        self.assertEqual(ds.limit(10).count_rows(), 10)


if __name__ == '__main__':
    unittest.main()
