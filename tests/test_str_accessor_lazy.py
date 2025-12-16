"""
Test Series.str accessor lazy execution behavior.

Verifies that:
1. SQL-based string methods remain lazy until materialization
2. Materializing methods (cat, extractall, get_dummies, partition, rpartition)
   explicitly execute and return proper results
3. Mixed usage patterns work correctly
"""

import pytest
import pandas as pd
from datastore import DataStore
from datastore.column_expr import ColumnExpr, ColumnExprStringAccessor


class TestStrAccessorLazy:
    """Test that .str accessor methods maintain lazy evaluation where appropriate."""

    @pytest.fixture
    def ds(self):
        """Create a test DataStore from DataFrame."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'name': ['John|Doe', 'Jane|Smith', 'Bob|Brown'],
                'text': ['hello world', 'foo bar baz', 'test string'],
                'numbers': ['abc123def456', 'xyz789', 'test123test456'],
            }
        )
        return DataStore.from_df(df)

    # ==================== SQL-Based Methods (Lazy) ====================

    def test_upper_is_lazy(self, ds):
        """Test that .str.upper() returns ColumnExpr (lazy)."""
        result = ds['name'].str.upper()
        assert isinstance(result, ColumnExpr)
        # Should not have executed yet - just built expression

    def test_lower_is_lazy(self, ds):
        """Test that .str.lower() returns ColumnExpr (lazy)."""
        result = ds['name'].str.lower()
        assert isinstance(result, ColumnExpr)

    def test_len_is_lazy(self, ds):
        """Test that .str.len() returns ColumnExpr (lazy)."""
        result = ds['name'].str.len()
        assert isinstance(result, ColumnExpr)

    def test_contains_is_lazy(self, ds):
        """Test that .str.contains() returns ColumnExpr (lazy)."""
        result = ds['name'].str.contains('John')
        assert isinstance(result, ColumnExpr)

    def test_replace_is_lazy(self, ds):
        """Test that .str.replace() returns ColumnExpr (lazy)."""
        result = ds['name'].str.replace('|', '-')
        assert isinstance(result, ColumnExpr)

    def test_split_is_lazy(self, ds):
        """Test that .str.split() returns ColumnExpr (lazy)."""
        result = ds['name'].str.split('|')
        assert isinstance(result, ColumnExpr)

    def test_strip_is_lazy(self, ds):
        """Test that .str.strip() returns ColumnExpr (lazy)."""
        result = ds['text'].str.strip()
        assert isinstance(result, ColumnExpr)

    def test_slice_is_lazy(self, ds):
        """Test that .str.slice() returns ColumnExpr (lazy)."""
        result = ds['text'].str.slice(0, 5)
        assert isinstance(result, ColumnExpr)

    def test_get_is_lazy(self, ds):
        """Test that .str.get() returns ColumnExpr (lazy)."""
        result = ds['text'].str.get(0)
        assert isinstance(result, ColumnExpr)

    def test_count_is_lazy(self, ds):
        """Test that .str.count() returns ColumnExpr (lazy)."""
        result = ds['text'].str.count('o')
        assert isinstance(result, ColumnExpr)

    # ==================== Lazy Execution Verification ====================

    def test_lazy_chain_no_execution(self, ds):
        """Test that chaining lazy str methods doesn't trigger execution."""
        # Chain multiple operations
        result = ds['name'].str.upper().str.lower().str.len()

        # All should be lazy - returning ColumnExpr
        assert isinstance(result, ColumnExpr)

        # No execution should have happened - verify by checking _lazy_ops
        # Original ds should still have minimal lazy ops
        assert len(ds._lazy_ops) <= 2  # Initial DataFrame source + maybe one more

    def test_lazy_column_assignment_with_str(self, ds):
        """Test that assigning str result to column is lazy."""
        # This should be recorded as lazy operation
        ds['upper_name'] = ds['name'].str.upper()

        # Should have recorded a lazy operation
        has_lazy_assignment = any(op.__class__.__name__ == 'LazyColumnAssignment' for op in ds._lazy_ops)
        assert has_lazy_assignment

    def test_lazy_str_executes_on_to_df(self, ds):
        """Test that lazy str operations execute when calling to_df()."""
        ds['upper_name'] = ds['name'].str.upper()

        # Execute
        df = ds.to_df()

        # Verify result
        assert 'upper_name' in df.columns

    # ==================== Materializing Methods ====================

    def test_partition_materializes(self, ds):
        """Test that .str.partition() materializes and returns DataStore."""
        result = ds['name'].str.partition('|')

        # Should return a DataStore (materialized)
        assert isinstance(result, DataStore)

        # Verify result structure
        df = result.to_df()
        assert df.shape[1] == 3  # Three columns: left, sep, right
        assert len(df) == 3  # Three rows

    def test_rpartition_materializes(self, ds):
        """Test that .str.rpartition() materializes and returns DataStore."""
        result = ds['name'].str.rpartition('|')

        # Should return a DataStore (materialized)
        assert isinstance(result, DataStore)

        # Verify result
        df = result.to_df()
        assert df.shape[1] == 3

    def test_get_dummies_materializes(self, ds):
        """Test that .str.get_dummies() materializes and returns DataStore."""
        result = ds['name'].str.get_dummies('|')

        # Should return a DataStore (materialized)
        assert isinstance(result, DataStore)

        # Verify result has dummy columns
        df = result.to_df()
        assert df.shape[1] > 0

    def test_extractall_materializes(self, ds):
        """Test that .str.extractall() materializes and returns DataStore."""
        result = ds['numbers'].str.extractall(r'(\d+)')

        # Should return a DataStore (materialized)
        assert isinstance(result, DataStore)

        # Verify result
        df = result.to_df()
        assert len(df) > 0  # Should have extracted numbers

    def test_cat_materializes_to_string(self, ds):
        """Test that .str.cat() materializes and returns string."""
        result = ds['name'].str.cat(sep='-')

        # Should return a string (fully materialized)
        assert isinstance(result, str)
        assert 'John|Doe' in result
        assert '-' in result

    def test_partition_expand_false_returns_series(self, ds):
        """Test that partition with expand=False returns Series."""
        result = ds['name'].str.partition('|', expand=False)

        # Should return a Series of tuples
        assert isinstance(result, pd.Series)
        assert all(isinstance(x, tuple) for x in result)

    # ==================== Mixed Usage ====================

    def test_lazy_after_materializing_method(self, ds):
        """Test that lazy operations work after materializing method."""
        # First materialize with partition
        partitioned = ds['name'].str.partition('|')
        assert isinstance(partitioned, DataStore)

        # Get column names from the partitioned result
        df = partitioned.to_df()
        first_col = df.columns[0]

        # Create fresh DataStore from the DataFrame for lazy ops test
        partitioned2 = DataStore.from_df(df)
        partitioned2['first_upper'] = partitioned2[str(first_col)].str.upper()

        # Should have lazy assignment
        has_lazy = any(op.__class__.__name__ == 'LazyColumnAssignment' for op in partitioned2._lazy_ops)
        assert has_lazy

    def test_materializing_method_preserves_parent(self, ds):
        """Test that materializing method doesn't modify parent DataStore."""
        original_lazy_ops_count = len(ds._lazy_ops)

        # Call materializing method
        result = ds['name'].str.partition('|')

        # Parent should be unchanged
        assert len(ds._lazy_ops) == original_lazy_ops_count

        # Can still do operations on parent
        ds['new_col'] = ds['text'].str.upper()
        assert len(ds._lazy_ops) == original_lazy_ops_count + 1

    # ==================== Correctness ====================

    def test_lazy_str_upper_correct_result(self, ds):
        """Test that lazy .str.upper() produces correct result."""
        ds['upper_name'] = ds['name'].str.upper()
        df = ds.order_by('id').to_df()

        expected = ['JOHN|DOE', 'JANE|SMITH', 'BOB|BROWN']
        assert list(df['upper_name']) == expected

    def test_lazy_str_len_correct_result(self, ds):
        """Test that lazy .str.len() produces correct result."""
        ds['name_len'] = ds['name'].str.len()
        df = ds.order_by('id').to_df()

        # Lengths: 'John|Doe'=8, 'Jane|Smith'=10, 'Bob|Brown'=9
        assert list(df['name_len']) == [8, 10, 9]

    def test_partition_correct_result(self, ds):
        """Test that .str.partition() produces correct result."""
        result = ds['name'].str.partition('|')
        df = result.to_df()

        assert list(df[0]) == ['John', 'Jane', 'Bob']
        assert list(df[1]) == ['|', '|', '|']
        assert list(df[2]) == ['Doe', 'Smith', 'Brown']

    def test_get_dummies_correct_result(self, ds):
        """Test that .str.get_dummies() produces correct result."""
        result = ds['name'].str.get_dummies('|')
        df = result.to_df()

        # Should have columns for each unique value
        assert 'John' in df.columns
        assert 'Doe' in df.columns
        assert 'Jane' in df.columns

    def test_extractall_correct_result(self, ds):
        """Test that .str.extractall() produces correct result."""
        result = ds['numbers'].str.extractall(r'(\d+)')
        df = result.to_df()

        # First row 'abc123def456' should have matches '123', '456'
        first_row_matches = df[df['level_0'] == 0][0].tolist()
        assert '123' in first_row_matches
        assert '456' in first_row_matches


class TestStrAccessorEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test str accessor on empty DataFrame."""
        df = pd.DataFrame({'name': pd.Series([], dtype=str)})
        ds = DataStore.from_df(df)

        # Lazy method should work
        result = ds['name'].str.upper()
        assert isinstance(result, ColumnExpr)

        # Materializing method should work
        result = ds['name'].str.partition('|')
        assert isinstance(result, DataStore)
        assert len(result.to_df()) == 0

    def test_null_values(self):
        """Test str accessor with null values."""
        df = pd.DataFrame({'name': ['hello', None, 'world']})
        ds = DataStore.from_df(df)

        # Should handle nulls gracefully
        result = ds['name'].str.partition(' ')
        assert isinstance(result, DataStore)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
