"""
Test column assignment operations (pandas-style).

This module tests the ability to update columns using the syntax:
    ds['column'] = value
"""

import pytest
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datastore import DataStore


class TestColumnAssignment:
    """Test column assignment operations."""

    def test_column_assignment_constant(self):
        """Test assigning a constant value to a column."""
        # Create a simple DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Create DataStore from DataFrame
        ds = DataStore('chdb')
        ds._cached_df = df.copy()
        ds._materialized = True
        ds._cache_invalidated = False

        # Assign constant value to new column (pandas-style)
        ds['c'] = 10

        # Verify the result
        result_df = ds.to_df()
        assert 'c' in result_df.columns
        assert all(result_df['c'] == 10)
        assert len(result_df) == 3

    def test_column_assignment_expression(self):
        """Test assigning an expression to a column."""
        # Create a simple DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Create DataStore from DataFrame
        ds = DataStore('chdb')
        ds._cached_df = df.copy()
        ds._materialized = True
        ds._cache_invalidated = False

        # Assign expression to new column (pandas-style)
        ds['c'] = ds['a'] * 2

        # Verify the result
        result_df = ds.to_df()
        assert 'c' in result_df.columns
        pd.testing.assert_series_equal(result_df['c'], pd.Series([2, 4, 6], name='c'))

    def test_column_update_in_place(self):
        """Test updating an existing column (like pandas)."""
        # Create a simple DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Create DataStore from DataFrame
        ds = DataStore('chdb')
        ds._cached_df = df.copy()
        ds._materialized = True
        ds._cache_invalidated = False

        # Update existing column (pandas-style: nat["n_nationkey"] = nat["n_nationkey"] - 1)
        ds['a'] = ds['a'] - 1

        # Verify the result
        result_df = ds.to_df()
        pd.testing.assert_series_equal(result_df['a'], pd.Series([0, 1, 2], name='a'))

    def test_column_assignment_from_file(self):
        """Test column assignment with data loaded from file."""
        # Use example dataset
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')

        if not os.path.exists(dataset_path):
            pytest.skip(f"Dataset not found: {dataset_path}")

        # Load data
        ds = DataStore.from_file(dataset_path)
        ds = ds.connect()

        # Get original data for comparison
        original_age = ds['age'].copy()

        # Assign new column (pandas-style)
        ds['age_plus_10'] = ds['age'] + 10

        # Verify the result
        result_df = ds.to_df()
        assert 'age_plus_10' in result_df.columns
        # Verify values are correct
        assert all(result_df['age_plus_10'] == original_age + 10)

    def test_column_assignment_series_compatibility(self):
        """Test that column assignment works with pandas Series."""
        # Create a simple DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Create DataStore from DataFrame
        ds = DataStore('chdb')
        ds._cached_df = df.copy()
        ds._materialized = True
        ds._cache_invalidated = False

        # Create a pandas Series
        new_values = pd.Series([7, 8, 9])

        # Assign Series to new column (pandas-style)
        ds['c'] = new_values

        # Verify the result
        result_df = ds.to_df()
        assert 'c' in result_df.columns
        assert list(result_df['c']) == [7, 8, 9]

    def test_column_assignment_list_compatibility(self):
        """Test that column assignment works with lists."""
        # Create a simple DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Create DataStore from DataFrame
        ds = DataStore('chdb')
        ds._cached_df = df.copy()
        ds._materialized = True
        ds._cache_invalidated = False

        # Create a list of values
        new_values = [7, 8, 9]

        # Assign list to new column (pandas-style)
        ds['c'] = new_values

        # Verify the result
        result_df = ds.to_df()
        assert 'c' in result_df.columns
        assert list(result_df['c']) == new_values

    def test_column_assignment_multiple_operations(self):
        """Test chaining multiple column assignments (pandas-style)."""
        # Create a simple DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Create DataStore from DataFrame
        ds = DataStore('chdb')
        ds._cached_df = df.copy()
        ds._materialized = True
        ds._cache_invalidated = False

        # Chain multiple assignments (pandas-style)
        ds['c'] = ds['a'] + ds['b']
        ds['d'] = ds['c'] * 2

        # Verify the result
        result_df = ds.to_df()
        assert 'c' in result_df.columns
        assert 'd' in result_df.columns
        pd.testing.assert_series_equal(result_df['c'], pd.Series([5, 7, 9], name='c'))
        pd.testing.assert_series_equal(result_df['d'], pd.Series([10, 14, 18], name='d'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
