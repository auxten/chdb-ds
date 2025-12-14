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
        # Use example dataset (lazy execution compatible)
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')

        # Load data with lazy execution
        ds = DataStore.from_file(dataset_path)

        # Assign constant value to new column (lazy)
        ds['constant_col'] = 10

        # Trigger execution
        result_df = ds.to_df()

        # Verify the result
        assert 'constant_col' in result_df.columns
        assert all(result_df['constant_col'] == 10)
        assert len(result_df) > 0  # Should have data

    def test_column_assignment_expression(self):
        """Test assigning an expression to a column."""
        # Use example dataset (lazy execution compatible)
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')

        # Load data with lazy execution
        ds = DataStore.from_file(dataset_path)

        # Assign expression to new column (lazy)
        ds['age_doubled'] = ds['age'] * 2

        # Trigger execution
        result_df = ds.to_df()

        # Verify the result
        assert 'age_doubled' in result_df.columns
        assert (result_df['age_doubled'] == result_df['age'] * 2).all()

    def test_column_update_in_place(self):
        """Test updating an existing column (lazy execution)."""
        # Use example dataset
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')

        # Load data with lazy execution
        ds = DataStore.from_file(dataset_path)

        # Update existing column (lazy: nat["n_nationkey"] = nat["n_nationkey"] - 1)
        ds['age'] = ds['age'] - 1

        # Trigger execution
        result_df = ds.to_df()

        # Verify: age should be decreased by 1 from original
        # Load original data to compare
        original_df = pd.read_csv(dataset_path)
        assert (result_df['age'] == original_df['age'] - 1).all()

    def test_column_assignment_from_file(self):
        """Test column assignment with data loaded from file."""
        # Use example dataset
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')

        # Load data
        ds = DataStore.from_file(dataset_path)

        # Assign new column (lazy execution)
        ds['age_plus_10'] = ds['age'] + 10

        # Trigger execution
        result_df = ds.to_df()

        # Verify the result
        assert 'age_plus_10' in result_df.columns
        # Verify values are correct
        original_df = pd.read_csv(dataset_path)
        assert (result_df['age_plus_10'] == original_df['age'] + 10).all()

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
        """Test chaining multiple column assignments (lazy execution)."""
        # Use example dataset
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')

        # Load data
        ds = DataStore.from_file(dataset_path)

        # Chain multiple assignments (lazy)
        ds['age_plus_10'] = ds['age'] + 10
        ds['age_doubled'] = ds['age_plus_10'] * 2

        # Trigger execution
        result_df = ds.to_df()

        # Verify the result
        assert 'age_plus_10' in result_df.columns
        assert 'age_doubled' in result_df.columns

        # Verify calculations
        original_df = pd.read_csv(dataset_path)
        assert (result_df['age_plus_10'] == original_df['age'] + 10).all()
        assert (result_df['age_doubled'] == (original_df['age'] + 10) * 2).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
