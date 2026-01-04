"""
Test cases for assign() method with mixed SQL expressions and lambda/pandas expressions.

These tests verify that assign() works correctly when:
1. Only SQL expressions are provided
2. Only lambda/pandas expressions are provided  
3. Mixed SQL expressions and lambda expressions are provided together
"""

import pandas as pd
import pytest
from datastore import DataStore


class TestAssignMixed:
    """Test assign() with mixed expression types."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame and DataStore for testing."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })
        ds = DataStore(df.copy())
        return df, ds

    def test_assign_only_sql_expression(self, sample_data):
        """Test assign with only SQL expressions (ColumnExpr)."""
        df, ds = sample_data
        
        # DataStore with SQL expression
        ds_result = ds.assign(D=ds['A'] + ds['B'])
        
        # pandas equivalent
        pd_result = df.assign(D=df['A'] + df['B'])
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_assign_only_lambda(self, sample_data):
        """Test assign with only lambda expressions."""
        df, ds = sample_data
        
        # DataStore with lambda
        ds_result = ds.assign(D=lambda x: x['A'] * 2)
        
        # pandas equivalent
        pd_result = df.assign(D=lambda x: x['A'] * 2)
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_assign_mixed_expr_and_lambda(self, sample_data):
        """Test assign with both SQL expression and lambda."""
        df, ds = sample_data
        
        # DataStore with mixed expressions
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],        # SQL expression
            E=lambda x: x['C'] * 2      # Lambda
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=lambda x: x['C'] * 2
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_assign_mixed_multiple_of_each(self, sample_data):
        """Test assign with multiple SQL expressions and multiple lambdas."""
        df, ds = sample_data
        
        # DataStore with multiple mixed expressions
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],        # SQL expression 1
            E=ds['A'] * ds['B'],        # SQL expression 2
            F=lambda x: x['C'] - 50,    # Lambda 1
            G=lambda x: x['A'] ** 2     # Lambda 2
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=df['A'] * df['B'],
            F=lambda x: x['C'] - 50,
            G=lambda x: x['A'] ** 2
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_assign_with_scalar_value(self, sample_data):
        """Test assign with scalar values (treated as pandas expression)."""
        df, ds = sample_data
        
        # DataStore with scalar
        ds_result = ds.assign(constant=42)
        
        # pandas equivalent
        pd_result = df.assign(constant=42)
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_assign_mixed_expr_lambda_scalar(self, sample_data):
        """Test assign with SQL expression, lambda, and scalar mixed together."""
        df, ds = sample_data
        
        # DataStore with all types mixed
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],        # SQL expression
            E=lambda x: x['C'] * 2,     # Lambda
            F=999                       # Scalar
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=lambda x: x['C'] * 2,
            F=999
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_assign_preserves_original_columns(self, sample_data):
        """Test that assign preserves all original columns."""
        df, ds = sample_data
        
        original_cols = ds.columns.tolist()
        
        # Assign new columns
        ds_result = ds.assign(
            D=ds['A'] + 1,
            E=lambda x: x['B'] + 1
        )
        
        # Check all original columns are present
        for col in original_cols:
            assert col in ds_result.columns.tolist(), f"Original column {col} missing"
        
        # Check new columns are present
        assert 'D' in ds_result.columns.tolist()
        assert 'E' in ds_result.columns.tolist()

    def test_assign_chain_operations(self, sample_data):
        """Test chained assign operations."""
        df, ds = sample_data
        
        # Chain assigns
        ds_result = ds.assign(D=ds['A'] + ds['B']).assign(E=lambda x: x['D'] * 2)
        
        # pandas equivalent
        pd_result = df.assign(D=df['A'] + df['B']).assign(E=lambda x: x['D'] * 2)
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_assign_lambda_references_new_sql_column(self, sample_data):
        """Test that lambda can reference columns created by SQL expressions in same assign."""
        df, ds = sample_data
        
        # In mixed assign, SQL expressions are processed first
        # Then lambda can reference the new columns
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],
            E=lambda x: x['D'] * 2  # References D created by SQL expr
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=lambda x: x['D'] * 2
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        pd.testing.assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
