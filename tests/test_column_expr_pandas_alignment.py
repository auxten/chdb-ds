"""
Tests for ColumnExpr pandas alignment.

These tests verify that DataStore's ColumnExpr behaves similarly to pandas
when accessing columns and performing operations.

Key behaviors tested:
- ds["col"] displays actual values like pandas Series
- ds["col"] + 1 returns computed values like pandas
- ds["col"].str.upper() returns string operation results
- ds["col"] > 10 returns Condition (for filtering, unlike pandas boolean Series)
"""

import unittest
import pandas as pd
import numpy as np

from datastore import DataStore, Field, ColumnExpr
from datastore.conditions import BinaryCondition, Condition
from datastore.lazy_ops import LazyDataFrameSource


class TestColumnExprPandasAlignment(unittest.TestCase):
    """Test ColumnExpr alignment with pandas behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
                'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
                'department': ['HR', 'Engineering', 'Engineering', 'Management', 'HR'],
                'active': [True, True, False, True, True],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    # ========== Basic Column Access ==========

    def test_column_access_returns_column_expr(self):
        """Test that ds['col'] returns ColumnExpr."""
        ds = self.create_ds()
        result = ds['age']
        self.assertIsInstance(result, ColumnExpr)

    def test_column_access_wraps_field(self):
        """Test that ColumnExpr wraps a Field."""
        ds = self.create_ds()
        result = ds['age']
        self.assertIsInstance(result._expr, Field)
        self.assertEqual(result._expr.name, 'age')

    def test_column_access_materializes_correctly(self):
        """Test that column access materializes to correct values."""
        ds = self.create_ds()
        result = list(ds['age']._materialize())
        expected = list(self.df['age'])
        self.assertEqual(result, expected)

    def test_attribute_access_returns_column_expr(self):
        """Test that ds.col returns ColumnExpr."""
        ds = self.create_ds()
        result = ds.age
        self.assertIsInstance(result, ColumnExpr)

    def test_attribute_access_materializes_correctly(self):
        """Test that attribute access materializes to correct values."""
        ds = self.create_ds()
        result = list(ds.age._materialize())
        expected = list(self.df['age'])
        self.assertEqual(result, expected)

    # ========== Arithmetic Operations ==========

    def test_addition(self):
        """Test column + scalar."""
        ds = self.create_ds()
        ds_result = list(ds['age'] + 10)
        pd_result = list(self.df['age'] + 10)
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_subtraction(self):
        """Test column - scalar."""
        ds = self.create_ds()
        ds_result = list(ds['age'] - 5)
        pd_result = list(self.df['age'] - 5)
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_multiplication(self):
        """Test column * scalar."""
        ds = self.create_ds()
        ds_result = list(ds['salary'] * 1.1)
        pd_result = list(self.df['salary'] * 1.1)
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))

    def test_division(self):
        """Test column / scalar."""
        ds = self.create_ds()
        ds_result = list(ds['salary'] / 1000)
        pd_result = list(self.df['salary'] / 1000)
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))

    def test_floor_division(self):
        """Test column // scalar."""
        ds = self.create_ds()
        ds_result = list(ds['age'] // 10)
        pd_result = list(self.df['age'] // 10)
        # Compare as sorted integers
        self.assertEqual(sorted([int(x) for x in ds_result]), sorted([int(x) for x in pd_result]))

    def test_modulo(self):
        """Test column % scalar."""
        ds = self.create_ds()
        ds_result = list(ds['age'] % 10)
        pd_result = list(self.df['age'] % 10)
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_power(self):
        """Test column ** scalar."""
        ds = self.create_ds()
        ds_result = list(ds['age'] ** 2)
        pd_result = list(self.df['age'] ** 2)
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    # ========== Reverse Arithmetic Operations ==========

    def test_reverse_addition(self):
        """Test scalar + column."""
        ds = self.create_ds()
        ds_result = list(100 + ds['age'])
        pd_result = list(100 + self.df['age'])
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_reverse_subtraction(self):
        """Test scalar - column."""
        ds = self.create_ds()
        ds_result = list(1000 - ds['age'])
        pd_result = list(1000 - self.df['age'])
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_reverse_multiplication(self):
        """Test scalar * column."""
        ds = self.create_ds()
        ds_result = list(2 * ds['salary'])
        pd_result = list(2 * self.df['salary'])
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))

    # ========== Unary Operations ==========

    def test_negation(self):
        """Test -column."""
        ds = self.create_ds()
        ds_result = list(-ds['age'])
        pd_result = list(-self.df['age'])
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    # ========== Column-Column Operations ==========

    def test_column_column_addition(self):
        """Test column + column."""
        ds = self.create_ds()
        ds_result = list(ds['age'] + ds['salary'] / 1000)
        pd_result = list(self.df['age'] + self.df['salary'] / 1000)
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))

    # ========== Chained Operations ==========

    def test_chained_arithmetic(self):
        """Test (column - scalar) * scalar + scalar."""
        ds = self.create_ds()
        ds_result = list((ds['age'] - 20) * 2 + 10)
        pd_result = list((self.df['age'] - 20) * 2 + 10)
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_complex_chained_operations(self):
        """Test complex chained operations."""
        ds = self.create_ds()
        ds_result = list(ds['salary'] / 1000 - ds['age'])
        pd_result = list(self.df['salary'] / 1000 - self.df['age'])
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))


class TestColumnExprStringOperations(unittest.TestCase):
    """Test ColumnExpr string operations alignment with pandas."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'text': ['Hello World', 'UPPER CASE', 'lower case', 'Mixed Case', '  spaces  '],
                'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_str_upper(self):
        """Test str.upper()."""
        ds = self.create_ds()
        ds_result = list(ds['name'].str.upper())
        pd_result = list(self.df['name'].str.upper())
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_str_lower(self):
        """Test str.lower()."""
        ds = self.create_ds()
        ds_result = list(ds['name'].str.lower())
        pd_result = list(self.df['name'].str.lower())
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_str_length(self):
        """Test str.length() / str.len()."""
        ds = self.create_ds()
        ds_result = list(ds['name'].str.length())
        pd_result = list(self.df['name'].str.len())
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_str_trim(self):
        """Test str.trim() / str.strip()."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.trim())
        pd_result = list(self.df['text'].str.strip())
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_str_left(self):
        """Test str.left(n)."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.left(5))
        pd_result = list(self.df['text'].str[:5])
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_str_right(self):
        """Test str.right(n)."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.right(3))
        pd_result = list(self.df['text'].str[-3:])
        self.assertEqual(sorted(ds_result), sorted(pd_result))

    def test_str_reverse(self):
        """Test str.reverse()."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.reverse())
        pd_result = list(self.df['text'].apply(lambda x: x[::-1]))
        self.assertEqual(sorted(ds_result), sorted(pd_result))


class TestColumnExprComparisonOperations(unittest.TestCase):
    """Test that comparison operations return Conditions (not boolean Series)."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_greater_than_returns_condition(self):
        """Test that > returns BinaryCondition."""
        ds = self.create_ds()
        result = ds['age'] > 25
        self.assertIsInstance(result, BinaryCondition)

    def test_greater_equal_returns_condition(self):
        """Test that >= returns BinaryCondition."""
        ds = self.create_ds()
        result = ds['age'] >= 28
        self.assertIsInstance(result, BinaryCondition)

    def test_less_than_returns_condition(self):
        """Test that < returns BinaryCondition."""
        ds = self.create_ds()
        result = ds['age'] < 30
        self.assertIsInstance(result, BinaryCondition)

    def test_less_equal_returns_condition(self):
        """Test that <= returns BinaryCondition."""
        ds = self.create_ds()
        result = ds['age'] <= 29
        self.assertIsInstance(result, BinaryCondition)

    def test_equal_returns_condition(self):
        """Test that == returns BinaryCondition."""
        ds = self.create_ds()
        result = ds['age'] == 28
        self.assertIsInstance(result, BinaryCondition)

    def test_not_equal_returns_condition(self):
        """Test that != returns BinaryCondition."""
        ds = self.create_ds()
        result = ds['age'] != 28
        self.assertIsInstance(result, BinaryCondition)


class TestColumnExprConditionMethods(unittest.TestCase):
    """Test condition methods on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [100, 200, 150, 300, 50],
                'category': ['A', 'B', 'A', 'C', 'B'],
                'text': ['Hello World', 'test', 'example', 'world', 'hello'],
                'nullable': [1.0, None, 3.0, None, 5.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_isin_returns_condition(self):
        """Test that isin() returns Condition with IN."""
        ds = self.create_ds()
        result = ds['category'].isin(['A', 'B'])
        self.assertIn('IN', str(result))

    def test_between_returns_condition(self):
        """Test that between() returns Condition with BETWEEN."""
        ds = self.create_ds()
        result = ds['value'].between(100, 200)
        self.assertIn('BETWEEN', str(result))

    def test_like_returns_condition(self):
        """Test that like() returns Condition with LIKE."""
        ds = self.create_ds()
        result = ds['text'].like('%World%')
        self.assertIn('LIKE', str(result))

    def test_isnull_returns_condition(self):
        """Test that isnull() returns Condition with IS NULL."""
        ds = self.create_ds()
        result = ds['nullable'].isnull()
        self.assertIn('IS NULL', str(result))

    def test_notnull_returns_condition(self):
        """Test that notnull() returns Condition with IS NOT NULL."""
        ds = self.create_ds()
        result = ds['nullable'].notnull()
        self.assertIn('IS NOT NULL', str(result))


class TestColumnExprTypeConversion(unittest.TestCase):
    """Test type conversion operations on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'int_col': [1, 2, 3, 4, 5],
                'float_col': [1.5, 2.5, 3.5, 4.5, 5.5],
                'str_col': ['10', '20', '30', '40', '50'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_cast_to_float(self):
        """Test cast to Float64."""
        ds = self.create_ds()
        result = list(ds['int_col'].cast('Float64'))
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_to_string(self):
        """Test to_string()."""
        ds = self.create_ds()
        result = list(ds['int_col'].to_string())
        self.assertTrue(all(isinstance(x, str) for x in result))
        self.assertEqual(sorted(result), ['1', '2', '3', '4', '5'])


class TestColumnExprMathFunctions(unittest.TestCase):
    """Test math functions on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [100.5, -200.3, 150.7, -300.2, 50.1],
                'positive': [4.0, 9.0, 16.0, 25.0, 36.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_abs(self):
        """Test abs()."""
        ds = self.create_ds()
        ds_result = list(ds['value'].abs())
        pd_result = list(self.df['value'].abs())
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))

    def test_round(self):
        """Test round()."""
        ds = self.create_ds()
        ds_result = list(ds['value'].round(0))
        pd_result = list(self.df['value'].round(0))
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))

    def test_sqrt(self):
        """Test sqrt()."""
        ds = self.create_ds()
        ds_result = list(ds['positive'].sqrt())
        pd_result = list(np.sqrt(self.df['positive']))
        self.assertTrue(np.allclose(sorted(ds_result), sorted(pd_result)))


class TestColumnExprDisplayBehavior(unittest.TestCase):
    """Test that ColumnExpr displays like pandas Series."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie'],
                'value': [100.5, 200.3, 150.7],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_repr_shows_values(self):
        """Test that repr() shows actual values."""
        ds = self.create_ds()
        result = repr(ds['value'])
        self.assertIn('100.5', result)
        self.assertIn('200.3', result)

    def test_str_shows_values(self):
        """Test that str() shows actual values."""
        ds = self.create_ds()
        result = str(ds['value'])
        self.assertIn('100.5', result)

    def test_len(self):
        """Test len() returns correct length."""
        ds = self.create_ds()
        self.assertEqual(len(ds['name']), 3)

    def test_iteration(self):
        """Test iteration over ColumnExpr."""
        ds = self.create_ds()
        values = list(ds['name'])
        self.assertEqual(values, ['Alice', 'Bob', 'Charlie'])

    def test_tolist(self):
        """Test tolist() method."""
        ds = self.create_ds()
        values = ds['name'].tolist()
        self.assertEqual(values, ['Alice', 'Bob', 'Charlie'])


class TestColumnExprFilterIntegration(unittest.TestCase):
    """Test that ColumnExpr works correctly with filter()."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_filter_with_column_expr_comparison(self):
        """Test filter with ds['col'] > value."""
        ds = self.create_ds()
        filtered = ds.filter(ds['age'] > 28).to_df()
        expected = self.df[self.df['age'] > 28]
        self.assertEqual(list(filtered['age']), list(expected['age']))

    def test_filter_with_attribute_comparison(self):
        """Test filter with ds.col > value."""
        ds = self.create_ds()
        filtered = ds.filter(ds.age >= 29).to_df()
        expected = self.df[self.df['age'] >= 29]
        self.assertEqual(list(filtered['age']), list(expected['age']))

    def test_filter_with_multiple_conditions(self):
        """Test filter with combined conditions."""
        ds = self.create_ds()
        filtered = ds.filter((ds.age > 25) & (ds.salary > 50000)).to_df()
        expected = self.df[(self.df['age'] > 25) & (self.df['salary'] > 50000)]
        self.assertEqual(list(filtered['name']), list(expected['name']))


class TestColumnExprAssignment(unittest.TestCase):
    """Test column assignment with ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [28, 31, 29],
                'salary': [50000.0, 75000.0, 60000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_assign_arithmetic_result(self):
        """Test ds['new'] = ds['col'] * 2."""
        ds = self.create_ds()
        ds['age_doubled'] = ds['age'] * 2
        result = ds.to_df()
        expected = list(self.df['age'] * 2)
        self.assertEqual(list(result['age_doubled']), expected)

    def test_assign_complex_expression(self):
        """Test ds['new'] = (col1 / 1000) + (col2 * 2)."""
        ds = self.create_ds()
        ds['complex'] = (ds['salary'] / 1000) + (ds['age'] * 2)
        result = ds.to_df()
        expected = list((self.df['salary'] / 1000) + (self.df['age'] * 2))
        self.assertTrue(np.allclose(list(result['complex']), expected))

    def test_assign_string_operation(self):
        """Test ds['new'] = ds['col'].str.upper()."""
        ds = self.create_ds()
        ds['name_upper'] = ds['name'].str.upper()
        result = ds.to_df()
        expected = list(self.df['name'].str.upper())
        self.assertEqual(sorted(list(result['name_upper'])), sorted(expected))

    def test_assign_type_cast(self):
        """Test ds['new'] = ds['col'].cast('Float64')."""
        ds = self.create_ds()
        ds['age_float'] = ds['age'].cast('Float64')
        result = ds.to_df()
        self.assertTrue(all(isinstance(x, float) for x in result['age_float']))


class TestColumnExprCombinedPipeline(unittest.TestCase):
    """Test combined pipelines with ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_filter_then_assign(self):
        """Test filter -> assign pipeline."""
        ds = self.create_ds()
        filtered = ds.filter(ds['salary'] > 50000)
        filtered['bonus'] = filtered['salary'] * 0.1
        result = filtered.to_df()

        # Verify filter
        expected_df = self.df[self.df['salary'] > 50000].copy()
        expected_df['bonus'] = expected_df['salary'] * 0.1

        self.assertTrue(np.allclose(list(result['bonus']), list(expected_df['bonus'])))

    def test_assign_then_filter(self):
        """Test assign -> filter pipeline."""
        ds = self.create_ds()
        ds['age_doubled'] = ds['age'] * 2
        filtered = ds.filter(ds['age_doubled'] > 50)
        result = filtered.to_df()

        # Verify
        temp_df = self.df.copy()
        temp_df['age_doubled'] = temp_df['age'] * 2
        expected = temp_df[temp_df['age_doubled'] > 50]

        self.assertEqual(list(result['name']), list(expected['name']))

    def test_access_column_from_filtered_result(self):
        """Test accessing column from filtered DataStore."""
        ds = self.create_ds()
        filtered = ds.filter(ds.salary > 50000)
        col_result = list(filtered['salary']._materialize())
        expected = list(self.df[self.df['salary'] > 50000]['salary'])
        self.assertTrue(np.allclose(col_result, expected))


if __name__ == '__main__':
    unittest.main()
