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
        result = ds['age']._materialize()
        expected = self.df['age']
        np.testing.assert_array_equal(result, expected)

    def test_attribute_access_returns_column_expr(self):
        """Test that ds.col returns ColumnExpr."""
        ds = self.create_ds()
        result = ds.age
        self.assertIsInstance(result, ColumnExpr)

    def test_attribute_access_materializes_correctly(self):
        """Test that attribute access materializes to correct values."""
        ds = self.create_ds()
        result = ds.age._materialize()
        expected = self.df['age']
        np.testing.assert_array_equal(result, expected)

    # ========== Arithmetic Operations ==========

    def test_addition(self):
        """Test column + scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] + 10
        pd_result = self.df['age'] + 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_subtraction(self):
        """Test column - scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] - 5
        pd_result = self.df['age'] - 5
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_multiplication(self):
        """Test column * scalar."""
        ds = self.create_ds()
        ds_result = ds['salary'] * 1.1
        pd_result = self.df['salary'] * 1.1
        np.testing.assert_allclose(ds_result, pd_result)

    def test_division(self):
        """Test column / scalar."""
        ds = self.create_ds()
        ds_result = ds['salary'] / 1000
        pd_result = self.df['salary'] / 1000
        np.testing.assert_allclose(ds_result, pd_result)

    def test_floor_division(self):
        """Test column // scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] // 10
        pd_result = self.df['age'] // 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_modulo(self):
        """Test column % scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] % 10
        pd_result = self.df['age'] % 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_power(self):
        """Test column ** scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] ** 2
        pd_result = self.df['age'] ** 2
        np.testing.assert_array_equal(ds_result, pd_result)

    # ========== Reverse Arithmetic Operations ==========

    def test_reverse_addition(self):
        """Test scalar + column."""
        ds = self.create_ds()
        ds_result = 100 + ds['age']
        pd_result = 100 + self.df['age']
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_reverse_subtraction(self):
        """Test scalar - column."""
        ds = self.create_ds()
        ds_result = 1000 - ds['age']
        pd_result = 1000 - self.df['age']
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_reverse_multiplication(self):
        """Test scalar * column."""
        ds = self.create_ds()
        ds_result = 2 * ds['salary']
        pd_result = 2 * self.df['salary']
        np.testing.assert_allclose(ds_result, pd_result)

    # ========== Unary Operations ==========

    def test_negation(self):
        """Test -column."""
        ds = self.create_ds()
        ds_result = -ds['age']
        pd_result = -self.df['age']
        np.testing.assert_array_equal(ds_result, pd_result)

    # ========== Column-Column Operations ==========

    def test_column_column_addition(self):
        """Test column + column."""
        ds = self.create_ds()
        ds_result = ds['age'] + ds['salary'] / 1000
        pd_result = self.df['age'] + self.df['salary'] / 1000
        np.testing.assert_allclose(ds_result, pd_result)

    # ========== Chained Operations ==========

    def test_chained_arithmetic(self):
        """Test (column - scalar) * scalar + scalar."""
        ds = self.create_ds()
        ds_result = (ds['age'] - 20) * 2 + 10
        pd_result = (self.df['age'] - 20) * 2 + 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_complex_chained_operations(self):
        """Test complex chained operations."""
        ds = self.create_ds()
        ds_result = ds['salary'] / 1000 - ds['age']
        pd_result = self.df['salary'] / 1000 - self.df['age']
        np.testing.assert_allclose(ds_result, pd_result)


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
        self.assertEqual(ds_result, pd_result)

    def test_str_lower(self):
        """Test str.lower()."""
        ds = self.create_ds()
        ds_result = list(ds['name'].str.lower())
        pd_result = list(self.df['name'].str.lower())
        self.assertEqual(ds_result, pd_result)

    def test_str_length(self):
        """Test str.length() / str.len()."""
        ds = self.create_ds()
        ds_result = list(ds['name'].str.length())
        pd_result = list(self.df['name'].str.len())
        self.assertEqual(ds_result, pd_result)

    def test_str_trim(self):
        """Test str.trim() / str.strip()."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.trim())
        pd_result = list(self.df['text'].str.strip())
        self.assertEqual(ds_result, pd_result)

    def test_str_left(self):
        """Test str.left(n)."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.left(5))
        pd_result = list(self.df['text'].str[:5])
        self.assertEqual(ds_result, pd_result)

    def test_str_right(self):
        """Test str.right(n)."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.right(3))
        pd_result = list(self.df['text'].str[-3:])
        self.assertEqual(ds_result, pd_result)

    def test_str_reverse(self):
        """Test str.reverse()."""
        ds = self.create_ds()
        ds_result = list(ds['text'].str.reverse())
        pd_result = list(self.df['text'].apply(lambda x: x[::-1]))
        self.assertEqual(ds_result, pd_result)


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


class TestColumnExprNullMethods(unittest.TestCase):
    """Test isnull/notnull methods on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [100, 200, 150, 300, 50],
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_notnull_returns_column_expr(self):
        """Test that notnull() returns ColumnExpr wrapping isNotNull()."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds['value'].notnull()
        # notnull() returns ColumnExpr wrapping isNotNull() function
        self.assertIsInstance(result, ColumnExpr)

    def test_isnull_returns_column_expr(self):
        """Test that isnull() returns ColumnExpr wrapping isNull()."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds['value'].isnull()
        # isnull() returns ColumnExpr wrapping isNull() function
        self.assertIsInstance(result, ColumnExpr)

    def test_notnull_astype_int_matches_pandas(self):
        """Test notnull().astype(int) matches pandas behavior (no NaN in data)."""
        ds = self.create_ds()
        result = ds['value'].notnull().astype(int)
        expected = self.df['value'].notnull().astype(int)
        # Compare with pandas result (this works when there's no NaN in data)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_isnull_astype_int_matches_pandas(self):
        """Test isnull().astype(int) matches pandas behavior (no NaN in data)."""
        ds = self.create_ds()
        result = ds['value'].isnull().astype(int)
        expected = self.df['value'].isnull().astype(int)
        # Compare with pandas result (this works when there's no NaN in data)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_notnull_condition_for_filtering(self):
        """Test notnull_condition() returns Condition for filtering."""
        ds = self.create_ds()
        result = ds['value'].notnull_condition()
        # Should be a Condition, not ColumnExpr
        from datastore.conditions import Condition

        self.assertIsInstance(result, Condition)

    def test_isnull_condition_for_filtering(self):
        """Test isnull_condition() returns Condition for filtering."""
        ds = self.create_ds()
        result = ds['value'].isnull_condition()
        from datastore.conditions import Condition

        self.assertIsInstance(result, Condition)


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

    def test_isnull_matches_pandas(self):
        """Test that isnull() result matches pandas when materialized."""
        ds = self.create_ds()
        # Materialize and compare with pandas
        ds_result = ds['nullable'].isnull().astype(int)
        pd_result = self.df['nullable'].isnull().astype(int)
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_notnull_matches_pandas(self):
        """Test that notnull() result matches pandas when materialized."""
        ds = self.create_ds()
        # Materialize and compare with pandas
        ds_result = ds['nullable'].notnull().astype(int)
        pd_result = self.df['nullable'].notnull().astype(int)
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_isnull_generates_sql(self):
        """Test that isnull() generates proper SQL with isNull function."""
        ds = self.create_ds()
        result = ds['nullable'].isnull()
        sql = result.to_sql()
        self.assertIn('isNull', sql)

    def test_notnull_generates_sql(self):
        """Test that notnull() generates proper SQL with isNotNull function."""
        ds = self.create_ds()
        result = ds['nullable'].notnull()
        sql = result.to_sql()
        self.assertIn('isNotNull', sql)


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
        self.assertEqual(result, ['1', '2', '3', '4', '5'])


class TestColumnExprAggregateFunctions(unittest.TestCase):
    """Test aggregate functions return correct values like pandas."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [28, 31, 29, 45, 22],
                'with_nan': [28.0, 31.0, np.nan, 45.0, 22.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_mean_returns_scalar(self):
        """Test mean() returns a LazyAggregate that behaves like a scalar."""
        from datastore.column_expr import LazyAggregate

        ds = self.create_ds()
        result = ds['value'].mean()
        expected = self.df['value'].mean()
        # mean() returns LazyAggregate which displays as scalar and can be converted
        self.assertIsInstance(result, LazyAggregate)
        # Should be able to compare with expected value (triggers execution)
        self.assertAlmostEqual(float(result), expected)

    def test_mean_with_nan_skipna(self):
        """Test mean() correctly skips NaN values like pandas."""
        ds = self.create_ds()
        result = ds['with_nan'].mean()
        expected = self.df['with_nan'].mean()  # pandas skips NaN by default
        self.assertAlmostEqual(result, expected)

    def test_sum_returns_scalar(self):
        """Test sum() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].sum()
        expected = self.df['value'].sum()
        self.assertEqual(result, expected)

    def test_min_returns_scalar(self):
        """Test min() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].min()
        expected = self.df['value'].min()
        self.assertEqual(result, expected)

    def test_max_returns_scalar(self):
        """Test max() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].max()
        expected = self.df['value'].max()
        self.assertEqual(result, expected)

    def test_std_returns_scalar(self):
        """Test std() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].std()
        expected = self.df['value'].std()
        self.assertAlmostEqual(result, expected, places=4)

    def test_count_returns_int(self):
        """Test count() returns an integer like pandas."""
        ds = self.create_ds()
        result = ds['value'].count()
        expected = self.df['value'].count()
        self.assertEqual(result, expected)

    def test_median_returns_scalar(self):
        """Test median() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].median()
        expected = self.df['value'].median()
        self.assertEqual(result, expected)

    def test_round_mean(self):
        """Test round(mean(), 2) works with scalar result."""
        ds = self.create_ds()
        result = round(ds['value'].mean(), 2)
        expected = round(self.df['value'].mean(), 2)
        self.assertEqual(result, expected)

    def test_mean_sql_returns_column_expr(self):
        """Test mean_sql() returns ColumnExpr for SQL building."""
        ds = self.create_ds()
        result = ds['value'].mean_sql()
        self.assertIsInstance(result, ColumnExpr)
        self.assertIn('avg', result.to_sql().lower())


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
        ds_result = ds['value'].abs()
        pd_result = self.df['value'].abs()
        np.testing.assert_allclose(ds_result, pd_result)

    def test_round(self):
        """Test round()."""
        ds = self.create_ds()
        ds_result = ds['value'].round(0)
        pd_result = self.df['value'].round(0)
        np.testing.assert_allclose(ds_result, pd_result)

    def test_sqrt(self):
        """Test sqrt()."""
        ds = self.create_ds()
        ds_result = ds['positive'].sqrt()
        pd_result = np.sqrt(self.df['positive'])
        np.testing.assert_allclose(ds_result, pd_result)

    def test_builtin_round(self):
        """Test Python's built-in round() function on ColumnExpr."""
        ds = self.create_ds()
        ds_result = round(ds['value'], 1)
        # Should return ColumnExpr
        self.assertIsInstance(ds_result, ColumnExpr)
        # Materialize and compare
        pd_result = self.df['value'].round(1)
        np.testing.assert_allclose(list(ds_result), list(pd_result))

    def test_builtin_round_no_decimals(self):
        """Test round() without decimal places."""
        ds = self.create_ds()
        ds_result = round(ds['value'])
        self.assertIsInstance(ds_result, ColumnExpr)
        pd_result = self.df['value'].round(0)
        np.testing.assert_allclose(list(ds_result), list(pd_result))

    def test_builtin_round_on_aggregate(self):
        """Test round() on aggregate result like mean()."""
        from datastore.column_expr import LazyAggregate

        ds = self.create_ds()
        # round(ds['value'].mean(), 2) should work
        mean_expr = ds['value'].mean()
        # mean() returns LazyAggregate which supports both display and SQL building
        self.assertIsInstance(mean_expr, (ColumnExpr, LazyAggregate))
        rounded = round(mean_expr, 2)
        self.assertIsInstance(rounded, ColumnExpr)
        # The SQL should contain round(avg(...))
        sql = rounded.to_sql()
        self.assertIn('round', sql.lower())
        self.assertIn('avg', sql.lower())

    def test_fillna_with_aggregate_expression(self):
        """Test fillna() with aggregate expression like mean()."""
        ds = self.create_ds()
        # fillna with mean should work
        result = ds['value'].fillna(ds['value'].mean())
        # Now returns pandas Series (always uses pandas fillna)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)


class TestColumnExprFillna(unittest.TestCase):
    """Test fillna() method for proper NaN handling."""

    def test_fillna_string_column(self):
        """Test fillna() on string column with NaN values."""
        df = pd.DataFrame(
            {
                'Cabin': ['A1', np.nan, 'B2', np.nan, 'C3'],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Cabin'].fillna('Unknown')
        expected = df['Cabin'].fillna('Unknown')

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(list(result), list(expected))

    def test_fillna_numeric_column(self):
        """Test fillna() on numeric column with NaN values."""
        df = pd.DataFrame(
            {
                'Age': [28.0, np.nan, 29.0, np.nan, 22.0],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Age'].fillna(0)
        expected = df['Age'].fillna(0)

        self.assertEqual(list(result), list(expected))

    def test_fillna_with_mean(self):
        """Test fillna() with mean value."""
        df = pd.DataFrame(
            {
                'Age': [28.0, np.nan, 29.0, np.nan, 22.0],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Age'].fillna(ds['Age'].mean())
        expected = df['Age'].fillna(df['Age'].mean())

        np.testing.assert_array_almost_equal(list(result), list(expected))

    def test_fillna_with_mode(self):
        """Test fillna() with mode value for string column."""
        df = pd.DataFrame(
            {
                'Embarked': ['S', 'C', np.nan, 'S', np.nan, 'S'],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Embarked'].fillna(ds['Embarked'].mode()[0])
        expected = df['Embarked'].fillna(df['Embarked'].mode()[0])

        self.assertEqual(list(result), list(expected))


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

    def test_getitem_index(self):
        """Test subscripting with integer index."""
        ds = self.create_ds()
        # ds['col'][0] should return first value
        first_value = ds['value'][0]
        self.assertEqual(first_value, 100.5)

    def test_getitem_slice(self):
        """Test subscripting with slice."""
        ds = self.create_ds()
        values = ds['value'][:2]
        self.assertEqual(len(values), 2)

    def test_mode_returns_series(self):
        """Test mode() returns a pandas Series."""
        ds = self.create_ds()
        result = ds['name'].mode()
        self.assertIsInstance(result, pd.Series)

    def test_mode_subscript(self):
        """Test mode()[0] pattern - regression test for TypeError."""
        ds = self.create_ds()
        # This used to fail with: TypeError: 'ColumnExpr' object is not subscriptable
        first_mode = ds['name'].mode()[0]
        # Should return a scalar value
        self.assertIsInstance(first_mode, str)


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
        expected = (self.df['salary'] / 1000) + (self.df['age'] * 2)
        np.testing.assert_allclose(result['complex'], expected)

    def test_assign_string_operation(self):
        """Test ds['new'] = ds['col'].str.upper()."""
        ds = self.create_ds()
        ds['name_upper'] = ds['name'].str.upper()
        result = ds.to_df()
        expected = list(self.df['name'].str.upper())
        self.assertEqual(list(result['name_upper']), expected)

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

        np.testing.assert_allclose(result['bonus'], expected_df['bonus'])

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
        col_result = filtered['salary']._materialize()
        expected = self.df[self.df['salary'] > 50000]['salary']
        np.testing.assert_allclose(col_result, expected)


class TestLazySlice(unittest.TestCase):
    """Test LazySlice class for lazy head()/tail() operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
                'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_column_head_returns_lazy_slice(self):
        """Test that ColumnExpr.head() returns LazySlice."""
        from datastore.lazy_result import LazySlice

        ds = self.create_ds()
        result = ds['value'].head(5)

        # Should return LazySlice, not pd.Series
        self.assertIsInstance(result, LazySlice)

    def test_column_tail_returns_lazy_slice(self):
        """Test that ColumnExpr.tail() returns LazySlice."""
        from datastore.lazy_result import LazySlice

        ds = self.create_ds()
        result = ds['value'].tail(5)

        # Should return LazySlice, not pd.Series
        self.assertIsInstance(result, LazySlice)

    def test_lazy_slice_materializes_on_repr(self):
        """Test that LazySlice materializes when displayed."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # repr() should trigger materialization
        repr_str = repr(result)

        # Should show actual values
        self.assertIn('10', repr_str)

    def test_lazy_slice_to_pandas(self):
        """Test explicit materialization with to_pandas()."""
        ds = self.create_ds()
        result = ds['value'].head(3).to_pandas()

        # Should return pd.Series
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)

    def test_lazy_slice_chainable(self):
        """Test that LazySlice.head() and tail() return new LazySlice."""
        from datastore.lazy_result import LazySlice

        ds = self.create_ds()
        result = ds['value'].head(5).tail(2)

        # Should still be LazySlice
        self.assertIsInstance(result, LazySlice)

        # Should work correctly
        final = result.to_pandas()
        self.assertEqual(len(final), 2)

    def test_lazy_slice_iteration(self):
        """Test iteration triggers materialization."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # Iteration should work
        values = list(result)
        self.assertEqual(len(values), 3)

    def test_lazy_slice_indexing(self):
        """Test indexing triggers materialization."""
        ds = self.create_ds()
        result = ds['value'].head(5)

        # Indexing should work
        first_value = result[0]
        self.assertEqual(first_value, 10)

    def test_lazy_slice_len(self):
        """Test len() triggers materialization."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        self.assertEqual(len(result), 3)

    def test_lazy_slice_properties(self):
        """Test properties like values, index, dtype."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # Properties should work
        self.assertEqual(len(result.values), 3)
        self.assertEqual(len(result.index), 3)
        self.assertIsNotNone(result.dtype)

    def test_lazy_slice_arithmetic(self):
        """Test arithmetic on LazySlice."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # Arithmetic should work
        doubled = result * 2
        self.assertEqual(list(doubled), [20, 40, 60])

    def test_lazy_aggregate_head_returns_lazy_slice(self):
        """Test that LazyAggregate.head() returns LazySlice."""
        from datastore.lazy_result import LazySlice
        from datastore.column_expr import LazyAggregate

        ds = self.create_ds()
        agg_result = ds.groupby('category')['value'].mean()

        # Should be LazyAggregate
        self.assertIsInstance(agg_result, LazyAggregate)

        # head() should return LazySlice
        head_result = agg_result.head(3)
        self.assertIsInstance(head_result, LazySlice)

        # Materialized result should have correct length
        self.assertEqual(len(head_result), 3)

    def test_lazy_aggregate_tail_returns_lazy_slice(self):
        """Test that LazyAggregate.tail() returns LazySlice."""
        from datastore.lazy_result import LazySlice

        ds = self.create_ds()
        result = ds.groupby('category')['value'].sum().tail(2)

        self.assertIsInstance(result, LazySlice)
        self.assertEqual(len(result), 2)

    def test_lazy_slice_on_scalar_aggregate(self):
        """Test LazySlice on scalar aggregate (no groupby)."""
        ds = self.create_ds()
        result = ds['value'].mean().head(5)

        # Scalar aggregates don't have head(), should return scalar
        # The scalar value should be preserved
        expected = self.df['value'].mean()
        self.assertAlmostEqual(float(result), expected)

    def test_lazy_slice_aggregations(self):
        """Test calling aggregation methods on LazySlice."""
        ds = self.create_ds()
        slice_result = ds['value'].head(5)

        # Aggregation methods should work
        self.assertEqual(slice_result.sum(), 10 + 20 + 30 + 40 + 50)
        self.assertEqual(slice_result.mean(), 30.0)
        self.assertEqual(slice_result.min(), 10)
        self.assertEqual(slice_result.max(), 50)


if __name__ == '__main__':
    unittest.main()
