"""
Pandas Alignment Tests
======================

This file tests pandas API alignment and marks known chDB issues as xfail.

Known chDB Issues:
- Issue #447: NaN/NULL handling in Python() table function
- Issue #448: datetime64 handling corruption

Issues that need to be fixed in datastore:
- Slice indexing (df[:3]) returns all data
- GroupBy.agg({'col': 'func'}) loses grouping
- DateTime accessor returns expression string instead of values
- String accessor Series name becomes __result__
"""

import pytest
import pandas as pd
import numpy as np
import datastore as ds


@pytest.fixture
def test_data():
    """Create test data for compatibility testing."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None, 'Grace'],
        'age': [25, 30, 35, None, 28, 32, 45, 27],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Sydney', 'Toronto', None],
        'salary': [50000, 60000, None, 75000, 55000, 80000, 70000, 52000],
        'department': ['HR', 'IT', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance'],
        'hire_date': [
            '2020-01-15',
            '2019-06-20',
            '2018-03-10',
            '2021-09-05',
            '2020-11-12',
            '2019-02-28',
            '2017-08-15',
            '2021-04-30',
        ],
        'performance_score': [8.5, 7.2, 9.1, 6.8, 8.0, 9.5, 7.5, 8.3],
    }
    return data


@pytest.fixture
def pandas_df(test_data):
    """Create pandas DataFrame with proper types."""
    df = pd.DataFrame(test_data)
    df['hire_date'] = pd.to_datetime(df['hire_date'])
    return df


@pytest.fixture
def datastore_df(test_data):
    """Create datastore DataFrame."""
    return ds.DataFrame(test_data)


class TestDataFrameCreation:
    """Test DataFrame creation operations."""

    def test_create_from_dict(self):
        """Create DataFrame from dict - PASS"""
        result = ds.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd.testing.assert_frame_equal(result.to_df(), expected)

    def test_create_with_index(self):
        """Create DataFrame with index - PASS"""
        result = ds.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        expected = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        pd.testing.assert_frame_equal(result.to_df(), expected)


class TestDataSelection:
    """Test data selection operations."""

    def test_select_single_column(self, test_data):
        """Select single column - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name']
        expected = df_pd['name']
        # Natural trigger via np.testing
        np.testing.assert_array_equal(result, expected)

    def test_select_multiple_columns(self, test_data):
        """Select multiple columns - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)
        pd.testing.assert_frame_equal(df_ds[['name', 'age']].to_df(), df_pd[['name', 'age']])

    def test_select_rows_by_slice(self, test_data):
        """Select rows by slice - needs fix in datastore."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds[:3].to_df()
        expected = df_pd[:3]

        # This should return first 3 rows, not all rows
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_select_with_boolean_indexing(self, test_data):
        """Select with boolean indexing - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds[df_ds['age'] > 30].to_df()
        expected = df_pd[df_pd['age'] > 30]

        assert len(result) == len(expected)
        # Compare ignoring index and minor formatting differences
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)

    def test_select_with_loc(self, test_data):
        """Select with loc - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.loc[0:2, ['name', 'age']]
        expected = df_pd.loc[0:2, ['name', 'age']]
        pd.testing.assert_frame_equal(result, expected)

    def test_select_with_iloc(self, test_data):
        """Select with iloc - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.iloc[0:3, 0:2]
        expected = df_pd.iloc[0:3, 0:2]
        pd.testing.assert_frame_equal(result, expected)


class TestStatistics:
    """Test statistical operations."""

    def test_compute_mean(self, test_data):
        """Compute mean - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['age'].mean())
        expected = df_pd['age'].mean()
        assert abs(result - expected) < 0.001

    def test_compute_sum(self, test_data):
        """Compute sum - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['salary'].sum())
        expected = df_pd['salary'].sum()
        assert result == expected

    def test_compute_median(self, test_data):
        """Compute median - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['age'].median())
        expected = df_pd['age'].median()
        assert result == expected

    def test_compute_std(self, test_data):
        """Compute std - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['age'].std())
        expected = df_pd['age'].std()
        assert abs(result - expected) < 0.001

    def test_value_counts(self, test_data):
        """Value counts - PASS

        Note: value_counts() returns LazySeries for lazy execution.
        Accessing .values and .index triggers execution naturally.
        """
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['department'].value_counts()
        expected = df_pd['department'].value_counts()

        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(result, expected)
        assert result.index.equals(expected.index)


class TestAggregation:
    """Test aggregation operations.

    Uses np.testing with __array__ protocol for natural execution trigger.
    """

    def test_groupby_single_aggregation(self, test_data):
        """GroupBy with single aggregation."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department')['salary'].mean()
        expected = df_pd.groupby('department')['salary'].mean()

        # Natural trigger via __array__ protocol
        np.testing.assert_array_almost_equal(result, expected)

    def test_groupby_multiple_aggregations(self, test_data):
        """GroupBy with multiple aggregations - returns lazy DataStore."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department').agg({'salary': 'mean', 'age': 'max'})
        expected = df_pd.groupby('department').agg({'salary': 'mean', 'age': 'max'})

        # Natural trigger via == comparison (__eq__)
        assert result == expected

    def test_groupby_sum(self, test_data):
        """GroupBy with sum."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department')['salary'].sum()
        expected = df_pd.groupby('department')['salary'].sum()

        # Natural trigger via __array__ protocol
        np.testing.assert_array_almost_equal(result, expected)

    def test_groupby_size(self, test_data):
        """GroupBy with size - returns LazyGroupBySize (pd.Series compatible)."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department').size()
        expected = df_pd.groupby('department').size()

        # Natural trigger via __array__ protocol
        np.testing.assert_array_equal(result, expected)


class TestStringOperations:
    """Test string operations."""

    def test_string_contains(self, test_data):
        """String contains - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name'].str.contains('a', na=False)
        expected = df_pd['name'].str.contains('a', na=False)

        # Natural trigger via np.testing
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.xfail(reason="chDB Issue #447: NULL becomes empty string instead of None", strict=False)
    def test_string_upper_null_handling(self, test_data):
        """String upper NULL handling - chDB Issue #447"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name'].str.upper()
        expected = df_pd['name'].str.upper()

        # Check that NULL/None values are preserved as None, not empty string
        result_series = result.to_series() if hasattr(result, 'to_series') else result
        assert result_series.iloc[6] is None or pd.isna(
            result_series.iloc[6]
        ), f"Expected None/NaN at index 6, got '{result_series.iloc[6]}'"

    @pytest.mark.xfail(reason="chDB Issue #447: str.len() returns 0 for NULL instead of NaN", strict=False)
    def test_string_length_null_handling(self, test_data):
        """String length NULL handling - chDB Issue #447"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name'].str.len()
        expected = df_pd['name'].str.len()

        result_series = result.to_series() if hasattr(result, 'to_series') else result
        # NULL should return NaN, not 0
        assert pd.isna(result_series.iloc[6]), f"Expected NaN at index 6, got {result_series.iloc[6]}"

    def test_string_accessor_preserves_series_name(self, test_data):
        """String accessor should preserve Series name, not use __result__."""
        df_ds = ds.DataFrame(test_data)

        result = df_ds['name'].str.upper()

        # Natural trigger via .name property
        assert result.name == 'name', f"Expected Series name 'name', got '{result.name}'"


class TestMerging:
    """Test merge/join operations."""

    def test_concat_dataframes(self):
        """Concat DataFrames - PASS"""
        df1_ds = ds.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2_ds = ds.DataFrame({'a': [5, 6], 'b': [7, 8]})
        result = ds.concat([df1_ds, df2_ds])

        df1_pd = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2_pd = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        expected = pd.concat([df1_pd, df2_pd])

        pd.testing.assert_frame_equal(result.to_df() if hasattr(result, 'to_df') else result, expected)

    def test_merge_dataframes(self):
        """Merge DataFrames - PASS"""
        df1_ds = ds.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2_ds = ds.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        result = ds.merge(df1_ds, df2_ds, on='key', how='inner')

        df1_pd = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2_pd = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        expected = pd.merge(df1_pd, df2_pd, on='key', how='inner')

        pd.testing.assert_frame_equal(result.to_df() if hasattr(result, 'to_df') else result, expected)


class TestDateTimeOperations:
    """Test datetime operations."""

    @pytest.mark.xfail(reason="chDB Issue #448: datetime64 handling corruption", strict=False)
    def test_extract_year_from_date(self, pandas_df, datastore_df):
        """Extract year from date - chDB Issue #448"""
        # First convert to datetime
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.year
        expected = pandas_df['hire_date'].dt.year

        result_series = result.to_series() if hasattr(result, 'to_series') else result

        # Should return actual year values, not expression string
        assert not isinstance(result_series.iloc[0], str), f"Expected numeric year, got string: {result_series.iloc[0]}"
        pd.testing.assert_series_equal(result_series, expected, check_names=False)

    @pytest.mark.xfail(reason="chDB Issue #448: datetime64 handling corruption", strict=False)
    def test_extract_month_from_date(self, pandas_df, datastore_df):
        """Extract month from date - chDB Issue #448"""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.month
        expected = pandas_df['hire_date'].dt.month

        result_series = result.to_series() if hasattr(result, 'to_series') else result

        assert not isinstance(
            result_series.iloc[0], str
        ), f"Expected numeric month, got string: {result_series.iloc[0]}"

    @pytest.mark.xfail(reason="chDB Issue #448: datetime64 handling corruption", strict=False)
    def test_date_strftime(self, pandas_df, datastore_df):
        """Date formatting with strftime - chDB Issue #448"""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.strftime('%Y-%m')
        expected = pandas_df['hire_date'].dt.strftime('%Y-%m')

        result_series = result.to_series() if hasattr(result, 'to_series') else result

        # Should return formatted date strings, not expression string
        assert (
            result_series.iloc[0] == expected.iloc[0]
        ), f"Expected '{expected.iloc[0]}', got '{result_series.iloc[0]}'"


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_dropna(self, test_data):
        """Drop NA values - check row count matches."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.dropna().to_df()
        expected = df_pd.dropna()

        assert len(result) == len(expected), f"Expected {len(expected)} rows after dropna, got {len(result)}"

    def test_fillna(self, test_data):
        """Fill NA values - check no NaN remains in numeric columns."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        # Fill only numeric columns with 0
        result = df_ds.fillna(0).to_df()
        expected = df_pd.fillna(0)

        # Check that numeric columns have no NaN
        for col in ['age', 'salary']:
            assert not result[col].isna().any(), f"Column {col} still has NaN after fillna"


class TestDataTransformation:
    """Test data transformation operations."""

    def test_map_values(self, test_data):
        """Map values - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        mapping = {'HR': 1, 'IT': 2, 'Finance': 3}
        result = df_ds['department'].map(mapping)
        expected = df_pd['department'].map(mapping)

        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(result, expected)

    def test_astype(self, test_data):
        """Convert type with astype - PASS"""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['age'].astype(str)
        expected = df_pd['age'].astype(str)

        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(result, expected)


class TestSorting:
    """Test sorting operations."""

    def test_sort_by_single_column(self, test_data):
        """Sort by single column - check order is correct."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.sort_values('age').to_df()
        expected = df_pd.sort_values('age')

        # Compare the order of age values (ignoring NaN position differences)
        result_ages = result['age'].dropna().tolist()
        expected_ages = expected['age'].dropna().tolist()
        assert result_ages == expected_ages, f"Age order mismatch: {result_ages} vs {expected_ages}"

    def test_sort_descending(self, test_data):
        """Sort descending - check order is correct."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.sort_values('salary', ascending=False).to_df()
        expected = df_pd.sort_values('salary', ascending=False)

        # Compare the order of salary values (ignoring NaN position differences)
        result_salaries = result['salary'].dropna().tolist()
        expected_salaries = expected['salary'].dropna().tolist()
        assert result_salaries == expected_salaries, f"Salary order mismatch: {result_salaries} vs {expected_salaries}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
