"""
Pandas Compatibility Tests
==========================

Test suite to verify datastore's compatibility with pandas API.
Each test compares the result of an operation in both pandas and datastore.

Uses the `==` operator for comparison, which is implemented in DataStore,
ColumnExpr, and LazyAggregate to handle smart comparison with pandas objects.
"""

import pytest
import pandas as pd
import numpy as np
import datastore as ds


# =============================================================================
# Test Data Fixture
# =============================================================================


@pytest.fixture
def test_data():
    """Comprehensive test dataset."""
    return {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None, 'Grace'],
        'age': [25, 30, 35, None, 28, 32, 45, 27],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Sydney', 'Toronto', None],
        'salary': [50000, 60000, None, 75000, 55000, 80000, 70000, 52000],
        'department': ['HR', 'IT', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance'],
        'hire_date': [
            '2020-01-15', '2019-06-20', '2018-03-10', '2021-09-05',
            '2020-11-12', '2019-02-28', '2017-08-15', '2021-04-30',
        ],
        'performance_score': [8.5, 7.2, 9.1, 6.8, 8.0, 9.5, 7.5, 8.3],
    }


@pytest.fixture
def pd_df(test_data):
    """Pandas DataFrame with test data."""
    return pd.DataFrame(test_data)


@pytest.fixture
def ds_df(test_data):
    """DataStore DataFrame with test data."""
    return ds.DataFrame(test_data)


# =============================================================================
# DataFrame Creation Tests
# =============================================================================


class TestDataFrameCreation:
    """Test DataFrame creation operations."""

    def test_create_from_dict(self):
        """Create DataFrame from dict."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd_result = pd.DataFrame(data)
        ds_result = ds.DataFrame(data)
        assert ds_result == pd_result

    def test_create_with_index(self):
        """Create DataFrame with index."""
        data = {'a': [1, 2, 3]}
        index = ['x', 'y', 'z']
        pd_result = pd.DataFrame(data, index=index)
        ds_result = ds.DataFrame(data, index=index)
        assert ds_result == pd_result


# =============================================================================
# Data Selection Tests
# =============================================================================


class TestDataSelection:
    """Test data selection operations."""

    def test_select_single_column(self, pd_df, ds_df):
        """Select single column."""
        pd_result = pd_df['name']
        ds_result = ds_df['name']
        assert ds_result == pd_result

    def test_select_multiple_columns(self, pd_df, ds_df):
        """Select multiple columns."""
        pd_result = pd_df[['name', 'age']]
        ds_result = ds_df[['name', 'age']]
        assert ds_result == pd_result

    def test_select_rows_by_slice(self, pd_df, ds_df):
        """Select rows by slice."""
        pd_result = pd_df[:3]
        ds_result = ds_df[:3]
        assert ds_result == pd_result

    def test_select_with_boolean_indexing(self, pd_df, ds_df):
        """Select with boolean indexing."""
        pd_result = pd_df[pd_df['age'] > 30]
        ds_result = ds_df[ds_df['age'] > 30]
        assert ds_result == pd_result

    def test_select_with_loc(self, pd_df, ds_df):
        """Select with loc."""
        pd_result = pd_df.loc[0:2, ['name', 'age']]
        ds_result = ds_df.loc[0:2, ['name', 'age']]
        # loc returns pandas DataFrame directly
        pd.testing.assert_frame_equal(ds_result, pd_result)

    def test_select_with_iloc(self, pd_df, ds_df):
        """Select with iloc."""
        pd_result = pd_df.iloc[0:3, 0:2]
        ds_result = ds_df.iloc[0:3, 0:2]
        # iloc returns pandas DataFrame directly
        pd.testing.assert_frame_equal(ds_result, pd_result)


# =============================================================================
# Data Cleaning Tests
# =============================================================================


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_dropna(self, pd_df, ds_df):
        """Drop NA values."""
        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()
        assert ds_result == pd_result

    def test_fillna(self, pd_df, ds_df):
        """Fill NA values."""
        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)
        assert ds_result == pd_result

    def test_drop_duplicates(self, pd_df, ds_df):
        """Drop duplicates."""
        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()
        assert ds_result == pd_result

    def test_replace_values(self, pd_df, ds_df):
        """Replace values."""
        pd_result = pd_df.replace('HR', 'Human Resources')
        ds_result = ds_df.replace('HR', 'Human Resources')
        assert ds_result == pd_result

    def test_drop_column(self, pd_df, ds_df):
        """Drop column."""
        pd_result = pd_df.drop('salary', axis=1)
        ds_result = ds_df.drop('salary', axis=1)
        assert ds_result == pd_result


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistical operations."""

    def test_mean(self, pd_df, ds_df):
        """Compute mean."""
        pd_result = pd_df['age'].mean()
        ds_result = ds_df['age'].mean()
        assert ds_result == pd_result

    def test_sum(self, pd_df, ds_df):
        """Compute sum."""
        pd_result = pd_df['salary'].sum()
        ds_result = ds_df['salary'].sum()
        assert ds_result == pd_result

    def test_median(self, pd_df, ds_df):
        """Compute median."""
        pd_result = pd_df['age'].median()
        ds_result = ds_df['age'].median()
        assert ds_result == pd_result

    def test_std(self, pd_df, ds_df):
        """Compute std."""
        pd_result = pd_df['age'].std()
        ds_result = ds_df['age'].std()
        assert ds_result == pd_result

    def test_describe(self, pd_df, ds_df):
        """Describe DataFrame."""
        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        assert ds_result == pd_result

    def test_value_counts(self, pd_df, ds_df):
        """Value counts."""
        pd_result = pd_df['department'].value_counts()
        ds_result = ds_df['department'].value_counts()
        # value_counts returns pandas Series
        pd.testing.assert_series_equal(ds_result.sort_index(), pd_result.sort_index())

    def test_correlation(self, pd_df, ds_df):
        """Correlation."""
        pd_result = pd_df[['age', 'salary']].corr()
        ds_result = ds_df[['age', 'salary']].corr()
        # corr returns pandas DataFrame
        pd.testing.assert_frame_equal(ds_result, pd_result)


# =============================================================================
# Data Transformation Tests
# =============================================================================


class TestDataTransformation:
    """Test data transformation operations."""

    def test_apply_function(self, pd_df, ds_df):
        """Apply function to column."""
        pd_result = pd_df['age'].apply(lambda x: x * 2 if pd.notna(x) else x)
        ds_result = ds_df['age'].apply(lambda x: x * 2 if pd.notna(x) else x)
        # apply returns pandas Series
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_map_values(self, pd_df, ds_df):
        """Map values."""
        mapping = {'HR': 1, 'IT': 2, 'Finance': 3}
        pd_result = pd_df['department'].map(mapping)
        ds_result = ds_df['department'].map(mapping)
        # map returns pandas Series
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_rename_columns(self, pd_df, ds_df):
        """Rename columns."""
        pd_result = pd_df.rename(columns={'name': 'employee_name'})
        ds_result = ds_df.rename(columns={'name': 'employee_name'})
        assert ds_result == pd_result

    def test_astype(self, pd_df, ds_df):
        """Convert type (astype)."""
        pd_result = pd_df['age'].astype(str)
        ds_result = ds_df['age'].astype(str)
        # astype returns pandas Series
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_add_new_column(self, pd_df, ds_df):
        """Add new column."""
        pd_df = pd_df.copy()
        pd_df['bonus'] = pd_df['salary'] * 0.1

        ds_df = ds_df.copy() if hasattr(ds_df, 'copy') else ds.DataFrame(ds_df.to_df())
        ds_df['bonus'] = ds_df['salary'] * 0.1

        assert ds_df == pd_df


# =============================================================================
# Sorting Tests
# =============================================================================


class TestSorting:
    """Test sorting operations."""

    def test_sort_by_single_column(self, pd_df, ds_df):
        """Sort by single column."""
        pd_result = pd_df.sort_values('age')
        ds_result = ds_df.sort_values('age')
        assert ds_result == pd_result

    def test_sort_by_multiple_columns(self, pd_df, ds_df):
        """Sort by multiple columns."""
        pd_result = pd_df.sort_values(['department', 'age'])
        ds_result = ds_df.sort_values(['department', 'age'])
        assert ds_result == pd_result

    def test_sort_descending(self, pd_df, ds_df):
        """Sort descending."""
        pd_result = pd_df.sort_values('salary', ascending=False)
        ds_result = ds_df.sort_values('salary', ascending=False)
        assert ds_result == pd_result

    def test_sort_index(self, pd_df, ds_df):
        """Sort index."""
        pd_result = pd_df.sort_index()
        ds_result = ds_df.sort_index()
        assert ds_result == pd_result


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregation:
    """Test aggregation operations."""

    def test_groupby_single_aggregation(self, pd_df, ds_df):
        """GroupBy with single aggregation."""
        pd_result = pd_df.groupby('department')['salary'].mean()
        ds_result = ds_df.groupby('department')['salary'].mean()
        assert ds_result == pd_result

    def test_groupby_multiple_aggregations(self, pd_df, ds_df):
        """GroupBy with multiple aggregations."""
        pd_result = pd_df.groupby('department').agg({'salary': 'mean', 'age': 'max'})
        ds_result = ds_df.groupby('department').agg({'salary': 'mean', 'age': 'max'})
        # agg returns pandas DataFrame
        pd.testing.assert_frame_equal(ds_result.sort_index(), pd_result.sort_index())

    def test_groupby_sum(self, pd_df, ds_df):
        """GroupBy with sum."""
        pd_result = pd_df.groupby('department')['salary'].sum()
        ds_result = ds_df.groupby('department')['salary'].sum()
        assert ds_result == pd_result

    def test_groupby_count(self, pd_df, ds_df):
        """GroupBy with count (size)."""
        pd_result = pd_df.groupby('department').size()
        ds_result = ds_df.groupby('department').size()
        # size returns pandas Series
        pd.testing.assert_series_equal(ds_result.sort_index(), pd_result.sort_index())


# =============================================================================
# String Operations Tests
# =============================================================================


class TestStringOperations:
    """Test string operations."""

    @pytest.mark.xfail(reason="chDB NULL handling issue #447 - NULL becomes empty string")
    def test_str_upper(self, pd_df, ds_df):
        """String upper."""
        pd_result = pd_df['name'].str.upper()
        ds_result = ds_df['name'].str.upper()
        assert ds_result == pd_result

    @pytest.mark.xfail(reason="chDB NULL handling issue #447 - NULL becomes empty string")
    def test_str_lower(self, pd_df, ds_df):
        """String lower."""
        pd_result = pd_df['city'].str.lower()
        ds_result = ds_df['city'].str.lower()
        assert ds_result == pd_result

    def test_str_contains(self, pd_df, ds_df):
        """String contains."""
        pd_result = pd_df['name'].str.contains('a', na=False)
        ds_result = ds_df['name'].str.contains('a', na=False)
        assert ds_result == pd_result

    @pytest.mark.xfail(reason="chDB NULL handling issue #447 - NULL becomes empty string")
    def test_str_len(self, pd_df, ds_df):
        """String length."""
        pd_result = pd_df['name'].str.len()
        ds_result = ds_df['name'].str.len()
        assert ds_result == pd_result

    @pytest.mark.xfail(reason="chDB NULL handling issue #447 - NULL becomes empty string")
    def test_str_replace(self, pd_df, ds_df):
        """String replace."""
        pd_result = pd_df['city'].str.replace('York', 'Amsterdam')
        ds_result = ds_df['city'].str.replace('York', 'Amsterdam')
        assert ds_result == pd_result


# =============================================================================
# Merging Tests
# =============================================================================


class TestMerging:
    """Test merge/join operations."""

    def test_concat(self):
        """Concat DataFrames."""
        df1_pd = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2_pd = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        pd_result = pd.concat([df1_pd, df2_pd])

        df1_ds = ds.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2_ds = ds.DataFrame({'a': [5, 6], 'b': [7, 8]})
        ds_result = ds.concat([df1_ds, df2_ds])

        assert ds_result == pd_result

    def test_merge(self):
        """Merge DataFrames."""
        df1_pd = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2_pd = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        pd_result = pd.merge(df1_pd, df2_pd, on='key', how='inner')

        df1_ds = ds.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2_ds = ds.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        ds_result = ds.merge(df1_ds, df2_ds, on='key', how='inner')

        assert ds_result == pd_result


# =============================================================================
# DateTime Operations Tests
# =============================================================================


class TestDateTimeOperations:
    """Test datetime operations."""

    def test_to_datetime(self, pd_df, ds_df):
        """Convert to datetime."""
        pd_result = pd.to_datetime(pd_df['hire_date'])
        ds_result = ds.to_datetime(ds_df['hire_date'])
        # to_datetime may return DatetimeIndex or Series - compare values
        np.testing.assert_array_equal(
            np.array(ds_result, dtype='datetime64[ns]'),
            np.array(pd_result, dtype='datetime64[ns]')
        )

    @pytest.mark.xfail(reason="chDB datetime issue #448 - string column needs datetime conversion")
    def test_dt_year(self, pd_df, ds_df):
        """Extract year from date."""
        pd_result = pd_df['hire_date'].dt.year
        ds_result = ds_df['hire_date'].dt.year
        assert ds_result == pd_result

    @pytest.mark.xfail(reason="chDB datetime issue #448 - string column needs datetime conversion")
    def test_dt_month(self, pd_df, ds_df):
        """Extract month from date."""
        pd_result = pd_df['hire_date'].dt.month
        ds_result = ds_df['hire_date'].dt.month
        assert ds_result == pd_result

    @pytest.mark.xfail(reason="chDB datetime issue #448 - string column needs datetime conversion")
    def test_dt_strftime(self, pd_df, ds_df):
        """Date formatting."""
        pd_result = pd_df['hire_date'].dt.strftime('%Y-%m')
        ds_result = ds_df['hire_date'].dt.strftime('%Y-%m')
        assert ds_result == pd_result


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_dataframe(self):
        """Test empty DataFrame operations."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = ds.DataFrame({'a': [], 'b': []})
        assert ds_df == pd_df

    def test_single_row(self):
        """Test single row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = ds.DataFrame({'a': [1], 'b': [2]})
        assert ds_df == pd_df

    def test_single_column(self):
        """Test single column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = ds.DataFrame({'a': [1, 2, 3]})
        assert ds_df == pd_df

    def test_all_null_column(self):
        """Test column with all nulls."""
        pd_df = pd.DataFrame({'a': [None, None, None]})
        ds_df = ds.DataFrame({'a': [None, None, None]})
        # Compare shapes at least
        assert ds_df.shape == pd_df.shape

    def test_mixed_types(self):
        """Test DataFrame with mixed types."""
        data = {
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataFrame(data)
        assert ds_df == pd_df

    def test_chained_operations(self, pd_df, ds_df):
        """Test chained operations."""
        pd_result = pd_df[pd_df['age'] > 25][['name', 'age']].sort_values('age')
        ds_result = ds_df[ds_df['age'] > 25][['name', 'age']].sort_values('age')
        assert ds_result == pd_result

