"""
Pandas Alignment Tests
======================

Comprehensive test suite for pandas API alignment.
Tests use natural execution triggers (__eq__, __array__, len(), etc.)
following the lazy execution design principle.

Known chDB Issues:
- Issue #447: NaN/NULL handling in Python() table function
- Issue #448: datetime64 handling corruption
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
def large_test_data():
    """Create larger test data for comprehensive testing."""
    np.random.seed(42)
    n = 100
    data = {
        'id': range(1, n + 1),
        'name': [f'Name_{i}' for i in range(n)],
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'value': np.random.randn(n) * 100,
        'quantity': np.random.randint(1, 100, n),
        'price': np.random.uniform(10, 1000, n),
        'date': pd.date_range('2024-01-01', periods=n, freq='D'),
        'is_active': np.random.choice([True, False], n),
        'score': np.random.uniform(0, 100, n),
    }
    df = pd.DataFrame(data)
    df.loc[10:14, 'value'] = np.nan
    df.loc[20:24, 'score'] = np.nan
    return df


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


class TestBasicProperties:
    """Test basic DataFrame properties."""

    def test_shape(self, test_data):
        """DataFrame shape property."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)
        assert df_ds.shape == df_pd.shape

    def test_size(self, test_data):
        """DataFrame size property."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)
        assert df_ds.size == df_pd.size

    def test_columns(self, test_data):
        """DataFrame columns property."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)
        assert list(df_ds.columns) == list(df_pd.columns)

    def test_dtypes(self, test_data):
        """DataFrame dtypes property."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)
        # Natural trigger via Series.equals()
        assert df_ds.dtypes.equals(df_pd.dtypes)

    def test_empty(self, test_data):
        """DataFrame empty property."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)
        assert df_ds.empty == df_pd.empty

        # Test with empty DataFrame
        empty_ds = ds.DataFrame({'a': []})
        empty_pd = pd.DataFrame({'a': []})
        assert empty_ds.empty == empty_pd.empty


class TestDataFrameCreation:
    """Test DataFrame creation operations."""

    def test_create_from_dict(self):
        """Create DataFrame from dict."""
        result = ds.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert result == expected

    def test_create_with_index(self):
        """Create DataFrame with index."""
        result = ds.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        expected = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        assert result == expected


class TestDataSelection:
    """Test data selection operations."""

    def test_select_single_column(self, test_data):
        """Select single column."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name']
        expected = df_pd['name']
        np.testing.assert_array_equal(result, expected)

    def test_select_multiple_columns(self, test_data):
        """Select multiple columns."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)
        assert df_ds[['name', 'age']] == df_pd[['name', 'age']]

    def test_select_rows_by_slice(self, test_data):
        """Select rows by slice."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds[:3]
        expected = df_pd[:3]

        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
        assert result == expected

    def test_select_with_boolean_indexing(self, test_data):
        """Select with boolean indexing."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds[df_ds['age'] > 30]
        expected = df_pd[df_pd['age'] > 30]

        assert len(result) == len(expected)
        assert result == expected

    def test_select_with_loc(self, test_data):
        """Select with loc."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.loc[0:2, ['name', 'age']]
        expected = df_pd.loc[0:2, ['name', 'age']]
        pd.testing.assert_frame_equal(result, expected)

    def test_select_with_iloc(self, test_data):
        """Select with iloc."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.iloc[0:3, 0:2]
        expected = df_pd.iloc[0:3, 0:2]
        pd.testing.assert_frame_equal(result, expected)

    def test_head(self, test_data):
        """DataFrame head()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.head(5)
        expected = df_pd.head(5)
        assert result == expected

    def test_tail(self, test_data):
        """DataFrame tail()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.tail(5)
        expected = df_pd.tail(5)
        assert result == expected

    def test_nlargest(self, large_test_data):
        """DataFrame nlargest()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds.nlargest(10, 'value')
        expected = df_pd.nlargest(10, 'value')
        assert result == expected

    def test_nsmallest(self, large_test_data):
        """DataFrame nsmallest()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds.nsmallest(10, 'value')
        expected = df_pd.nsmallest(10, 'value')
        assert result == expected


class TestStatistics:
    """Test statistical operations."""

    def test_mean(self, test_data):
        """Column mean()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['age'].mean())
        expected = df_pd['age'].mean()
        assert abs(result - expected) < 0.001

    def test_sum(self, test_data):
        """Column sum()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['salary'].sum())
        expected = df_pd['salary'].sum()
        assert result == expected

    def test_median(self, test_data):
        """Column median()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['age'].median())
        expected = df_pd['age'].median()
        assert result == expected

    def test_std(self, test_data):
        """Column std()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['age'].std())
        expected = df_pd['age'].std()
        assert abs(result - expected) < 0.001

    def test_min(self, test_data):
        """Column min()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['salary'].min())
        expected = df_pd['salary'].min()
        assert result == expected

    def test_max(self, test_data):
        """Column max()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = float(df_ds['salary'].max())
        expected = df_pd['salary'].max()
        assert result == expected

    def test_count(self, test_data):
        """Column count()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = int(df_ds['age'].count())
        expected = df_pd['age'].count()
        assert result == expected

    def test_describe(self, large_test_data):
        """DataFrame describe()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds.describe()
        expected = df_pd.describe()
        assert result == expected

    def test_value_counts(self, test_data):
        """Column value_counts()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['department'].value_counts()
        expected = df_pd['department'].value_counts()

        np.testing.assert_array_equal(result, expected)


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
        """GroupBy with size - returns LazySeries (pd.Series compatible)."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department').size()
        expected = df_pd.groupby('department').size()

        # Natural trigger via __array__ protocol
        np.testing.assert_array_equal(result, expected)

    # ========== GroupBy Return Type Alignment Tests ==========

    def test_groupby_column_sum_series(self, test_data):
        """df.groupby('cat')['col'].sum() - same as pandas."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department')['salary'].sum()
        expected = df_pd.groupby('department')['salary'].sum()

        # Natural trigger via __array__ protocol
        np.testing.assert_array_almost_equal(result, expected)

    def test_groupby_agg_dict(self, test_data):
        """df.groupby('cat').agg({'col': 'func'}) - same as pandas."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department').agg({'salary': 'sum'})
        expected = df_pd.groupby('department').agg({'salary': 'sum'})

        # Natural trigger via __eq__
        assert result == expected

    def test_groupby_agg_multiple_funcs(self, test_data):
        """df.groupby('cat').agg({'col': ['func1', 'func2']}) - same as pandas."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department').agg({'salary': ['sum', 'mean']})
        expected = df_pd.groupby('department').agg({'salary': ['sum', 'mean']})

        # Natural trigger via __eq__
        assert result == expected

    def test_groupby_column_mean(self, test_data):
        """df.groupby('cat')['col'].mean() - same as pandas."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.groupby('department')['salary'].mean()
        expected = df_pd.groupby('department')['salary'].mean()

        # Natural trigger via __array__ protocol
        np.testing.assert_array_almost_equal(result, expected)


class TestStringOperations:
    """Test string operations."""

    def test_str_contains(self, test_data):
        """str.contains()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name'].str.contains('a', na=False)
        expected = df_pd['name'].str.contains('a', na=False)

        np.testing.assert_array_equal(result, expected)

    def test_str_upper(self, large_test_data):
        """str.upper()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds['name'].str.upper()
        expected = df_pd['name'].str.upper()

        np.testing.assert_array_equal(result, expected)

    def test_str_lower(self, large_test_data):
        """str.lower()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds['name'].str.lower()
        expected = df_pd['name'].str.lower()

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.xfail(reason="chDB Issue #447: NULL becomes empty string instead of None", strict=False)
    def test_str_upper_null_handling(self, test_data):
        """str.upper() NULL handling - chDB Issue #447."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name'].str.upper()
        expected = df_pd['name'].str.upper()

        # Check that NULL/None values are preserved as None, not empty string
        assert result.iloc[6] is None or pd.isna(
            result.iloc[6]
        ), f"Expected None/NaN at index 6, got '{result.iloc[6]}'"

    @pytest.mark.xfail(reason="chDB Issue #447: str.len() returns 0 for NULL instead of NaN", strict=False)
    def test_str_len_null_handling(self, test_data):
        """str.len() NULL handling - chDB Issue #447."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['name'].str.len()
        expected = df_pd['name'].str.len()

        # NULL should return NaN, not 0
        assert pd.isna(result.iloc[6]), f"Expected NaN at index 6, got {result.iloc[6]}"


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

        # Natural trigger via __eq__
        assert result == expected

    def test_merge_dataframes(self):
        """Merge DataFrames - PASS"""
        df1_ds = ds.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2_ds = ds.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        result = ds.merge(df1_ds, df2_ds, on='key', how='inner')

        df1_pd = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2_pd = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        expected = pd.merge(df1_pd, df2_pd, on='key', how='inner')

        # Natural trigger via __eq__
        assert result == expected


class TestDateTimeOperations:
    """Test datetime operations - values, name, and dtype alignment with pandas."""

    # ========== Basic dt accessor properties ==========

    def test_dt_year(self, pandas_df, datastore_df):
        """dt.year - values, name, and dtype alignment."""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.year
        expected = pandas_df['hire_date'].dt.year

        # Values match
        np.testing.assert_array_equal(result, expected)
        # Name preserved
        assert result.name == expected.name, f"Name mismatch: {result.name} vs {expected.name}"
        # Dtype aligned
        assert str(result.dtype) == str(expected.dtype), f"Dtype mismatch: {result.dtype} vs {expected.dtype}"

    def test_dt_month(self, pandas_df, datastore_df):
        """dt.month - values, name, and dtype alignment."""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.month
        expected = pandas_df['hire_date'].dt.month

        np.testing.assert_array_equal(result, expected)
        assert result.name == expected.name
        assert str(result.dtype) == str(expected.dtype)

    def test_dt_day(self, pandas_df, datastore_df):
        """dt.day - values, name, and dtype alignment."""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.day
        expected = pandas_df['hire_date'].dt.day

        np.testing.assert_array_equal(result, expected)
        assert result.name == expected.name
        assert str(result.dtype) == str(expected.dtype)

    def test_dt_dayofweek(self, pandas_df, datastore_df):
        """dt.dayofweek - values, name, and dtype alignment (Monday=0)."""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.dayofweek
        expected = pandas_df['hire_date'].dt.dayofweek

        np.testing.assert_array_equal(result, expected)
        assert result.name == expected.name
        assert str(result.dtype) == str(expected.dtype)

    def test_dt_quarter(self, pandas_df, datastore_df):
        """dt.quarter - values, name, and dtype alignment."""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.quarter
        expected = pandas_df['hire_date'].dt.quarter

        np.testing.assert_array_equal(result, expected)
        assert result.name == expected.name
        assert str(result.dtype) == str(expected.dtype)

    def test_dt_dayofyear(self, pandas_df, datastore_df):
        """dt.dayofyear - values, name, and dtype alignment."""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.dayofyear
        expected = pandas_df['hire_date'].dt.dayofyear

        np.testing.assert_array_equal(result, expected)
        assert result.name == expected.name
        assert str(result.dtype) == str(expected.dtype)

    # ========== dt methods ==========

    def test_dt_strftime(self, pandas_df, datastore_df):
        """dt.strftime - format datetime as string."""
        datastore_df['hire_date'] = ds.to_datetime(datastore_df['hire_date'])

        result = datastore_df['hire_date'].dt.strftime('%Y-%m')
        expected = pandas_df['hire_date'].dt.strftime('%Y-%m')

        np.testing.assert_array_equal(result, expected)

    # ========== Different column name preservation ==========

    def test_dt_accessor_preserves_custom_column_name(self):
        """dt accessor preserves custom column name, not __result__."""
        data = {'my_custom_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10'])}
        df_ds = ds.DataFrame(data)
        df_pd = pd.DataFrame(data)

        result = df_ds['my_custom_date'].dt.year
        expected = df_pd['my_custom_date'].dt.year

        # Must preserve the original column name
        assert result.name == 'my_custom_date', f"Expected 'my_custom_date', got '{result.name}'"
        assert result.name == expected.name


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_dropna(self, test_data):
        """dropna()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.dropna()
        expected = df_pd.dropna()

        assert len(result) == len(expected), f"Expected {len(expected)} rows after dropna, got {len(result)}"

    def test_fillna(self, test_data):
        """fillna()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.fillna(0)
        expected = df_pd.fillna(0)

        assert result == expected

    def test_isnull(self, large_test_data):
        """isnull().sum()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds.isna().sum()
        expected = df_pd.isna().sum()

        # Compare as Series
        assert result.equals(expected)


class TestDataManipulation:
    """Test data manipulation operations."""

    def test_drop_columns(self, test_data):
        """drop(columns=...)."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.drop(columns=['salary'])
        expected = df_pd.drop(columns=['salary'])

        assert result == expected

    def test_rename_columns(self, test_data):
        """rename(columns=...)."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.rename(columns={'salary': 'income'})
        expected = df_pd.rename(columns={'salary': 'income'})

        assert result == expected

    def test_drop_duplicates(self, test_data):
        """drop_duplicates()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.drop_duplicates(subset=['department'])
        expected = df_pd.drop_duplicates(subset=['department'])

        assert result == expected

    def test_assign(self, large_test_data):
        """assign()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds.assign(revenue=lambda x: x['price'] * x['quantity']).head(5)
        expected = df_pd.assign(revenue=lambda x: x['price'] * x['quantity']).head(5)

        assert result == expected


class TestDataTransformation:
    """Test data transformation operations."""

    def test_map_values(self, test_data):
        """Series.map()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        mapping = {'HR': 1, 'IT': 2, 'Finance': 3}
        result = df_ds['department'].map(mapping)
        expected = df_pd['department'].map(mapping)

        np.testing.assert_array_equal(result, expected)

    def test_astype_column(self, test_data):
        """Series.astype()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds['age'].astype(str)
        expected = df_pd['age'].astype(str)

        np.testing.assert_array_equal(result, expected)

    def test_astype_dataframe(self, large_test_data):
        """DataFrame.astype()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds.astype({'quantity': 'float64'})
        expected = df_pd.astype({'quantity': 'float64'})

        assert result == expected


class TestSorting:
    """Test sorting operations."""

    def test_sort_values_single_column(self, test_data):
        """sort_values() single column."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.sort_values('age')
        expected = df_pd.sort_values('age')

        assert result == expected

    def test_sort_values_descending(self, test_data):
        """sort_values() descending."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds.sort_values('salary', ascending=False)
        expected = df_pd.sort_values('salary', ascending=False)

        assert result == expected

    def test_sort_values_head(self, large_test_data):
        """sort_values().head()."""
        df_ds = ds.DataFrame(large_test_data)
        df_pd = large_test_data

        result = df_ds.sort_values('value', ascending=False).head(5)
        expected = df_pd.sort_values('value', ascending=False).head(5)

        assert result == expected


class TestIO:
    """Test I/O operations."""

    def test_to_dict(self, test_data):
        """to_dict()."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        # DataStore to_dict returns records format by default
        result = df_ds.head(5).to_dict()
        expected = df_pd.head(5).to_dict('records')

        assert len(result) == len(expected)

    def test_to_numpy_shape(self, test_data):
        """to_numpy() shape."""
        df_ds = ds.DataFrame(test_data)
        df_pd = pd.DataFrame(test_data)

        result = df_ds[['age', 'salary']].head(5).to_numpy().shape
        expected = df_pd[['age', 'salary']].head(5).to_numpy().shape

        assert result == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
