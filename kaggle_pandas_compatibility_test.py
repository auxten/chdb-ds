"""
Kaggle Pandas Compatibility Test Suite

This script tests common pandas operations found in Kaggle notebooks
by comparing pandas behavior with datastore's pandas-compatible API.

Based on analysis of popular Kaggle notebooks including:
- Topic 1. Exploratory Data Analysis with Pandas (kashnitsky)
- Comprehensive Data Analysis with Pandas (prashant111)
- EDA: Exploratory Data Analysis notebook (udutta)
- And patterns from top 100+ Kaggle notebooks

Operations tested cover the most frequent use cases in data science workflows.
"""

import sys
import traceback
from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd

# Monkey patch: Import datastore as pd alias for compatibility testing
try:
    import datastore as ds
except ImportError:
    print("ERROR: datastore module not found. Make sure it's installed.")
    sys.exit(1)


class CompatibilityTester:
    """Test framework for comparing pandas and datastore operations."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.errors = 0

    def test(self, name: str, pandas_func: Callable, datastore_func: Callable,
             compare_func: Callable = None):
        """
        Run a single test comparing pandas and datastore behavior.

        Args:
            name: Test name
            pandas_func: Function that executes pandas operation
            datastore_func: Function that executes equivalent datastore operation
            compare_func: Optional custom comparison function
        """
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")

        try:
            # Execute pandas version
            print("  [1/3] Running pandas operation...")
            pandas_result = pandas_func()
            print(f"  ✓ Pandas result type: {type(pandas_result)}")

            # Execute datastore version
            print("  [2/3] Running datastore operation...")
            datastore_result = datastore_func()
            print(f"  ✓ DataStore result type: {type(datastore_result)}")

            # Compare results
            print("  [3/3] Comparing results...")
            if compare_func:
                passed = compare_func(pandas_result, datastore_result)
            else:
                passed = self._default_compare(pandas_result, datastore_result)

            if passed:
                print(f"  ✓ PASSED: Results match")
                self.passed += 1
                self.results.append({
                    'test': name,
                    'status': 'PASSED',
                    'pandas_result': pandas_result,
                    'datastore_result': datastore_result
                })
            else:
                print(f"  ✗ FAILED: Results don't match")
                print(f"    Pandas result: {pandas_result}")
                print(f"    DataStore result: {datastore_result}")
                self.failed += 1
                self.results.append({
                    'test': name,
                    'status': 'FAILED',
                    'pandas_result': pandas_result,
                    'datastore_result': datastore_result,
                    'reason': 'Results mismatch'
                })

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            print(f"  Traceback:\n{traceback.format_exc()}")
            self.errors += 1
            self.results.append({
                'test': name,
                'status': 'ERROR',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def _default_compare(self, pandas_result: Any, datastore_result: Any) -> bool:
        """Default comparison logic."""
        try:
            # Handle DataFrames
            if isinstance(pandas_result, pd.DataFrame):
                if hasattr(datastore_result, 'to_pandas'):
                    ds_df = datastore_result.to_pandas()
                elif isinstance(datastore_result, pd.DataFrame):
                    ds_df = datastore_result
                else:
                    return False
                return pandas_result.equals(ds_df)

            # Handle Series
            elif isinstance(pandas_result, pd.Series):
                if hasattr(datastore_result, 'to_pandas'):
                    ds_series = datastore_result.to_pandas()
                elif isinstance(datastore_result, pd.Series):
                    ds_series = datastore_result
                else:
                    return False
                return pandas_result.equals(ds_series)

            # Handle scalar values
            elif isinstance(pandas_result, (int, float, str, bool, type(None))):
                return pandas_result == datastore_result

            # Handle numpy arrays
            elif isinstance(pandas_result, np.ndarray):
                if isinstance(datastore_result, np.ndarray):
                    return np.array_equal(pandas_result, datastore_result)
                return False

            # Default: exact match
            else:
                return pandas_result == datastore_result

        except Exception as e:
            print(f"    Comparison error: {e}")
            return False

    def print_summary(self):
        """Print test summary."""
        print(f"\n\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {len(self.results)}")
        print(f"✓ Passed: {self.passed}")
        print(f"✗ Failed: {self.failed}")
        print(f"⚠ Errors: {self.errors}")
        print(f"Success rate: {self.passed / len(self.results) * 100:.1f}%" if self.results else "N/A")

        if self.failed > 0 or self.errors > 0:
            print(f"\n{'='*80}")
            print(f"ISSUES FOUND")
            print(f"{'='*80}")
            for result in self.results:
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"\n{result['status']}: {result['test']}")
                    if 'error' in result:
                        print(f"  Error: {result['error']}")
                    elif 'reason' in result:
                        print(f"  Reason: {result['reason']}")


def run_tests():
    """Run comprehensive pandas compatibility tests."""

    tester = CompatibilityTester()

    # ========================================================================
    # Category 1: DataFrame Creation and Basic IO (Most Common in Kaggle)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 1: DataFrame Creation and Basic IO")
    print("="*80)

    # Test 1: Create DataFrame from dict
    tester.test(
        "1.1 Create DataFrame from dict",
        lambda: pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),
        lambda: ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    )

    # Test 2: Create DataFrame with index
    tester.test(
        "1.2 Create DataFrame with custom index",
        lambda: pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c']),
        lambda: ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
    )

    # Test 3: Create Series
    tester.test(
        "1.3 Create Series",
        lambda: pd.Series([1, 2, 3, 4], name='test_series'),
        lambda: ds.Series([1, 2, 3, 4], name='test_series')
    )

    # Create test CSV file
    test_csv_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin'],
        'salary': [50000, 60000, 70000, 55000, 65000]
    })
    test_csv_data.to_csv('/tmp/test_data.csv', index=False)

    # Test 4: Read CSV
    tester.test(
        "1.4 Read CSV file",
        lambda: pd.read_csv('/tmp/test_data.csv'),
        lambda: ds.read_csv('/tmp/test_data.csv').to_pandas()
    )

    # ========================================================================
    # Category 2: Basic DataFrame Inspection (Very Common in EDA)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 2: Basic DataFrame Inspection")
    print("="*80)

    # Create test DataFrame for inspection
    df_pandas = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    df_ds = ds.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })

    # Test 5: head()
    tester.test(
        "2.1 head() - get first 3 rows",
        lambda: df_pandas.head(3),
        lambda: df_ds.head(3).to_pandas() if hasattr(df_ds.head(3), 'to_pandas') else df_ds.head(3)
    )

    # Test 6: tail()
    tester.test(
        "2.2 tail() - get last 2 rows",
        lambda: df_pandas.tail(2),
        lambda: df_ds.tail(2).to_pandas() if hasattr(df_ds.tail(2), 'to_pandas') else df_ds.tail(2)
    )

    # Test 7: shape
    tester.test(
        "2.3 shape - get dimensions",
        lambda: df_pandas.shape,
        lambda: df_ds.shape
    )

    # Test 8: columns
    tester.test(
        "2.4 columns - get column names",
        lambda: list(df_pandas.columns),
        lambda: list(df_ds.columns)
    )

    # Test 9: dtypes
    tester.test(
        "2.5 dtypes - get data types",
        lambda: df_pandas.dtypes.to_dict(),
        lambda: df_ds.dtypes.to_dict() if hasattr(df_ds, 'dtypes') else {}
    )

    # ========================================================================
    # Category 3: Column Selection and Indexing (Extremely Common)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 3: Column Selection and Indexing")
    print("="*80)

    # Test 10: Select single column
    tester.test(
        "3.1 Select single column with []",
        lambda: df_pandas['A'],
        lambda: df_ds['A'].to_pandas() if hasattr(df_ds['A'], 'to_pandas') else df_ds['A']
    )

    # Test 11: Select multiple columns
    tester.test(
        "3.2 Select multiple columns with []",
        lambda: df_pandas[['A', 'B']],
        lambda: df_ds[['A', 'B']].to_pandas() if hasattr(df_ds[['A', 'B']], 'to_pandas') else df_ds[['A', 'B']]
    )

    # Test 12: loc indexing
    tester.test(
        "3.3 loc - select rows by label",
        lambda: df_pandas.loc[0:2, ['A', 'B']],
        lambda: df_ds.loc[0:2, ['A', 'B']].to_pandas() if hasattr(df_ds, 'loc') else None
    )

    # Test 13: iloc indexing
    tester.test(
        "3.4 iloc - select rows by position",
        lambda: df_pandas.iloc[0:2, 0:2],
        lambda: df_ds.iloc[0:2, 0:2].to_pandas() if hasattr(df_ds, 'iloc') else None
    )

    # ========================================================================
    # Category 4: Filtering and Boolean Indexing (Very Common)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 4: Filtering and Boolean Indexing")
    print("="*80)

    # Test 14: Simple boolean filter
    tester.test(
        "4.1 Boolean filter - single condition",
        lambda: df_pandas[df_pandas['A'] > 2],
        lambda: df_ds[df_ds['A'] > 2].to_pandas() if hasattr(df_ds[df_ds['A'] > 2], 'to_pandas') else df_ds[df_ds['A'] > 2]
    )

    # Test 15: Multiple conditions with &
    tester.test(
        "4.2 Boolean filter - AND condition",
        lambda: df_pandas[(df_pandas['A'] > 2) & (df_pandas['B'] < 50)],
        lambda: df_ds[(df_ds['A'] > 2) & (df_ds['B'] < 50)].to_pandas() if hasattr(df_ds, '__getitem__') else None
    )

    # Test 16: Multiple conditions with |
    tester.test(
        "4.3 Boolean filter - OR condition",
        lambda: df_pandas[(df_pandas['A'] < 2) | (df_pandas['A'] > 4)],
        lambda: df_ds[(df_ds['A'] < 2) | (df_ds['A'] > 4)].to_pandas() if hasattr(df_ds, '__getitem__') else None
    )

    # Test 17: isin() filter
    tester.test(
        "4.4 isin() - filter with list of values",
        lambda: df_pandas[df_pandas['A'].isin([1, 3, 5])],
        lambda: df_ds[df_ds['A'].isin([1, 3, 5])].to_pandas() if hasattr(df_ds, '__getitem__') else None
    )

    # ========================================================================
    # Category 5: Missing Value Handling (Very Common in Real Data)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 5: Missing Value Handling")
    print("="*80)

    # Create DataFrame with missing values
    df_na_pandas = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    })
    df_na_ds = ds.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    })

    # Test 18: isna() / isnull()
    tester.test(
        "5.1 isna() - detect missing values",
        lambda: df_na_pandas.isna(),
        lambda: df_na_ds.isna().to_pandas() if hasattr(df_na_ds, 'isna') else None
    )

    # Test 19: fillna()
    tester.test(
        "5.2 fillna() - fill missing values with 0",
        lambda: df_na_pandas.fillna(0),
        lambda: df_na_ds.fillna(0).to_pandas() if hasattr(df_na_ds, 'fillna') else None
    )

    # Test 20: dropna()
    tester.test(
        "5.3 dropna() - drop rows with missing values",
        lambda: df_na_pandas.dropna(),
        lambda: df_na_ds.dropna().to_pandas() if hasattr(df_na_ds, 'dropna') else None
    )

    # ========================================================================
    # Category 6: GroupBy and Aggregation (Very Common in Analysis)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 6: GroupBy and Aggregation")
    print("="*80)

    # Create DataFrame for groupby tests
    df_group_pandas = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40, 50, 60],
        'count': [1, 2, 3, 4, 5, 6]
    })
    df_group_ds = ds.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40, 50, 60],
        'count': [1, 2, 3, 4, 5, 6]
    })

    # Test 21: groupby().sum()
    tester.test(
        "6.1 groupby().sum()",
        lambda: df_group_pandas.groupby('category')['value'].sum(),
        lambda: df_group_ds.groupby('category')['value'].sum().to_pandas() if hasattr(df_group_ds, 'groupby') else None
    )

    # Test 22: groupby().mean()
    tester.test(
        "6.2 groupby().mean()",
        lambda: df_group_pandas.groupby('category')['value'].mean(),
        lambda: df_group_ds.groupby('category')['value'].mean().to_pandas() if hasattr(df_group_ds, 'groupby') else None
    )

    # Test 23: groupby().count()
    tester.test(
        "6.3 groupby().count()",
        lambda: df_group_pandas.groupby('category')['value'].count(),
        lambda: df_group_ds.groupby('category')['value'].count().to_pandas() if hasattr(df_group_ds, 'groupby') else None
    )

    # Test 24: groupby().agg() with multiple functions
    tester.test(
        "6.4 groupby().agg() - multiple aggregations",
        lambda: df_group_pandas.groupby('category')['value'].agg(['sum', 'mean', 'count']),
        lambda: df_group_ds.groupby('category')['value'].agg(['sum', 'mean', 'count']).to_pandas() if hasattr(df_group_ds, 'groupby') else None
    )

    # ========================================================================
    # Category 7: Sorting (Common in Data Exploration)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 7: Sorting")
    print("="*80)

    # Test 25: sort_values() ascending
    tester.test(
        "7.1 sort_values() - ascending",
        lambda: df_pandas.sort_values('A'),
        lambda: df_ds.sort_values('A').to_pandas() if hasattr(df_ds, 'sort_values') else None
    )

    # Test 26: sort_values() descending
    tester.test(
        "7.2 sort_values() - descending",
        lambda: df_pandas.sort_values('B', ascending=False),
        lambda: df_ds.sort_values('B', ascending=False).to_pandas() if hasattr(df_ds, 'sort_values') else None
    )

    # Test 27: sort_values() multiple columns
    tester.test(
        "7.3 sort_values() - multiple columns",
        lambda: df_group_pandas.sort_values(['category', 'value']),
        lambda: df_group_ds.sort_values(['category', 'value']).to_pandas() if hasattr(df_group_ds, 'sort_values') else None
    )

    # ========================================================================
    # Category 8: Statistical Operations (Common in EDA)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 8: Statistical Operations")
    print("="*80)

    # Test 28: sum()
    tester.test(
        "8.1 sum() - column sum",
        lambda: df_pandas['A'].sum(),
        lambda: df_ds['A'].sum() if hasattr(df_ds['A'], 'sum') else None
    )

    # Test 29: mean()
    tester.test(
        "8.2 mean() - column mean",
        lambda: df_pandas['B'].mean(),
        lambda: df_ds['B'].mean() if hasattr(df_ds['B'], 'mean') else None
    )

    # Test 30: min() and max()
    tester.test(
        "8.3 min() and max()",
        lambda: (df_pandas['A'].min(), df_pandas['A'].max()),
        lambda: (df_ds['A'].min(), df_ds['A'].max()) if hasattr(df_ds['A'], 'min') else None
    )

    # Test 31: describe()
    tester.test(
        "8.4 describe() - summary statistics",
        lambda: df_pandas[['A', 'B']].describe(),
        lambda: df_ds[['A', 'B']].describe().to_pandas() if hasattr(df_ds, 'describe') else None
    )

    # Test 32: value_counts()
    tester.test(
        "8.5 value_counts() - count unique values",
        lambda: df_group_pandas['category'].value_counts(),
        lambda: df_group_ds['category'].value_counts().to_pandas() if hasattr(df_group_ds['category'], 'value_counts') else None
    )

    # ========================================================================
    # Category 9: Data Transformation (Common Operations)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 9: Data Transformation")
    print("="*80)

    # Test 33: rename columns
    tester.test(
        "9.1 rename() - rename columns",
        lambda: df_pandas.rename(columns={'A': 'col_A', 'B': 'col_B'}),
        lambda: df_ds.rename(columns={'A': 'col_A', 'B': 'col_B'}).to_pandas() if hasattr(df_ds, 'rename') else None
    )

    # Test 34: drop columns
    tester.test(
        "9.2 drop() - drop columns",
        lambda: df_pandas.drop(columns=['C']),
        lambda: df_ds.drop(columns=['C']).to_pandas() if hasattr(df_ds, 'drop') else None
    )

    # Test 35: reset_index()
    tester.test(
        "9.3 reset_index() - reset index",
        lambda: df_pandas.reset_index(drop=True),
        lambda: df_ds.reset_index(drop=True).to_pandas() if hasattr(df_ds, 'reset_index') else None
    )

    # Test 36: apply() with lambda
    tester.test(
        "9.4 apply() - apply function to column",
        lambda: df_pandas['A'].apply(lambda x: x * 2),
        lambda: df_ds['A'].apply(lambda x: x * 2).to_pandas() if hasattr(df_ds['A'], 'apply') else None
    )

    # ========================================================================
    # Category 10: String Operations (Common with Text Data)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 10: String Operations")
    print("="*80)

    # Create DataFrame with strings
    df_str_pandas = pd.DataFrame({
        'text': ['hello', 'WORLD', 'Python', 'pandas', 'DataStore']
    })
    df_str_ds = ds.DataFrame({
        'text': ['hello', 'WORLD', 'Python', 'pandas', 'DataStore']
    })

    # Test 37: str.lower()
    tester.test(
        "10.1 str.lower() - convert to lowercase",
        lambda: df_str_pandas['text'].str.lower(),
        lambda: df_str_ds['text'].str.lower().to_pandas() if hasattr(df_str_ds['text'], 'str') else None
    )

    # Test 38: str.upper()
    tester.test(
        "10.2 str.upper() - convert to uppercase",
        lambda: df_str_pandas['text'].str.upper(),
        lambda: df_str_ds['text'].str.upper().to_pandas() if hasattr(df_str_ds['text'], 'str') else None
    )

    # Test 39: str.contains()
    tester.test(
        "10.3 str.contains() - check if contains substring",
        lambda: df_str_pandas['text'].str.contains('a'),
        lambda: df_str_ds['text'].str.contains('a').to_pandas() if hasattr(df_str_ds['text'], 'str') else None
    )

    # Test 40: str.len()
    tester.test(
        "10.4 str.len() - get string length",
        lambda: df_str_pandas['text'].str.len(),
        lambda: df_str_ds['text'].str.len().to_pandas() if hasattr(df_str_ds['text'], 'str') else None
    )

    # ========================================================================
    # Category 11: DateTime Operations (Common in Time Series)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 11: DateTime Operations")
    print("="*80)

    # Create DataFrame with dates
    df_dt_pandas = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5)
    })
    df_dt_ds = ds.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5)
    })

    # Test 41: dt.year
    tester.test(
        "11.1 dt.year - extract year",
        lambda: df_dt_pandas['date'].dt.year,
        lambda: df_dt_ds['date'].dt.year.to_pandas() if hasattr(df_dt_ds['date'], 'dt') else None
    )

    # Test 42: dt.month
    tester.test(
        "11.2 dt.month - extract month",
        lambda: df_dt_pandas['date'].dt.month,
        lambda: df_dt_ds['date'].dt.month.to_pandas() if hasattr(df_dt_ds['date'], 'dt') else None
    )

    # Test 43: dt.day
    tester.test(
        "11.3 dt.day - extract day",
        lambda: df_dt_pandas['date'].dt.day,
        lambda: df_dt_ds['date'].dt.day.to_pandas() if hasattr(df_dt_ds['date'], 'dt') else None
    )

    # Test 44: dt.dayofweek
    tester.test(
        "11.4 dt.dayofweek - extract day of week",
        lambda: df_dt_pandas['date'].dt.dayofweek,
        lambda: df_dt_ds['date'].dt.dayofweek.to_pandas() if hasattr(df_dt_ds['date'], 'dt') else None
    )

    # ========================================================================
    # Category 12: Merge and Join Operations (Common in Data Integration)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 12: Merge and Join Operations")
    print("="*80)

    # Create DataFrames for merge
    df1_pandas = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
    df2_pandas = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
    df1_ds = ds.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
    df2_ds = ds.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

    # Test 45: merge() inner join
    tester.test(
        "12.1 merge() - inner join",
        lambda: pd.merge(df1_pandas, df2_pandas, on='key', how='inner'),
        lambda: ds.merge(df1_ds, df2_ds, on='key', how='inner').to_pandas() if hasattr(ds, 'merge') else None
    )

    # Test 46: merge() left join
    tester.test(
        "12.2 merge() - left join",
        lambda: pd.merge(df1_pandas, df2_pandas, on='key', how='left'),
        lambda: ds.merge(df1_ds, df2_ds, on='key', how='left').to_pandas() if hasattr(ds, 'merge') else None
    )

    # Test 47: concat() vertical
    tester.test(
        "12.3 concat() - vertical concatenation",
        lambda: pd.concat([df1_pandas, df1_pandas], ignore_index=True),
        lambda: ds.concat([df1_ds, df1_ds], ignore_index=True).to_pandas() if hasattr(ds, 'concat') else None
    )

    # ========================================================================
    # Category 13: Unique and Duplicates (Common Data Quality Checks)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 13: Unique and Duplicates")
    print("="*80)

    # Create DataFrame with duplicates
    df_dup_pandas = pd.DataFrame({
        'A': [1, 2, 2, 3, 3, 3],
        'B': ['a', 'b', 'b', 'c', 'c', 'c']
    })
    df_dup_ds = ds.DataFrame({
        'A': [1, 2, 2, 3, 3, 3],
        'B': ['a', 'b', 'b', 'c', 'c', 'c']
    })

    # Test 48: unique()
    tester.test(
        "13.1 unique() - get unique values",
        lambda: sorted(df_dup_pandas['A'].unique().tolist()),
        lambda: sorted(df_dup_ds['A'].unique().tolist()) if hasattr(df_dup_ds['A'], 'unique') else None
    )

    # Test 49: nunique()
    tester.test(
        "13.2 nunique() - count unique values",
        lambda: df_dup_pandas['A'].nunique(),
        lambda: df_dup_ds['A'].nunique() if hasattr(df_dup_ds['A'], 'nunique') else None
    )

    # Test 50: drop_duplicates()
    tester.test(
        "13.3 drop_duplicates() - remove duplicate rows",
        lambda: df_dup_pandas.drop_duplicates(),
        lambda: df_dup_ds.drop_duplicates().to_pandas() if hasattr(df_dup_ds, 'drop_duplicates') else None
    )

    # Print summary
    tester.print_summary()

    return tester


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           Kaggle Pandas Compatibility Test Suite for DataStore              ║
║                                                                              ║
║  Testing 50+ common pandas operations from top Kaggle notebooks             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    tester = run_tests()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nThis test suite identified compatibility gaps between pandas and datastore.")
    print("Review the failed/error tests above to identify improvement opportunities.")

    sys.exit(0 if tester.failed == 0 and tester.errors == 0 else 1)
