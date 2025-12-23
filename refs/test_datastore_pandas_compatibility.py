"""
DataStore vs Pandas Compatibility Test Suite

This script tests datastore's compatibility with pandas by:
1. Running the same operations on both pandas and datastore
2. Comparing results to identify differences
3. Documenting compatibility issues and improvement opportunities

Usage:
    python test_datastore_pandas_compatibility.py
"""

import sys
import warnings
import traceback
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

# Import datastore
try:
    import datastore as ds
    from datastore import DataStore
    HAS_DATASTORE = True
except ImportError:
    print("WARNING: datastore module not found. Install with: pip install chdb-ds")
    HAS_DATASTORE = False

warnings.filterwarnings('ignore')

# ============================================================================
# TEST DATA SETUP
# ============================================================================

def create_test_data() -> Tuple[pd.DataFrame, Any]:
    """Create test DataFrame for both pandas and datastore."""
    np.random.seed(42)

    # Create comprehensive test dataset
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
        'score': np.random.uniform(0, 100, n),  # Changed to float to support NaN
    }

    df_pandas = pd.DataFrame(data)

    # Add some NaN values
    df_pandas.loc[10:14, 'value'] = np.nan
    df_pandas.loc[20:24, 'score'] = np.nan

    # Create datastore version
    if HAS_DATASTORE:
        # Use from_df to create DataStore from pandas DataFrame
        ds_datastore = DataStore.from_df(df_pandas, name='test_data')
    else:
        ds_datastore = None

    return df_pandas, ds_datastore


# ============================================================================
# TEST FRAMEWORK
# ============================================================================

class CompatibilityTest:
    """Base class for compatibility tests."""

    def __init__(self, name: str, category: str, complexity: str):
        self.name = name
        self.category = category
        self.complexity = complexity
        self.pandas_result = None
        self.datastore_result = None
        self.pandas_error = None
        self.datastore_error = None
        self.compatible = None
        self.notes = ""

    def run_pandas(self, df_pandas: pd.DataFrame) -> Any:
        """Run test on pandas DataFrame."""
        raise NotImplementedError

    def run_datastore(self, ds_datastore: Any) -> Any:
        """Run test on datastore."""
        raise NotImplementedError

    def compare_results(self) -> bool:
        """Compare results from pandas and datastore."""
        if self.pandas_error and self.datastore_error:
            return True  # Both failed - considered compatible

        if self.pandas_error or self.datastore_error:
            return False  # One failed - not compatible

        # Try to compare results
        try:
            if isinstance(self.pandas_result, pd.DataFrame):
                if hasattr(self.datastore_result, 'to_df'):
                    ds_df = self.datastore_result.to_df()
                elif hasattr(self.datastore_result, '_get_df'):
                    ds_df = self.datastore_result._get_df()
                else:
                    ds_df = self.datastore_result

                return self.pandas_result.equals(ds_df)

            elif isinstance(self.pandas_result, pd.Series):
                if hasattr(self.datastore_result, 'to_df'):
                    return np.allclose(self.pandas_result.values,
                                     self.datastore_result.to_df().squeeze().values,
                                     equal_nan=True)
                else:
                    return np.allclose(self.pandas_result.values,
                                     self.datastore_result.values if hasattr(self.datastore_result, 'values') else self.datastore_result,
                                     equal_nan=True)

            elif isinstance(self.pandas_result, (int, float, np.number)):
                ds_val = self.datastore_result
                if hasattr(ds_val, 'item'):
                    ds_val = ds_val.item()
                return np.isclose(self.pandas_result, ds_val, equal_nan=True)

            else:
                return self.pandas_result == self.datastore_result

        except Exception as e:
            self.notes = f"Comparison error: {str(e)}"
            return False

    def run(self, df_pandas: pd.DataFrame, ds_datastore: Any) -> Dict:
        """Run test on both pandas and datastore."""
        # Run pandas test
        try:
            self.pandas_result = self.run_pandas(df_pandas)
        except Exception as e:
            self.pandas_error = f"{type(e).__name__}: {str(e)}"

        # Run datastore test
        if ds_datastore is not None:
            try:
                self.datastore_result = self.run_datastore(ds_datastore)
            except Exception as e:
                self.datastore_error = f"{type(e).__name__}: {str(e)}"
        else:
            self.datastore_error = "DataStore not available"

        # Compare results
        self.compatible = self.compare_results()

        return {
            'name': self.name,
            'category': self.category,
            'complexity': self.complexity,
            'compatible': self.compatible,
            'pandas_error': self.pandas_error,
            'datastore_error': self.datastore_error,
            'notes': self.notes
        }


# ============================================================================
# TEST CASES - CATEGORY 1: BASIC PROPERTIES
# ============================================================================

class TestShape(CompatibilityTest):
    def run_pandas(self, df): return df.shape
    def run_datastore(self, ds): return ds.shape

class TestSize(CompatibilityTest):
    def run_pandas(self, df): return df.size
    def run_datastore(self, ds): return ds.size

class TestColumns(CompatibilityTest):
    def run_pandas(self, df): return list(df.columns)
    def run_datastore(self, ds): return list(ds.columns)

class TestDtypes(CompatibilityTest):
    def run_pandas(self, df): return df.dtypes
    def run_datastore(self, ds): return ds.dtypes

class TestEmpty(CompatibilityTest):
    def run_pandas(self, df): return df.empty
    def run_datastore(self, ds): return ds.empty

# ============================================================================
# TEST CASES - CATEGORY 2: STATISTICAL METHODS
# ============================================================================

class TestMean(CompatibilityTest):
    def run_pandas(self, df): return df['value'].mean()
    def run_datastore(self, ds): return ds.mean(numeric_only=True).get('value')

class TestMedian(CompatibilityTest):
    def run_pandas(self, df): return df['value'].median()
    def run_datastore(self, ds): return ds.median(numeric_only=True).get('value')

class TestSum(CompatibilityTest):
    def run_pandas(self, df): return df['quantity'].sum()
    def run_datastore(self, ds): return ds.sum(numeric_only=True).get('quantity')

class TestStd(CompatibilityTest):
    def run_pandas(self, df): return df['value'].std()
    def run_datastore(self, ds): return ds.std(numeric_only=True).get('value')

class TestMin(CompatibilityTest):
    def run_pandas(self, df): return df['price'].min()
    def run_datastore(self, ds): return ds.min(numeric_only=True).get('price')

class TestMax(CompatibilityTest):
    def run_pandas(self, df): return df['price'].max()
    def run_datastore(self, ds): return ds.max(numeric_only=True).get('price')

class TestCount(CompatibilityTest):
    def run_pandas(self, df): return df['value'].count()
    def run_datastore(self, ds): return ds.count().get('value')

class TestDescribe(CompatibilityTest):
    def run_pandas(self, df): return df.describe()
    def run_datastore(self, ds): return ds.describe()

# ============================================================================
# TEST CASES - CATEGORY 3: SELECTION & INDEXING
# ============================================================================

class TestHead(CompatibilityTest):
    def run_pandas(self, df): return df.head(5)
    def run_datastore(self, ds): return ds.head(5)

class TestTail(CompatibilityTest):
    def run_pandas(self, df): return df.tail(5)
    def run_datastore(self, ds): return ds.tail(5)

class TestColumnSelection(CompatibilityTest):
    def run_pandas(self, df): return df['name']
    def run_datastore(self, ds): return ds['name']

class TestMultiColumnSelection(CompatibilityTest):
    def run_pandas(self, df): return df[['name', 'category', 'value']]
    def run_datastore(self, ds): return ds[['name', 'category', 'value']]

class TestNLargest(CompatibilityTest):
    def run_pandas(self, df): return df.nlargest(10, 'value')
    def run_datastore(self, ds): return ds.nlargest(10, 'value')

class TestNSmallest(CompatibilityTest):
    def run_pandas(self, df): return df.nsmallest(10, 'value')
    def run_datastore(self, ds): return ds.nsmallest(10, 'value')

# ============================================================================
# TEST CASES - CATEGORY 4: DATA MANIPULATION
# ============================================================================

class TestDrop(CompatibilityTest):
    def run_pandas(self, df): return df.drop(columns=['score'])
    def run_datastore(self, ds): return ds.drop(columns=['score'])

class TestRename(CompatibilityTest):
    def run_pandas(self, df): return df.rename(columns={'value': 'amount'})
    def run_datastore(self, ds): return ds.rename(columns={'value': 'amount'})

class TestSortValues(CompatibilityTest):
    def run_pandas(self, df): return df.sort_values('value', ascending=False).head(5)
    def run_datastore(self, ds): return ds.sort_values('value', ascending=False).head(5)

class TestDropDuplicates(CompatibilityTest):
    def run_pandas(self, df): return df.drop_duplicates(subset=['category'])
    def run_datastore(self, ds): return ds.drop_duplicates(subset=['category'])

class TestAssign(CompatibilityTest):
    def run_pandas(self, df):
        return df.assign(revenue=lambda x: x['price'] * x['quantity']).head(5)
    def run_datastore(self, ds):
        return ds.assign(revenue=lambda x: x['price'] * x['quantity']).head(5)

class TestAstype(CompatibilityTest):
    def run_pandas(self, df): return df.astype({'quantity': 'float64'})
    def run_datastore(self, ds): return ds.astype({'quantity': 'float64'})

# ============================================================================
# TEST CASES - CATEGORY 5: MISSING DATA
# ============================================================================

class TestIsNull(CompatibilityTest):
    def run_pandas(self, df): return df.isnull().sum()
    def run_datastore(self, ds): return ds.isna().sum()

class TestDropNA(CompatibilityTest):
    def run_pandas(self, df): return df.dropna()
    def run_datastore(self, ds): return ds.dropna()

class TestFillNA(CompatibilityTest):
    def run_pandas(self, df): return df.fillna(0)
    def run_datastore(self, ds): return ds.fillna(0)

# ============================================================================
# TEST CASES - CATEGORY 6: AGGREGATION
# ============================================================================

class TestAgg(CompatibilityTest):
    def run_pandas(self, df):
        return df.agg({'value': ['sum', 'mean'], 'quantity': 'sum'})
    def run_datastore(self, ds):
        return ds.agg({'value': ['sum', 'mean'], 'quantity': 'sum'})

class TestGroupBySum(CompatibilityTest):
    def run_pandas(self, df):
        return df.groupby('category')['value'].sum()
    def run_datastore(self, ds):
        # DataStore groupby returns different structure
        result = ds.groupby('category').agg({'value': 'sum'})
        if hasattr(result, 'to_df'):
            return result.to_df()
        return result

class TestGroupByAgg(CompatibilityTest):
    def run_pandas(self, df):
        return df.groupby('category').agg({'value': ['sum', 'mean'], 'quantity': 'count'})
    def run_datastore(self, ds):
        # DataStore groupby with multiple aggregations
        result = ds.groupby('category').agg({'value': ['sum', 'mean'], 'quantity': 'count'})
        if hasattr(result, 'to_df'):
            return result.to_df()
        return result

# ============================================================================
# TEST CASES - CATEGORY 7: I/O OPERATIONS
# ============================================================================

class TestToDict(CompatibilityTest):
    def run_pandas(self, df):
        return len(df.head(10).to_dict('records'))
    def run_datastore(self, ds):
        result = ds.head(10).to_dict()
        if isinstance(result, list):
            return len(result)
        return len(result.get('records', []))

class TestToNumpy(CompatibilityTest):
    def run_pandas(self, df):
        return df[['value', 'quantity']].head(5).to_numpy().shape
    def run_datastore(self, ds):
        return ds[['value', 'quantity']].head(5).to_numpy().shape

# ============================================================================
# TEST CASES - CATEGORY 8: STRING OPERATIONS
# ============================================================================

class TestStrUpper(CompatibilityTest):
    def run_pandas(self, df):
        return df['name'].str.upper().head(5)
    def run_datastore(self, ds):
        result = ds['name'].str.upper()
        # Convert to pandas Series for comparison
        if hasattr(result, 'to_df'):
            return result.to_df().squeeze()
        return result

class TestStrLower(CompatibilityTest):
    def run_pandas(self, df):
        return df['name'].str.lower().head(5)
    def run_datastore(self, ds):
        result = ds['name'].str.lower()
        if hasattr(result, 'to_df'):
            return result.to_df().squeeze()
        return result

class TestStrContains(CompatibilityTest):
    def run_pandas(self, df):
        return df[df['name'].str.contains('1')].shape[0]
    def run_datastore(self, ds):
        # DataStore str.contains may work differently
        result = ds.filter(ds['name'].str.contains('1'))
        if hasattr(result, 'shape'):
            return result.shape[0]
        if hasattr(result, 'to_df'):
            return result.to_df().shape[0]
        return len(result)

# ============================================================================
# TEST CASES - CATEGORY 9: DATETIME OPERATIONS
# ============================================================================

class TestDtYear(CompatibilityTest):
    def run_pandas(self, df):
        return df['date'].dt.year.head(5)
    def run_datastore(self, ds):
        result = ds['date'].dt.year
        if hasattr(result, 'to_df'):
            return result.to_df().squeeze()
        return result

class TestDtMonth(CompatibilityTest):
    def run_pandas(self, df):
        return df['date'].dt.month.head(5)
    def run_datastore(self, ds):
        result = ds['date'].dt.month
        if hasattr(result, 'to_df'):
            return result.to_df().squeeze()
        return result

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all compatibility tests."""
    print("=" * 100)
    print("DATASTORE VS PANDAS COMPATIBILITY TEST SUITE")
    print("=" * 100)

    if not HAS_DATASTORE:
        print("\nERROR: DataStore module not found. Cannot run tests.")
        return

    # Create test data
    print("\nCreating test data...")
    df_pandas, ds_datastore = create_test_data()
    print(f"Test data created: {df_pandas.shape[0]} rows, {df_pandas.shape[1]} columns")

    # Define all test cases
    test_cases = [
        # Category 1: Basic Properties
        TestShape("shape", "Properties", "basic"),
        TestSize("size", "Properties", "basic"),
        TestColumns("columns", "Properties", "basic"),
        TestDtypes("dtypes", "Properties", "basic"),
        TestEmpty("empty", "Properties", "basic"),

        # Category 2: Statistical Methods
        TestMean("mean()", "Statistics", "basic"),
        TestMedian("median()", "Statistics", "basic"),
        TestSum("sum()", "Statistics", "basic"),
        TestStd("std()", "Statistics", "basic"),
        TestMin("min()", "Statistics", "basic"),
        TestMax("max()", "Statistics", "basic"),
        TestCount("count()", "Statistics", "basic"),
        TestDescribe("describe()", "Statistics", "basic"),

        # Category 3: Selection & Indexing
        TestHead("head()", "Selection", "basic"),
        TestTail("tail()", "Selection", "basic"),
        TestColumnSelection("df['col']", "Selection", "basic"),
        TestMultiColumnSelection("df[['col1', 'col2']]", "Selection", "basic"),
        TestNLargest("nlargest()", "Selection", "basic"),
        TestNSmallest("nsmallest()", "Selection", "basic"),

        # Category 4: Data Manipulation
        TestDrop("drop()", "Manipulation", "basic"),
        TestRename("rename()", "Manipulation", "basic"),
        TestSortValues("sort_values()", "Manipulation", "basic"),
        TestDropDuplicates("drop_duplicates()", "Manipulation", "basic"),
        TestAssign("assign()", "Manipulation", "intermediate"),
        TestAstype("astype()", "Manipulation", "basic"),

        # Category 5: Missing Data
        TestIsNull("isnull()", "Missing Data", "basic"),
        TestDropNA("dropna()", "Missing Data", "basic"),
        TestFillNA("fillna()", "Missing Data", "basic"),

        # Category 6: Aggregation
        TestAgg("agg()", "Aggregation", "intermediate"),
        TestGroupBySum("groupby().sum()", "Aggregation", "intermediate"),
        TestGroupByAgg("groupby().agg()", "Aggregation", "intermediate"),

        # Category 7: I/O
        TestToDict("to_dict()", "I/O", "basic"),
        TestToNumpy("to_numpy()", "I/O", "basic"),

        # Category 8: String Operations
        TestStrUpper("str.upper()", "String Ops", "intermediate"),
        TestStrLower("str.lower()", "String Ops", "intermediate"),
        TestStrContains("str.contains()", "String Ops", "intermediate"),

        # Category 9: DateTime Operations
        TestDtYear("dt.year", "DateTime", "intermediate"),
        TestDtMonth("dt.month", "DateTime", "intermediate"),
    ]

    # Run all tests
    print(f"\nRunning {len(test_cases)} compatibility tests...\n")
    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing {test.name}...", end=' ')
        result = test.run(df_pandas, ds_datastore)
        results.append(result)

        if result['compatible']:
            print("✓ PASS")
        else:
            print("✗ FAIL")
            if result['datastore_error']:
                print(f"    DataStore Error: {result['datastore_error']}")
            if result['notes']:
                print(f"    Notes: {result['notes']}")

    # Summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)

    results_df = pd.DataFrame(results)

    total_tests = len(results)
    passed_tests = results_df['compatible'].sum()
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Pass Rate: {pass_rate:.1f}%")

    print(f"\nResults by Category:")
    category_summary = results_df.groupby('category')['compatible'].agg(['sum', 'count'])
    category_summary['pass_rate'] = (category_summary['sum'] / category_summary['count'] * 100).round(1)
    category_summary.columns = ['Passed', 'Total', 'Pass Rate %']
    print(category_summary)

    # Failed tests details
    failed_df = results_df[~results_df['compatible']]
    if len(failed_df) > 0:
        print(f"\n" + "=" * 100)
        print("FAILED TESTS DETAILS")
        print("=" * 100)

        for idx, row in failed_df.iterrows():
            print(f"\n{row['name']} ({row['category']})")
            if row['datastore_error']:
                print(f"  Error: {row['datastore_error']}")
            if row['notes']:
                print(f"  Notes: {row['notes']}")

    # Save results
    results_df.to_csv('datastore_pandas_compatibility_results.csv', index=False)
    print(f"\n\nDetailed results saved to 'datastore_pandas_compatibility_results.csv'")

    return results_df


# ============================================================================
# IMPROVEMENT RECOMMENDATIONS
# ============================================================================

def generate_improvement_report(results_df: pd.DataFrame):
    """Generate a report of improvement opportunities."""
    print("\n" + "=" * 100)
    print("IMPROVEMENT RECOMMENDATIONS FOR DATASTORE")
    print("=" * 100)

    failed_df = results_df[~results_df['compatible']]

    if len(failed_df) == 0:
        print("\n✓ Excellent! All tested operations are compatible.")
        return

    print(f"\nFound {len(failed_df)} compatibility issues to address:\n")

    # Group by category
    for category in failed_df['category'].unique():
        category_failures = failed_df[failed_df['category'] == category]
        print(f"\n{category} ({len(category_failures)} issues):")
        print("-" * 100)

        for idx, row in category_failures.iterrows():
            print(f"\n  • {row['name']}")
            if row['datastore_error']:
                error_msg = row['datastore_error']
                if 'AttributeError' in error_msg:
                    print(f"    Issue: Method/property not implemented")
                    print(f"    Recommendation: Implement {row['name']} method in DataStore class")
                elif 'TypeError' in error_msg:
                    print(f"    Issue: Incompatible parameters or return type")
                    print(f"    Recommendation: Review parameter handling and return type consistency")
                else:
                    print(f"    Issue: {error_msg}")
            if row['notes']:
                print(f"    Additional notes: {row['notes']}")

    print("\n" + "=" * 100)
    print("PRIORITY IMPROVEMENTS")
    print("=" * 100)

    print("""
Based on the test results, here are the recommended priorities:

1. HIGH PRIORITY - Basic Operations (Properties, Statistics, Selection)
   These are the most commonly used pandas operations and should work seamlessly.

2. MEDIUM PRIORITY - Data Manipulation (drop, rename, sort, assign)
   These operations are frequently used in data preprocessing pipelines.

3. MEDIUM PRIORITY - Aggregation & GroupBy
   Critical for data analysis workflows. Ensure groupby().agg() works correctly.

4. LOW PRIORITY - Advanced Operations (MultiIndex, Complex Transformations)
   Less frequently used but important for power users.

KEY RECOMMENDATIONS:

1. Ensure all properties (shape, size, columns, dtypes, empty) return values
   identical to pandas.

2. Statistical methods (mean, median, sum, std, etc.) should accept the same
   parameters as pandas (numeric_only, skipna, etc.) and return compatible types.

3. Selection operations (head, tail, nlargest, nsmallest) should return DataStore
   objects that can be further chained or converted to pandas.

4. String operations (.str.*) and datetime operations (.dt.*) should be fully
   implemented with the same API as pandas.

5. GroupBy operations should support the full pandas API including:
   - Multiple aggregation functions
   - Named aggregations
   - Transform and filter methods

6. I/O operations (to_csv, to_json, to_dict, to_numpy) should support all
   pandas parameters and return compatible formats.
""")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results_df = run_all_tests()

    if results_df is not None:
        generate_improvement_report(results_df)

        print("\n" + "=" * 100)
        print("TEST SUITE COMPLETED")
        print("=" * 100)
        print("\nNext Steps:")
        print("1. Review failed tests in 'datastore_pandas_compatibility_results.csv'")
        print("2. Implement missing methods/properties in DataStore")
        print("3. Fix parameter handling for incompatible methods")
        print("4. Re-run tests to verify improvements")
        print("5. Add more test cases for comprehensive coverage")
