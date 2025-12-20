#!/usr/bin/env python3
"""
Pandas Compatibility Test Script
==================================

This script tests the compatibility between pandas and datastore by running
common pandas operations extracted from popular Jupyter notebooks and comparing
the results.

Usage:
    python pandas_compatibility_test.py

The script will generate a compatibility report showing:
- Which operations work the same in both libraries
- Which operations have differences
- Which operations are not supported by datastore
- Recommendations for improvements
"""

import json
import traceback
import sys
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class PandasCompatibilityTester:
    """Test pandas compatibility with datastore"""

    def __init__(self):
        self.test_results = []
        self.pandas_df = None
        self.datastore_df = None

    def setup_test_data(self):
        """Create test data for compatibility testing"""
        # Import both pandas and datastore
        import pandas as pd
        import numpy as np

        # Create comprehensive test dataset
        self.test_data = {
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None, 'Grace'],
            'age': [25, 30, 35, None, 28, 32, 45, 27],
            'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Sydney', 'Toronto', None],
            'salary': [50000, 60000, None, 75000, 55000, 80000, 70000, 52000],
            'department': ['HR', 'IT', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance'],
            'hire_date': ['2020-01-15', '2019-06-20', '2018-03-10', '2021-09-05',
                         '2020-11-12', '2019-02-28', '2017-08-15', '2021-04-30'],
            'performance_score': [8.5, 7.2, 9.1, 6.8, 8.0, 9.5, 7.5, 8.3]
        }

        # Create pandas DataFrame
        self.pandas_df = pd.DataFrame(self.test_data)
        self.pandas_df['hire_date'] = pd.to_datetime(self.pandas_df['hire_date'])

        print("✓ Test data created successfully")
        return self.pandas_df

    def test_operation(self, category: str, description: str,
                      pandas_code: str, datastore_code: str = None) -> Dict[str, Any]:
        """
        Test a single operation in both pandas and datastore

        Args:
            category: Operation category (e.g., 'data_selection', 'aggregation')
            description: Human-readable description of the operation
            pandas_code: Code to execute with pandas
            datastore_code: Code to execute with datastore (if different from pandas_code)

        Returns:
            Dictionary with test results
        """
        if datastore_code is None:
            datastore_code = pandas_code

        result = {
            'category': category,
            'description': description,
            'pandas_code': pandas_code,
            'datastore_code': datastore_code,
            'pandas_success': False,
            'datastore_success': False,
            'pandas_result': None,
            'datastore_result': None,
            'pandas_error': None,
            'datastore_error': None,
            'results_match': False,
            'notes': []
        }

        # Test with pandas
        try:
            import pandas as pd
            import numpy as np
            df = self.pandas_df.copy()

            # Execute pandas code
            local_vars = {'pd': pd, 'np': np, 'df': df}
            exec(pandas_code, {}, local_vars)

            # Get result (could be in 'result' variable or last expression)
            if 'result' in local_vars:
                result['pandas_result'] = local_vars['result']
            else:
                # Try to evaluate as expression
                try:
                    result['pandas_result'] = eval(pandas_code, {}, local_vars)
                except:
                    result['pandas_result'] = "Operation completed (no return value)"

            result['pandas_success'] = True

        except Exception as e:
            result['pandas_error'] = f"{type(e).__name__}: {str(e)}"

        # Test with datastore (as pd)
        try:
            import datastore as pd
            import numpy as np

            # For datastore, we need to create DataFrame differently
            # Try to use datastore's pandas-compatible API
            df = pd.DataFrame(self.test_data)

            # Execute datastore code
            local_vars = {'pd': pd, 'np': np, 'df': df}
            exec(datastore_code, {}, local_vars)

            # Get result
            if 'result' in local_vars:
                result['datastore_result'] = local_vars['result']
            else:
                try:
                    result['datastore_result'] = eval(datastore_code, {}, local_vars)
                except:
                    result['datastore_result'] = "Operation completed (no return value)"

            result['datastore_success'] = True

        except Exception as e:
            result['datastore_error'] = f"{type(e).__name__}: {str(e)}"
            result['notes'].append(f"Datastore error: {str(e)}")

        # Compare results if both succeeded
        if result['pandas_success'] and result['datastore_success']:
            try:
                # Basic comparison - could be enhanced
                pandas_str = str(result['pandas_result'])
                datastore_str = str(result['datastore_result'])
                result['results_match'] = pandas_str == datastore_str

                if not result['results_match']:
                    result['notes'].append("Results differ between pandas and datastore")
            except:
                result['notes'].append("Could not compare results")

        return result

    def run_all_tests(self):
        """Run comprehensive compatibility tests based on notebook analysis"""

        print("\n" + "="*80)
        print("PANDAS COMPATIBILITY TEST SUITE")
        print("="*80 + "\n")

        # Setup test data
        self.setup_test_data()

        # ==================== DataFrame Creation Tests ====================
        print("\n[1/10] Testing DataFrame Creation Operations...")

        tests = [
            ('dataframe_creation', 'Create DataFrame from dict',
             "result = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})"),

            ('dataframe_creation', 'Create DataFrame with index',
             "result = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== Data Selection Tests ====================
        print("[2/10] Testing Data Selection Operations...")

        tests = [
            ('data_selection', 'Select single column',
             "result = df['name']"),

            ('data_selection', 'Select multiple columns',
             "result = df[['name', 'age']]"),

            ('data_selection', 'Select rows by slice',
             "result = df[:3]"),

            ('data_selection', 'Select with boolean indexing',
             "result = df[df['age'] > 30]"),

            ('data_selection', 'Select with loc',
             "result = df.loc[0:2, ['name', 'age']]"),

            ('data_selection', 'Select with iloc',
             "result = df.iloc[0:3, 0:2]"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== Data Cleaning Tests ====================
        print("[3/10] Testing Data Cleaning Operations...")

        tests = [
            ('data_cleaning', 'Drop NA values',
             "result = df.dropna()"),

            ('data_cleaning', 'Fill NA values',
             "result = df.fillna(0)"),

            ('data_cleaning', 'Drop duplicates',
             "result = df.drop_duplicates()"),

            ('data_cleaning', 'Replace values',
             "result = df.replace('HR', 'Human Resources')"),

            ('data_cleaning', 'Drop column',
             "result = df.drop('salary', axis=1)"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== Statistics Tests ====================
        print("[4/10] Testing Statistical Operations...")

        tests = [
            ('statistics', 'Compute mean',
             "result = df['age'].mean()"),

            ('statistics', 'Compute sum',
             "result = df['salary'].sum()"),

            ('statistics', 'Compute median',
             "result = df['age'].median()"),

            ('statistics', 'Compute std',
             "result = df['age'].std()"),

            ('statistics', 'Describe DataFrame',
             "result = df.describe()"),

            ('statistics', 'Value counts',
             "result = df['department'].value_counts()"),

            ('statistics', 'Correlation',
             "result = df[['age', 'salary']].corr()"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== Data Transformation Tests ====================
        print("[5/10] Testing Data Transformation Operations...")

        tests = [
            ('data_transformation', 'Apply function to column',
             "result = df['age'].apply(lambda x: x * 2 if pd.notna(x) else x)"),

            ('data_transformation', 'Map values',
             "result = df['department'].map({'HR': 1, 'IT': 2, 'Finance': 3})"),

            ('data_transformation', 'Rename columns',
             "result = df.rename(columns={'name': 'employee_name'})"),

            ('data_transformation', 'Convert type (astype)',
             "result = df['age'].astype(str)"),

            ('data_transformation', 'Add new column',
             "df['bonus'] = df['salary'] * 0.1; result = df"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== Sorting Tests ====================
        print("[6/10] Testing Sorting Operations...")

        tests = [
            ('sorting', 'Sort by single column',
             "result = df.sort_values('age')"),

            ('sorting', 'Sort by multiple columns',
             "result = df.sort_values(['department', 'age'])"),

            ('sorting', 'Sort descending',
             "result = df.sort_values('salary', ascending=False)"),

            ('sorting', 'Sort index',
             "result = df.sort_index()"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== Aggregation Tests ====================
        print("[7/10] Testing Aggregation Operations...")

        tests = [
            ('aggregation', 'GroupBy with single aggregation',
             "result = df.groupby('department')['salary'].mean()"),

            ('aggregation', 'GroupBy with multiple aggregations',
             "result = df.groupby('department').agg({'salary': 'mean', 'age': 'max'})"),

            ('aggregation', 'GroupBy with sum',
             "result = df.groupby('department')['salary'].sum()"),

            ('aggregation', 'GroupBy with count',
             "result = df.groupby('department').size()"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== String Operations Tests ====================
        print("[8/10] Testing String Operations...")

        tests = [
            ('string_operations', 'String upper',
             "result = df['name'].str.upper()"),

            ('string_operations', 'String lower',
             "result = df['city'].str.lower()"),

            ('string_operations', 'String contains',
             "result = df['name'].str.contains('a', na=False)"),

            ('string_operations', 'String length',
             "result = df['name'].str.len()"),

            ('string_operations', 'String replace',
             "result = df['city'].str.replace('York', 'Amsterdam')"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== Merging Tests ====================
        print("[9/10] Testing Merge/Join Operations...")

        tests = [
            ('merging', 'Concat DataFrames',
             """
df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
result = pd.concat([df1, df2])
"""),

            ('merging', 'Merge DataFrames',
             """
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
result = pd.merge(df1, df2, on='key', how='inner')
"""),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        # ==================== DateTime Operations Tests ====================
        print("[10/10] Testing DateTime Operations...")

        tests = [
            ('datetime_operations', 'Convert to datetime',
             "result = pd.to_datetime(df['hire_date'])"),

            ('datetime_operations', 'Extract year from date',
             "result = df['hire_date'].dt.year"),

            ('datetime_operations', 'Extract month from date',
             "result = df['hire_date'].dt.month"),

            ('datetime_operations', 'Date formatting',
             "result = df['hire_date'].dt.strftime('%Y-%m')"),
        ]

        for test in tests:
            self.test_results.append(self.test_operation(*test))

        print("\n✓ All tests completed!\n")

    def generate_report(self) -> str:
        """Generate a detailed compatibility report"""

        report = []
        report.append("\n" + "="*80)
        report.append("PANDAS COMPATIBILITY REPORT")
        report.append("="*80 + "\n")

        # Summary statistics
        total_tests = len(self.test_results)
        pandas_success = sum(1 for r in self.test_results if r['pandas_success'])
        datastore_success = sum(1 for r in self.test_results if r['datastore_success'])
        both_success = sum(1 for r in self.test_results if r['pandas_success'] and r['datastore_success'])
        results_match = sum(1 for r in self.test_results if r['results_match'])

        report.append(f"Total Tests: {total_tests}")
        report.append(f"Pandas Success Rate: {pandas_success}/{total_tests} ({100*pandas_success/total_tests:.1f}%)")
        report.append(f"DataStore Success Rate: {datastore_success}/{total_tests} ({100*datastore_success/total_tests:.1f}%)")
        report.append(f"Both Libraries Succeeded: {both_success}/{total_tests} ({100*both_success/total_tests:.1f}%)")
        match_rate = (100*results_match/both_success if both_success else 0)
        report.append(f"Results Match (when both succeed): {results_match}/{both_success} ({match_rate:.1f}%)")
        report.append("\n" + "="*80 + "\n")

        # Group results by category
        from collections import defaultdict
        by_category = defaultdict(list)
        for result in self.test_results:
            by_category[result['category']].append(result)

        # Detailed results by category
        for category, results in sorted(by_category.items()):
            report.append(f"\n{category.upper().replace('_', ' ')}")
            report.append("-" * 80)

            for i, result in enumerate(results, 1):
                report.append(f"\n{i}. {result['description']}")
                report.append(f"   Code: {result['pandas_code'][:60]}...")

                status = []
                if result['pandas_success']:
                    status.append("✓ Pandas")
                else:
                    status.append(f"✗ Pandas ({result['pandas_error']})")

                if result['datastore_success']:
                    status.append("✓ DataStore")
                else:
                    status.append(f"✗ DataStore ({result['datastore_error']})")

                report.append(f"   Status: {' | '.join(status)}")

                if result['pandas_success'] and result['datastore_success']:
                    if result['results_match']:
                        report.append("   Results: ✓ Match")
                    else:
                        report.append("   Results: ✗ Differ")

                if result['notes']:
                    report.append(f"   Notes: {'; '.join(result['notes'])}")

        # Recommendations section
        report.append("\n" + "="*80)
        report.append("COMPATIBILITY IMPROVEMENT RECOMMENDATIONS")
        report.append("="*80 + "\n")

        recommendations = self.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")

        return "\n".join(report)

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving datastore compatibility"""

        recommendations = []

        # Analyze failures
        failed_operations = [r for r in self.test_results if not r['datastore_success']]

        # Group by error type
        from collections import Counter
        error_types = Counter()
        for result in failed_operations:
            if result['datastore_error']:
                error_msg = result['datastore_error'].split(':')[0]
                error_types[error_msg] += 1

        # Generate specific recommendations based on failures
        categories_with_issues = set(r['category'] for r in failed_operations)

        if 'dataframe_creation' in categories_with_issues:
            recommendations.append(
                "Improve DataFrame creation API to match pandas initialization patterns more closely"
            )

        if 'data_selection' in categories_with_issues:
            recommendations.append(
                "Enhance data selection operations (loc, iloc, boolean indexing) to match pandas behavior"
            )

        if 'data_cleaning' in categories_with_issues:
            recommendations.append(
                "Implement missing data cleaning methods (dropna, fillna, drop_duplicates) with pandas-compatible signatures"
            )

        if 'statistics' in categories_with_issues:
            recommendations.append(
                "Add statistical methods (mean, median, std, describe, value_counts) with identical behavior to pandas"
            )

        if 'data_transformation' in categories_with_issues:
            recommendations.append(
                "Implement apply, map, and other transformation methods with support for lambda functions"
            )

        if 'sorting' in categories_with_issues:
            recommendations.append(
                "Enhance sort_values and sort_index methods to support all pandas parameters"
            )

        if 'aggregation' in categories_with_issues:
            recommendations.append(
                "Improve groupby operations to support all pandas aggregation functions and multiple aggregations"
            )

        if 'string_operations' in categories_with_issues:
            recommendations.append(
                "Implement .str accessor with all common string methods (upper, lower, contains, replace, len)"
            )

        if 'merging' in categories_with_issues:
            recommendations.append(
                "Add merge, join, and concat operations with pandas-compatible parameters and behavior"
            )

        if 'datetime_operations' in categories_with_issues:
            recommendations.append(
                "Implement .dt accessor with date/time extraction and formatting methods"
            )

        # Error-specific recommendations
        if 'AttributeError' in error_types:
            recommendations.append(
                "Add missing methods and attributes that are commonly used in pandas but not yet implemented in datastore"
            )

        if 'TypeError' in error_types:
            recommendations.append(
                "Review method signatures to ensure they accept the same parameter types as pandas"
            )

        if 'NotImplementedError' in error_types:
            recommendations.append(
                "Prioritize implementation of operations that are currently marked as not implemented"
            )

        # General recommendations
        recommendations.append(
            "Add comprehensive unit tests covering all pandas-compatible operations"
        )

        recommendations.append(
            "Create a compatibility matrix documenting which pandas operations are supported and their limitations"
        )

        recommendations.append(
            "Implement lazy evaluation where possible to maintain compatibility with pandas' eager evaluation model"
        )

        return recommendations

    def save_detailed_results(self, filename: str = "pandas_compatibility_results.json"):
        """Save detailed test results to JSON file"""

        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\n✓ Detailed results saved to {filename}")


def main():
    """Main entry point"""

    tester = PandasCompatibilityTester()

    try:
        # Run all tests
        tester.run_all_tests()

        # Generate and display report
        report = tester.generate_report()
        print(report)

        # Save detailed results
        tester.save_detailed_results()

        # Also save report to file
        with open('pandas_compatibility_report.txt', 'w') as f:
            f.write(report)
        print("✓ Report saved to pandas_compatibility_report.txt")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
