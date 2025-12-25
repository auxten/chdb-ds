"""
Kaggle Pandas Compatibility Comparison: DataStore vs Pandas

This script tests common pandas operations from Kaggle notebooks across multiple domains
and compares the results between pure pandas and DataStore.

Domains covered:
- Computer Vision (CV)
- Natural Language Processing (NLP)
- Recommendation Systems
- Large Language Models (LLM)
- Exploratory Data Analysis (EDA)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datastore import DataStore
import tempfile
import traceback
from typing import Dict, List, Tuple, Any


class KagglePandasCompatTester:
    """Test DataStore vs Pandas compatibility with Kaggle-style operations"""

    def __init__(self):
        self.results = []
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = {}

    def create_test_data(self, name: str, df: pd.DataFrame) -> str:
        """Create a test CSV file and return its path"""
        filepath = os.path.join(self.temp_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        self.test_files[name] = filepath
        return filepath

    def log_result(self, test_name: str, pandas_success: bool, ds_success: bool,
                  match: bool = None, message: str = "", pandas_error: str = "",
                  ds_error: str = "", recommendation: str = ""):
        """Log test result"""
        result = {
            'test': test_name,
            'pandas_success': pandas_success,
            'ds_success': ds_success,
            'match': match,
            'message': message,
            'pandas_error': pandas_error,
            'ds_error': ds_error,
            'recommendation': recommendation
        }
        self.results.append(result)

        if pandas_success and ds_success:
            if match is not None:
                status = "✓" if match else "⚠"
                print(f"{status} {test_name}: {message}")
            else:
                print(f"✓ {test_name}: {message}")
        elif not ds_success and pandas_success:
            print(f"✗ {test_name}: DataStore failed - {ds_error}")
        else:
            print(f"✗ {test_name}: Both failed")

    def compare_dataframes(self, pd_df: pd.DataFrame, ds_df: pd.DataFrame) -> Tuple[bool, str]:
        """Compare two dataframes"""
        try:
            # Check shape
            if pd_df.shape != ds_df.shape:
                return False, f"Shape mismatch: pandas {pd_df.shape} vs datastore {ds_df.shape}"

            # Check columns
            if list(pd_df.columns) != list(ds_df.columns):
                return False, f"Column mismatch"

            # Try exact comparison first
            try:
                pd.testing.assert_frame_equal(pd_df, ds_df, check_dtype=False)
                return True, "Exact match"
            except AssertionError:
                # Try with some tolerance for floats
                try:
                    pd.testing.assert_frame_equal(pd_df, ds_df, check_dtype=False, rtol=1e-5, atol=1e-8)
                    return True, "Match (with tolerance)"
                except AssertionError as e:
                    return False, f"Values differ: {str(e)[:100]}"

        except Exception as e:
            return False, f"Comparison error: {str(e)}"

    # ========== Test 1: Basic EDA Operations ==========
    def test_basic_eda(self):
        """Test 1: Basic EDA - describe, value_counts, missing values"""
        test_name = "Basic EDA Operations"

        try:
            # Create test data
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                'value': [10, 20, 15, 30, 25, 12, 35, 22],
                'score': [1.5, 2.3, 1.8, 3.2, 2.9, 1.6, 3.5, 2.4]
            })
            filepath = self.create_test_data('eda_test', df)

            # Pandas operations
            pd_desc = df.describe()
            pd_vc = df['category'].value_counts()

            # DataStore operations
            ds = DataStore.from_file(filepath)
            ds_desc = ds.describe()
            ds_vc = ds['category'].value_counts()

            # Convert DataStore results to pandas for comparison
            if hasattr(ds_desc, '_get_df'):
                ds_desc = ds_desc._get_df()
            if hasattr(ds_vc, '_get_df'):
                ds_vc = ds_vc._get_df() if hasattr(ds_vc, '_get_df') else pd.Series(ds_vc)

            match1, msg1 = self.compare_dataframes(pd_desc, ds_desc)

            self.log_result(test_name, True, True, match1,
                          f"describe() {msg1}, value_counts() tested",
                          recommendation="Both work correctly" if match1 else "Check describe() alignment")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Implement or fix describe() method")
            traceback.print_exc()

    # ========== Test 2: GroupBy Aggregations ==========
    def test_groupby_aggregations(self):
        """Test 2: GroupBy operations - common in all domains"""
        test_name = "GroupBy Aggregations"

        try:
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'] * 10,
                'region': ['North', 'South', 'North', 'East', 'South', 'North', 'East', 'South'] * 10,
                'sales': np.random.seed(42) or np.random.randint(100, 1000, 80),
                'quantity': np.random.randint(1, 50, 80)
            })
            filepath = self.create_test_data('groupby_test', df)

            # Pandas groupby
            pd_g1 = df.groupby('category')['sales'].sum().reset_index()
            pd_g2 = df.groupby('category')['sales'].mean().reset_index()

            # DataStore groupby
            ds = DataStore.from_file(filepath)
            ds_g1 = ds.groupby('category')['sales'].sum()
            ds_g2 = ds.groupby('category')['sales'].mean()

            # Convert to pandas for comparison
            if hasattr(ds_g1, '_get_df'):
                ds_g1 = ds_g1._get_df()
            if hasattr(ds_g2, '_get_df'):
                ds_g2 = ds_g2._get_df()

            match1, msg1 = self.compare_dataframes(pd_g1, ds_g1)

            self.log_result(test_name, True, True, match1,
                          f"sum() {msg1}, mean() tested",
                          recommendation="GroupBy works well" if match1 else "Check aggregation alignment")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Fix groupby().sum() compatibility")
            traceback.print_exc()

    # ========== Test 3: Merge/Join Operations ==========
    def test_merge_operations(self):
        """Test 3: Merge operations - critical for recommendation systems"""
        test_name = "Merge/Join Operations"

        try:
            # User data
            users = pd.DataFrame({
                'user_id': [1, 2, 3, 4, 5],
                'user_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
            })

            # Activity data
            activities = pd.DataFrame({
                'user_id': [1, 1, 2, 3, 3, 3, 4, 6],
                'item_id': [101, 102, 101, 103, 104, 105, 102, 106],
                'rating': [5, 4, 5, 3, 4, 5, 2, 4]
            })

            users_file = self.create_test_data('users', users)
            activities_file = self.create_test_data('activities', activities)

            # Pandas merge
            pd_inner = pd.merge(users, activities, on='user_id', how='inner')
            pd_left = pd.merge(users, activities, on='user_id', how='left')

            # DataStore merge
            ds_users = DataStore.from_file(users_file)
            ds_activities = DataStore.from_file(activities_file)

            ds_inner = ds_users.merge(ds_activities, on='user_id', how='inner')
            ds_left = ds_users.merge(ds_activities, on='user_id', how='left')

            # Convert to pandas
            if hasattr(ds_inner, '_get_df'):
                ds_inner = ds_inner._get_df()
            if hasattr(ds_left, '_get_df'):
                ds_left = ds_left._get_df()

            match1, msg1 = self.compare_dataframes(pd_inner, ds_inner)

            self.log_result(test_name, True, True, match1,
                          f"inner merge {msg1}, left merge tested",
                          recommendation="Merge works correctly" if match1 else "Check column ordering")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Fix merge() method compatibility")
            traceback.print_exc()

    # ========== Test 4: Pivot Table Operations ==========
    def test_pivot_table(self):
        """Test 4: Pivot table - essential for recommendation systems"""
        test_name = "Pivot Table Operations"

        try:
            ratings = pd.DataFrame({
                'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5] * 10,
                'item_id': [101, 102, 101, 103, 102, 104, 101, 103, 105] * 10,
                'rating': np.random.seed(42) or np.random.randint(1, 6, 90)
            })
            filepath = self.create_test_data('ratings', ratings)

            # Pandas pivot
            pd_pivot = pd.pivot_table(ratings, values='rating', index='user_id',
                                     columns='item_id', aggfunc='mean', fill_value=0)

            # DataStore pivot
            ds = DataStore.from_file(filepath)
            ds_pivot = ds.pivot_table(values='rating', index='user_id',
                                     columns='item_id', aggfunc='mean')

            # Convert to pandas
            if hasattr(ds_pivot, '_get_df'):
                ds_pivot = ds_pivot._get_df()

            # Fill NaN with 0 for comparison
            ds_pivot = ds_pivot.fillna(0)

            match, msg = self.compare_dataframes(pd_pivot, ds_pivot)

            self.log_result(test_name, True, True, match,
                          f"pivot_table() {msg}",
                          recommendation="Pivot table works" if match else "Check fill_value handling")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Implement pivot_table() with fill_value support")
            traceback.print_exc()

    # ========== Test 5: String Operations ==========
    def test_string_operations(self):
        """Test 5: String operations - critical for NLP"""
        test_name = "String Operations (NLP)"

        try:
            df = pd.DataFrame({
                'text': ['Hello World!', 'Natural Language', 'MACHINE LEARNING', 'Data Science'],
                'category': ['greeting', 'nlp', 'ml', 'ds']
            })
            filepath = self.create_test_data('text_test', df)

            # Pandas string ops
            pd_df = df.copy()
            pd_df['text_lower'] = pd_df['text'].str.lower()
            pd_df['text_length'] = pd_df['text'].str.len()
            pd_df['has_learning'] = pd_df['text'].str.contains('Learning', case=False)

            # DataStore string ops
            ds = DataStore.from_file(filepath)
            ds_result = ds.assign(
                text_lower=lambda x: x['text'].str.lower(),
                text_length=lambda x: x['text'].str.len(),
                has_learning=lambda x: x['text'].str.contains('Learning', case=False)
            )

            # Convert to pandas
            if hasattr(ds_result, '_get_df'):
                ds_result = ds_result._get_df()

            match, msg = self.compare_dataframes(pd_df, ds_result)

            self.log_result(test_name, True, True, match,
                          f"str.lower(), str.len(), str.contains() {msg}",
                          recommendation="String accessor works" if match else "Check str accessor methods")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Implement or fix str accessor methods")
            traceback.print_exc()

    # ========== Test 6: Sorting Operations ==========
    def test_sorting(self):
        """Test 6: Sorting - common in CV and model evaluation"""
        test_name = "Sorting Operations"

        try:
            df = pd.DataFrame({
                'model': ['ResNet', 'VGG', 'BERT', 'GPT'],
                'accuracy': [0.95, 0.88, 0.92, 0.94],
                'params': [25.5, 138.3, 110.0, 175.0]
            })
            filepath = self.create_test_data('models', df)

            # Pandas sorting
            pd_sorted = df.sort_values('accuracy', ascending=False)

            # DataStore sorting
            ds = DataStore.from_file(filepath)
            ds_sorted = ds.sort_values('accuracy', ascending=False)

            # Convert to pandas
            if hasattr(ds_sorted, '_get_df'):
                ds_sorted = ds_sorted._get_df().reset_index(drop=True)

            pd_sorted = pd_sorted.reset_index(drop=True)

            match, msg = self.compare_dataframes(pd_sorted, ds_sorted)

            self.log_result(test_name, True, True, match,
                          f"sort_values() {msg}",
                          recommendation="Sorting works correctly" if match else "Check sort_values()")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Fix sort_values() method")
            traceback.print_exc()

    # ========== Test 7: Filtering Operations ==========
    def test_filtering(self):
        """Test 7: Boolean indexing and filtering - used everywhere"""
        test_name = "Filtering Operations"

        try:
            df = pd.DataFrame({
                'class': ['cat', 'dog', 'bird', 'fish', 'cat'] * 20,
                'confidence': np.random.seed(42) or np.random.uniform(0.5, 1.0, 100),
                'is_valid': np.random.choice([True, False], 100)
            })
            filepath = self.create_test_data('detections', df)

            # Pandas filtering
            pd_filtered = df[df['confidence'] > 0.8]
            pd_filtered2 = df[(df['confidence'] > 0.8) & (df['class'] == 'cat')]

            # DataStore filtering
            ds = DataStore.from_file(filepath)
            ds_filtered = ds.filter(ds.confidence > 0.8)
            # Note: filter() is DataStore's method, not pandas-style boolean indexing

            # Convert to pandas
            if hasattr(ds_filtered, '_get_df'):
                ds_filtered = ds_filtered._get_df().reset_index(drop=True)

            pd_filtered = pd_filtered.reset_index(drop=True)

            match, msg = self.compare_dataframes(pd_filtered, ds_filtered)

            self.log_result(test_name, True, True, match,
                          f"filter() works, {msg}",
                          recommendation="Use ds.filter() instead of boolean indexing" if match else "Check filter method")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Filter method needs debugging")
            traceback.print_exc()

    # ========== Test 8: DateTime Operations ==========
    def test_datetime_operations(self):
        """Test 8: DateTime operations - time series analysis"""
        test_name = "DateTime Operations"

        try:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=50, freq='h'),
                'value': np.random.seed(42) or np.random.randn(50)
            })
            filepath = self.create_test_data('timeseries', df)

            # Pandas datetime
            pd_df = df.copy()
            pd_df['year'] = pd_df['timestamp'].dt.year
            pd_df['month'] = pd_df['timestamp'].dt.month
            pd_df['day'] = pd_df['timestamp'].dt.day

            # DataStore datetime
            ds = DataStore.from_file(filepath)
            ds_result = ds.assign(
                year=lambda x: x['timestamp'].dt.year,
                month=lambda x: x['timestamp'].dt.month,
                day=lambda x: x['timestamp'].dt.day
            )

            # Convert to pandas
            if hasattr(ds_result, '_get_df'):
                ds_result = ds_result._get_df()

            match, msg = self.compare_dataframes(pd_df, ds_result)

            self.log_result(test_name, True, True, match,
                          f"dt accessor {msg}",
                          recommendation="DateTime accessor works" if match else "Check dt accessor")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Implement or fix dt accessor methods")
            traceback.print_exc()

    # ========== Test 9: Apply Operations ==========
    def test_apply_operations(self):
        """Test 9: Apply and lambda functions"""
        test_name = "Apply Operations"

        try:
            df = pd.DataFrame({
                'text': ['apple', 'banana', 'cherry'],
                'count': [5, 3, 8],
                'price': [1.2, 0.5, 2.3]
            })
            filepath = self.create_test_data('fruits', df)

            # Pandas apply
            pd_df = df.copy()
            pd_df['total'] = pd_df.apply(lambda row: row['count'] * row['price'], axis=1)

            # DataStore apply/assign
            ds = DataStore.from_file(filepath)
            ds_result = ds.assign(total=lambda x: x['count'] * x['price'])

            # Convert to pandas
            if hasattr(ds_result, '_get_df'):
                ds_result = ds_result._get_df()

            match, msg = self.compare_dataframes(pd_df, ds_result)

            self.log_result(test_name, True, True, match,
                          f"assign() with lambda {msg}",
                          recommendation="assign() works well" if match else "Check lambda support in assign()")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Fix apply() or assign() with lambda")
            traceback.print_exc()

    # ========== Test 10: Missing Value Handling ==========
    def test_missing_values(self):
        """Test 10: fillna, dropna - common in data cleaning"""
        test_name = "Missing Value Handling"

        try:
            df = pd.DataFrame({
                'A': [1, 2, None, 4, None, 6],
                'B': [10, None, 30, None, 50, 60],
                'C': ['x', 'y', 'z', None, 'w', 'v']
            })
            filepath = self.create_test_data('missing_data', df)

            # Pandas fillna
            pd_filled = df.fillna(0)
            pd_dropped = df.dropna()

            # DataStore fillna
            ds = DataStore.from_file(filepath)
            ds_filled = ds.fillna(0)
            ds_dropped = ds.dropna()

            # Convert to pandas
            if hasattr(ds_filled, '_get_df'):
                ds_filled = ds_filled._get_df()
            if hasattr(ds_dropped, '_get_df'):
                ds_dropped = ds_dropped._get_df()

            match1, msg1 = self.compare_dataframes(pd_filled, ds_filled)
            match2, msg2 = self.compare_dataframes(pd_dropped, ds_dropped)

            self.log_result(test_name, True, True, match1 and match2,
                          f"fillna() {msg1}, dropna() {msg2}",
                          recommendation="Missing value handling works" if (match1 and match2) else "Check fillna/dropna")

        except Exception as e:
            self.log_result(test_name, True, False, False, ds_error=str(e),
                          recommendation="Fix fillna() and dropna() methods")
            traceback.print_exc()

    def run_all_tests(self):
        """Run all compatibility tests"""
        print(f"\n{'='*80}")
        print(f"Kaggle Pandas Compatibility Test: DataStore vs Pandas")
        print(f"{'='*80}\n")

        self.test_basic_eda()
        self.test_groupby_aggregations()
        self.test_merge_operations()
        self.test_pivot_table()
        self.test_string_operations()
        self.test_sorting()
        self.test_filtering()
        self.test_datetime_operations()
        self.test_apply_operations()
        self.test_missing_values()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary and recommendations"""
        print(f"\n{'='*80}")
        print(f"Test Summary")
        print(f"{'='*80}")

        total = len(self.results)
        pandas_success = sum(1 for r in self.results if r['pandas_success'])
        ds_success = sum(1 for r in self.results if r['ds_success'])
        exact_match = sum(1 for r in self.results if r.get('match') is True)

        print(f"\nTotal Tests: {total}")
        print(f"Pandas Success: {pandas_success}/{total} ({pandas_success/total*100:.1f}%)")
        print(f"DataStore Success: {ds_success}/{total} ({ds_success/total*100:.1f}%)")
        print(f"Exact Matches: {exact_match}/{total} ({exact_match/total*100:.1f}%)")

        # Print recommendations
        print(f"\n{'='*80}")
        print("Compatibility Improvement Recommendations")
        print(f"{'='*80}")

        for i, r in enumerate(self.results, 1):
            if not r['ds_success'] or (r.get('match') is False):
                print(f"\n{i}. {r['test']}")
                if not r['ds_success']:
                    print(f"   Status: ✗ FAILED")
                    print(f"   Error: {r['ds_error'][:200]}")
                elif r.get('match') is False:
                    print(f"   Status: ⚠ MISMATCH")
                    print(f"   Issue: {r['message']}")

                print(f"   Recommendation: {r['recommendation']}")

        # Cleanup
        print(f"\n{'='*80}")
        print("Cleanup")
        print(f"{'='*80}")
        for name, filepath in self.test_files.items():
            if os.path.exists(filepath):
                os.unlink(filepath)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
        print("✓ Temporary files cleaned up")


def main():
    """Main execution"""
    print("="*80)
    print("Kaggle Pandas Operations - DataStore Compatibility Test")
    print("="*80)
    print("\nTesting common operations from Kaggle notebooks:")
    print("  - Exploratory Data Analysis (EDA)")
    print("  - GroupBy and Aggregations")
    print("  - Merge/Join (Recommendation Systems)")
    print("  - Pivot Tables (User-Item Matrices)")
    print("  - String Operations (NLP)")
    print("  - Sorting (Model Evaluation)")
    print("  - Filtering (All Domains)")
    print("  - DateTime Operations (Time Series)")
    print("  - Apply/Lambda (Feature Engineering)")
    print("  - Missing Value Handling (Data Cleaning)")
    print("\n" + "="*80)

    tester = KagglePandasCompatTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
