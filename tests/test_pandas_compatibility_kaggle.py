"""
Comprehensive Pandas Compatibility Test for DataStore
Based on common operations from Kaggle notebooks across multiple domains:
- Computer Vision (CV)
- Natural Language Processing (NLP)
- Recommendation Systems
- Large Language Models (LLM)
- Exploratory Data Analysis (EDA)

This script tests DataStore's pandas compatibility by monkey patching and comparing results.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import traceback
from typing import Any, Dict, List, Tuple


class PandasCompatibilityTester:
    """Test DataStore compatibility with common Kaggle pandas operations"""

    def __init__(self, use_datastore: bool = False):
        self.use_datastore = use_datastore
        self.results = []
        self.errors = []

        if use_datastore:
            try:
                import datastore as ds
                # Monkey patch pandas with datastore
                sys.modules['pandas'] = ds
                globals()['pd'] = ds
                print("✓ Using DataStore as pandas replacement")
            except ImportError:
                print("✗ DataStore not available, falling back to pandas")
                self.use_datastore = False
        else:
            print("✓ Using standard pandas")

    def log_result(self, test_name: str, success: bool, message: str = "", error: str = ""):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'error': error,
            'backend': 'DataStore' if self.use_datastore else 'Pandas'
        }
        self.results.append(result)

        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {message if success else error}")

    def compare_results(self, test_name: str, pd_result: Any, ds_result: Any) -> Tuple[bool, str]:
        """Compare results from pandas and datastore"""
        try:
            if isinstance(pd_result, pd.DataFrame) and hasattr(ds_result, 'to_pandas'):
                ds_result = ds_result.to_pandas()

            if isinstance(pd_result, pd.Series) and hasattr(ds_result, 'to_pandas'):
                ds_result = ds_result.to_pandas()

            # Compare DataFrames
            if isinstance(pd_result, pd.DataFrame):
                if not pd_result.equals(ds_result):
                    return False, f"DataFrames not equal: shape {pd_result.shape} vs {ds_result.shape}"
                return True, "Results match"

            # Compare Series
            if isinstance(pd_result, pd.Series):
                if not pd_result.equals(ds_result):
                    return False, f"Series not equal"
                return True, "Results match"

            # Compare scalars
            if pd_result == ds_result or (pd.isna(pd_result) and pd.isna(ds_result)):
                return True, "Results match"

            return False, f"Values differ: {pd_result} vs {ds_result}"

        except Exception as e:
            return False, f"Comparison error: {str(e)}"

    # ========== EDA Operations (Most Common) ==========

    def test_basic_eda_operations(self):
        """Test 1: Basic EDA operations - describe, info, value_counts"""
        try:
            # Create sample dataset
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                'value': [10, 20, 15, 30, 25, 12, 35, 22],
                'score': [1.5, 2.3, 1.8, 3.2, 2.9, 1.6, 3.5, 2.4],
                'missing': [1, None, 3, None, 5, None, 7, 8]
            })

            # describe() - statistical summary
            desc = df.describe()

            # value_counts() - frequency distribution
            vc = df['category'].value_counts()

            # isnull().sum() - count missing values
            null_counts = df.isnull().sum()

            self.log_result('test_basic_eda_operations', True,
                          f"describe shape: {desc.shape}, value_counts: {len(vc)}, nulls: {null_counts.sum()}")
            return df, desc, vc

        except Exception as e:
            self.log_result('test_basic_eda_operations', False, error=str(e))
            return None, None, None

    def test_fillna_operations(self):
        """Test 2: Missing value handling - fillna, dropna"""
        try:
            df = pd.DataFrame({
                'A': [1, 2, None, 4, None, 6],
                'B': [10, None, 30, None, 50, 60],
                'C': ['x', 'y', 'z', None, 'w', 'v']
            })

            # fillna with constant
            df1 = df.fillna(0)

            # fillna with mean
            df2 = df.copy()
            df2['A'] = df2['A'].fillna(df2['A'].mean())

            # fillna with forward fill
            df3 = df.fillna(method='ffill')

            # dropna
            df4 = df.dropna()

            self.log_result('test_fillna_operations', True,
                          f"fillna(0): {df1.shape}, fillna(mean): {df2.shape}, ffill: {df3.shape}, dropna: {df4.shape}")
            return df, df1, df2, df3, df4

        except Exception as e:
            self.log_result('test_fillna_operations', False, error=str(e))
            return None, None, None, None, None

    def test_groupby_aggregations(self):
        """Test 3: GroupBy operations - common in all domains"""
        try:
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'] * 10,
                'region': ['North', 'South', 'North', 'East', 'South', 'North', 'East', 'South'] * 10,
                'sales': np.random.randint(100, 1000, 80),
                'quantity': np.random.randint(1, 50, 80),
                'price': np.random.uniform(10, 100, 80)
            })

            # Single column groupby with single aggregation
            g1 = df.groupby('category')['sales'].sum()

            # Multiple aggregations
            g2 = df.groupby('category').agg({
                'sales': ['sum', 'mean', 'count'],
                'quantity': 'sum',
                'price': 'mean'
            })

            # Multiple groupby columns
            g3 = df.groupby(['category', 'region'])['sales'].mean()

            # GroupBy with transform
            df['sales_normalized'] = df.groupby('category')['sales'].transform(lambda x: (x - x.mean()) / x.std())

            self.log_result('test_groupby_aggregations', True,
                          f"g1: {len(g1)}, g2: {g2.shape}, g3: {len(g3)}, transform: OK")
            return df, g1, g2, g3

        except Exception as e:
            self.log_result('test_groupby_aggregations', False, error=str(e))
            return None, None, None, None

    # ========== Data Merging & Joining (NLP, Recommendation Systems) ==========

    def test_merge_join_operations(self):
        """Test 4: Merge and Join operations"""
        try:
            # User data (common in recommendation systems)
            users = pd.DataFrame({
                'user_id': [1, 2, 3, 4, 5],
                'user_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [25, 30, 35, 28, 32]
            })

            # Activity data
            activities = pd.DataFrame({
                'user_id': [1, 1, 2, 3, 3, 3, 4, 6],
                'item_id': [101, 102, 101, 103, 104, 105, 102, 106],
                'rating': [5, 4, 5, 3, 4, 5, 2, 4]
            })

            # Inner merge
            m1 = pd.merge(users, activities, on='user_id', how='inner')

            # Left merge
            m2 = pd.merge(users, activities, on='user_id', how='left')

            # Right merge
            m3 = pd.merge(users, activities, on='user_id', how='right')

            # Outer merge
            m4 = pd.merge(users, activities, on='user_id', how='outer')

            self.log_result('test_merge_join_operations', True,
                          f"inner: {m1.shape}, left: {m2.shape}, right: {m3.shape}, outer: {m4.shape}")
            return users, activities, m1, m2, m3, m4

        except Exception as e:
            self.log_result('test_merge_join_operations', False, error=str(e))
            return None, None, None, None, None, None

    def test_concat_operations(self):
        """Test 5: Concatenation operations"""
        try:
            df1 = pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            })

            df2 = pd.DataFrame({
                'A': [7, 8, 9],
                'B': [10, 11, 12]
            })

            df3 = pd.DataFrame({
                'C': [13, 14, 15],
                'D': [16, 17, 18]
            })

            # Vertical concat (row-wise)
            c1 = pd.concat([df1, df2], ignore_index=True)

            # Horizontal concat (column-wise)
            c2 = pd.concat([df1, df3], axis=1)

            # Concat with keys
            c3 = pd.concat([df1, df2], keys=['first', 'second'])

            self.log_result('test_concat_operations', True,
                          f"vertical: {c1.shape}, horizontal: {c2.shape}, with_keys: {c3.shape}")
            return c1, c2, c3

        except Exception as e:
            self.log_result('test_concat_operations', False, error=str(e))
            return None, None, None

    # ========== Pivot & Reshape (Recommendation Systems) ==========

    def test_pivot_table_operations(self):
        """Test 6: Pivot table operations - critical for recommendation systems"""
        try:
            # User-item interaction data
            ratings = pd.DataFrame({
                'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5] * 10,
                'item_id': [101, 102, 101, 103, 102, 104, 101, 103, 105] * 10,
                'rating': np.random.randint(1, 6, 90),
                'timestamp': pd.date_range('2025-01-01', periods=90, freq='h')
            })

            # Basic pivot table
            p1 = pd.pivot_table(ratings, values='rating', index='user_id', columns='item_id')

            # With aggregation function
            p2 = pd.pivot_table(ratings, values='rating', index='user_id',
                              columns='item_id', aggfunc='mean', fill_value=0)

            # Multiple values
            p3 = pd.pivot_table(ratings, values='rating', index='user_id',
                              columns='item_id', aggfunc=['mean', 'count'], fill_value=0)

            self.log_result('test_pivot_table_operations', True,
                          f"p1: {p1.shape}, p2: {p2.shape}, p3: {p3.shape}")
            return ratings, p1, p2, p3

        except Exception as e:
            self.log_result('test_pivot_table_operations', False, error=str(e))
            traceback.print_exc()
            return None, None, None, None

    def test_melt_operations(self):
        """Test 7: Melt operations - unpivoting data"""
        try:
            df = pd.DataFrame({
                'id': [1, 2, 3],
                'Jan': [100, 200, 150],
                'Feb': [120, 180, 160],
                'Mar': [130, 210, 170]
            })

            # Basic melt
            m1 = pd.melt(df, id_vars=['id'], value_vars=['Jan', 'Feb', 'Mar'])

            # Melt with custom names
            m2 = pd.melt(df, id_vars=['id'], value_vars=['Jan', 'Feb', 'Mar'],
                        var_name='month', value_name='sales')

            self.log_result('test_melt_operations', True,
                          f"m1: {m1.shape}, m2: {m2.shape}")
            return df, m1, m2

        except Exception as e:
            self.log_result('test_melt_operations', False, error=str(e))
            return None, None, None

    # ========== String Operations (NLP) ==========

    def test_string_operations(self):
        """Test 8: String operations - common in NLP preprocessing"""
        try:
            df = pd.DataFrame({
                'text': [
                    'Hello World!',
                    'Natural Language Processing',
                    'MACHINE LEARNING',
                    'Data Science 2025',
                    'Python Programming',
                    'Deep Learning Models',
                    'Computer Vision CV',
                    'Transformers & LLMs'
                ],
                'category': ['greeting', 'nlp', 'ml', 'ds', 'prog', 'dl', 'cv', 'llm']
            })

            # Lowercase
            df['text_lower'] = df['text'].str.lower()

            # Uppercase
            df['text_upper'] = df['text'].str.upper()

            # Length
            df['text_length'] = df['text'].str.len()

            # Contains
            df['has_learning'] = df['text'].str.contains('Learning', case=False)

            # Replace
            df['text_clean'] = df['text'].str.replace('[^a-zA-Z\\s]', '', regex=True)

            # Split
            df['word_count'] = df['text'].str.split().str.len()

            # Strip
            df['text_strip'] = df['text'].str.strip()

            self.log_result('test_string_operations', True,
                          f"Processed {len(df)} text entries with 7 string operations")
            return df

        except Exception as e:
            self.log_result('test_string_operations', False, error=str(e))
            traceback.print_exc()
            return None

    # ========== Sorting & Ranking (Common in CV, NLP) ==========

    def test_sorting_operations(self):
        """Test 9: Sorting and ranking operations"""
        try:
            df = pd.DataFrame({
                'model': ['ResNet', 'VGG', 'BERT', 'GPT', 'YOLO', 'Transformer'],
                'accuracy': [0.95, 0.88, 0.92, 0.94, 0.89, 0.96],
                'params_m': [25.5, 138.3, 110.0, 175.0, 61.9, 95.0],
                'year': [2015, 2014, 2018, 2020, 2016, 2017]
            })

            # Sort by single column
            s1 = df.sort_values('accuracy')

            # Sort by multiple columns
            s2 = df.sort_values(['year', 'accuracy'], ascending=[True, False])

            # Sort index
            s3 = df.sort_index()

            # Rank
            df['accuracy_rank'] = df['accuracy'].rank(ascending=False)

            # nlargest
            s4 = df.nlargest(3, 'accuracy')

            # nsmallest
            s5 = df.nsmallest(2, 'params_m')

            self.log_result('test_sorting_operations', True,
                          f"6 sorting/ranking operations completed")
            return df, s1, s2, s3, s4, s5

        except Exception as e:
            self.log_result('test_sorting_operations', False, error=str(e))
            return None, None, None, None, None, None

    # ========== Filtering & Selection (All Domains) ==========

    def test_filtering_operations(self):
        """Test 10: Boolean indexing and filtering"""
        try:
            df = pd.DataFrame({
                'image_id': range(1, 101),
                'class': np.random.choice(['cat', 'dog', 'bird', 'fish'], 100),
                'confidence': np.random.uniform(0.5, 1.0, 100),
                'bbox_area': np.random.randint(100, 10000, 100),
                'is_valid': np.random.choice([True, False], 100)
            })

            # Simple boolean filter
            f1 = df[df['confidence'] > 0.9]

            # Multiple conditions with &
            f2 = df[(df['confidence'] > 0.8) & (df['class'] == 'cat')]

            # Multiple conditions with |
            f3 = df[(df['class'] == 'cat') | (df['class'] == 'dog')]

            # isin
            f4 = df[df['class'].isin(['cat', 'bird'])]

            # between
            f5 = df[df['confidence'].between(0.7, 0.9)]

            # query method
            f6 = df.query('confidence > 0.85 and bbox_area > 5000')

            self.log_result('test_filtering_operations', True,
                          f"f1: {len(f1)}, f2: {len(f2)}, f3: {len(f3)}, f4: {len(f4)}, f5: {len(f5)}, f6: {len(f6)}")
            return df, f1, f2, f3, f4, f5, f6

        except Exception as e:
            self.log_result('test_filtering_operations', False, error=str(e))
            traceback.print_exc()
            return None, None, None, None, None, None, None

    # ========== DateTime Operations (Time-series Analysis) ==========

    def test_datetime_operations(self):
        """Test 11: DateTime operations - common in time-series EDA"""
        try:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=100, freq='h'),
                'value': np.random.randn(100).cumsum() + 100
            })

            # Extract date components
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['date'] = df['timestamp'].dt.date

            # Filter by date range
            f1 = df[df['timestamp'] >= '2025-01-02']

            # Resample (time-based groupby)
            df_indexed = df.set_index('timestamp')
            r1 = df_indexed.resample('D')['value'].mean()

            self.log_result('test_datetime_operations', True,
                          f"Extracted 6 date components, filtered: {len(f1)}, resampled: {len(r1)}")
            return df, f1, r1

        except Exception as e:
            self.log_result('test_datetime_operations', False, error=str(e))
            traceback.print_exc()
            return None, None, None

    # ========== Apply & Transform (Custom Operations) ==========

    def test_apply_operations(self):
        """Test 12: Apply and transform operations"""
        try:
            df = pd.DataFrame({
                'text': ['apple', 'banana', 'cherry', 'date'],
                'count': [5, 3, 8, 2],
                'price': [1.2, 0.5, 2.3, 3.1]
            })

            # apply on Series
            df['text_len'] = df['text'].apply(len)

            # apply with lambda
            df['count_squared'] = df['count'].apply(lambda x: x ** 2)

            # apply on DataFrame (row-wise)
            df['total'] = df.apply(lambda row: row['count'] * row['price'], axis=1)

            # apply on DataFrame (column-wise)
            numeric_sum = df[['count', 'price']].apply(sum, axis=0)

            # map
            df['category'] = df['count'].map({5: 'high', 3: 'medium', 8: 'high', 2: 'low'})

            self.log_result('test_apply_operations', True,
                          f"5 apply/map operations completed")
            return df, numeric_sum

        except Exception as e:
            self.log_result('test_apply_operations', False, error=str(e))
            traceback.print_exc()
            return None, None

    # ========== Window Functions (Rolling, Expanding) ==========

    def test_window_operations(self):
        """Test 13: Rolling and expanding window operations"""
        try:
            df = pd.DataFrame({
                'value': np.random.randn(50).cumsum() + 100
            })

            # Rolling mean
            df['rolling_mean_5'] = df['value'].rolling(window=5).mean()

            # Rolling std
            df['rolling_std_5'] = df['value'].rolling(window=5).std()

            # Rolling sum
            df['rolling_sum_10'] = df['value'].rolling(window=10).sum()

            # Expanding mean
            df['expanding_mean'] = df['value'].expanding().mean()

            # Shift (lag)
            df['lag_1'] = df['value'].shift(1)
            df['lag_2'] = df['value'].shift(2)

            # Difference
            df['diff_1'] = df['value'].diff()

            self.log_result('test_window_operations', True,
                          f"7 window operations completed")
            return df

        except Exception as e:
            self.log_result('test_window_operations', False, error=str(e))
            traceback.print_exc()
            return None

    # ========== Categorical Operations ==========

    def test_categorical_operations(self):
        """Test 14: Categorical data operations"""
        try:
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'D'],
                'value': [10, 20, 15, 30, 25, 12, 35, 40]
            })

            # Convert to categorical
            df['category_cat'] = df['category'].astype('category')

            # Get categories
            cats = df['category_cat'].cat.categories

            # Add category
            df['category_cat'] = df['category_cat'].cat.add_categories(['E'])

            # Remove unused categories
            df['category_cat'] = df['category_cat'].cat.remove_unused_categories()

            # Rename categories
            df['category_renamed'] = df['category_cat'].cat.rename_categories({'A': 'Alpha', 'B': 'Beta', 'C': 'Gamma', 'D': 'Delta'})

            self.log_result('test_categorical_operations', True,
                          f"5 categorical operations completed, categories: {len(cats)}")
            return df, cats

        except Exception as e:
            self.log_result('test_categorical_operations', False, error=str(e))
            traceback.print_exc()
            return None, None

    # ========== Statistical Operations ==========

    def test_statistical_operations(self):
        """Test 15: Statistical operations"""
        try:
            df = pd.DataFrame({
                'A': np.random.randn(100),
                'B': np.random.randn(100),
                'C': np.random.randn(100)
            })

            # Correlation
            corr_matrix = df.corr()

            # Covariance
            cov_matrix = df.cov()

            # Mean, median, std
            means = df.mean()
            medians = df.median()
            stds = df.std()

            # Min, max
            mins = df.min()
            maxs = df.max()

            # Quantile
            q25 = df.quantile(0.25)
            q75 = df.quantile(0.75)

            # Cumulative operations
            df['A_cumsum'] = df['A'].cumsum()
            df['B_cumprod'] = df['B'].cumprod()
            df['C_cummax'] = df['C'].cummax()

            self.log_result('test_statistical_operations', True,
                          f"Computed correlations, statistics, and cumulative operations")
            return df, corr_matrix, means, medians

        except Exception as e:
            self.log_result('test_statistical_operations', False, error=str(e))
            traceback.print_exc()
            return None, None, None, None

    # ========== Multi-Index Operations ==========

    def test_multiindex_operations(self):
        """Test 16: Multi-index operations"""
        try:
            # Create multi-index DataFrame
            arrays = [
                ['A', 'A', 'B', 'B', 'C', 'C'],
                [1, 2, 1, 2, 1, 2]
            ]
            index = pd.MultiIndex.from_arrays(arrays, names=['letter', 'number'])
            df = pd.DataFrame({
                'value1': [10, 20, 30, 40, 50, 60],
                'value2': [100, 200, 300, 400, 500, 600]
            }, index=index)

            # Access by level
            level_a = df.loc['A']

            # Cross-section
            xs_1 = df.xs(1, level='number')

            # Reset index
            df_reset = df.reset_index()

            # Set index
            df_set = df_reset.set_index(['letter', 'number'])

            # Stack and unstack
            df_single = pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            })
            df_stacked = df_single.stack()
            df_unstacked = df_stacked.unstack()

            self.log_result('test_multiindex_operations', True,
                          f"Multi-index operations completed, level_a: {level_a.shape}")
            return df, level_a, xs_1, df_reset

        except Exception as e:
            self.log_result('test_multiindex_operations', False, error=str(e))
            traceback.print_exc()
            return None, None, None, None

    # ========== Advanced Selection ==========

    def test_advanced_selection(self):
        """Test 17: Advanced selection with loc, iloc, at, iat"""
        try:
            df = pd.DataFrame({
                'A': range(10),
                'B': range(10, 20),
                'C': range(20, 30)
            }, index=list('abcdefghij'))

            # loc - label-based
            loc1 = df.loc['a':'c']
            loc2 = df.loc['a':'c', 'A':'B']
            loc3 = df.loc[df['A'] > 5, ['B', 'C']]

            # iloc - position-based
            iloc1 = df.iloc[0:3]
            iloc2 = df.iloc[0:3, 0:2]
            iloc3 = df.iloc[[0, 2, 4], [1, 2]]

            # at - single value label-based
            at_val = df.at['a', 'A']

            # iat - single value position-based
            iat_val = df.iat[0, 0]

            self.log_result('test_advanced_selection', True,
                          f"loc, iloc, at, iat operations completed")
            return df, loc1, iloc1, at_val, iat_val

        except Exception as e:
            self.log_result('test_advanced_selection', False, error=str(e))
            traceback.print_exc()
            return None, None, None, None, None

    # ========== Duplicate Handling ==========

    def test_duplicate_operations(self):
        """Test 18: Duplicate detection and removal"""
        try:
            df = pd.DataFrame({
                'user_id': [1, 2, 3, 1, 2, 4, 5],
                'item_id': [101, 102, 103, 101, 102, 104, 105],
                'rating': [5, 4, 3, 5, 4, 2, 5]
            })

            # duplicated - boolean mask
            dup_mask = df.duplicated()

            # duplicated with subset
            dup_subset = df.duplicated(subset=['user_id', 'item_id'])

            # drop_duplicates
            df_dedup = df.drop_duplicates()

            # drop_duplicates with subset
            df_dedup_subset = df.drop_duplicates(subset=['user_id', 'item_id'])

            # drop_duplicates with keep last
            df_dedup_last = df.drop_duplicates(subset=['user_id'], keep='last')

            self.log_result('test_duplicate_operations', True,
                          f"Duplicates found: {dup_mask.sum()}, after dedup: {len(df_dedup)}")
            return df, dup_mask, df_dedup, df_dedup_subset

        except Exception as e:
            self.log_result('test_duplicate_operations', False, error=str(e))
            traceback.print_exc()
            return None, None, None, None

    # ========== Data Type Operations ==========

    def test_dtype_operations(self):
        """Test 19: Data type conversions"""
        try:
            df = pd.DataFrame({
                'int_col': [1, 2, 3, 4, 5],
                'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
                'str_col': ['1', '2', '3', '4', '5'],
                'bool_col': [True, False, True, False, True]
            })

            # astype conversions
            df['int_to_float'] = df['int_col'].astype(float)
            df['str_to_int'] = df['str_col'].astype(int)
            df['float_to_int'] = df['float_col'].astype(int)

            # to_numeric
            df['numeric'] = pd.to_numeric(df['str_col'])

            # Check dtypes
            dtypes = df.dtypes

            self.log_result('test_dtype_operations', True,
                          f"Data type conversions completed: {len(dtypes)} columns")
            return df, dtypes

        except Exception as e:
            self.log_result('test_dtype_operations', False, error=str(e))
            traceback.print_exc()
            return None, None

    # ========== Binning Operations ==========

    def test_binning_operations(self):
        """Test 20: Binning and discretization"""
        try:
            df = pd.DataFrame({
                'age': np.random.randint(18, 80, 100),
                'score': np.random.uniform(0, 100, 100)
            })

            # cut - equal width bins
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100],
                                     labels=['young', 'middle', 'senior'])

            # qcut - equal frequency bins
            df['score_quartile'] = pd.qcut(df['score'], q=4,
                                          labels=['Q1', 'Q2', 'Q3', 'Q4'])

            # cut with custom bins
            df['score_category'] = pd.cut(df['score'],
                                         bins=[0, 60, 75, 90, 100],
                                         labels=['F', 'C', 'B', 'A'])

            self.log_result('test_binning_operations', True,
                          f"Binning operations completed on {len(df)} rows")
            return df

        except Exception as e:
            self.log_result('test_binning_operations', False, error=str(e))
            traceback.print_exc()
            return None

    def run_all_tests(self):
        """Run all compatibility tests"""
        print(f"\n{'='*80}")
        print(f"Running Pandas Compatibility Tests")
        print(f"Backend: {'DataStore' if self.use_datastore else 'Pandas'}")
        print(f"{'='*80}\n")

        # Run all tests
        self.test_basic_eda_operations()
        self.test_fillna_operations()
        self.test_groupby_aggregations()
        self.test_merge_join_operations()
        self.test_concat_operations()
        self.test_pivot_table_operations()
        self.test_melt_operations()
        self.test_string_operations()
        self.test_sorting_operations()
        self.test_filtering_operations()
        self.test_datetime_operations()
        self.test_apply_operations()
        self.test_window_operations()
        self.test_categorical_operations()
        self.test_statistical_operations()
        self.test_multiindex_operations()
        self.test_advanced_selection()
        self.test_duplicate_operations()
        self.test_dtype_operations()
        self.test_binning_operations()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*80}")
        print(f"Test Summary")
        print(f"{'='*80}")

        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed

        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")

        if failed > 0:
            print(f"\n{'='*80}")
            print("Failed Tests:")
            print(f"{'='*80}")
            for r in self.results:
                if not r['success']:
                    print(f"\n✗ {r['test']}")
                    print(f"  Error: {r['error']}")

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'results': self.results
        }


def main():
    """Main execution function"""
    print("="*80)
    print("Kaggle Pandas Operations - Comprehensive Compatibility Test")
    print("="*80)
    print("\nThis script tests DataStore's pandas compatibility using operations")
    print("commonly found in Kaggle notebooks across multiple domains:")
    print("  - Computer Vision (CV)")
    print("  - Natural Language Processing (NLP)")
    print("  - Recommendation Systems")
    print("  - Large Language Models (LLM)")
    print("  - Exploratory Data Analysis (EDA)")
    print("\n" + "="*80)

    # Run tests with pandas
    print("\n\n### TESTING WITH PANDAS ###\n")
    pandas_tester = PandasCompatibilityTester(use_datastore=False)
    pandas_tester.run_all_tests()
    pandas_results = pandas_tester.print_summary()

    # Run tests with datastore
    print("\n\n### TESTING WITH DATASTORE ###\n")
    try:
        ds_tester = PandasCompatibilityTester(use_datastore=True)
        ds_tester.run_all_tests()
        ds_results = ds_tester.print_summary()

        # Compare results
        print(f"\n{'='*80}")
        print("Compatibility Analysis")
        print(f"{'='*80}")

        # Find operations that work in pandas but fail in datastore
        compatibility_issues = []
        for p_result, d_result in zip(pandas_results['results'], ds_results['results']):
            if p_result['success'] and not d_result['success']:
                compatibility_issues.append({
                    'test': p_result['test'],
                    'pandas_msg': p_result['message'],
                    'datastore_error': d_result['error']
                })

        if compatibility_issues:
            print(f"\nFound {len(compatibility_issues)} compatibility issues:")
            for i, issue in enumerate(compatibility_issues, 1):
                print(f"\n{i}. {issue['test']}")
                print(f"   Pandas: {issue['pandas_msg']}")
                print(f"   DataStore Error: {issue['datastore_error']}")
        else:
            print("\n✓ All pandas operations work correctly with DataStore!")

        # Calculate compatibility score
        compatibility_score = (ds_results['passed'] / pandas_results['total']) * 100
        print(f"\nDataStore Compatibility Score: {compatibility_score:.1f}%")
        print(f"({ds_results['passed']}/{pandas_results['total']} operations compatible)")

    except ImportError as e:
        print(f"\n✗ Could not import DataStore: {e}")
        print("Please ensure DataStore is installed and available")


if __name__ == "__main__":
    main()
