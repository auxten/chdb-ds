"""
Kaggle LLM/NLP-Specific Pandas Compatibility Test Suite

This test suite focuses on pandas operations commonly used in LLM and NLP
competitions on Kaggle, based on analysis of:
- LLM Classification Finetuning Competition
- LLM - Detect AI Generated Text Competition
- NLP text preprocessing notebooks
- Transformers and fine-tuning workflows

Common LLM/NLP data preparation patterns tested:
1. Loading multiple datasets and combining them
2. Text preprocessing (cleaning, lowercasing, deduplication)
3. Train/test splitting with flags
4. String operations for text cleaning
5. Duplicate detection and removal
6. Missing value handling in text columns
7. Feature engineering for text data
8. Creating submission DataFrames
"""

import sys
import traceback
import numpy as np
import pandas as pd

try:
    import datastore as ds
except ImportError:
    print("ERROR: datastore module not found")
    sys.exit(1)


class LLMCompatibilityTester:
    """Test framework for LLM/NLP-specific pandas operations."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.errors = 0

    def test(self, name: str, pandas_func, datastore_func, compare_func=None):
        """Run a single test."""
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")

        try:
            print("  [1/3] Running pandas operation...")
            pandas_result = pandas_func()
            print(f"  ✓ Pandas result type: {type(pandas_result)}")

            print("  [2/3] Running datastore operation...")
            datastore_result = datastore_func()
            print(f"  ✓ DataStore result type: {type(datastore_result)}")

            print("  [3/3] Comparing results...")
            if compare_func:
                passed = compare_func(pandas_result, datastore_result)
            else:
                passed = self._default_compare(pandas_result, datastore_result)

            if passed:
                print(f"  ✓ PASSED")
                self.passed += 1
                self.results.append({'test': name, 'status': 'PASSED'})
            else:
                print(f"  ✗ FAILED: Results don't match")
                self.failed += 1
                self.results.append({
                    'test': name,
                    'status': 'FAILED',
                    'pandas_result': str(pandas_result)[:200],
                    'datastore_result': str(datastore_result)[:200]
                })

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            self.errors += 1
            self.results.append({
                'test': name,
                'status': 'ERROR',
                'error': str(e)
            })

    def _default_compare(self, pandas_result, datastore_result):
        """Default comparison logic."""
        try:
            if isinstance(pandas_result, pd.DataFrame):
                if hasattr(datastore_result, 'to_pandas'):
                    ds_df = datastore_result.to_pandas()
                elif isinstance(datastore_result, pd.DataFrame):
                    ds_df = datastore_result
                else:
                    return False
                return pandas_result.equals(ds_df)
            elif isinstance(pandas_result, pd.Series):
                if hasattr(datastore_result, 'to_pandas'):
                    ds_series = datastore_result.to_pandas()
                elif isinstance(datastore_result, pd.Series):
                    ds_series = datastore_result
                else:
                    return False
                return pandas_result.equals(ds_series)
            else:
                return pandas_result == datastore_result
        except Exception:
            return False

    def print_summary(self):
        """Print test summary."""
        print(f"\n\n{'='*80}")
        print(f"LLM/NLP TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {len(self.results)}")
        print(f"✓ Passed: {self.passed}")
        print(f"✗ Failed: {self.failed}")
        print(f"⚠ Errors: {self.errors}")
        if self.results:
            print(f"Success rate: {self.passed / len(self.results) * 100:.1f}%")

        if self.failed > 0 or self.errors > 0:
            print(f"\n{'='*80}")
            print(f"ISSUES FOUND")
            print(f"{'='*80}")
            for result in self.results:
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"\n{result['status']}: {result['test']}")
                    if 'error' in result:
                        print(f"  Error: {result['error']}")


def run_llm_tests():
    """Run LLM/NLP-specific pandas compatibility tests."""

    tester = LLMCompatibilityTester()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          LLM/NLP Pandas Compatibility Test Suite for DataStore              ║
║                                                                              ║
║  Testing pandas operations common in Kaggle LLM/NLP competitions            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # ========================================================================
    # Category 1: Multi-Dataset Loading and Combining (Very Common in LLM Tasks)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 1: Multi-Dataset Loading and Combining")
    print("="*80)

    # Create sample train and test CSV files
    train_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': ['Hello world', 'Machine learning', 'Deep learning', 'NLP tasks', 'Transformers'],
        'label': [0, 1, 1, 0, 1]
    })
    test_data = pd.DataFrame({
        'id': [6, 7, 8],
        'text': ['Test sentence', 'Another test', 'Final test'],
    })
    train_data.to_csv('/tmp/train_llm.csv', index=False)
    test_data.to_csv('/tmp/test_llm.csv', index=False)

    # Test 1: Load train and test, add flag, concatenate (LLM competition pattern)
    def test_concat_with_flag_pandas():
        training = pd.read_csv('/tmp/train_llm.csv')
        test = pd.read_csv('/tmp/test_llm.csv')
        training['train_test'] = 1
        test['train_test'] = 0
        all_data = pd.concat([training, test], ignore_index=True)
        return all_data

    def test_concat_with_flag_ds():
        training = ds.read_csv('/tmp/train_llm.csv')
        test = ds.read_csv('/tmp/test_llm.csv')
        # Try to add column and concatenate
        if isinstance(training, pd.DataFrame):
            training['train_test'] = 1
            test['train_test'] = 0
            return pd.concat([training, test], ignore_index=True)
        else:
            # DataStore path
            training_df = training.to_pandas() if hasattr(training, 'to_pandas') else training
            test_df = test.to_pandas() if hasattr(test, 'to_pandas') else test
            training_df['train_test'] = 1
            test_df['train_test'] = 0
            return pd.concat([training_df, test_df], ignore_index=True)

    tester.test(
        "1.1 Load train/test CSV, add flag column, concatenate",
        test_concat_with_flag_pandas,
        test_concat_with_flag_ds
    )

    # ========================================================================
    # Category 2: Text Preprocessing Operations (Core NLP Pattern)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 2: Text Preprocessing Operations")
    print("="*80)

    # Create DataFrame with text data for preprocessing
    text_df_pandas = pd.DataFrame({
        'text': ['Hello World!', 'MACHINE Learning.', 'Deep-Learning', 'NLP Tasks?', 'Transformers!!!'],
        'id': [1, 2, 3, 4, 5]
    })
    text_df_ds = ds.DataFrame({
        'text': ['Hello World!', 'MACHINE Learning.', 'Deep-Learning', 'NLP Tasks?', 'Transformers!!!'],
        'id': [1, 2, 3, 4, 5]
    })

    # Test 2: Convert text to lowercase (most common preprocessing)
    def test_lowercase_pandas():
        df = text_df_pandas.copy()
        df['text_lower'] = df['text'].str.lower()
        return df['text_lower']

    def test_lowercase_ds():
        df = text_df_ds
        result = df['text'].str.lower()
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "2.1 Convert text to lowercase with str.lower()",
        test_lowercase_pandas,
        test_lowercase_ds
    )

    # Test 3: Remove punctuation with str.replace() and regex
    def test_remove_punct_pandas():
        df = text_df_pandas.copy()
        df['text_clean'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
        return df['text_clean']

    def test_remove_punct_ds():
        df = text_df_ds
        result = df['text'].str.replace(r'[^\w\s]', '', regex=True)
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "2.2 Remove punctuation with str.replace() regex",
        test_remove_punct_pandas,
        test_remove_punct_ds
    )

    # Test 4: Strip whitespace
    def test_strip_pandas():
        df = pd.DataFrame({'text': ['  hello  ', ' world ', '  test  ']})
        df['text_stripped'] = df['text'].str.strip()
        return df['text_stripped']

    def test_strip_ds():
        df = ds.DataFrame({'text': ['  hello  ', ' world ', '  test  ']})
        result = df['text'].str.strip()
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "2.3 Strip whitespace with str.strip()",
        test_strip_pandas,
        test_strip_ds
    )

    # Test 5: Get text length (for feature engineering)
    def test_text_len_pandas():
        df = text_df_pandas.copy()
        df['text_len'] = df['text'].str.len()
        return df['text_len']

    def test_text_len_ds():
        df = text_df_ds
        result = df['text'].str.len()
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "2.4 Get text length with str.len()",
        test_text_len_pandas,
        test_text_len_ds
    )

    # ========================================================================
    # Category 3: Duplicate Detection and Removal (Data Quality)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 3: Duplicate Detection and Removal")
    print("="*80)

    # Create DataFrame with duplicates
    dup_df_pandas = pd.DataFrame({
        'id': [1, 2, 3, 2, 4, 3],
        'text': ['a', 'b', 'c', 'b', 'd', 'c'],
        'label': [0, 1, 0, 1, 1, 0]
    })
    dup_df_ds = ds.DataFrame({
        'id': [1, 2, 3, 2, 4, 3],
        'text': ['a', 'b', 'c', 'b', 'd', 'c'],
        'label': [0, 1, 0, 1, 1, 0]
    })

    # Test 6: Detect duplicates in specific column
    def test_duplicated_pandas():
        return dup_df_pandas.duplicated(subset=['id'])

    def test_duplicated_ds():
        result = dup_df_ds.duplicated(subset=['id'])
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "3.1 Detect duplicates with duplicated(subset=)",
        test_duplicated_pandas,
        test_duplicated_ds
    )

    # Test 7: Count duplicates
    def test_dup_sum_pandas():
        return dup_df_pandas.duplicated(subset=['id']).sum()

    def test_dup_sum_ds():
        result = dup_df_ds.duplicated(subset=['id'])
        if isinstance(result, pd.Series):
            return result.sum()
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas().sum()
        elif hasattr(result, 'sum'):
            return result.sum()
        return None

    tester.test(
        "3.2 Count duplicates with duplicated().sum()",
        test_dup_sum_pandas,
        test_dup_sum_ds
    )

    # ========================================================================
    # Category 4: Missing Value Handling in Text Columns
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 4: Missing Value Handling in Text Columns")
    print("="*80)

    # Create DataFrame with missing text values
    na_text_df_pandas = pd.DataFrame({
        'text': ['Hello', None, 'World', np.nan, 'Test'],
        'id': [1, 2, 3, 4, 5]
    })
    na_text_df_ds = ds.DataFrame({
        'text': ['Hello', None, 'World', np.nan, 'Test'],
        'id': [1, 2, 3, 4, 5]
    })

    # Test 8: Drop rows with missing text (common in NLP)
    def test_dropna_subset_pandas():
        return na_text_df_pandas.dropna(subset=['text'])

    def test_dropna_subset_ds():
        result = na_text_df_ds.dropna(subset=['text'])
        if isinstance(result, pd.DataFrame):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "4.1 Drop rows with missing text using dropna(subset=)",
        test_dropna_subset_pandas,
        test_dropna_subset_ds
    )

    # Test 9: Fill missing text with empty string
    def test_fillna_text_pandas():
        df = na_text_df_pandas.copy()
        df['text'] = df['text'].fillna('')
        return df

    def test_fillna_text_ds():
        df = na_text_df_ds
        result = df.fillna({'text': ''})
        if isinstance(result, pd.DataFrame):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "4.2 Fill missing text with empty string",
        test_fillna_text_pandas,
        test_fillna_text_ds
    )

    # ========================================================================
    # Category 5: Feature Engineering for Text Data
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 5: Feature Engineering for Text Data")
    print("="*80)

    # Test 10: Count words in text
    def test_word_count_pandas():
        df = text_df_pandas.copy()
        df['word_count'] = df['text'].str.split().str.len()
        return df['word_count']

    def test_word_count_ds():
        df = text_df_ds
        result = df['text'].str.split().str.len()
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "5.1 Count words with str.split().str.len()",
        test_word_count_pandas,
        test_word_count_ds
    )

    # Test 11: Check if text contains specific word
    def test_contains_pandas():
        df = text_df_pandas.copy()
        df['has_learning'] = df['text'].str.contains('learning', case=False)
        return df['has_learning']

    def test_contains_ds():
        df = text_df_ds
        result = df['text'].str.contains('learning', case=False)
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "5.2 Check if text contains word with str.contains(case=False)",
        test_contains_pandas,
        test_contains_ds
    )

    # Test 12: Extract first word
    def test_extract_pandas():
        df = text_df_pandas.copy()
        df['first_word'] = df['text'].str.split().str[0]
        return df['first_word']

    def test_extract_ds():
        df = text_df_ds
        result = df['text'].str.split().str[0]
        if isinstance(result, pd.Series):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "5.3 Extract first word with str.split().str[0]",
        test_extract_pandas,
        test_extract_ds
    )

    # ========================================================================
    # Category 6: Creating Submission DataFrames (Competition Pattern)
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 6: Creating Submission DataFrames")
    print("="*80)

    # Test 13: Create submission DataFrame from predictions
    def test_submission_pandas():
        predictions = [0, 1, 1, 0, 1]
        ids = [1, 2, 3, 4, 5]
        submission = pd.DataFrame({
            'id': ids,
            'prediction': predictions
        })
        return submission

    def test_submission_ds():
        predictions = [0, 1, 1, 0, 1]
        ids = [1, 2, 3, 4, 5]
        submission = ds.DataFrame({
            'id': ids,
            'prediction': predictions
        })
        if isinstance(submission, pd.DataFrame):
            return submission
        elif hasattr(submission, 'to_pandas'):
            return submission.to_pandas()
        return submission

    tester.test(
        "6.1 Create submission DataFrame from predictions",
        test_submission_pandas,
        test_submission_ds
    )

    # ========================================================================
    # Category 7: Sampling and Data Splitting
    # ========================================================================

    print("\n" + "="*80)
    print("CATEGORY 7: Sampling and Data Splitting")
    print("="*80)

    # Test 14: Sample n rows
    def test_sample_pandas():
        df = text_df_pandas.copy()
        return df.sample(n=3, random_state=42)

    def test_sample_ds():
        df = text_df_ds
        result = df.sample(n=3, random_state=42)
        if isinstance(result, pd.DataFrame):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "7.1 Sample n rows with sample(n=3, random_state=42)",
        test_sample_pandas,
        test_sample_ds
    )

    # Test 15: Sample fraction of rows
    def test_sample_frac_pandas():
        df = text_df_pandas.copy()
        return df.sample(frac=0.6, random_state=42)

    def test_sample_frac_ds():
        df = text_df_ds
        result = df.sample(frac=0.6, random_state=42)
        if isinstance(result, pd.DataFrame):
            return result
        elif hasattr(result, 'to_pandas'):
            return result.to_pandas()
        return result

    tester.test(
        "7.2 Sample fraction of rows with sample(frac=0.6)",
        test_sample_frac_pandas,
        test_sample_frac_ds
    )

    # Print summary
    tester.print_summary()
    return tester


if __name__ == "__main__":
    tester = run_llm_tests()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nLLM/NLP-specific pandas operations tested.")
    print("These operations are commonly used in Kaggle LLM competitions for:")
    print("  - Data loading and combining (train/test)")
    print("  - Text preprocessing and cleaning")
    print("  - Duplicate detection")
    print("  - Missing value handling")
    print("  - Feature engineering")
    print("  - Creating submission files")

    sys.exit(0 if tester.failed == 0 and tester.errors == 0 else 1)
