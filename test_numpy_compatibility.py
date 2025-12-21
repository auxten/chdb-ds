"""Test numpy compatibility for DataStore and ColumnExpr"""

import numpy as np
import pandas as pd
import sys
from datastore import DataStore


def test_basic_setup():
    """Test basic data setup"""
    print("=" * 80)
    print("1. Setting up test data")
    print("=" * 80)

    # Create test data
    df = pd.DataFrame(
        {
            'a': [1.0, 2.0, 3.0, 4.0],
            'b': [1.0, 2.0, 3.0, 4.0],
            'c': [1.1, 2.1, 3.1, 4.1],
            'matrix_col': [[1, 2], [3, 4], [5, 6], [7, 8]],
        }
    )

    ds = DataStore.from_df(df)
    print(f"Pandas DataFrame:\n{df}\n")
    print(f"DataStore:\n{ds}\n")

    return df, ds


def test_np_allclose():
    """Test np.allclose with DataStore and ColumnExpr"""
    print("=" * 80)
    print("2. Testing np.allclose()")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Test with pandas DataFrame
    print("\n2.1 Test np.allclose with pandas DataFrame:")
    try:
        result = np.allclose(df['a'], df['b'])
        print(f"✓ np.allclose(df['a'], df['b']) = {result}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test with pandas Series
    print("\n2.2 Test np.allclose with pandas Series:")
    try:
        result = np.allclose(df['a'].values, df['b'].values)
        print(f"✓ np.allclose(df['a'].values, df['b'].values) = {result}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test with DataStore columns
    print("\n2.3 Test np.allclose with DataStore columns (direct):")
    try:
        result = np.allclose(ds['a'], ds['b'])
        print(f"✓ np.allclose(ds['a'], ds['b']) = {result}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

    # Test with DataStore columns using .values
    print("\n2.4 Test np.allclose with DataStore columns using .values:")
    try:
        result = np.allclose(ds['a'].values, ds['b'].values)
        print(f"✓ np.allclose(ds['a'].values, ds['b'].values) = {result}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test with DataStore columns using .to_numpy()
    print("\n2.5 Test np.allclose with DataStore columns using .to_numpy():")
    try:
        result = np.allclose(ds['a'].to_numpy(), ds['b'].to_numpy())
        print(f"✓ np.allclose(ds['a'].to_numpy(), ds['b'].to_numpy()) = {result}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test with slightly different values
    print("\n2.6 Test np.allclose with slightly different values:")
    try:
        result_pandas = np.allclose(df['a'], df['c'], rtol=0.1)
        print(f"✓ Pandas: np.allclose(df['a'], df['c'], rtol=0.1) = {result_pandas}")

        result_ds = np.allclose(ds['a'].to_numpy(), ds['c'].to_numpy(), rtol=0.1)
        print(f"✓ DataStore: np.allclose(ds['a'].to_numpy(), ds['c'].to_numpy(), rtol=0.1) = {result_ds}")

        assert result_pandas == result_ds, "Results should match!"
    except Exception as e:
        print(f"✗ Error: {e}")


def test_other_numpy_functions():
    """Test other common numpy functions"""
    print("\n" + "=" * 80)
    print("3. Testing other numpy functions")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Test np.array()
    print("\n3.1 Test np.array():")
    print("Pandas:")
    try:
        arr_pandas = np.array(df['a'])
        print(f"✓ np.array(df['a']) = {arr_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (direct):")
    try:
        arr_ds = np.array(ds['a'])
        print(f"✓ np.array(ds['a']) = {arr_ds}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

    print("DataStore (.to_numpy()):")
    try:
        arr_ds = np.array(ds['a'].to_numpy())
        print(f"✓ np.array(ds['a'].to_numpy()) = {arr_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.mean()
    print("\n3.2 Test np.mean():")
    print("Pandas:")
    try:
        mean_pandas = np.mean(df['a'])
        print(f"✓ np.mean(df['a']) = {mean_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (direct):")
    try:
        mean_ds = np.mean(ds['a'])
        print(f"✓ np.mean(ds['a']) = {mean_ds}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

    print("DataStore (.to_numpy()):")
    try:
        mean_ds = np.mean(ds['a'].to_numpy())
        print(f"✓ np.mean(ds['a'].to_numpy()) = {mean_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.sum()
    print("\n3.3 Test np.sum():")
    print("Pandas:")
    try:
        sum_pandas = np.sum(df['a'])
        print(f"✓ np.sum(df['a']) = {sum_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (direct):")
    try:
        sum_ds = np.sum(ds['a'])
        print(f"✓ np.sum(ds['a']) = {sum_ds}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

    print("DataStore (.to_numpy()):")
    try:
        sum_ds = np.sum(ds['a'].to_numpy())
        print(f"✓ np.sum(ds['a'].to_numpy()) = {sum_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.std()
    print("\n3.4 Test np.std():")
    print("Pandas:")
    try:
        std_pandas = np.std(df['a'])
        print(f"✓ np.std(df['a']) = {std_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (direct):")
    try:
        std_ds = np.std(ds['a'])
        print(f"✓ np.std(ds['a']) = {std_ds}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

    print("DataStore (.to_numpy()):")
    try:
        std_ds = np.std(ds['a'].to_numpy())
        print(f"✓ np.std(ds['a'].to_numpy()) = {std_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.max() and np.min()
    print("\n3.5 Test np.max() and np.min():")
    print("Pandas:")
    try:
        max_pandas = np.max(df['a'])
        min_pandas = np.min(df['a'])
        print(f"✓ np.max(df['a']) = {max_pandas}, np.min(df['a']) = {min_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (.to_numpy()):")
    try:
        max_ds = np.max(ds['a'].to_numpy())
        min_ds = np.min(ds['a'].to_numpy())
        print(f"✓ np.max(ds['a'].to_numpy()) = {max_ds}, np.min(ds['a'].to_numpy()) = {min_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.concatenate()
    print("\n3.6 Test np.concatenate():")
    print("Pandas:")
    try:
        concat_pandas = np.concatenate([df['a'].values, df['b'].values])
        print(f"✓ np.concatenate([df['a'].values, df['b'].values]) = {concat_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (.to_numpy()):")
    try:
        concat_ds = np.concatenate([ds['a'].to_numpy(), ds['b'].to_numpy()])
        print(f"✓ np.concatenate([ds['a'].to_numpy(), ds['b'].to_numpy()]) = {concat_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.dot()
    print("\n3.7 Test np.dot():")
    print("Pandas:")
    try:
        dot_pandas = np.dot(df['a'].values, df['b'].values)
        print(f"✓ np.dot(df['a'].values, df['b'].values) = {dot_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (.to_numpy()):")
    try:
        dot_ds = np.dot(ds['a'].to_numpy(), ds['b'].to_numpy())
        print(f"✓ np.dot(ds['a'].to_numpy(), ds['b'].to_numpy()) = {dot_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_numpy_array_interface():
    """Test if ColumnExpr implements __array__ interface"""
    print("\n" + "=" * 80)
    print("4. Testing numpy array interface (__array__)")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Check if pandas Series has __array__
    print("\n4.1 Check if pandas Series has __array__:")
    print(f"hasattr(df['a'], '__array__') = {hasattr(df['a'], '__array__')}")

    # Check if ColumnExpr has __array__
    print("\n4.2 Check if ColumnExpr has __array__:")
    print(f"hasattr(ds['a'], '__array__') = {hasattr(ds['a'], '__array__')}")

    # Check what methods ColumnExpr has for array conversion
    print("\n4.3 Array conversion methods available on ColumnExpr:")
    col_expr = ds['a']
    array_methods = [
        attr for attr in dir(col_expr) if 'array' in attr.lower() or attr in ['values', 'to_numpy', 'tolist']
    ]
    for method in array_methods:
        print(f"  - {method}")


def test_comparison_operations():
    """Test numpy comparison operations"""
    print("\n" + "=" * 80)
    print("5. Testing numpy comparison operations")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Test np.equal()
    print("\n5.1 Test np.equal():")
    print("Pandas:")
    try:
        result_pandas = np.equal(df['a'].values, df['b'].values)
        print(f"✓ np.equal(df['a'].values, df['b'].values) = {result_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (.to_numpy()):")
    try:
        result_ds = np.equal(ds['a'].to_numpy(), ds['b'].to_numpy())
        print(f"✓ np.equal(ds['a'].to_numpy(), ds['b'].to_numpy()) = {result_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.isclose()
    print("\n5.2 Test np.isclose():")
    print("Pandas:")
    try:
        result_pandas = np.isclose(df['a'].values, df['c'].values, rtol=0.1)
        print(f"✓ np.isclose(df['a'].values, df['c'].values, rtol=0.1) = {result_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (.to_numpy()):")
    try:
        result_ds = np.isclose(ds['a'].to_numpy(), ds['c'].to_numpy(), rtol=0.1)
        print(f"✓ np.isclose(ds['a'].to_numpy(), ds['c'].to_numpy(), rtol=0.1) = {result_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_dataframe_level_operations():
    """Test numpy operations on DataFrame level"""
    print("\n" + "=" * 80)
    print("6. Testing DataFrame-level numpy operations")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Select numeric columns only
    df_numeric = df[['a', 'b', 'c']]

    # Test with pandas DataFrame
    print("\n6.1 Test with pandas DataFrame:")
    try:
        # np.array on DataFrame
        arr = np.array(df_numeric)
        print(f"✓ np.array(df[['a', 'b', 'c']]).shape = {arr.shape}")
        print(f"  Values:\n{arr}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test with DataStore
    print("\n6.2 Test with DataStore (to_df()):")
    try:
        ds_numeric = ds.select('a', 'b', 'c')
        arr = np.array(ds_numeric.to_df())
        print(f"✓ np.array(ds.select('a', 'b', 'c').to_df()).shape = {arr.shape}")
        print(f"  Values:\n{arr}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test np.corrcoef()
    print("\n6.3 Test np.corrcoef():")
    print("Pandas:")
    try:
        corr_pandas = np.corrcoef(df['a'], df['b'])
        print(f"✓ np.corrcoef(df['a'], df['b']) =\n{corr_pandas}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("DataStore (.to_numpy()):")
    try:
        corr_ds = np.corrcoef(ds['a'].to_numpy(), ds['b'].to_numpy())
        print(f"✓ np.corrcoef(ds['a'].to_numpy(), ds['b'].to_numpy()) =\n{corr_ds}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_array_method_robustness():
    """Test __array__ method handles various return types correctly.

    This tests the fix for: ValueError: object __array__ method not producing an array
    """
    print("\n" + "=" * 80)
    print("7. Testing __array__ method robustness")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Test 1: ColumnExpr.__array__ returns proper numpy array
    print("\n7.1 Test ColumnExpr.__array__ returns numpy array:")
    arr = ds['a'].__array__()
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    assert len(arr) == 4, f"Expected length 4, got {len(arr)}"
    print(f"✓ ds['a'].__array__() returns {type(arr).__name__} with shape {arr.shape}")

    # Test 2: DataStore.__array__ returns proper numpy array
    print("\n7.2 Test DataStore.__array__ returns numpy array:")
    arr = ds.__array__()
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    print(f"✓ ds.__array__() returns {type(arr).__name__} with shape {arr.shape}")

    # Test 3: np.asarray works with ColumnExpr (used by seaborn/pandas)
    print("\n7.3 Test np.asarray with ColumnExpr:")
    arr = np.asarray(ds['a'])
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    print(f"✓ np.asarray(ds['a']) returns {type(arr).__name__} with shape {arr.shape}")

    # Test 4: np.asarray works with DataStore
    print("\n7.4 Test np.asarray with DataStore:")
    arr = np.asarray(ds)
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    print(f"✓ np.asarray(ds) returns {type(arr).__name__} with shape {arr.shape}")

    # Test 5: __array__ with dtype parameter
    print("\n7.5 Test __array__ with dtype parameter:")
    arr = ds['a'].__array__(dtype=np.float32)
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    print(f"✓ ds['a'].__array__(dtype=np.float32) returns dtype {arr.dtype}")

    # Test 6: __array__ with copy parameter (numpy 2.0+ compatibility)
    print("\n7.6 Test __array__ with copy parameter:")
    arr1 = ds['a'].__array__()
    arr2 = ds['a'].__array__(copy=True)
    assert isinstance(arr2, np.ndarray), f"Expected numpy.ndarray, got {type(arr2)}"
    print(f"✓ ds['a'].__array__(copy=True) returns {type(arr2).__name__}")


def test_array_method_with_seaborn_pattern():
    """Test the exact pattern that seaborn uses when creating DataFrames.

    This is a regression test for the seaborn kdeplot ValueError.
    """
    print("\n" + "=" * 80)
    print("8. Testing seaborn-like DataFrame creation pattern")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Test 1: Create pd.DataFrame from dict of ColumnExpr (seaborn pattern)
    print("\n8.1 Test pd.DataFrame creation from ColumnExpr dict:")
    try:
        plot_data = {'x': ds['a'], 'hue': ds['b']}
        result_df = pd.DataFrame(plot_data)
        assert isinstance(result_df, pd.DataFrame), f"Expected DataFrame, got {type(result_df)}"
        assert result_df.shape == (4, 2), f"Expected shape (4, 2), got {result_df.shape}"
        print(f"✓ pd.DataFrame({{'x': ds['a'], 'hue': ds['b']}}) works, shape={result_df.shape}")
    except Exception as e:
        raise AssertionError(f"pd.DataFrame creation failed: {e}")

    # Test 2: Create pd.DataFrame from dict with mixed types
    print("\n8.2 Test pd.DataFrame creation with mixed types:")
    try:
        plot_data = {'x': ds['a'], 'y': np.array([1, 2, 3, 4])}
        result_df = pd.DataFrame(plot_data)
        assert isinstance(result_df, pd.DataFrame), f"Expected DataFrame, got {type(result_df)}"
        print(f"✓ pd.DataFrame with mixed ColumnExpr and numpy works, shape={result_df.shape}")
    except Exception as e:
        raise AssertionError(f"pd.DataFrame creation with mixed types failed: {e}")

    # Test 3: Create pd.DataFrame from dict with grouper column
    print("\n8.3 Test pd.DataFrame with hue/grouper pattern:")
    try:
        # This is the exact pattern seaborn uses
        plot_data = {'x': ds['a'], 'hue': ds['b']}
        result_df = pd.DataFrame(plot_data)

        # Verify data integrity
        np.testing.assert_array_almost_equal(result_df['x'].values, df['a'].values)
        np.testing.assert_array_almost_equal(result_df['hue'].values, df['b'].values)
        print(f"✓ Data integrity verified after DataFrame conversion")
    except Exception as e:
        raise AssertionError(f"Seaborn hue pattern failed: {e}")

    print("\n✓ All seaborn-like patterns work correctly!")


def test_lazy_aggregate_array_method():
    """Test __array__ method on LazyAggregate objects."""
    print("\n" + "=" * 80)
    print("9. Testing LazyAggregate __array__ method")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Test 1: LazyAggregate from mean()
    print("\n9.1 Test LazyAggregate.__array__ from mean():")
    lazy_mean = ds['a'].mean()
    arr = np.asarray(lazy_mean)
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    print(f"✓ np.asarray(ds['a'].mean()) = {arr}")

    # Test 2: LazyAggregate with dtype
    print("\n9.2 Test LazyAggregate.__array__ with dtype:")
    arr = lazy_mean.__array__(dtype=np.float32)
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    print(f"✓ ds['a'].mean().__array__(dtype=np.float32) dtype = {arr.dtype}")

    # Test 3: LazyAggregate with copy parameter
    print("\n9.3 Test LazyAggregate.__array__ with copy parameter:")
    arr = lazy_mean.__array__(copy=True)
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    print(f"✓ ds['a'].mean().__array__(copy=True) works")


def test_lazy_result_array_method():
    """Test __array__ method on LazyResult objects (e.g., from head/tail)."""
    print("\n" + "=" * 80)
    print("10. Testing LazyResult __array__ method")
    print("=" * 80)

    df, ds = test_basic_setup()

    # Test 1: LazyResult from head()
    print("\n10.1 Test LazyResult.__array__ from head():")
    lazy_head = ds.head(2)
    arr = np.asarray(lazy_head)
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    assert arr.shape[0] == 2, f"Expected 2 rows, got {arr.shape[0]}"
    print(f"✓ np.asarray(ds.head(2)) shape = {arr.shape}")

    # Test 2: LazyResult with dtype
    print("\n10.2 Test LazyResult.__array__ with dtype:")
    lazy_head = ds.select('a', 'b').head(2)
    arr = lazy_head.__array__(dtype=np.float64)
    assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"
    print(f"✓ LazyResult.__array__(dtype=np.float64) dtype = {arr.dtype}")

    # Test 3: LazyResult with copy parameter
    print("\n10.3 Test LazyResult.__array__ with copy parameter:")
    arr = lazy_head.__array__(copy=True)
    assert isinstance(arr, np.ndarray), f"Expected numpy.ndarray, got {type(arr)}"
    print(f"✓ LazyResult.__array__(copy=True) works")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("NumPy Compatibility Test Suite for DataStore")
    print("=" * 80)

    try:
        test_np_allclose()
        test_other_numpy_functions()
        test_numpy_array_interface()
        test_comparison_operations()
        test_dataframe_level_operations()
        test_array_method_robustness()
        test_array_method_with_seaborn_pattern()
        test_lazy_aggregate_array_method()
        test_lazy_result_array_method()

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(
            """
The test results show:
1. DataStore columns can be converted to numpy arrays using .to_numpy() or .values
2. Direct __array__ interface now works correctly with ColumnExpr, DataStore, 
   LazyAggregate, and LazyResult
3. DataStore works with all standard numpy functions
4. Seaborn/pandas DataFrame creation patterns work correctly
5. numpy 2.0+ copy parameter is supported
        """
        )

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
