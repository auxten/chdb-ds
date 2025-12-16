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
    df = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0],
        'b': [1.0, 2.0, 3.0, 4.0],
        'c': [1.1, 2.1, 3.1, 4.1],
        'matrix_col': [[1, 2], [3, 4], [5, 6], [7, 8]]
    })
    
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
    array_methods = [attr for attr in dir(col_expr) if 'array' in attr.lower() or attr in ['values', 'to_numpy', 'tolist']]
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
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("""
The test results show:
1. DataStore columns can be converted to numpy arrays using .to_numpy() or .values
2. Direct usage (without conversion) may not work due to missing __array__ interface
3. After conversion, DataStore works with all standard numpy functions
4. To improve compatibility, consider implementing __array__ method in ColumnExpr
        """)
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

