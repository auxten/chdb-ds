"""
Test utility functions for DataStore vs Pandas comparison.

These utilities follow the Mirror Code Pattern and Complete Output Comparison
principles defined in .cursor/rules/chdb-ds.mdc
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional


# =============================================================================
# Known chDB dtype differences (see test_chdb_dtype_differences.py)
# =============================================================================
# When data passes through chDB's Python() table function:
# 1. float64 with NaN → Float64Dtype() (nullable)
# 2. Integer columns with None → Float64Dtype()
# 3. datetime64[ns] → datetime64[ns, <timezone>] (adds system timezone)
#
# TODO: Many tests fail due to these dtype differences. Use
#       assert_datastore_equals_pandas(..., check_dtype=False) or
#       assert_datastore_equals_pandas_chdb_compat() for tests affected by this.
# =============================================================================


def _normalize_chdb_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is only used for tests that we already know are chDB issues.
    Use this function means there is a known issue with chDB!!

    Normalize chDB output dtypes to standard pandas dtypes.

    Converts:
    - Float64 (nullable) → float64
    - Int64 (nullable) → int64 or float64 (if has NA)
    - Timezone-aware datetime → timezone-naive datetime

    This allows for value comparison when dtype differences are expected.

    Args:
        df: DataFrame potentially containing chDB nullable dtypes

    Returns:
        DataFrame with standard pandas dtypes
    """
    result = df.copy()
    for col in result.columns:
        dtype_str = str(result[col].dtype)

        # Handle nullable float (Float64 → float64)
        if dtype_str == "Float64":
            result[col] = result[col].astype("float64")

        # Handle nullable int (Int64 → int64 or float64 if has NA)
        elif dtype_str == "Int64":
            if result[col].isna().any():
                result[col] = result[col].astype("float64")
            else:
                result[col] = result[col].astype("int64")

        # Handle timezone-aware datetime → naive
        elif hasattr(result[col].dtype, "tz") and result[col].dtype.tz is not None:
            result[col] = result[col].dt.tz_localize(None)

    return result


def assert_datastore_equals_pandas(
    ds_result,
    pd_result: pd.DataFrame,
    check_column_order: bool = True,
    check_row_order: bool = True,
    check_dtype: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = "",
) -> None:
    """
    Complete comparison between DataStore result and pandas DataFrame.

    Compares: column names (with order), data values, row order.

    Args:
        ds_result: DataStore result (DataStore, LazySeries, or similar)
        pd_result: Expected pandas DataFrame or Series
        check_column_order: If True, column order must match exactly
        check_row_order: If True, row order must match exactly
                        Set to False for operations with undefined order (e.g., groupby without sort)
        check_dtype: If True, also verify dtypes match
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison
        msg: Additional message to include in assertion errors

    Raises:
        AssertionError: If any comparison fails

    Example:
        # pandas operations
        pd_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        pd_result = pd_df[pd_df['age'] > 20].sort_values('name')

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        ds_result = ds_df[ds_df['age'] > 20].sort_values('name')

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)
    """
    prefix = f"{msg}: " if msg else ""

    # Handle Series comparison
    if isinstance(pd_result, pd.Series):
        _assert_series_equals(ds_result, pd_result, check_row_order, check_dtype, rtol, atol, prefix)
        return

    # Get DataStore columns
    ds_columns = list(ds_result.columns)
    pd_columns = list(pd_result.columns)

    # 1. Compare column names
    if check_column_order:
        assert ds_columns == pd_columns, (
            f"{prefix}Column names or order don't match.\n"
            f"DataStore columns: {ds_columns}\n"
            f"Pandas columns:    {pd_columns}"
        )
    else:
        assert set(ds_columns) == set(pd_columns), (
            f"{prefix}Column names don't match.\n"
            f"DataStore columns: {set(ds_columns)}\n"
            f"Pandas columns:    {set(pd_columns)}"
        )

    # 2. Compare row count
    ds_len = len(ds_result)
    pd_len = len(pd_result)
    assert ds_len == pd_len, (
        f"{prefix}Row count doesn't match.\n" f"DataStore: {ds_len} rows\n" f"Pandas:    {pd_len} rows"
    )

    # 3. Compare data values for each column
    columns_to_check = pd_columns if check_column_order else sorted(pd_columns)

    for col in columns_to_check:
        ds_values = np.asarray(ds_result[col].values)
        pd_values = np.asarray(pd_result[col].values)

        if not check_row_order:
            # Sort both for comparison when order doesn't matter
            ds_values = np.sort(ds_values)
            pd_values = np.sort(pd_values)

        _assert_array_equal(ds_values, pd_values, f"{prefix}Column '{col}' values don't match", check_dtype, rtol, atol)

    # 4. Optionally check dtypes
    if check_dtype:
        for col in columns_to_check:
            ds_dtype = ds_result[col].dtype
            pd_dtype = pd_result[col].dtype
            assert ds_dtype == pd_dtype, (
                f"{prefix}Column '{col}' dtype doesn't match.\n"
                f"DataStore dtype: {ds_dtype}\n"
                f"Pandas dtype:    {pd_dtype}"
            )


def _assert_series_equals(
    ds_result,
    pd_result: pd.Series,
    check_order: bool,
    check_dtype: bool,
    rtol: float,
    atol: float,
    prefix: str,
) -> None:
    """Helper to compare DataStore Series-like result with pandas Series."""
    ds_values = np.asarray(ds_result.values)
    pd_values = np.asarray(pd_result.values)

    # Compare length
    assert len(ds_values) == len(pd_values), (
        f"{prefix}Series length doesn't match.\n" f"DataStore: {len(ds_values)}\n" f"Pandas:    {len(pd_values)}"
    )

    if not check_order:
        ds_values = np.sort(ds_values)
        pd_values = np.sort(pd_values)

    _assert_array_equal(ds_values, pd_values, f"{prefix}Series values don't match", check_dtype, rtol, atol)

    if check_dtype:
        ds_dtype = ds_result.dtype
        pd_dtype = pd_result.dtype
        assert ds_dtype == pd_dtype, (
            f"{prefix}Series dtype doesn't match.\n" f"DataStore dtype: {ds_dtype}\n" f"Pandas dtype:    {pd_dtype}"
        )


def _assert_array_equal(
    ds_values: np.ndarray,
    pd_values: np.ndarray,
    err_msg: str,
    check_dtype: bool,
    rtol: float,
    atol: float,
) -> None:
    """Helper to compare two numpy arrays with proper handling of different dtypes."""
    # Check for floating point types - use allclose for tolerance
    if np.issubdtype(ds_values.dtype, np.floating) or np.issubdtype(pd_values.dtype, np.floating):
        # Handle NaN values
        ds_nan_mask = pd.isna(ds_values)
        pd_nan_mask = pd.isna(pd_values)

        np.testing.assert_array_equal(ds_nan_mask, pd_nan_mask, err_msg=f"{err_msg} (NaN positions differ)")

        # Compare non-NaN values with tolerance
        if not np.all(ds_nan_mask):
            np.testing.assert_allclose(
                ds_values[~ds_nan_mask].astype(float),
                pd_values[~pd_nan_mask].astype(float),
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
    else:
        # For non-float types, use exact comparison
        # Handle object dtype (strings, mixed types)
        if ds_values.dtype == object or pd_values.dtype == object:
            # Convert to string for comparison to handle None/NaN consistently
            ds_str = np.array([str(x) if pd.notna(x) else None for x in ds_values])
            pd_str = np.array([str(x) if pd.notna(x) else None for x in pd_values])
            np.testing.assert_array_equal(ds_str, pd_str, err_msg=err_msg)
        else:
            np.testing.assert_array_equal(ds_values, pd_values, err_msg=err_msg)


def assert_column_values_equal(
    ds_result,
    pd_result: pd.DataFrame,
    column: str,
    check_order: bool = True,
    msg: str = "",
) -> None:
    """
    Compare a single column between DataStore and pandas results.

    Args:
        ds_result: DataStore result
        pd_result: pandas DataFrame
        column: Column name to compare
        check_order: If True, order must match
        msg: Additional error message
    """
    prefix = f"{msg}: " if msg else ""

    ds_values = np.asarray(ds_result[column].values)
    pd_values = np.asarray(pd_result[column].values)

    if not check_order:
        ds_values = np.sort(ds_values)
        pd_values = np.sort(pd_values)

    np.testing.assert_array_equal(ds_values, pd_values, err_msg=f"{prefix}Column '{column}' values don't match")


def assert_columns_match(
    ds_result,
    pd_result: pd.DataFrame,
    check_order: bool = True,
    msg: str = "",
) -> None:
    """
    Compare only column names between DataStore and pandas results.

    Args:
        ds_result: DataStore result
        pd_result: pandas DataFrame
        check_order: If True, column order must match
        msg: Additional error message
    """
    prefix = f"{msg}: " if msg else ""

    ds_columns = list(ds_result.columns)
    pd_columns = list(pd_result.columns)

    if check_order:
        assert ds_columns == pd_columns, (
            f"{prefix}Column names or order don't match.\n" f"DataStore: {ds_columns}\n" f"Pandas:    {pd_columns}"
        )
    else:
        assert set(ds_columns) == set(pd_columns), (
            f"{prefix}Column names don't match.\n" f"DataStore: {set(ds_columns)}\n" f"Pandas:    {set(pd_columns)}"
        )


def assert_row_count_match(
    ds_result,
    pd_result: Union[pd.DataFrame, pd.Series],
    msg: str = "",
) -> None:
    """
    Compare row counts between DataStore and pandas results.

    Args:
        ds_result: DataStore result
        pd_result: pandas DataFrame or Series
        msg: Additional error message
    """
    prefix = f"{msg}: " if msg else ""

    ds_len = len(ds_result)
    pd_len = len(pd_result)

    assert ds_len == pd_len, f"{prefix}Row count doesn't match.\n" f"DataStore: {ds_len}\n" f"Pandas:    {pd_len}"


def assert_datastore_equals_pandas_chdb_compat(
    ds_result,
    pd_result: pd.DataFrame,
    check_column_order: bool = True,
    check_row_order: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = "",
) -> None:
    """
    Compare DataStore result with pandas, tolerating chDB dtype differences.

    This is a convenience wrapper around assert_datastore_equals_pandas that:
    1. Normalizes chDB nullable dtypes (Float64 → float64, etc.)
    2. Skips dtype comparison (check_dtype=False)

    Use this for tests affected by chDB's dtype behavior.
    See test_chdb_dtype_differences.py for documentation of these differences.

    Args:
        ds_result: DataStore result (DataStore, LazySeries, or similar)
        pd_result: Expected pandas DataFrame or Series
        check_column_order: If True, column order must match exactly
        check_row_order: If True, row order must match exactly
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison
        msg: Additional message to include in assertion errors

    Example:
        # When dtype differences are acceptable
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)
    """
    # Get DataStore DataFrame
    if hasattr(ds_result, 'to_df'):
        ds_df = ds_result.to_df()
    elif hasattr(ds_result, '_get_df'):
        ds_df = ds_result._get_df()
    else:
        ds_df = ds_result

    # Normalize chDB dtypes
    ds_df_normalized = _normalize_chdb_dtypes(ds_df)

    # Compare with dtype checking disabled
    assert_datastore_equals_pandas(
        ds_df_normalized,
        pd_result,
        check_column_order=check_column_order,
        check_row_order=check_row_order,
        check_dtype=False,
        rtol=rtol,
        atol=atol,
        msg=msg,
    )


def normalize_dataframe_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame for comparison, handling chDB dtype differences.

    Public wrapper for _normalize_chdb_dtypes.

    Args:
        df: DataFrame to normalize

    Returns:
        Normalized DataFrame with standard pandas dtypes
    """
    return _normalize_chdb_dtypes(df)
