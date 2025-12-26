"""
Test documenting dtype differences between chDB and pandas.

When data passes through chDB's Python() table function, the returned DataFrame
may have different dtypes than the original:

1. Float columns with NaN: pandas `float64` → chDB `Float64Dtype()` (nullable)
2. Integer columns with None: converted to `Float64Dtype()`
3. Datetime columns: naive `datetime64[ns]` → timezone-aware `datetime64[ns, <timezone>]`

These are known chDB behaviors documented here for tracking.
See: https://github.com/chdb-io/chdb/issues/XXX (to be filed)

The values are semantically equivalent (NaN positions match, values match),
but the dtype representation differs.
"""

import numpy as np
import pandas as pd
import pytest

import chdb


class TestChDBDtypeDifferences:
    """Document known dtype differences between chDB output and pandas."""

    @pytest.mark.xfail(
        reason="chDB converts float64 with NaN to nullable Float64Dtype",
        strict=True,
    )
    def test_float_nan_dtype_preservation(self):
        """
        chDB converts float64 columns containing NaN to nullable Float64Dtype.

        Original: float64 with NaN (numpy.nan)
        After chDB: Float64 with <NA> (pandas NA)

        The values are equivalent, but DataFrame.equals() returns False due to dtype.
        """
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        assert df["a"].dtype == np.float64

        result = chdb.query("SELECT * FROM Python(df)", "DataFrame")

        # This will fail: chDB returns Float64Dtype() instead of float64
        assert result["a"].dtype == np.float64

    @pytest.mark.xfail(
        reason="chDB converts integer-like columns with None to Float64Dtype",
        strict=True,
    )
    def test_integer_none_dtype_preservation(self):
        """
        chDB converts columns with None to nullable Float64Dtype.

        Note: pandas converts int-like list with None to float64 (not int64),
        and chDB further converts it to Float64Dtype.
        """
        df = pd.DataFrame({"a": [1, 2, None, 4]})
        original_dtype = df["a"].dtype  # float64 (because of None)

        result = chdb.query("SELECT * FROM Python(df)", "DataFrame")

        # This will fail: chDB returns Float64Dtype()
        assert result["a"].dtype == original_dtype

    @pytest.mark.xfail(
        reason="chDB adds timezone to naive datetime columns",
        strict=True,
    )
    def test_datetime_timezone_preservation(self):
        """
        chDB adds system timezone to naive datetime columns.

        Original: datetime64[ns] (naive)
        After chDB: datetime64[ns, <system_timezone>] (timezone-aware)
        """
        df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01", "2021-01-02"])})
        assert df["dt"].dtype == "datetime64[ns]"

        result = chdb.query("SELECT * FROM Python(df)", "DataFrame")

        # This will fail: chDB adds timezone
        assert result["dt"].dtype == df["dt"].dtype

class TestChDBArrayNullableLimitation:
    """Document ClickHouse's Array(T) in Nullable limitation and workaround."""

    def test_chdb_direct_split_null_becomes_empty_array(self):
        """
        When executing splitByWhitespace directly in chDB with ifNull workaround,
        NULL becomes empty array [].

        This is the raw chDB behavior without DataStore's pandas execution path.
        """
        df = pd.DataFrame({'text': ['hello world', None, 'foo bar']})

        # Direct chDB execution with ifNull workaround
        sql = "SELECT splitByWhitespace(ifNull(text, '')) AS result FROM Python(df)"
        result = chdb.query(sql, 'DataFrame')

        # chDB converts NULL -> '' -> [] (empty array)
        result_list = result['result'].tolist()
        assert len(result_list) == 3
        assert list(result_list[0]) == ['hello', 'world']
        assert list(result_list[1]) == []  # NULL became empty array
        assert list(result_list[2]) == ['foo', 'bar']

    def test_datastore_split_preserves_none_via_pandas_fallback(self):
        """
        DataStore's str.split() may use pandas fallback execution,
        which preserves None values like pandas does.

        This is actually the CORRECT pandas-compatible behavior.
        """
        from datastore import DataStore

        df = pd.DataFrame({'text': ['hello world', None, 'foo bar']})

        # pandas behavior
        pd_result = df['text'].str.split()

        # DataStore behavior (may use pandas fallback)
        ds = DataStore.from_df(df)
        ds_result = ds['text'].str.split().to_pandas()

        # Both should preserve None for NULL values
        assert pd_result.iloc[1] is None
        assert ds_result.iloc[1] is None

    @pytest.mark.xfail(
        reason="chDB doesn't support Nullable(Array), raw SQL needs ifNull workaround",
        strict=True,
    )
    def test_raw_sql_split_without_ifnull_fails(self):
        """
        Using splitByWhitespace directly in SQL without ifNull fails
        with 'Nested type Array(String) cannot be inside Nullable type' error.

        This documents the ClickHouse limitation that requires the ifNull workaround.
        """
        df = pd.DataFrame({'text': ['hello world', None, 'foo bar']})

        # This should fail: no ifNull wrapper
        result = chdb.query("SELECT splitByWhitespace(text) FROM Python(df)", 'DataFrame')
        assert len(result) == 3
