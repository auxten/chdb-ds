"""
Benchmark: Cross-DataStore operations - SQL vs Pandas

Compares performance of:
1. Same DataStore (pure SQL path)
2. Cross-DataStore current impl (executor mode -> pandas fallback)
3. Pure pandas baseline

Tests at different data scales to understand when SQL vs pandas is better.
"""

import time
import pandas as pd
import numpy as np
from datastore import DataStore


def benchmark(func, name, warmup=1, runs=5):
    """Run benchmark with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        # Force execution if lazy
        if hasattr(result, '_execute'):
            result = result._execute()
        elif hasattr(result, '__len__'):
            _ = len(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg = np.mean(times)
    std = np.std(times)
    print(f"  {name}: {avg:.2f} ms (±{std:.2f})")
    return avg


def run_benchmark_suite(n_rows):
    """Run benchmark suite for given number of rows."""
    print(f"\n{'='*60}")
    print(f"Data size: {n_rows:,} rows")
    print('='*60)
    
    # Create test data
    np.random.seed(42)
    data1 = {
        'a': np.random.randn(n_rows),
        'b': np.random.randn(n_rows),
    }
    data2 = {
        'c': np.random.randn(n_rows),
        'd': np.random.randn(n_rows),
    }
    
    # Create DataStores and DataFrames
    ds1 = DataStore(data1)
    ds2 = DataStore(data2)
    ds_combined = DataStore({**data1, **data2})
    
    pdf1 = pd.DataFrame(data1)
    pdf2 = pd.DataFrame(data2)
    
    print("\n--- Single arithmetic operation ---")
    
    # 1. Same DataStore (SQL path)
    def same_ds_add():
        return ds_combined['a'] + ds_combined['c']
    
    # 2. Cross DataStore (current impl: executor -> pandas)
    def cross_ds_add():
        return ds1['a'] + ds2['c']
    
    # 3. Pure pandas baseline
    def pandas_add():
        return pdf1['a'] + pdf2['c']
    
    t_same = benchmark(same_ds_add, "Same DataStore (SQL)")
    t_cross = benchmark(cross_ds_add, "Cross DataStore (executor)")
    t_pandas = benchmark(pandas_add, "Pure pandas")
    
    print(f"\n  Ratio: Cross/Same = {t_cross/t_same:.2f}x, Cross/Pandas = {t_cross/t_pandas:.2f}x")
    
    print("\n--- Chained operations: (a + b) * 2 + 100 ---")
    
    # 1. Same DataStore - all SQL
    def same_ds_chain():
        return (ds_combined['a'] + ds_combined['c']) * 2 + 100
    
    # 2. Cross DataStore - first op executor, rest pandas
    def cross_ds_chain():
        return (ds1['a'] + ds2['c']) * 2 + 100
    
    # 3. Pure pandas
    def pandas_chain():
        return (pdf1['a'] + pdf2['c']) * 2 + 100
    
    t_same = benchmark(same_ds_chain, "Same DataStore (SQL)")
    t_cross = benchmark(cross_ds_chain, "Cross DataStore (executor->pandas)")
    t_pandas = benchmark(pandas_chain, "Pure pandas")
    
    print(f"\n  Ratio: Cross/Same = {t_cross/t_same:.2f}x, Cross/Pandas = {t_cross/t_pandas:.2f}x")
    
    print("\n--- Multiple cross-DS ops: ds1['a'] + ds2['c'] + ds1['b'] ---")
    
    # 1. Same DataStore - all SQL
    def same_ds_multi():
        return ds_combined['a'] + ds_combined['c'] + ds_combined['b']
    
    # 2. Cross DataStore - cascading
    def cross_ds_multi():
        return ds1['a'] + ds2['c'] + ds1['b']
    
    # 3. Pure pandas
    def pandas_multi():
        return pdf1['a'] + pdf2['c'] + pdf1['b']
    
    t_same = benchmark(same_ds_multi, "Same DataStore (SQL)")
    t_cross = benchmark(cross_ds_multi, "Cross DataStore (cascading)")
    t_pandas = benchmark(pandas_multi, "Pure pandas")
    
    print(f"\n  Ratio: Cross/Same = {t_cross/t_same:.2f}x, Cross/Pandas = {t_cross/t_pandas:.2f}x")
    
    print("\n--- Comparison: (a > c) & (b < d) ---")
    
    # 1. Same DataStore - SQL
    def same_ds_cmp():
        return (ds_combined['a'] > ds_combined['c']) & (ds_combined['b'] < ds_combined['d'])
    
    # 2. Cross DataStore
    def cross_ds_cmp():
        return (ds1['a'] > ds2['c']) & (ds1['b'] < ds2['d'])
    
    # 3. Pure pandas
    def pandas_cmp():
        return (pdf1['a'] > pdf2['c']) & (pdf1['b'] < pdf2['d'])
    
    t_same = benchmark(same_ds_cmp, "Same DataStore (SQL)")
    t_cross = benchmark(cross_ds_cmp, "Cross DataStore")
    t_pandas = benchmark(pandas_cmp, "Pure pandas")
    
    print(f"\n  Ratio: Cross/Same = {t_cross/t_same:.2f}x, Cross/Pandas = {t_cross/t_pandas:.2f}x")
    
    return {
        'n_rows': n_rows,
        't_same': t_same,
        't_cross': t_cross,
        't_pandas': t_pandas,
    }


def run_sql_continuation_comparison(n_rows):
    """
    Compare: After cross-DS op, should we continue with SQL or pandas?
    
    Scenario: result = (ds1['a'] + ds2['c']) * factor + offset
    
    Current: cross_ds -> executor mode -> subsequent ops use pandas
    Alternative: cross_ds -> materialize to new DataStore -> continue SQL
    """
    print(f"\n{'='*60}")
    print(f"SQL Continuation Analysis: {n_rows:,} rows")
    print('='*60)
    
    np.random.seed(42)
    data1 = {'a': np.random.randn(n_rows)}
    data2 = {'c': np.random.randn(n_rows)}
    
    ds1 = DataStore(data1)
    ds2 = DataStore(data2)
    pdf1 = pd.DataFrame(data1)
    pdf2 = pd.DataFrame(data2)
    
    print("\n--- Current impl: executor -> pandas fallback ---")
    
    def current_impl():
        step1 = ds1['a'] + ds2['c']  # executor mode
        step2 = step1 * 2            # method mode (pandas)
        step3 = step2 + 100          # method mode (pandas)
        return step3
    
    t_current = benchmark(current_impl, "Current (executor->pandas)")
    
    print("\n--- Alternative: materialize to DataStore -> SQL ---")
    
    def materialize_to_ds():
        # Step 1: cross-DS op
        step1_result = (ds1['a'] + ds2['c'])._execute()
        # Step 2: wrap in new DataStore, continue with SQL
        ds_temp = DataStore({'val': step1_result.values})
        step2 = ds_temp['val'] * 2 + 100
        return step2
    
    t_materialize = benchmark(materialize_to_ds, "Materialize -> SQL")
    
    print("\n--- Pure pandas baseline ---")
    
    def pure_pandas():
        step1 = pdf1['a'] + pdf2['c']
        step2 = step1 * 2
        step3 = step2 + 100
        return step3
    
    t_pandas = benchmark(pure_pandas, "Pure pandas")
    
    print(f"\n  Summary:")
    print(f"    Current (executor->pandas): {t_current:.2f} ms")
    print(f"    Materialize -> SQL:         {t_materialize:.2f} ms")
    print(f"    Pure pandas:                {t_pandas:.2f} ms")
    print(f"    Ratio Materialize/Current:  {t_materialize/t_current:.2f}x")


def main():
    print("Cross-DataStore Operations Benchmark")
    print("=" * 60)
    
    # Test different scales
    scales = [1_000, 10_000, 100_000, 1_000_000]
    
    for n in scales:
        run_benchmark_suite(n)
    
    print("\n" + "=" * 60)
    print("SQL CONTINUATION ANALYSIS")
    print("=" * 60)
    
    for n in [10_000, 100_000, 1_000_000]:
        run_sql_continuation_comparison(n)
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
Based on the benchmarks, consider:
1. If Cross/Pandas ratio ≈ 1: current impl is fine
2. If Cross/Pandas ratio >> 1: overhead from executor mode
3. If Materialize->SQL < Current: worth optimizing to continue SQL

Key insight: The overhead comes from:
- Creating executor closure
- Double execution (self + other)
- pandas Series alignment

For large data (>100K rows), SQL typically wins for:
- Complex expressions
- Multiple chained operations
- Aggregations
""")


if __name__ == '__main__':
    main()
