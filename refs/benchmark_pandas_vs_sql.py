#!/usr/bin/env python3
"""
Benchmark: Pandas vs chDB SQL operations on DataFrames

This benchmark compares the performance of:
1. Pure Pandas operations (current implementation)
2. chDB SQL execution on DataFrames (using Python() table function)

Operations tested:
- Filter (single condition)
- Filter (multiple conditions)
- Select columns
- Sort
- Aggregation (GROUP BY)
- Combined operations
"""

import time
import pandas as pd
import numpy as np
import chdb
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    operation: str
    data_size: int
    pandas_time: float
    chdb_time: float

    @property
    def speedup(self) -> float:
        if self.chdb_time == 0:
            return float('inf')
        return self.pandas_time / self.chdb_time

    @property
    def faster(self) -> str:
        if self.pandas_time < self.chdb_time:
            return f"Pandas ({self.chdb_time/self.pandas_time:.2f}x)"
        else:
            return f"chDB ({self.pandas_time/self.chdb_time:.2f}x)"


def generate_test_data(n_rows: int) -> pd.DataFrame:
    """Generate test DataFrame with various column types."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            'id': range(n_rows),
            'int_col': np.random.randint(0, 1000, n_rows),
            'float_col': np.random.uniform(0, 1000, n_rows),
            'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'category': np.random.choice(
                ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10'], n_rows
            ),
            'bool_col': np.random.choice([True, False], n_rows),
            'date_col': pd.date_range('2020-01-01', periods=n_rows, freq='s')[:n_rows],
        }
    )

    return df


def time_operation(func: Callable, n_runs: int = 5) -> float:
    """Time an operation, return average time in milliseconds."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func()
        # Force evaluation for lazy results
        if isinstance(result, pd.DataFrame):
            _ = len(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Remove outliers and return average
    times.sort()
    if len(times) > 2:
        times = times[1:-1]  # Remove min and max
    return sum(times) / len(times)


# Global connection for reuse (initialized in main())
conn = None


class Benchmark:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_rows = len(df)

    # ==================== Filter Operations ====================

    def pandas_filter_single(self) -> pd.DataFrame:
        return self.df[self.df['int_col'] > 500]

    def chdb_filter_single(self) -> pd.DataFrame:
        df = self.df  # Make df available in local scope for chDB
        return conn.query("SELECT * FROM Python(df) WHERE int_col > 500", 'DataFrame')

    def pandas_filter_multiple(self) -> pd.DataFrame:
        return self.df[(self.df['int_col'] > 300) & (self.df['int_col'] < 700) & (self.df['str_col'] == 'A')]

    def chdb_filter_multiple(self) -> pd.DataFrame:
        df = self.df
        return conn.query(
            "SELECT * FROM Python(df) WHERE int_col > 300 AND int_col < 700 AND str_col = 'A'", 'DataFrame'
        )

    # ==================== Select Operations ====================

    def pandas_select_columns(self) -> pd.DataFrame:
        return self.df[['id', 'int_col', 'str_col']]

    def chdb_select_columns(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT id, int_col, str_col FROM Python(df)", 'DataFrame')

    # ==================== Sort Operations ====================

    def pandas_sort_single(self) -> pd.DataFrame:
        return self.df.sort_values('int_col')

    def chdb_sort_single(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT * FROM Python(df) ORDER BY int_col", 'DataFrame')

    def pandas_sort_multiple(self) -> pd.DataFrame:
        return self.df.sort_values(['str_col', 'int_col'], ascending=[True, False])

    def chdb_sort_multiple(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT * FROM Python(df) ORDER BY str_col ASC, int_col DESC", 'DataFrame')

    # ==================== Aggregation Operations ====================

    def pandas_groupby_count(self) -> pd.DataFrame:
        return self.df.groupby('str_col').size().reset_index(name='count')

    def chdb_groupby_count(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT str_col, count(*) as count FROM Python(df) GROUP BY str_col", 'DataFrame')

    def pandas_groupby_agg(self) -> pd.DataFrame:
        return (
            self.df.groupby('category')
            .agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']})
            .reset_index()
        )

    def chdb_groupby_agg(self) -> pd.DataFrame:
        df = self.df
        return conn.query(
            """SELECT category, 
                      sum(int_col) as int_sum, 
                      avg(int_col) as int_mean,
                      max(int_col) as int_max,
                      sum(float_col) as float_sum,
                      avg(float_col) as float_mean
               FROM Python(df) 
               GROUP BY category""",
            'DataFrame',
        )

    # ==================== Combined Operations ====================

    def pandas_combined(self) -> pd.DataFrame:
        result = self.df[self.df['int_col'] > 200]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.head(100)

    def chdb_combined(self) -> pd.DataFrame:
        df = self.df
        return conn.query(
            """SELECT id, int_col, str_col, float_col 
               FROM Python(df) 
               WHERE int_col > 200 
               ORDER BY int_col DESC 
               LIMIT 100""",
            'DataFrame',
        )

    # ==================== Head/Limit Operations ====================

    def pandas_head(self) -> pd.DataFrame:
        return self.df.head(1000)

    def chdb_head(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT * FROM Python(df) LIMIT 1000", 'DataFrame')

    # ==================== Multi-Step Operations ====================
    # These simulate the DataStore lazy execution pattern where multiple
    # operations are chained together

    def pandas_multi_filter(self) -> pd.DataFrame:
        """Multiple filter operations in sequence (like DataStore Phase 2)."""
        result = self.df[self.df['int_col'] > 200]
        result = result[result['int_col'] < 800]
        result = result[result['str_col'].isin(['A', 'B', 'C'])]
        result = result[result['float_col'] > 100]
        return result

    def chdb_multi_filter(self) -> pd.DataFrame:
        """Single SQL with all filters combined."""
        df = self.df
        return conn.query(
            """SELECT * FROM Python(df) 
               WHERE int_col > 200 
                 AND int_col < 800 
                 AND str_col IN ('A', 'B', 'C')
                 AND float_col > 100""",
            'DataFrame',
        )

    def pandas_filter_select_sort(self) -> pd.DataFrame:
        """Filter -> Select columns -> Sort (common pattern)."""
        result = self.df[self.df['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result

    def chdb_filter_select_sort(self) -> pd.DataFrame:
        """Single SQL for filter + select + sort."""
        df = self.df
        return conn.query(
            """SELECT id, int_col, str_col, float_col 
               FROM Python(df) 
               WHERE int_col > 300 
               ORDER BY int_col DESC""",
            'DataFrame',
        )

    def pandas_filter_groupby_sort(self) -> pd.DataFrame:
        """Filter -> GroupBy -> Sort (analytics pattern)."""
        result = self.df[self.df['int_col'] > 200]
        result = result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
        result = result.sort_values('int_col', ascending=False)
        return result

    def chdb_filter_groupby_sort(self) -> pd.DataFrame:
        """Single SQL for filter + groupby + sort."""
        df = self.df
        return conn.query(
            """SELECT category, sum(int_col) as int_col, avg(float_col) as float_col
               FROM Python(df) 
               WHERE int_col > 200 
               GROUP BY category
               ORDER BY int_col DESC""",
            'DataFrame',
        )

    def pandas_complex_pipeline(self) -> pd.DataFrame:
        """Complex multi-step pipeline (simulates real DataStore usage)."""
        # Step 1: Filter
        result = self.df[self.df['int_col'] > 100]
        # Step 2: Add computed column
        result = result.copy()
        result['computed'] = result['int_col'] * 2 + result['float_col']
        # Step 3: Another filter on computed column
        result = result[result['computed'] > 500]
        # Step 4: Select columns
        result = result[['id', 'int_col', 'str_col', 'computed']]
        # Step 5: Sort
        result = result.sort_values('computed', ascending=False)
        # Step 6: Limit
        return result.head(500)

    def chdb_complex_pipeline(self) -> pd.DataFrame:
        """Single SQL for complex pipeline."""
        df = self.df
        return conn.query(
            """SELECT id, int_col, str_col, (int_col * 2 + float_col) as computed
               FROM Python(df) 
               WHERE int_col > 100 
                 AND (int_col * 2 + float_col) > 500
               ORDER BY computed DESC
               LIMIT 500""",
            'DataFrame',
        )

    def pandas_chain_5_filters(self) -> pd.DataFrame:
        """5 sequential filter operations."""
        result = self.df[self.df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['float_col'] < 950]
        result = result[result['str_col'] != 'E']
        return result

    def chdb_chain_5_filters(self) -> pd.DataFrame:
        """Single SQL with 5 conditions."""
        df = self.df
        return conn.query(
            """SELECT * FROM Python(df) 
               WHERE int_col > 100 
                 AND int_col < 900
                 AND float_col > 50
                 AND float_col < 950
                 AND str_col != 'E'""",
            'DataFrame',
        )


def run_benchmarks(data_sizes: List[int], n_runs: int = 5) -> List[BenchmarkResult]:
    """Run all benchmarks for different data sizes."""
    results = []

    operations = [
        # Single operations
        ('Filter (single)', 'pandas_filter_single', 'chdb_filter_single'),
        ('Filter (multiple)', 'pandas_filter_multiple', 'chdb_filter_multiple'),
        ('Select columns', 'pandas_select_columns', 'chdb_select_columns'),
        ('Sort (single)', 'pandas_sort_single', 'chdb_sort_single'),
        ('Sort (multiple)', 'pandas_sort_multiple', 'chdb_sort_multiple'),
        ('GroupBy count', 'pandas_groupby_count', 'chdb_groupby_count'),
        ('GroupBy agg', 'pandas_groupby_agg', 'chdb_groupby_agg'),
        ('Combined ops', 'pandas_combined', 'chdb_combined'),
        ('Head/Limit', 'pandas_head', 'chdb_head'),
        # Multi-step operations (simulates DataStore lazy execution)
        ('Multi-filter (4x)', 'pandas_multi_filter', 'chdb_multi_filter'),
        ('Filter+Select+Sort', 'pandas_filter_select_sort', 'chdb_filter_select_sort'),
        ('Filter+GroupBy+Sort', 'pandas_filter_groupby_sort', 'chdb_filter_groupby_sort'),
        ('Complex pipeline', 'pandas_complex_pipeline', 'chdb_complex_pipeline'),
        ('Chain 5 filters', 'pandas_chain_5_filters', 'chdb_chain_5_filters'),
    ]

    for size in data_sizes:
        print(f"\n{'='*60}")
        print(f"Generating {size:,} rows of test data...")
        df = generate_test_data(size)
        benchmark = Benchmark(df)

        print(f"Running benchmarks (n_runs={n_runs})...")

        for op_name, pandas_method, chdb_method in operations:
            pandas_func = getattr(benchmark, pandas_method)
            chdb_func = getattr(benchmark, chdb_method)

            # Warm up
            try:
                pandas_func()
                chdb_func()
            except Exception as e:
                print(f"  Skipping {op_name}: {e}")
                continue

            # Benchmark
            pandas_time = time_operation(pandas_func, n_runs)
            chdb_time = time_operation(chdb_func, n_runs)

            result = BenchmarkResult(operation=op_name, data_size=size, pandas_time=pandas_time, chdb_time=chdb_time)
            results.append(result)

            print(f"  {op_name:20s}: Pandas={pandas_time:8.2f}ms, chDB={chdb_time:8.2f}ms -> {result.faster}")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print summary table of results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    # Group by data size
    sizes = sorted(set(r.data_size for r in results))
    operations = sorted(set(r.operation for r in results), key=lambda x: [r.operation for r in results].index(x))

    # Print header
    print(f"\n{'Operation':<20}", end='')
    for size in sizes:
        print(f" | {size:>12,} rows", end='')
    print()
    print("-" * 20, end='')
    for _ in sizes:
        print("-" + "-" * 17, end='')
    print()

    # Print results
    for op in operations:
        print(f"{op:<20}", end='')
        for size in sizes:
            matching = [r for r in results if r.operation == op and r.data_size == size]
            if matching:
                r = matching[0]
                if r.pandas_time < r.chdb_time:
                    ratio = r.chdb_time / r.pandas_time
                    print(f" | Pandas {ratio:>5.1f}x  ", end='')
                else:
                    ratio = r.pandas_time / r.chdb_time
                    print(f" |  chDB {ratio:>5.1f}x  ", end='')
            else:
                print(f" | {'N/A':>12}", end='')
        print()

    # Print detailed times
    print("\n" + "=" * 100)
    print("DETAILED TIMES (milliseconds)")
    print("=" * 100)

    print(f"\n{'Operation':<20} | {'Size':>10} | {'Pandas':>10} | {'chDB':>10} | {'Winner':>15}")
    print("-" * 75)

    for r in results:
        print(
            f"{r.operation:<20} | {r.data_size:>10,} | {r.pandas_time:>10.2f} | {r.chdb_time:>10.2f} | {r.faster:>15}"
        )


def main():
    global conn

    print("=" * 60)
    print("Pandas vs chDB SQL Benchmark")
    print("=" * 60)

    # Initialize chdb session to avoid ~15ms per-query initialization overhead
    conn = chdb.connect()
    print(f"chdb version: {chdb.__version__}")
    print("chdb session initialized (using conn.query() for best performance)")

    # Test different data sizes
    data_sizes = [1_000, 10_000, 100_000, 1_000_000]

    # Run benchmarks
    results = run_benchmarks(data_sizes, n_runs=5)

    # Print summary
    print_summary(results)

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    # Analyze results
    pandas_wins = sum(1 for r in results if r.pandas_time < r.chdb_time)
    chdb_wins = len(results) - pandas_wins

    print(f"\nOverall: Pandas won {pandas_wins}/{len(results)} tests, chDB won {chdb_wins}/{len(results)} tests")

    # Group by size
    for size in sorted(set(r.data_size for r in results)):
        size_results = [r for r in results if r.data_size == size]
        pandas_better = sum(1 for r in size_results if r.pandas_time < r.chdb_time)
        print(f"\n  {size:>10,} rows: Pandas better in {pandas_better}/{len(size_results)} operations")

    # Specific recommendations
    print("\nPer-operation analysis:")
    for op in sorted(set(r.operation for r in results), key=lambda x: [r.operation for r in results].index(x)):
        op_results = [r for r in results if r.operation == op]
        pandas_better = sum(1 for r in op_results if r.pandas_time < r.chdb_time)

        if pandas_better > len(op_results) / 2:
            rec = "Use Pandas"
        elif pandas_better < len(op_results) / 2:
            rec = "Use chDB"
        else:
            rec = "Depends on data size"

        print(f"  {op:<20}: {rec}")


if __name__ == '__main__':
    main()
