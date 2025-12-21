#!/usr/bin/env python3
"""
Benchmark: DataStore (chDB lazy mode) vs Pure Pandas

This benchmark compares the performance of:
1. Pure Pandas operations (eager evaluation - each step executes immediately)
2. DataStore lazy execution (multiple operations merged into single SQL via chDB)

Key insight: DataStore's lazy execution can merge multiple operations (filter, sort,
groupby, etc.) into a single SQL query, which is especially advantageous for:
- Complex multi-step pipelines
- Chained filter operations
- Filter + GroupBy + Sort patterns

Operations tested:
- Single operations (filter, sort, groupby)
- Multi-step pipelines (where DataStore shines)
"""

import time
import tempfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
from dataclasses import dataclass
from collections import Counter

# Import DataStore
from datastore import DataStore


@dataclass
class BenchmarkResult:
    operation: str
    data_size: int
    pandas_time: float
    datastore_time: float

    @property
    def fastest(self) -> str:
        times = {'Pandas': self.pandas_time, 'DataStore': self.datastore_time}
        winner = min(times, key=times.get)
        winner_time = times[winner]
        loser_time = max(times.values())
        speedup = loser_time / winner_time if winner_time > 0 else float('inf')
        return f"{winner} ({speedup:.2f}x)"

    @property
    def speedup(self) -> float:
        """Speedup of DataStore over Pandas (>1 means DataStore is faster)"""
        return self.pandas_time / self.datastore_time if self.datastore_time > 0 else float('inf')


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
        elif hasattr(result, 'to_df'):
            # DataStore - force materialization
            _ = len(result.to_df())
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Remove outliers and return average
    times.sort()
    if len(times) > 2:
        times = times[1:-1]  # Remove min and max
    return sum(times) / len(times)


class Benchmark:
    """
    Benchmark class for comparing Pandas vs DataStore performance.

    IMPORTANT: For fair comparison, both should operate on the same data source.
    - When comparing file operations: both read from file
    - When comparing in-memory operations: both use in-memory data

    The benchmark uses parquet files as the source, which is the typical
    DataStore use case (lazy SQL execution on file sources).
    """

    def __init__(self, df: pd.DataFrame, parquet_path: str):
        self.df = df
        self.parquet_path = parquet_path
        self.n_rows = len(df)
        # Pre-connect DataStore to avoid connection overhead in tight loops
        self._ds_template = DataStore.from_file(self.parquet_path)
        self._ds_template.connect()

    def _fresh_ds(self) -> DataStore:
        """Create a fresh DataStore from the same file source."""
        return DataStore.from_file(self.parquet_path)

    # ==================== Filter Operations ====================

    def pandas_filter_single(self) -> pd.DataFrame:
        # Fair comparison: read from file like DataStore does
        df = pd.read_parquet(self.parquet_path)
        return df[df['int_col'] > 500]

    def datastore_filter_single(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds[ds['int_col'] > 500].to_df()

    def pandas_filter_multiple(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df[(df['int_col'] > 300) & (df['int_col'] < 700) & (df['str_col'] == 'A')]

    def datastore_filter_multiple(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds[(ds['int_col'] > 300) & (ds['int_col'] < 700) & (ds['str_col'] == 'A')].to_df()

    # ==================== Sort Operations ====================

    def pandas_sort_single(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.sort_values('int_col')

    def datastore_sort_single(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.sort_values('int_col').to_df()

    def pandas_sort_multiple(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.sort_values(['str_col', 'int_col'], ascending=[True, False])

    def datastore_sort_multiple(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.sort_values(['str_col', 'int_col'], ascending=[True, False]).to_df()

    # ==================== GroupBy Operations ====================

    def pandas_groupby_count(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.groupby('str_col').size().reset_index(name='count')

    def datastore_groupby_count(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.groupby('str_col').size().reset_index(name='count')

    def pandas_groupby_agg(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return (
            df.groupby('category').agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']}).reset_index()
        )

    def datastore_groupby_agg(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return (
            ds.groupby('category').agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']}).reset_index()
        )

    # ==================== Head/Limit Operations ====================

    def pandas_head(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.head(1000)

    def datastore_head(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.head(1000).to_df()

    # ==================== Combined Operations ====================

    def pandas_combined(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.head(100)

    def datastore_combined(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.head(100).to_df()

    # ==================== Multi-Step Operations (DataStore Advantage) ====================
    # DataStore merges multiple operations into single SQL query

    def pandas_multi_filter(self) -> pd.DataFrame:
        """Multiple filter operations - Pandas executes each step separately."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result[result['int_col'] < 800]
        result = result[result['str_col'].isin(['A', 'B', 'C'])]
        result = result[result['float_col'] > 100]
        return result

    def datastore_multi_filter(self) -> pd.DataFrame:
        """DataStore merges all filters into single SQL WHERE clause."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        result = result[result['int_col'] < 800]
        result = result[result['str_col'].isin(['A', 'B', 'C'])]
        result = result[result['float_col'] > 100]
        return result.to_df()

    def pandas_filter_select_sort(self) -> pd.DataFrame:
        """Filter -> Select -> Sort: Pandas processes step by step."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result

    def datastore_filter_select_sort(self) -> pd.DataFrame:
        """DataStore merges filter + select + sort into single optimized SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.to_df()

    def pandas_filter_groupby_sort(self) -> pd.DataFrame:
        """Filter -> GroupBy -> Sort: Analytics pattern."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
        result.columns = ['category', 'int_sum', 'float_avg']
        result = result.sort_values('int_sum', ascending=False)
        return result

    def datastore_filter_groupby_sort(self) -> pd.DataFrame:
        """DataStore: filter before groupby is pushed down to SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        result = result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
        result.columns = ['category', 'int_sum', 'float_avg']
        return result.sort_values('int_sum', ascending=False)

    def pandas_complex_pipeline(self) -> pd.DataFrame:
        """Complex multi-step pipeline (simulates real usage)."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result.copy()
        result['computed'] = result['int_col'] * 2 + result['float_col']
        result = result[result['computed'] > 500]
        result = result[['id', 'int_col', 'str_col', 'computed']]
        result = result.sort_values('computed', ascending=False)
        return result.head(500)

    def datastore_complex_pipeline(self) -> pd.DataFrame:
        """DataStore handles complex pipeline with lazy execution."""
        result = self._fresh_ds()
        result = result[result['int_col'] > 100]
        result['computed'] = result['int_col'] * 2 + result['float_col']
        result = result[result['computed'] > 500]
        result = result[['id', 'int_col', 'str_col', 'computed']]
        result = result.sort_values('computed', ascending=False)
        return result.head(500).to_df()

    def pandas_chain_5_filters(self) -> pd.DataFrame:
        """5 sequential filter operations - each creates intermediate DataFrame."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['float_col'] < 950]
        result = result[result['str_col'] != 'E']
        return result

    def datastore_chain_5_filters(self) -> pd.DataFrame:
        """DataStore merges 5 filters into single SQL WHERE with AND."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['float_col'] < 950]
        result = result[result['str_col'] != 'E']
        return result.to_df()

    # ==================== Ultra-Complex Pipeline (DataStore Advantage) ====================

    def pandas_ultra_complex(self) -> pd.DataFrame:
        """Ultra-complex pipeline with 10+ operations."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        result = result[result['bool_col'] == True]  # noqa: E712
        result = result[['id', 'int_col', 'float_col', 'str_col', 'category']]
        result = result.sort_values(['category', 'int_col'], ascending=[True, False])
        return result.head(1000)

    def datastore_ultra_complex(self) -> pd.DataFrame:
        """DataStore merges all operations into single optimized SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        result = result[result['bool_col'] == True]  # noqa: E712
        result = result[['id', 'int_col', 'float_col', 'str_col', 'category']]
        result = result.sort_values(['category', 'int_col'], ascending=[True, False])
        return result.head(1000).to_df()

    # ==================== Pandas-Style Lazy API (Now Optimized!) ====================
    # After optimization, pandas-style API also uses lazy SQL execution

    def pandas_lazy_filter_sort_limit(self) -> pd.DataFrame:
        """Filter + Sort + Limit: Pandas baseline."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result.sort_values('int_col', ascending=False)
        return result.head(100)

    def datastore_lazy_filter_sort_limit(self) -> pd.DataFrame:
        """DataStore pandas-style: filter+sort_values+head merged into single SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        result = result.sort_values('int_col', ascending=False)
        return result.head(100).to_df()

    def pandas_lazy_multi_filter_sort_limit(self) -> pd.DataFrame:
        """Multi-filter + Sort + Limit: Pandas baseline."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        result = result.sort_values('int_col', ascending=False)
        return result.head(500)

    def datastore_lazy_multi_filter_sort_limit(self) -> pd.DataFrame:
        """DataStore pandas-style: all filters+sort_values+head merged into single SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        result = result.sort_values('int_col', ascending=False)
        return result.head(500).to_df()

    def pandas_lazy_select_filter_sort(self) -> pd.DataFrame:
        """Select + Filter + Sort: Pandas baseline."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col']]
        result = result.sort_values('int_col', ascending=False)
        return result

    def datastore_lazy_select_filter_sort(self) -> pd.DataFrame:
        """DataStore pandas-style: filter+column_select+sort_values merged into SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.to_df()


def run_benchmarks(data_sizes: List[int], temp_dir: str, n_runs: int = 5) -> List[BenchmarkResult]:
    """Run all benchmarks for different data sizes."""
    results = []

    operations = [
        # Single operations (pandas-style API)
        ('Filter (single)', 'pandas_filter_single', 'datastore_filter_single'),
        ('Filter (multiple AND)', 'pandas_filter_multiple', 'datastore_filter_multiple'),
        ('Sort (single)', 'pandas_sort_single', 'datastore_sort_single'),
        ('Sort (multiple)', 'pandas_sort_multiple', 'datastore_sort_multiple'),
        ('GroupBy count', 'pandas_groupby_count', 'datastore_groupby_count'),
        ('GroupBy agg', 'pandas_groupby_agg', 'datastore_groupby_agg'),
        ('Head/Limit', 'pandas_head', 'datastore_head'),
        ('Combined ops', 'pandas_combined', 'datastore_combined'),
        # Multi-step operations (pandas-style API)
        ('Multi-filter (4x)', 'pandas_multi_filter', 'datastore_multi_filter'),
        ('Filter+Select+Sort', 'pandas_filter_select_sort', 'datastore_filter_select_sort'),
        ('Filter+GroupBy+Sort', 'pandas_filter_groupby_sort', 'datastore_filter_groupby_sort'),
        ('Complex pipeline', 'pandas_complex_pipeline', 'datastore_complex_pipeline'),
        ('Chain 5 filters', 'pandas_chain_5_filters', 'datastore_chain_5_filters'),
        ('Ultra-complex (10+ ops)', 'pandas_ultra_complex', 'datastore_ultra_complex'),
        # Pandas-style lazy API (now optimized - sort_values/head use lazy SQL)
        ('Pandas-style: Filter+Sort+Head', 'pandas_lazy_filter_sort_limit', 'datastore_lazy_filter_sort_limit'),
        (
            'Pandas-style: MultiFilter+Sort+Head',
            'pandas_lazy_multi_filter_sort_limit',
            'datastore_lazy_multi_filter_sort_limit',
        ),
        ('Pandas-style: Select+Filter+Sort', 'pandas_lazy_select_filter_sort', 'datastore_lazy_select_filter_sort'),
    ]

    for size in data_sizes:
        print(f"\n{'='*60}")
        print(f"Generating {size:,} rows of test data...")
        df = generate_test_data(size)

        # Save to parquet file
        parquet_path = os.path.join(temp_dir, f"test_data_{size}.parquet")
        df.to_parquet(parquet_path)
        print(f"Saved to {parquet_path}")

        benchmark = Benchmark(df, parquet_path)

        print(f"Running benchmarks (n_runs={n_runs})...")

        for op_name, pandas_method, datastore_method in operations:
            pandas_func = getattr(benchmark, pandas_method)
            datastore_func = getattr(benchmark, datastore_method)

            # Warm up
            try:
                pandas_func()
                datastore_func()
            except Exception as e:
                print(f"  Skipping {op_name}: {e}")
                continue

            # Benchmark
            pandas_time = time_operation(pandas_func, n_runs)
            datastore_time = time_operation(datastore_func, n_runs)

            result = BenchmarkResult(
                operation=op_name,
                data_size=size,
                pandas_time=pandas_time,
                datastore_time=datastore_time,
            )
            results.append(result)

            print(
                f"  {op_name:25s}: Pandas={pandas_time:8.2f}ms, DataStore={datastore_time:8.2f}ms -> {result.fastest}"
            )

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
    print(f"\n{'Operation':<25}", end='')
    for size in sizes:
        print(f" | {size:>12,} rows", end='')
    print()
    print("-" * 25, end='')
    for _ in sizes:
        print("-" + "-" * 17, end='')
    print()

    # Print results
    for op in operations:
        print(f"{op:<25}", end='')
        for size in sizes:
            matching = [r for r in results if r.operation == op and r.data_size == size]
            if matching:
                r = matching[0]
                times = {'Pandas': r.pandas_time, 'DataStore': r.datastore_time}
                winner = min(times, key=times.get)
                winner_time = times[winner]
                loser_time = max(times.values())
                ratio = loser_time / winner_time if winner_time > 0 else float('inf')
                print(f" | {winner:>8} {ratio:>5.1f}x ", end='')
            else:
                print(f" | {'N/A':>14}", end='')
        print()

    # Print detailed times
    print("\n" + "=" * 100)
    print("DETAILED TIMES (milliseconds)")
    print("=" * 100)

    print(f"\n{'Operation':<25} | {'Size':>10} | {'Pandas':>10} | {'DataStore':>10} | {'Winner':>16}")
    print("-" * 85)

    for r in results:
        print(
            f"{r.operation:<25} | {r.data_size:>10,} | {r.pandas_time:>10.2f} | {r.datastore_time:>10.2f} | {r.fastest:>16}"
        )


def plot_benchmark_results(results: List[BenchmarkResult], output_prefix: str = 'benchmark_pandas_datastore'):
    """Generate benchmark visualization plot."""
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12

    # Convert results to DataFrame
    data = []
    for r in results:
        times = {'Pandas': r.pandas_time, 'DataStore': r.datastore_time}
        winner = min(times, key=times.get)
        data.append(
            {
                'op': r.operation,
                'size': r.data_size,
                'pandas': r.pandas_time,
                'datastore': r.datastore_time,
                'winner': winner,
                'speedup': r.speedup,
            }
        )

    df = pd.DataFrame(data)

    # Define colors
    colors = {'DataStore': '#5B8FF9', 'Pandas': '#5AD8A6'}  # Blue  # Teal

    # Get unique sizes and operations
    sizes = sorted(df['size'].unique())
    size_labels = [f'{s//1000}K' if s < 1000000 else f'{s//1000000}M' for s in sizes]
    all_ops = list(df['op'].unique())

    # Calculate wins for each size
    wins_by_size = {}
    for size in sizes:
        df_size = df[df['size'] == size]
        wins_by_size[size] = {
            'Pandas': len(df_size[df_size['winner'] == 'Pandas']),
            'DataStore': len(df_size[df_size['winner'] == 'DataStore']),
        }

    n_ops = len(all_ops)
    n_sizes = len(sizes)

    # Group spacing parameters
    width = 0.35
    gap_between_sizes = 0.2
    gap_between_ops = 1.0
    size_group_width = 2 * width
    total_size_group_width = n_sizes * size_group_width + (n_sizes - 1) * gap_between_sizes

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Calculate positions
    x_positions = np.arange(n_ops) * (total_size_group_width + gap_between_ops)

    # Store positions for labels
    all_positions = []
    all_labels = []

    # Plot bars for all operations
    for op_idx, op in enumerate(all_ops):
        df_op = df[df['op'] == op]

        for size_idx, (size, size_label) in enumerate(zip(sizes, size_labels)):
            df_size = df_op[df_op['size'] == size]

            if len(df_size) == 0:
                continue

            row = df_size.iloc[0]

            # Calculate x position
            base_x = x_positions[op_idx] + size_idx * (size_group_width + gap_between_sizes)

            # Plot bars - Pandas, DataStore order
            pandas_bar = ax.bar(
                base_x, row['pandas'], width, color=colors['Pandas'], alpha=0.75, edgecolor='black', linewidth=0.5
            )
            datastore_bar = ax.bar(
                base_x + width,
                row['datastore'],
                width,
                color=colors['DataStore'],
                alpha=0.75,
                edgecolor='black',
                linewidth=0.5,
            )

            # Highlight winner
            if row['winner'] == 'Pandas':
                pandas_bar[0].set_alpha(1.0)
                pandas_bar[0].set_linewidth(2.0)
            elif row['winner'] == 'DataStore':
                datastore_bar[0].set_alpha(1.0)
                datastore_bar[0].set_linewidth(2.0)

            # Store position
            center_x = base_x + width * 0.5
            all_positions.append(center_x)
            all_labels.append(size_label)

    # Formatting
    ax.set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

    # Build table data
    table_data = [['Wins'] + size_labels]
    for engine in ['DataStore', 'Pandas']:
        row = [engine] + [str(wins_by_size[size][engine]) for size in sizes]
        table_data.append(row)

    # Table colors
    table_colors = [
        ['white'] * (n_sizes + 1),
        [colors['DataStore']] + ['white'] * n_sizes,
        [colors['Pandas']] + ['white'] * n_sizes,
    ]

    # Add table
    table = ax.table(
        cellText=table_data, cellLoc='center', loc='upper left', bbox=[0.02, 0.78, 0.12, 0.18], cellColours=table_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header row
    for i in range(n_sizes + 1):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', fontsize=8)
        cell.set_facecolor('#E8E8E8')

    # Style color column
    for i in range(1, 3):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold', fontsize=8, color='black')
        cell.set_alpha(0.8)

    # Table borders
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1.0)
        cell.set_edgecolor('black')

    # Title
    ax.set_title(
        'DataFrame Performance Benchmark: Pandas vs DataStore (chDB lazy mode)',
        fontweight='bold',
        fontsize=12,
        pad=15,
    )

    # Two-level x-axis labels
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=7, color='gray')
    ax.tick_params(axis='x', which='major', length=3)

    # Operation names on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    op_centers = x_positions + total_size_group_width / 2
    ax2.set_xticks(op_centers)
    ax2.set_xticklabels([op.replace(' ', '\n') for op in all_ops], fontsize=7, fontweight='bold')
    ax2.tick_params(axis='x', which='major', length=0)
    ax2.spines['top'].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 10))

    plt.tight_layout()

    # Save figures
    pdf_path = f'{output_prefix}.pdf'
    png_path = f'{output_prefix}.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {pdf_path} and {png_path}")

    # Print wins summary
    print("\nWins Summary by Data Size:")
    print("=" * 50)
    for size, label in zip(sizes, size_labels):
        wins = wins_by_size[size]
        print(f"{label:>5}: DataStore={wins['DataStore']}, Pandas={wins['Pandas']}")

    plt.show()


def main():
    print("=" * 60)
    print("Pandas vs DataStore (chDB Lazy Mode) Benchmark")
    print("=" * 60)

    # Create temporary directory for parquet files
    temp_dir = tempfile.mkdtemp(prefix='datastore_benchmark_')
    print(f"Using temp directory: {temp_dir}")

    try:
        # Test different data sizes
        data_sizes = [100_000, 1_000_000, 10_000_000]

        # Run benchmarks
        results = run_benchmarks(data_sizes, temp_dir, n_runs=5)

        # Print summary
        print_summary(results)

        # Recommendations
        print("\n" + "=" * 100)
        print("ANALYSIS: DataStore Lazy Execution Advantages")
        print("=" * 100)

        # Analyze results - count wins for each engine
        def get_winner(r):
            times = {'Pandas': r.pandas_time, 'DataStore': r.datastore_time}
            return min(times, key=times.get)

        pandas_wins = sum(1 for r in results if get_winner(r) == 'Pandas')
        datastore_wins = sum(1 for r in results if get_winner(r) == 'DataStore')

        print(f"\nOverall wins: Pandas={pandas_wins}/{len(results)}, DataStore={datastore_wins}/{len(results)}")

        # Group by size
        for size in sorted(set(r.data_size for r in results)):
            size_results = [r for r in results if r.data_size == size]
            pandas_better = sum(1 for r in size_results if get_winner(r) == 'Pandas')
            datastore_better = sum(1 for r in size_results if get_winner(r) == 'DataStore')
            print(f"\n  {size:>10,} rows: Pandas={pandas_better}, DataStore={datastore_better}")

        # Find operations where DataStore excels
        print("\n" + "-" * 60)
        print("Operations where DataStore excels (multi-step SQL merging):")
        print("-" * 60)

        multi_step_ops = [
            'Multi-filter (4x)',
            'Filter+Select+Sort',
            'Filter+GroupBy+Sort',
            'Complex pipeline',
            'Chain 5 filters',
            'Ultra-complex (10+ ops)',
            'Pandas-style: Filter+Sort+Head',
            'Pandas-style: MultiFilter+Sort+Head',
            'Pandas-style: Select+Filter+Sort',
        ]

        for op in multi_step_ops:
            op_results = [r for r in results if r.operation == op]
            if op_results:
                avg_speedup = sum(r.speedup for r in op_results) / len(op_results)
                if avg_speedup > 1:
                    print(f"  {op:<25}: DataStore is {avg_speedup:.2f}x faster on average")
                else:
                    print(f"  {op:<25}: Pandas is {1/avg_speedup:.2f}x faster on average")

        # Generate plot
        plot_benchmark_results(results, output_prefix='benchmark_pandas_datastore')

    finally:
        # Cleanup temporary files
        import shutil

        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"\nWarning: Could not clean up temp directory: {e}")


if __name__ == '__main__':
    main()
