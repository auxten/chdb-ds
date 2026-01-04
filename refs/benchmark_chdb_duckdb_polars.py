import pandas as pd
import numpy as np
import duckdb
import polars as pl
import chdb
import time

# Test different data sizes
sizes = [1_000_000, 5_000_000, 10_000_000, 20_000_000]

duck_conn = duckdb.connect()
chdb_conn = chdb.connect()

print('Benchmark: Simple Filter (SELECT a, c WHERE a > 500)')
print('=' * 70)
print(f'{\"Rows\":>15} {\"DuckDB\":>12} {\"Polars\":>12} {\"chDB\":>12}')
print('-' * 70)

for N in sizes:
    np.random.seed(42)
    df = pd.DataFrame({
        'a': np.random.randint(1, 1000, N),
        'b': np.random.randint(1, 100, N),
        'c': np.random.randn(N),
        'd': np.random.randint(1, 100, N),
    })
    
    chdb_query = 'SELECT a, c FROM Python(df) WHERE a > 500'
    duck_query = 'SELECT a, c FROM df WHERE a > 500'
    
    # Warmup
    duck_conn.sql(duck_query).df()
    pl.sql(duck_query, eager=True)
    chdb_conn.query(chdb_query, 'DataFrame')
    
    # DuckDB
    start = time.perf_counter()
    duck_res = duck_conn.sql(duck_query).df()
    duck_time = (time.perf_counter() - start) * 1000
    
    # Polars
    start = time.perf_counter()
    pl_res = pl.sql(duck_query, eager=True).to_pandas()
    pl_time = (time.perf_counter() - start) * 1000
    
    # chdb
    start = time.perf_counter()
    chdb_res = chdb_conn.query(chdb_query, 'DataFrame')
    chdb_time = (time.perf_counter() - start) * 1000
    
    print(f'{N:>15,} {duck_time:>10.2f}ms {pl_time:>10.2f}ms {chdb_time:>10.2f}ms')

print()
print('Benchmark: Aggregation (GROUP BY b, SUM, AVG)')
print('=' * 70)
print(f'{\"Rows\":>15} {\"DuckDB\":>12} {\"Polars\":>12} {\"chDB\":>12}')
print('-' * 70)

for N in sizes:
    np.random.seed(42)
    df = pd.DataFrame({
        'a': np.random.randint(1, 1000, N),
        'b': np.random.randint(1, 100, N),
        'c': np.random.randn(N),
        'd': np.random.randint(1, 100, N),
    })
    
    chdb_query = 'SELECT b, SUM(a) as sum_a, AVG(c) as avg_c FROM Python(df) GROUP BY b'
    duck_query = 'SELECT b, SUM(a) as sum_a, AVG(c) as avg_c FROM df GROUP BY b'
    
    # Warmup
    duck_conn.sql(duck_query).df()
    pl.sql(duck_query, eager=True)
    chdb_conn.query(chdb_query, 'DataFrame')
    
    # DuckDB
    start = time.perf_counter()
    duck_res = duck_conn.sql(duck_query).df()
    duck_time = (time.perf_counter() - start) * 1000
    
    # Polars
    start = time.perf_counter()
    pl_res = pl.sql(duck_query, eager=True).to_pandas()
    pl_time = (time.perf_counter() - start) * 1000
    
    # chdb
    start = time.perf_counter()
    chdb_res = chdb_conn.query(chdb_query, 'DataFrame')
    chdb_time = (time.perf_counter() - start) * 1000
    
    print(f'{N:>15,} {duck_time:>10.2f}ms {pl_time:>10.2f}ms {chdb_time:>10.2f}ms')

print()
print('Benchmark: Complex Filter + Sort (WHERE a > 500 AND d < 50 ORDER BY c LIMIT 1000)')
print('=' * 70)
print(f'{\"Rows\":>15} {\"DuckDB\":>12} {\"Polars\":>12} {\"chDB\":>12}')
print('-' * 70)

for N in sizes:
    np.random.seed(42)
    df = pd.DataFrame({
        'a': np.random.randint(1, 1000, N),
        'b': np.random.randint(1, 100, N),
        'c': np.random.randn(N),
        'd': np.random.randint(1, 100, N),
    })
    
    chdb_query = 'SELECT a, b, c FROM Python(df) WHERE a > 500 AND d < 50 ORDER BY c DESC LIMIT 1000'
    duck_query = 'SELECT a, b, c FROM df WHERE a > 500 AND d < 50 ORDER BY c DESC LIMIT 1000'
    
    # Warmup
    duck_conn.sql(duck_query).df()
    pl.sql(duck_query, eager=True)
    chdb_conn.query(chdb_query, 'DataFrame')
    
    # DuckDB
    start = time.perf_counter()
    duck_res = duck_conn.sql(duck_query).df()
    duck_time = (time.perf_counter() - start) * 1000
    
    # Polars
    start = time.perf_counter()
    pl_res = pl.sql(duck_query, eager=True).to_pandas()
    pl_time = (time.perf_counter() - start) * 1000
    
    # chdb
    start = time.perf_counter()
    chdb_res = chdb_conn.query(chdb_query, 'DataFrame')
    chdb_time = (time.perf_counter() - start) * 1000
    
    print(f'{N:>15,} {duck_time:>10.2f}ms {pl_time:>10.2f}ms {chdb_time:>10.2f}ms')
