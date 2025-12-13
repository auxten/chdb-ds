# Pandas DataFrame Compatibility

DataStore provides comprehensive pandas DataFrame API compatibility, allowing you to use familiar pandas methods directly on DataStore objects while maintaining the benefits of SQL-based query optimization.

## Overview

- **180+ Methods**: Full coverage of common pandas DataFrame operations
- **Seamless Integration**: Mix SQL-style queries with pandas transformations
- **Automatic Wrapping**: DataFrame/Series results automatically wrapped as DataStore
- **Immutable**: All operations return new instances (no `inplace=True`)
- **Smart Execution**: SQL operations build queries, pandas operations materialize and cache results
- **Correct Chaining**: Handles mixed SQL→pandas→pandas chains correctly

## Quick Start

```python
from datastore import DataStore

ds = DataStore.from_file("data.csv")

# Use any pandas method
df = ds.drop(columns=['unused'])
      .fillna(0)
      .assign(revenue=lambda x: x['price'] * x['quantity'])
      .sort_values('revenue', ascending=False)
      .head(10)

# Mix SQL and pandas
result = (ds
    .select('*')
    .filter(ds.price > 100)              # SQL-style
    .assign(margin=lambda x: x['profit'] / x['revenue'])  # pandas-style
    .query('margin > 0.2')               # pandas-style
    .groupby('category').agg({'revenue': 'sum'}))  # pandas-style
```

## Feature Checklist

### ✅ Attributes and Properties
- [x] `df.index` - Row labels
- [x] `df.columns` - Column labels
- [x] `df.dtypes` - Data types
- [x] `df.values` - NumPy array representation
- [x] `df.shape` - Dimensions (rows, cols)
- [x] `df.size` - Total elements
- [x] `df.ndim` - Number of dimensions
- [x] `df.empty` - Empty check
- [x] `df.T` - Transpose
- [x] `df.axes` - Axis labels

### ✅ Indexing and Selection
- [x] `df.loc[...]` - Label-based indexing
- [x] `df.iloc[...]` - Integer-based indexing
- [x] `df.at[...]` - Fast scalar access
- [x] `df.iat[...]` - Fast integer scalar access
- [x] `df['col']` - Column selection
- [x] `df[['col1', 'col2']]` - Multiple columns
- [x] `df.head(n)` - First n rows
- [x] `df.tail(n)` - Last n rows
- [x] `df.sample(n)` - Random sample
- [x] `df.select_dtypes()` - Select by dtype
- [x] `df.query()` - Query by expression
- [x] `df.where()` - Conditional replacement
- [x] `df.mask()` - Inverse where
- [x] `df.isin()` - Value membership
- [x] `df.get()` - Safe column access
- [x] `df.xs()` - Cross-section
- [x] `df.pop()` - Remove and return column
- [x] `df.insert()` - Insert column

### ✅ Statistical Methods
- [x] `df.describe()` - Summary statistics
- [x] `df.mean()` - Mean values
- [x] `df.median()` - Median values
- [x] `df.mode()` - Mode values
- [x] `df.std()` - Standard deviation
- [x] `df.var()` - Variance
- [x] `df.min()` / `df.max()` - Min/Max values
- [x] `df.sum()` - Sum
- [x] `df.prod()` - Product
- [x] `df.count()` - Non-null counts
- [x] `df.nunique()` - Unique counts
- [x] `df.value_counts()` - Value frequencies
- [x] `df.quantile()` - Quantiles
- [x] `df.corr()` - Correlation matrix
- [x] `df.cov()` - Covariance matrix
- [x] `df.corrwith()` - Pairwise correlation
- [x] `df.rank()` - Rank values
- [x] `df.abs()` - Absolute values
- [x] `df.round()` - Round values
- [x] `df.clip()` - Clip values
- [x] `df.cumsum()` - Cumulative sum
- [x] `df.cumprod()` - Cumulative product
- [x] `df.cummin()` - Cumulative min
- [x] `df.cummax()` - Cumulative max
- [x] `df.diff()` - Difference
- [x] `df.pct_change()` - Percent change
- [x] `df.skew()` - Skewness
- [x] `df.kurt()` - Kurtosis
- [x] `df.sem()` - Standard error
- [x] `df.all()` / `df.any()` - Boolean aggregation
- [x] `df.idxmin()` / `df.idxmax()` - Index of min/max
- [x] `df.eval()` - Expression evaluation

### ✅ Data Manipulation
- [x] `df.drop()` - Drop rows/columns
- [x] `df.drop_duplicates()` - Remove duplicates
- [x] `df.duplicated()` - Mark duplicates
- [x] `df.dropna()` - Remove missing
- [x] `df.fillna()` - Fill missing
- [x] `df.ffill()` / `df.bfill()` - Forward/backward fill
- [x] `df.interpolate()` - Interpolate values
- [x] `df.replace()` - Replace values
- [x] `df.rename()` - Rename labels
- [x] `df.rename_axis()` - Rename axis
- [x] `df.assign()` - Add columns
- [x] `df.astype()` - Convert types
- [x] `df.convert_dtypes()` - Infer types
- [x] `df.copy()` - Copy data

### ✅ Sorting and Ranking
- [x] `df.sort_values()` - Sort by values
- [x] `df.sort_index()` - Sort by index
- [x] `df.nlargest()` - N largest values
- [x] `df.nsmallest()` - N smallest values

### ✅ Reindexing
- [x] `df.reset_index()` - Reset index
- [x] `df.set_index()` - Set index
- [x] `df.reindex()` - Conform to new index
- [x] `df.reindex_like()` - Match another's index
- [x] `df.add_prefix()` - Add prefix to labels
- [x] `df.add_suffix()` - Add suffix to labels
- [x] `df.align()` - Align two objects
- [x] `df.set_axis()` - Set axis labels
- [x] `df.take()` - Select by positions
- [x] `df.truncate()` - Truncate by range

### ✅ Reshaping
- [x] `df.pivot()` - Pivot table
- [x] `df.pivot_table()` - Pivot with aggregation
- [x] `df.melt()` - Unpivot
- [x] `df.stack()` - Stack columns to index
- [x] `df.unstack()` - Unstack index to columns
- [x] `df.transpose()` / `df.T` - Transpose
- [x] `df.explode()` - Explode lists to rows
- [x] `df.squeeze()` - Reduce dimensions
- [x] `df.droplevel()` - Drop index level
- [x] `df.swaplevel()` - Swap index levels
- [x] `df.swapaxes()` - Swap axes
- [x] `df.reorder_levels()` - Reorder levels

### ✅ Combining / Joining / Merging
- [x] `df.append()` - Append rows
- [x] `df.merge()` - SQL-style merge
- [x] `df.join()` - Join on index
- [x] `df.concat()` - Concatenate
- [x] `df.compare()` - Compare differences
- [x] `df.update()` - Update values
- [x] `df.combine()` - Combine with function
- [x] `df.combine_first()` - Combine with priority

### ✅ Binary Operators
- [x] `df.add()` / `df.radd()` - Addition
- [x] `df.sub()` / `df.rsub()` - Subtraction
- [x] `df.mul()` / `df.rmul()` - Multiplication
- [x] `df.div()` / `df.rdiv()` - Division
- [x] `df.truediv()` / `df.rtruediv()` - True division
- [x] `df.floordiv()` / `df.rfloordiv()` - Floor division
- [x] `df.mod()` / `df.rmod()` - Modulo
- [x] `df.pow()` / `df.rpow()` - Power
- [x] `df.dot()` - Matrix multiplication

### ✅ Comparison Operators
- [x] `df.eq()` - Equal
- [x] `df.ne()` - Not equal
- [x] `df.lt()` - Less than
- [x] `df.le()` - Less than or equal
- [x] `df.gt()` - Greater than
- [x] `df.ge()` - Greater than or equal

### ✅ Function Application
- [x] `df.apply()` - Apply function
- [x] `df.applymap()` - Apply element-wise
- [x] `df.map()` - Apply element-wise (alias)
- [x] `df.agg()` / `df.aggregate()` - Aggregate
- [x] `df.transform()` - Transform
- [x] `df.pipe()` - Pipe functions
- [x] `df.groupby()` - Group by (returns GroupBy)

### ✅ Time Series
- [x] `df.rolling()` - Rolling window
- [x] `df.expanding()` - Expanding window
- [x] `df.ewm()` - Exponentially weighted
- [x] `df.resample()` - Resample time series
- [x] `df.shift()` - Shift values
- [x] `df.asfreq()` - Convert frequency
- [x] `df.asof()` - Latest value as of time
- [x] `df.at_time()` - Select at time
- [x] `df.between_time()` - Select time range
- [x] `df.first()` / `df.last()` - First/last periods
- [x] `df.first_valid_index()` - First valid index
- [x] `df.last_valid_index()` - Last valid index
- [x] `df.to_period()` - Convert to period
- [x] `df.to_timestamp()` - Convert to timestamp
- [x] `df.tz_convert()` - Convert timezone
- [x] `df.tz_localize()` - Localize timezone

### ✅ Missing Data
- [x] `df.isna()` / `df.isnull()` - Detect missing
- [x] `df.notna()` / `df.notnull()` - Detect non-missing
- [x] `df.dropna()` - Drop missing
- [x] `df.fillna()` - Fill missing
- [x] `df.ffill()` - Forward fill
- [x] `df.bfill()` - Backward fill
- [x] `df.backfill()` - Backward fill (alias)
- [x] `df.pad()` - Forward fill (alias)
- [x] `df.interpolate()` - Interpolate
- [x] `df.replace()` - Replace values

### ✅ Export / IO
- [x] `df.to_csv()` - Export to CSV
- [x] `df.to_json()` - Export to JSON
- [x] `df.to_excel()` - Export to Excel
- [x] `df.to_parquet()` - Export to Parquet
- [x] `df.to_feather()` - Export to Feather
- [x] `df.to_hdf()` - Export to HDF5
- [x] `df.to_sql()` - Export to SQL database
- [x] `df.to_stata()` - Export to Stata
- [x] `df.to_pickle()` - Pickle to file
- [x] `df.to_html()` - Render as HTML
- [x] `df.to_latex()` - Render as LaTeX
- [x] `df.to_markdown()` - Render as Markdown
- [x] `df.to_string()` - Render as string
- [x] `df.to_dict()` - Convert to dictionary
- [x] `df.to_records()` - Convert to records
- [x] `df.to_numpy()` - Convert to NumPy
- [x] `df.to_clipboard()` - Copy to clipboard
- [x] `df.to_xarray()` - Convert to xarray
- [x] `df.to_orc()` - Export to ORC
- [x] `df.to_gbq()` - Export to BigQuery

### ✅ Iteration
- [x] `df.items()` - Iterate (column, Series) pairs
- [x] `df.iterrows()` - Iterate (index, Series) pairs
- [x] `df.itertuples()` - Iterate as namedtuples

### ✅ Plotting
- [x] `df.plot` - Plotting accessor
- [x] `df.plot.*` - Various plot types
- [x] `df.hist()` - Histogram
- [x] `df.boxplot()` - Box plot

### ✅ Accessors
- [x] `df.str` - String accessor (for Series)
- [x] `df.dt` - Datetime accessor
- [x] `df.sparse` - Sparse accessor
- [x] `df.style` - Styling accessor

### ✅ Comparison
- [x] `df.equals()` - Test equality
- [x] `df.compare()` - Show differences

### ✅ Miscellaneous
- [x] `df.info()` - Print summary
- [x] `df.memory_usage()` - Memory usage
- [x] `df.copy()` - Copy DataFrame

## Key Differences from Pandas

### 1. Row Ordering is Not Guaranteed

**Important**: Unlike pandas, DataStore uses ClickHouse as its underlying SQL engine, which does **not guarantee row order** by default. This means:

```python
# ❌ Order may vary between executions
ds = DataStore.from_file("data.csv")
result = ds.filter(ds.value > 50).to_df()
# Row order is NOT guaranteed to match the original file order

# ✅ Explicitly specify ORDER BY for deterministic order
result = ds.filter(ds.value > 50).order_by('id').to_df()
# Rows are ordered by 'id' column
```

**Why?** ClickHouse is optimized for analytical workloads and may return rows in any order for better performance. This is standard SQL behavior - without `ORDER BY`, result order is undefined.

**Impact on comparisons**:
- When comparing results, sort both DataFrames first or use set-based comparisons
- Use `df.sort_values('col').reset_index(drop=True)` before `pd.testing.assert_frame_equal()`

### 2. Immutability
DataStore operations are immutable - `inplace=True` is not supported:

```python
# ❌ Not supported
df.drop(columns=['col'], inplace=True)

# ✅ Correct usage
df = df.drop(columns=['col'])
```

### 3. Return Types
Methods behavior matches pandas:

```python
# DataFrame methods return DataStore
result = ds.drop(columns=['col'])  # Returns DataStore
df = result.to_df()  # Get underlying DataFrame

# Series methods return Series  
series = ds['column']  # Returns pd.Series (not DataStore)
value = series.mean()  # Normal pandas Series operations

# Aggregations return appropriate types
series_result = ds.mean()  # Returns pd.Series
scalar_result = ds['age'].mean()  # Returns scalar
```

### 4. Series Handling
Operations that return Series in pandas also return Series in DataStore (not wrapped):

```python
# Returns pandas Series (as expected)
series = ds['column']  
print(type(series))  # <class 'pandas.core.series.Series'>

# Multiple columns return DataStore
datastore = ds[['col1', 'col2']]
print(type(datastore))  # <class 'datastore.core.DataStore'>
```

**Why?** This maintains pandas semantics and user expectations.

### 5. Method Naming
The INSERT VALUES method has been renamed to avoid conflicts:

```python
# Old (conflicts with df.values property)
ds.insert_into('id', 'name').values(1, 'Alice')

# New (recommended)
ds.insert_into('id', 'name').insert_values(1, 'Alice')
```

## Execution Model

DataStore implements a sophisticated **Mixed Execution Engine** that enables **arbitrary mixing** of SQL and pandas operations. 

### Key Innovation: SQL on DataFrames

After materialization, SQL-style operations use **chDB's `Python()` table function** to execute SQL directly on cached DataFrames, enabling true mixed execution.

### Three-Stage Execution

**Stage 1: SQL Query Building (Lazy)**
```python
ds = DataStore.from_file("data.csv")
ds1 = ds.select('*')                    # Builds: SELECT *
ds2 = ds1.filter(ds.age > 25)           # Adds: WHERE age > 25
# ds2._materialized = False (no execution yet)
```

**Stage 2: Materialization (First pandas Operation)**
```python
ds3 = ds2.add_prefix('emp_')            # ← Executes SQL here!
# ds3._materialized = True
# ds3._cached_df = DataFrame with filtered data and prefixed columns
```

**Stage 3: SQL on DataFrame (chDB Magic)**
```python
ds4 = ds3.filter(ds.emp_age > 30)       # SQL on DataFrame!
# Internally: SELECT * FROM Python(__datastore_cached_df__) WHERE emp_age > 30
# ds4._materialized = True (result cached)
```

### Arbitrary Mixing Examples

**Example 1: SQL → Pandas → SQL → Pandas**
```python
result = (ds
    .filter(ds.age > 25)                      # SQL query building
    .add_prefix('emp_')                       # Pandas (materializes)
    .filter(ds.emp_salary > 55000)            # SQL on DataFrame!
    .fillna(0))                               # Pandas on DataFrame
```

**Example 2: Pandas → SQL → Pandas → SQL**
```python
result = (ds
    .rename(columns={'id': 'ID'})             # Pandas (materializes)
    .filter(ds.ID > 5)                        # SQL on DataFrame
    .sort_values('salary')                    # Pandas
    .select('ID', 'name', 'salary'))          # SQL on DataFrame again!
```

**Example 3: Complex Mixed Chain**
```python
result = (ds
    .select('*')                              # SQL 1
    .filter(ds.status == 'active')            # SQL 2
    .assign(revenue=lambda x: x['price'] * x['qty'])  # Pandas (materializes)
    .filter(ds.revenue > 1000)                # SQL 3 on DataFrame
    .add_prefix('sales_')                     # Pandas
    .query('sales_revenue > 5000')            # Pandas
    .select('sales_id', 'sales_customer', 'sales_revenue'))  # SQL 4 on DataFrame
```

**For detailed technical documentation, see [Mixed Execution Engine Guide](MIXED_EXECUTION_ENGINE.md)**

## Performance Tips

### 1. Use SQL for Filtering
Let the query engine do heavy filtering before pandas operations:

```python
# ✅ Efficient
result = (ds
    .select('*')
    .filter(ds.date >= '2024-01-01')  # SQL filter
    .filter(ds.amount > 1000)         # SQL filter
    .assign(margin=lambda x: x['profit'] / x['revenue'])  # Pandas transform
    .groupby('category').agg({'revenue': 'sum'}))  # Pandas aggregation

# ❌ Less efficient
result = (ds
    .to_df()  # Load all data
    .query('date >= "2024-01-01" and amount > 1000'))  # Filter in memory
```

### 2. Understand Materialization
Once materialized (pandas operation applied), all subsequent operations use cached data:

```python
ds = DataStore.from_file("big_data.csv")

# SQL operations - build query (lazy)
ds_filtered = ds.select('*').filter(ds.value > 0)  # No execution yet

# First pandas operation - materializes
ds_prefixed = ds_filtered.add_prefix('col_')  # ← Query executes here!

# All subsequent operations use cached DataFrame
mean = ds_prefixed.mean()       # Uses cache, no SQL
std = ds_prefixed.std()         # Uses cache, no SQL
df = ds_prefixed.to_df()        # Returns cache, no SQL
```

### 3. Optimal Workflow Pattern

**Best Practice**: Filter in SQL, transform in pandas

```python
# ✅ Optimal: SQL filtering → Pandas transformation
result = (ds
    .select('*')
    .filter(ds.date >= '2024-01-01')    # SQL: Filters billions of rows
    .filter(ds.amount > 1000)           # SQL: More filtering
    # ↑ Query built but not executed yet
    
    .add_prefix('col_')                 # ← Executes SQL here, materializes
    .fillna(0)                          # Pandas: Works on cached result
    .assign(margin=lambda x: x['col_profit'] / x['col_revenue']))  # Pandas
```

### 4. Chain Operations
Chain multiple operations for better readability and potential optimization:

```python
result = (ds
    .drop(columns=['unused1', 'unused2'])
    .fillna(0)
    .assign(
        revenue=lambda x: x['price'] * x['quantity'],
        margin=lambda x: x['profit'] / x['revenue']
    )
    .query('margin > 0.2')
    .sort_values('revenue', ascending=False)
    .head(100))
```

## Examples

### Example 1: Data Cleaning
```python
cleaned = (ds
    .drop(columns=['temp_col'])
    .dropna(subset=['important_col'])
    .drop_duplicates()
    .fillna({'numeric_col': 0, 'string_col': 'unknown'})
    .astype({'id': 'int64', 'amount': 'float64'}))
```

### Example 2: Feature Engineering
```python
featured = ds.assign(
    revenue=lambda x: x['price'] * x['quantity'],
    profit=lambda x: x['revenue'] - x['cost'],
    margin=lambda x: x['profit'] / x['revenue'],
    high_value=lambda x: x['revenue'] > 1000
)
```

### Example 3: Time Series Analysis
```python
ts_result = (ds
    .set_index('date')
    .sort_index()
    .asfreq('D')
    .fillna(method='ffill')
    .rolling(window=7).mean()
    .shift(1))
```

### Example 4: Binary Operations
```python
# Calculate year-over-year growth
growth = (current_year
    .set_index('product')
    .sub(last_year.set_index('product'))
    .div(last_year.set_index('product'))
    .mul(100))
```

### Example 5: Conditional Operations
```python
# Complex filtering and transformation
result = (ds
    .query('age > 18 and income > 50000')
    .assign(
        segment=lambda x: pd.cut(x['income'], 
                                  bins=[0, 75000, 150000, float('inf')],
                                  labels=['Low', 'Medium', 'High'])
    )
    .where(lambda x: x['score'] > 0, 0)
    .groupby('segment')
    .agg({'income': 'mean', 'score': 'sum'}))
```

### Example 6: Mixing SQL and Pandas
```python
# Optimal workflow
result = (ds
    # Use SQL for heavy filtering
    .select('customer_id', 'order_date', 'amount', 'product_category')
    .filter(ds.order_date >= '2024-01-01')
    .filter(ds.order_date < '2024-02-01')
    .filter(ds.amount > 0)
    
    # Use pandas for complex transformations
    .assign(
        month=lambda x: pd.to_datetime(x['order_date']).dt.month,
        is_high_value=lambda x: x['amount'] > x['amount'].quantile(0.75)
    )
    .groupby(['customer_id', 'month'])
    .agg({
        'amount': ['sum', 'mean', 'count'],
        'is_high_value': 'sum'
    })
    .reset_index()
    
    # Export
    .to_parquet('customer_monthly_summary.parquet'))
```

## Limitations

### Not Implemented
- `inplace=True` parameter (DataStore is immutable)
- Some deprecated pandas methods
- Methods that don't make sense for DataStore (e.g., `from_dict`, `from_records` as instance methods)

### Partial Support
- `df.groupby()` - Returns pandas GroupBy object, not DataStore
- Class methods - Return pandas objects, not DataStore

## Getting Help

- **Documentation**: See [DataStore README](../README.md)
- **Examples**: Check [examples/example_pandas_extended.py](../examples/example_pandas_extended.py)
- **Pandas Docs**: https://pandas.pydata.org/docs/reference/frame.html

## Summary

DataStore provides **180+ pandas DataFrame methods** with seamless integration:

- ✅ All common pandas operations supported
- ✅ Mix SQL queries with pandas transformations
- ✅ Automatic DataFrame/Series wrapping
- ✅ Performance optimization through caching
- ✅ Immutable, thread-safe operations

Use DataStore when you need the power of pandas with the performance of SQL query optimization!
