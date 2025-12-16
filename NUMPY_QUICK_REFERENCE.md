# NumPy Compatibility Quick Reference

## Quick Answer

**✅ Yes, DataStore is fully compatible with NumPy!**

DataStore and ColumnExpr can be used **directly** with all common NumPy functions without any conversion!

## Usage

### 🚀 Direct Usage (Recommended)

```python
import numpy as np
from datastore import DataStore

ds = DataStore.from_file('data.csv')

# ✅ All of these work directly!
np.mean(ds['column'])                   # Compute mean
np.std(ds['column'])                    # Compute standard deviation
np.sum(ds['column'])                    # Compute sum
np.min(ds['column'])                    # Minimum value
np.max(ds['column'])                    # Maximum value
np.median(ds['column'])                 # Median
np.var(ds['column'])                    # Variance
np.allclose(ds['a'], ds['b'])           # Compare if two columns are close
np.corrcoef(ds['a'], ds['b'])           # Correlation coefficient
np.dot(ds['a'], ds['b'])                # Dot product
np.concatenate([ds['a'], ds['b']])      # Concatenate
np.percentile(ds['column'], [25, 50, 75])  # Percentiles
np.histogram(ds['column'], bins=10)     # Histogram
```

### 🔧 Optional: Explicit Conversion

```python
# If needed, you can also convert explicitly
arr = ds['column'].to_numpy()  # or ds['column'].values
result = np.mean(arr)
```

### 💡 Best Practice: SQL + NumPy Combination

```python
# Use SQL for filtering, then NumPy for computation
filtered = ds.filter(ds['age'] > 25)
ages = filtered['age']  # No need for .to_numpy()
normalized = (np.array(ages) - np.mean(ages)) / np.std(ages)
```

## Full Compatibility List

| NumPy Function | Direct Use | With .to_numpy() | Description |
|----------------|------------|------------------|-------------|
| `np.array()` | ✅ | ✅ | Convert to array |
| `np.allclose()` | ✅ | ✅ | Check if arrays are close |
| `np.equal()` | ✅ | ✅ | Element-wise equality |
| `np.isclose()` | ✅ | ✅ | Element-wise close comparison |
| `np.mean()` | ✅ | ✅ | Compute mean |
| `np.sum()` | ✅ | ✅ | Compute sum |
| `np.std()` | ✅ | ✅ | Compute standard deviation |
| `np.var()` | ✅ | ✅ | Compute variance |
| `np.max()` | ✅ | ✅ | Compute maximum |
| `np.min()` | ✅ | ✅ | Compute minimum |
| `np.median()` | ✅ | ✅ | Compute median |
| `np.prod()` | ✅ | ✅ | Compute product |
| `np.argmin()` | ✅ | ✅ | Index of minimum |
| `np.argmax()` | ✅ | ✅ | Index of maximum |
| `np.cumsum()` | ✅ | ✅ | Cumulative sum |
| `np.concatenate()` | ✅ | ✅ | Concatenate arrays |
| `np.dot()` | ✅ | ✅ | Dot product |
| `np.corrcoef()` | ✅ | ✅ | Correlation coefficient |
| `np.percentile()` | ✅ | ✅ | Percentiles |
| `np.histogram()` | ✅ | ✅ | Histogram |
| `np.sort()` | ✅ | ✅ | Sort array |
| `np.unique()` | ✅ | ✅ | Unique values |

## Practical Examples

### Statistical Analysis

```python
# Compute basic statistics for a column
print(f"Mean: {np.mean(ds['sales'])}")
print(f"Median: {np.median(ds['sales'])}")
print(f"Std Dev: {np.std(ds['sales'])}")
print(f"Range: {np.ptp(ds['sales'])}")
```

### Data Normalization

```python
values = ds['price']
normalized = (np.array(values) - np.mean(values)) / np.std(values)
```

### Correlation Analysis

```python
correlation = np.corrcoef(ds['feature1'], ds['feature2'])[0, 1]
print(f"Correlation: {correlation}")
```

### Array Operations

```python
# Combine multiple columns
combined = np.column_stack([
    np.array(ds['col1']),
    np.array(ds['col2'])
])

# Dot product
result = np.dot(ds['a'], ds['b'])
```

## Performance Tips

### ✅ Good Practice
```python
# Convert once, use multiple times (for many operations)
arr = ds['column'].to_numpy()
mean = np.mean(arr)
std = np.std(arr)
max_val = np.max(arr)
```

### ✅ Also Good (Direct Use)
```python
# Direct use works fine too!
mean = np.mean(ds['column'])
std = np.std(ds['column'])
max_val = np.max(ds['column'])
```

## Comparison with Pandas

```python
# Pandas
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
np.mean(df['a'])  # ✓ Works directly

# DataStore - Same experience!
ds = DataStore.from_df(df)
np.mean(ds['a'])  # ✓ Works directly too!
```

**DataStore now has the same level of NumPy compatibility as Pandas!**

## Technical Implementation

DataStore implements the following interfaces for NumPy support:

1. **`__array__` protocol**: Allows NumPy to convert DataStore/ColumnExpr to arrays
2. **`.values` property**: Returns underlying numpy array (pandas compatible)
3. **`.to_numpy()` method**: Explicit conversion method
4. **NumPy-compatible statistical methods**: `mean()`, `sum()`, `std()`, `var()`, `min()`, `max()`, `median()`, `prod()`, `argmin()`, `argmax()`, `cumsum()`, `cumprod()`, `any()`, `all()` - all accept NumPy-style parameters

## Related Documentation

- Test suite: [tests/test_numpy_compatibility.py](tests/test_numpy_compatibility.py)

## Summary

```python
# DataStore works directly with NumPy!
np.mean(ds['column'])      # ✓ Direct use
np.std(ds['column'])       # ✓ Direct use
np.corrcoef(ds['a'], ds['b'])  # ✓ Direct use

# Or convert if you prefer
arr = ds['column'].to_numpy()
result = np.mean(arr)
```

**DataStore + NumPy = SQL's powerful filtering + NumPy's scientific computing!** 🚀
