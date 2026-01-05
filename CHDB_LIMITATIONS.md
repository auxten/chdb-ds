# chDB Engine Limitations for DataStore

This document catalogs known limitations and behavioral differences between chDB/ClickHouse
and pandas that affect DataStore. Understanding these limitations helps users:

1. Know when to expect different behavior from pandas
2. Understand which workarounds are available
3. Track which limitations might be resolved in future chDB versions

**chDB Version**: 4.0.0b3 (as of 2026-01-05)

---

## Table of Contents

1. [Resolved Limitations](#resolved-limitations-chdb-400b3)
2. [Type Support Issues](#type-support-issues)
3. [NULL/NaN/NA Handling](#nullnanna-handling)
4. [DateTime/Timezone Issues](#datetimetimezone-issues)
5. [String Operation Limitations](#string-operation-limitations)
6. [SQL Semantic Differences](#sql-semantic-differences)
7. [Function Limitations](#function-limitations)
8. [DataStore Implementation Notes](#datastore-implementation-notes)
9. [Workarounds Reference](#workarounds-reference)

---

## Resolved Limitations (chDB 4.0.0b3)

The following limitations have been resolved in chDB 4.0.0b3:

| Issue | Previous Status | Resolution |
|-------|-----------------|------------|
| Categorical type | Not supported | Now supported (converted to string internally) |
| Timedelta type | Not supported | Now supported |
| Sum of all-NaN | Returned NA | Now returns 0 (pandas-compatible) |
| product() function | Not available | Now available |
| Unicode string filter | Encoding issues | Now works correctly |

---

## Type Support Issues

### Critical: Nullable Boolean (pd.BooleanDtype)

**Severity**: HIGH - DATA CORRUPTION RISK

| Aspect | Description |
|--------|-------------|
| Issue | Nullable Boolean type converts to uint8, NA values become 0 |
| Example | `pd.array([True, False, pd.NA], dtype='boolean')` -> `[1, 0, 0]` |
| Impact | Data corruption - NA silently becomes False |

```python
# DANGER: NA values are lost
df = pd.DataFrame({'flag': pd.array([True, False, pd.NA], dtype='boolean')})
# After chDB: flag = [1, 0, 0] (uint8) - pd.NA became 0!
```

**Workaround**: Convert nullable boolean to object type with explicit None handling before SQL operations.

---

### High: Nullable Int64 (pd.Int64Dtype)

| Aspect | Description |
|--------|-------------|
| Issue | Nullable Int64 loses nullability, converts to int64 |
| Example | `pd.array([1, pd.NA, 3], dtype='Int64')` -> `[1, 0, 3]` or comparison returns raw bytes |
| Impact | NA values may become 0, comparisons may fail |
| Test Marker | `chdb_nullable_int64_comparison` |

```python
# NA values may be silently converted to 0
df = pd.DataFrame({'val': pd.array([1, pd.NA, 3], dtype='Int64')})
# After chDB: val = int64, NA -> 0 or lost
```

**Workaround**: Convert to float64 which properly handles NaN.

---

### ~~Categorical Type Not Supported~~ RESOLVED

**Status**: RESOLVED in chDB 4.0.0b3

Categorical type is now supported. DataStore converts categorical to string internally.

---

### ~~Timedelta Type Not Supported~~ RESOLVED

**Status**: RESOLVED in chDB 4.0.0b3

Timedelta type is now supported.

---

### Array Type Cannot Be Nullable

| Aspect | Description |
|--------|-------------|
| Issue | Array(T) cannot be inside Nullable type in ClickHouse |
| Error | `Array(String) cannot be inside Nullable type` |
| Test Marker | `chdb_array_nullable` |

This affects operations like `str.findall()` which return arrays.

---

## NULL/NaN/NA Handling

### NaN Comparison Returns NULL (Not False)

| Aspect | Description |
|--------|-------------|
| Issue | SQL standard: NaN == x returns NULL, pandas returns False |
| Impact | Boolean masks may have unexpected NULL values |

```python
df = pd.DataFrame({'a': [1.0, np.nan], 'b': [1.0, 2.0]})
# Pandas: a == b -> [True, False]
# chDB:   a = b  -> [1, NULL]
```

---

### GroupBy Includes NULL as Separate Group

| Aspect | Description |
|--------|-------------|
| Issue | chDB GROUP BY includes NULL as a group, pandas dropna=True by default |
| Impact | Extra group in aggregation results |
| Test Marker | `chdb_null_in_groupby` |

```python
df = pd.DataFrame({'group': ['A', 'B', None, 'A'], 'val': [1, 2, 3, 4]})
# Pandas groupby (dropna=True, default): ['A', 'B']
# chDB GROUP BY: ['A', 'B', NULL] - includes NULL group
```

**Workaround**: Filter out NULL values before groupby, or post-process results.

---

### JOIN Does Not Match NULL Values

| Aspect | Description |
|--------|-------------|
| Issue | SQL standard: NULL != NULL, pandas merge matches None |
| Impact | Rows with NULL keys excluded from JOIN results |

```python
df1 = pd.DataFrame({'key': ['A', None], 'val': [1, 2]})
df2 = pd.DataFrame({'key': ['A', None], 'val': [10, 20]})
# Pandas merge (inner): 2 rows (matches None with None)
# chDB INNER JOIN: 1 row (NULL != NULL in SQL)
```

---

### NULL in IN Clause

| Aspect | Description |
|--------|-------------|
| Issue | SQL: `val IN (..., NULL)` never matches NULL |
| Impact | isin() with None values may exclude NULL rows |

```python
# Pandas: df['val'].isin(['A', None]) -> matches None values
# chDB: WHERE val IN ('A', NULL) -> never matches NULL
```

---

### ~~Sum of All-NaN Returns NA (Not 0)~~ RESOLVED

**Status**: RESOLVED in chDB 4.0.0b3

Sum of all-NaN column now correctly returns 0 like pandas.

---

## DateTime/Timezone Issues

### Timezone Handling Differences

| Aspect | Description |
|--------|-------------|
| Issue | chDB may interpret datetime in local timezone |
| Impact | Year/month/day/hour may be off by timezone offset |
| Test Marker | `chdb_datetime_timezone` |

```python
# A datetime at 2024-01-01 00:30:00 UTC
# In UTC+8 timezone: becomes 2024-01-01 08:30:00
# Extracting hour: pandas=0, chDB=8
```

**Workaround**: DataStore converts to UTC using `toTimezone(parseDateTimeBestEffort(...), 'UTC')`.

---

### NaT (Not a Time) Conversion

| Aspect | Description |
|--------|-------------|
| Issue | chDB cannot convert NaT to integer for datetime properties |
| Error | `ValueError: cannot convert NA to integer` |

**Workaround**: DataStore uses nullable Int32 dtype as fallback.

---

### strftime Format Codes Differ

| Aspect | Description |
|--------|-------------|
| Issue | ClickHouse formatDateTime uses different codes than Python |
| Example | `%M` = minutes in Python, month name in ClickHouse |
| Resolution | strftime uses pandas fallback |

---

### Multiple DateTime Extractions Conflict

| Aspect | Description |
|--------|-------------|
| Issue | Multiple dt.xxx extractions can cause column naming conflicts |
| Error | Block structure mismatch in chDB |
| Test Marker | `chdb_datetime_extraction_conflict` |

---

## String Operation Limitations

### Functions Not Available in chDB

| Function | Status | Fallback |
|----------|--------|----------|
| `str.extract(regex)` | Partial | pandas fallback for expand behavior |
| `str.extractall(regex)` | No | pandas fallback |
| `str.findall(regex)` | No | Array in Nullable error |
| `str.split(expand=True)` | No | pandas fallback |
| `str.wrap(width)` | No | pandas fallback |
| `str.normalize(form)` | No | normalizeUTF8NFD doesn't exist |

Test Markers: `chdb_no_normalize_utf8`, `chdb_array_nullable`

---

### ~~Unicode String Filtering~~ RESOLVED

**Status**: RESOLVED in chDB 4.0.0b3

Unicode string filtering now works correctly.

---

### str.strip() Whitespace Handling

| Aspect | Description |
|--------|-------------|
| Issue | str.strip() doesn't handle all whitespace types (tabs, newlines) |
| Test Marker | `chdb_strip_whitespace` |

---

## SQL Semantic Differences

### Duplicate Column Auto-Rename

| Aspect | Description |
|--------|-------------|
| Issue | SQL automatically renames duplicate columns (e.g., col, col_1) |
| Test Marker | `chdb_duplicate_column_rename` |

---

### CASE WHEN Bool Conversion

| Aspect | Description |
|--------|-------------|
| Issue | SQL CASE WHEN cannot convert Bool to Int64/String in some contexts |
| Test Marker | `chdb_case_bool_conversion` |

---

### Aggregate Functions in WHERE

| Aspect | Description |
|--------|-------------|
| Issue | Aggregate functions like median() in WHERE clause require subquery |
| Test Marker | `chdb_median_in_where` |

---

## Function Limitations

### ~~product() Not Available~~ RESOLVED

**Status**: RESOLVED in chDB 4.0.0b3

product() aggregate function is now available.

---

### quantile() with Array Parameter

| Aspect | Description |
|--------|-------------|
| Issue | quantile([0.25, 0.5, 0.75]) not supported |
| Test Marker | `chdb_no_quantile_array` |

**Workaround**: Call quantile() separately for each percentile.

---

## DataStore Implementation Notes

These are not chDB limitations but DataStore design decisions or incomplete implementations:

### Callable Index Not Supported

Test Marker: `datastore_callable_index`

### query() @variable Scope

Test Marker: `datastore_query_variable_scope`

Variables in `query('@threshold')` require local scope which is lost after `_get_df()`.

### loc Conditional Assignment

Test Marker: `datastore_loc_conditional_assignment`

`ds.loc[condition, 'col'] = value` with ColumnExpr not fully compatible.

### Index Not Preserved in Lazy Execution

Test Markers: `lazy_index_not_preserved`, `lazy_extractall_multiindex`

Index information is lost when operations go through SQL execution.

---

## Workarounds Reference

### Quick Reference Table

| Issue | Workaround |
|-------|------------|
| Nullable Boolean | Convert to object type with explicit None handling |
| Nullable Int64 | Convert to float64 |
| JOIN with NULL keys | Filter NULL before join, or use pandas merge |
| GroupBy NULL group | Filter NULL before groupby, or post-process |
| Timezone issues | Ensure consistent timezone handling |

### Detecting and Handling Limitations

```python
from datastore import DataStore

# Check if column has problematic type
def needs_type_conversion(df, col):
    dtype = df[col].dtype
    if str(dtype) == 'Int64':
        return 'nullable_int_to_float'
    if str(dtype) == 'boolean':
        return 'nullable_bool_to_object'
    return None
```

---

## Version History

| Date | chDB Version | Changes |
|------|--------------|---------|
| 2026-01-05 | 4.0.0b3 | Initial documentation; identified 5 resolved limitations |

---

## Related Files

- `tests/xfail_markers.py` - Centralized test markers for known limitations
- `tests/test_chdb_limitations_tracker.py` - Automated limitation tracking tests
- `tracking/discoveries/2026-01-05_chdb_nan_none_type_issues.md` - Detailed NaN/None testing
- `datastore/function_executor.py` - `PANDAS_ONLY_FUNCTIONS` set
- `datastore/function_executor.py` - `ACCESSOR_PARAM_PANDAS_FALLBACK` registry

---

## Contributing

When you discover a new chDB limitation:

1. Add test with appropriate xfail marker from `tests/xfail_markers.py`
2. Document the limitation in this file
3. Record discovery details in `tracking/discoveries/`
4. If workaround exists, implement it in DataStore and document here

When a limitation is resolved in a new chDB version:

1. Update the chDB version in this document
2. Move the limitation to "Resolved Limitations" section
3. Update `tests/test_chdb_limitations_tracker.py` to track the resolution
4. Remove or update the corresponding xfail marker if still needed elsewhere
5. Update the Version History section
