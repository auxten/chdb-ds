# DataStore Function Reference

DataStore provides ClickHouse SQL functions through three interfaces:

1. **Accessor Pattern** (`.str`, `.dt`) - Pandas-like API for domain-specific functions
2. **Expression Methods** - Direct methods on column expressions
3. **Function Namespace (`F`)** - Explicit function calls

## Quick Reference

```python
from datastore import DataStore, F, Field

ds = DataStore.from_file('data.csv')

# Accessor pattern (recommended for chaining)
ds['name'].str.upper()           # String function
ds['date'].dt.year               # DateTime function

# Expression methods
ds['value'].abs()                # Math function
ds['price'].sum()                # Aggregate function
ds['value'].cast('Float64')      # Type conversion

# F namespace (explicit)
F.upper(Field('name'))
F.sum(Field('value'))
```

---

## String Functions (`.str` accessor)

Access via `ds['column'].str.<function>()`.

### Case Conversion

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `upper()` | `upper(s)` | Convert to uppercase | `ds['name'].str.upper()` |
| `lower()` | `lower(s)` | Convert to lowercase | `ds['name'].str.lower()` |
| `capitalize()` | `initcap(s)` | Capitalize first letter | `ds['name'].str.capitalize()` |

### Length & Size

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `length()` / `len()` | `length(s)` | String length in bytes | `ds['name'].str.length()` |
| `char_length()` | `char_length(s)` | Length in Unicode code points | `ds['name'].str.char_length()` |
| `empty()` | `empty(s)` | Check if empty (returns 1/0) | `ds['name'].str.empty()` |
| `not_empty()` | `notEmpty(s)` | Check if not empty | `ds['name'].str.not_empty()` |

### Substring & Slicing

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `substring(offset, length)` | `substring(s, offset, length)` | Extract substring (1-indexed) | `ds['name'].str.substring(1, 5)` |
| `left(n)` | `left(s, n)` | Get leftmost N characters | `ds['name'].str.left(3)` |
| `right(n)` | `right(s, n)` | Get rightmost N characters | `ds['name'].str.right(3)` |

### Trimming

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `trim()` / `strip()` | `trim(s)` | Remove leading/trailing whitespace | `ds['name'].str.trim()` |
| `ltrim()` / `lstrip()` | `trimLeft(s)` | Remove leading whitespace | `ds['name'].str.ltrim()` |
| `rtrim()` / `rstrip()` | `trimRight(s)` | Remove trailing whitespace | `ds['name'].str.rtrim()` |

### Search & Match

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `contains(pattern)` | `position(s, pattern) > 0` | Check if contains substring | `ds['name'].str.contains('test')` |
| `startswith(prefix)` | `startsWith(s, prefix)` | Check if starts with | `ds['name'].str.startswith('Mr.')` |
| `endswith(suffix)` | `endsWith(s, suffix)` | Check if ends with | `ds['name'].str.endswith('.txt')` |
| `position(needle)` / `find()` | `position(s, needle)` | Find position (1-indexed) | `ds['email'].str.position('@')` |

### Replace & Transform

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `replace(old, new)` | `replace(s, old, new)` | Replace all occurrences | `ds['text'].str.replace('old', 'new')` |
| `replace_one(old, new)` | `replaceOne(s, old, new)` | Replace first occurrence | `ds['text'].str.replace_one('old', 'new')` |
| `replace_regex(pattern, repl)` | `replaceRegexpAll(s, p, r)` | Regex replace | `ds['text'].str.replace_regex(r'\d+', 'NUM')` |
| `reverse()` | `reverse(s)` | Reverse string | `ds['name'].str.reverse()` |

### Concatenation & Padding

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `concat(*others)` | `concat(s1, s2, ...)` | Concatenate strings | `ds['first'].str.concat(' ', ds['last'])` |
| `pad_left(len, char)` | `leftPad(s, len, char)` | Left-pad to length | `ds['id'].str.pad_left(5, '0')` |
| `pad_right(len, char)` | `rightPad(s, len, char)` | Right-pad to length | `ds['name'].str.pad_right(20)` |
| `zfill(width)` | `leftPad(s, width, '0')` | Zero-pad left | `ds['id'].str.zfill(5)` |

### Splitting

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `split(separator)` | `splitByString(sep, s)` | Split into array | `ds['tags'].str.split(',')` |
| `split_by_char(char)` | `splitByChar(char, s)` | Split by single char | `ds['path'].str.split_by_char('/')` |

### Regular Expressions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `match(pattern)` | `match(s, pattern)` | Check regex match | `ds['email'].str.match(r'^[\w]+@')` |
| `extract(pattern)` | `extract(s, pattern)` | Extract first group | `ds['url'].str.extract(r'https?://([^/]+)')` |
| `extract_all(pattern)` | `extractAll(s, pattern)` | Extract all matches | `ds['text'].str.extract_all(r'\d+')` |

### Encoding

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `base64_encode()` | `base64Encode(s)` | Encode to Base64 | `ds['data'].str.base64_encode()` |
| `base64_decode()` | `base64Decode(s)` | Decode from Base64 | `ds['encoded'].str.base64_decode()` |
| `hex()` | `hex(s)` | Encode to hex | `ds['data'].str.hex()` |
| `unhex()` | `unhex(s)` | Decode from hex | `ds['hex_data'].str.unhex()` |

### URL Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `url_encode()` | `encodeURLComponent(s)` | URL-encode | `ds['text'].str.url_encode()` |
| `url_decode()` | `decodeURLComponent(s)` | URL-decode | `ds['encoded'].str.url_decode()` |
| `domain()` | `domain(url)` | Extract domain | `ds['url'].str.domain()` |
| `path()` | `path(url)` | Extract path | `ds['url'].str.path()` |
| `extract_url_parameter(name)` | `extractURLParameter(url, name)` | Extract query param | `ds['url'].str.extract_url_parameter('id')` |

### Hash Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `md5()` | `MD5(s)` | MD5 hash | `ds['data'].str.md5()` |
| `sha256()` | `SHA256(s)` | SHA256 hash | `ds['data'].str.sha256()` |
| `city_hash64()` | `cityHash64(s)` | CityHash64 (fast) | `ds['data'].str.city_hash64()` |

---

## DateTime Functions (`.dt` accessor)

Access via `ds['column'].dt.<property>` or `ds['column'].dt.<method>()`.

### Date Part Extraction (Properties)

| Property | ClickHouse | Description | Example |
|----------|------------|-------------|---------|
| `year` | `toYear(dt)` | Extract year | `ds['date'].dt.year` |
| `month` | `toMonth(dt)` | Extract month (1-12) | `ds['date'].dt.month` |
| `day` | `toDayOfMonth(dt)` | Extract day (1-31) | `ds['date'].dt.day` |
| `hour` | `toHour(dt)` | Extract hour (0-23) | `ds['ts'].dt.hour` |
| `minute` | `toMinute(dt)` | Extract minute (0-59) | `ds['ts'].dt.minute` |
| `second` | `toSecond(dt)` | Extract second (0-59) | `ds['ts'].dt.second` |
| `millisecond` | `toMillisecond(dt)` | Extract millisecond | `ds['ts'].dt.millisecond` |
| `quarter` | `toQuarter(dt)` | Extract quarter (1-4) | `ds['date'].dt.quarter` |
| `day_of_week` / `dayofweek` | `toDayOfWeek(dt)` | Day of week (1=Mon) | `ds['date'].dt.day_of_week` |
| `day_of_year` / `dayofyear` | `toDayOfYear(dt)` | Day of year (1-366) | `ds['date'].dt.day_of_year` |
| `week` / `weekofyear` | `toISOWeek(dt)` | ISO week number | `ds['date'].dt.week` |
| `iso_year` | `toISOYear(dt)` | ISO year | `ds['date'].dt.iso_year` |

### Date Truncation (Methods)

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `to_start_of_day()` | `toStartOfDay(dt)` | Truncate to day start | `ds['ts'].dt.to_start_of_day()` |
| `to_start_of_week(mode)` | `toStartOfWeek(dt, mode)` | Truncate to week start | `ds['date'].dt.to_start_of_week()` |
| `to_start_of_month()` | `toStartOfMonth(dt)` | Truncate to month start | `ds['date'].dt.to_start_of_month()` |
| `to_start_of_quarter()` | `toStartOfQuarter(dt)` | Truncate to quarter start | `ds['date'].dt.to_start_of_quarter()` |
| `to_start_of_year()` | `toStartOfYear(dt)` | Truncate to year start | `ds['date'].dt.to_start_of_year()` |
| `to_start_of_hour()` | `toStartOfHour(dt)` | Truncate to hour start | `ds['ts'].dt.to_start_of_hour()` |
| `to_start_of_minute()` | `toStartOfMinute(dt)` | Truncate to minute start | `ds['ts'].dt.to_start_of_minute()` |
| `date_trunc(unit)` | `date_trunc(unit, dt)` | Truncate to unit | `ds['ts'].dt.date_trunc('month')` |

### Date Arithmetic

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `add_years(n)` | `addYears(dt, n)` | Add years | `ds['date'].dt.add_years(1)` |
| `add_months(n)` | `addMonths(dt, n)` | Add months | `ds['date'].dt.add_months(3)` |
| `add_weeks(n)` | `addWeeks(dt, n)` | Add weeks | `ds['date'].dt.add_weeks(2)` |
| `add_days(n)` | `addDays(dt, n)` | Add days | `ds['date'].dt.add_days(7)` |
| `add_hours(n)` | `addHours(dt, n)` | Add hours | `ds['ts'].dt.add_hours(24)` |
| `add_minutes(n)` | `addMinutes(dt, n)` | Add minutes | `ds['ts'].dt.add_minutes(30)` |
| `add_seconds(n)` | `addSeconds(dt, n)` | Add seconds | `ds['ts'].dt.add_seconds(60)` |
| `sub_years(n)` | `subtractYears(dt, n)` | Subtract years | `ds['date'].dt.sub_years(1)` |
| `sub_months(n)` | `subtractMonths(dt, n)` | Subtract months | `ds['date'].dt.sub_months(1)` |
| `sub_days(n)` | `subtractDays(dt, n)` | Subtract days | `ds['date'].dt.sub_days(7)` |

### Date Difference

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `diff(other, unit)` | `dateDiff(unit, dt1, dt2)` | Difference in unit | `ds['start'].dt.diff(ds['end'], 'day')` |
| `days_diff(other)` | `dateDiff('day', ...)` | Difference in days | `ds['start'].dt.days_diff(ds['end'])` |
| `months_diff(other)` | `dateDiff('month', ...)` | Difference in months | `ds['start'].dt.months_diff(ds['end'])` |
| `years_diff(other)` | `dateDiff('year', ...)` | Difference in years | `ds['start'].dt.years_diff(ds['end'])` |

### Formatting & Conversion

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `format(fmt)` / `strftime(fmt)` | `formatDateTime(dt, fmt)` | Format as string | `ds['date'].dt.format('%Y-%m-%d')` |
| `to_date()` | `toDate(dt)` | Convert to Date | `ds['ts'].dt.to_date()` |
| `to_datetime(tz)` | `toDateTime(dt, tz)` | Convert to DateTime | `ds['date'].dt.to_datetime('UTC')` |
| `to_timezone(tz)` | `toTimezone(dt, tz)` | Convert timezone | `ds['ts'].dt.to_timezone('UTC')` |
| `to_unix_timestamp()` | `toUnixTimestamp(dt)` | To Unix timestamp | `ds['ts'].dt.to_unix_timestamp()` |

### Utility Properties

| Property | ClickHouse | Description | Example |
|----------|------------|-------------|---------|
| `is_weekend` | `toDayOfWeek(dt) >= 6` | Check if weekend | `ds['date'].dt.is_weekend` |
| `is_weekday` | `toDayOfWeek(dt) < 6` | Check if weekday | `ds['date'].dt.is_weekday` |
| `is_leap_year` | `isLeapYear(year)` | Check leap year | `ds['date'].dt.is_leap_year` |
| `days_in_month` | `toDayOfMonth(toLastDayOfMonth(dt))` | Days in month | `ds['date'].dt.days_in_month` |

---

## Math Functions (Expression Methods)

Access via `ds['column'].<method>()`.

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `abs()` | `abs(x)` | Absolute value | `ds['value'].abs()` |
| `round(n)` | `round(x, n)` | Round to N decimals | `ds['price'].round(2)` |
| `floor()` | `floor(x)` | Round down | `ds['value'].floor()` |
| `ceil()` / `ceiling()` | `ceiling(x)` | Round up | `ds['value'].ceil()` |
| `sqrt()` | `sqrt(x)` | Square root | `ds['value'].sqrt()` |
| `exp()` | `exp(x)` | Exponential (e^x) | `ds['value'].exp()` |
| `log(base)` | `log(x)` / `log(base, x)` | Logarithm | `ds['value'].log()` |
| `log10()` | `log10(x)` | Base-10 logarithm | `ds['value'].log10()` |
| `log2()` | `log2(x)` | Base-2 logarithm | `ds['value'].log2()` |
| `sin()` | `sin(x)` | Sine | `ds['angle'].sin()` |
| `cos()` | `cos(x)` | Cosine | `ds['angle'].cos()` |
| `tan()` | `tan(x)` | Tangent | `ds['angle'].tan()` |
| `asin()` | `asin(x)` | Arc sine | `ds['value'].asin()` |
| `acos()` | `acos(x)` | Arc cosine | `ds['value'].acos()` |
| `atan()` | `atan(x)` | Arc tangent | `ds['value'].atan()` |
| `power(n)` | `pow(x, n)` | Raise to power | `ds['value'].power(2)` |
| `sign()` | `sign(x)` | Sign (-1, 0, 1) | `ds['value'].sign()` |

---

## Aggregate Functions (Expression Methods)

Access via `ds['column'].<method>()`. Used with `groupby()`.

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `sum()` | `sum(x)` | Sum | `ds['amount'].sum()` |
| `avg()` / `mean()` | `avg(x)` | Average | `ds['price'].avg()` |
| `count()` | `count(x)` | Count | `ds['id'].count()` |
| `min()` | `min(x)` | Minimum | `ds['price'].min()` |
| `max()` | `max(x)` | Maximum | `ds['price'].max()` |
| `count_distinct()` / `uniq()` | `uniq(x)` | Count distinct | `ds['user_id'].count_distinct()` |
| `stddev()` | `stddevPop(x)` | Std deviation (pop) | `ds['value'].stddev()` |
| `stddev_samp()` | `stddevSamp(x)` | Std deviation (sample) | `ds['value'].stddev_samp()` |
| `variance()` | `varPop(x)` | Variance (pop) | `ds['value'].variance()` |
| `var_samp()` | `varSamp(x)` | Variance (sample) | `ds['value'].var_samp()` |
| `median()` | `median(x)` | Median | `ds['value'].median()` |
| `quantile(level)` | `quantile(level)(x)` | Quantile | `ds['value'].quantile(0.95)` |
| `group_array()` | `groupArray(x)` | Collect to array | `ds['id'].group_array()` |

---

## Type Conversion (Expression Methods)

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `cast(type)` | `CAST(x AS type)` | Cast to type | `ds['value'].cast('Float64')` |
| `to_string()` | `toString(x)` | Convert to String | `ds['id'].to_string()` |
| `to_int(bits)` | `toInt64(x)` etc. | Convert to Int | `ds['value'].to_int(32)` |
| `to_float(bits)` | `toFloat64(x)` etc. | Convert to Float | `ds['value'].to_float(64)` |
| `to_date()` | `toDate(x)` | Convert to Date | `ds['str_date'].to_date()` |
| `to_datetime(tz)` | `toDateTime(x, tz)` | Convert to DateTime | `ds['str_date'].to_datetime()` |

---

## Conditional Functions (Expression Methods)

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `if_null(default)` | `ifNull(x, default)` | Return default if NULL | `ds['value'].if_null(0)` |
| `coalesce(*values)` | `coalesce(x, ...)` | First non-NULL | `ds['v1'].coalesce(ds['v2'], 0)` |
| `null_if(value)` | `nullIf(x, value)` | NULL if equals value | `ds['status'].null_if('')` |

---

## F Namespace

For explicit function calls when you need more control:

```python
from datastore import F, Field

# String functions
F.upper(Field('name'))
F.lower(Field('name'))
F.length(Field('name'))
F.concat(Field('first'), ' ', Field('last'))
F.substring(Field('name'), 1, 5)
F.replace(Field('text'), 'old', 'new')
F.trim(Field('name'))

# Math functions
F.abs(Field('value'))
F.round(Field('price'), 2)
F.floor(Field('value'))
F.ceil(Field('value'))
F.sqrt(Field('value'))
F.pow(Field('value'), 2)

# Aggregate functions
F.sum(Field('amount'))
F.avg(Field('price'))
F.count()
F.count_distinct(Field('user_id'))
F.max(Field('value'))
F.min(Field('value'))

# Date/Time functions
F.now()
F.today()
F.year(Field('date'))
F.month(Field('date'))
F.to_date(Field('string_date'))

# Conditional functions
F.if_(Field('age') > 18, 'adult', 'minor')
F.if_null(Field('value'), 0)
F.coalesce(Field('a'), Field('b'), 0)
F.case().when(Field('status') == 1, 'active').else_('inactive')

# Type conversion
F.cast(Field('value'), 'Float64')
F.to_string(Field('id'))
F.to_int64(Field('str_num'))
```

---

## Hybrid Execution Configuration

Control whether overlapping functions use chDB SQL or Pandas:

### Global Execution Engine (Recommended)

Use `config` to set the global execution engine for all operations:

```python
from datastore import DataStore, config

# Set execution engine globally
config.set_execution_engine('pandas')      # Force Pandas for all operations
config.set_execution_engine('clickhouse')  # Force ClickHouse/chDB
config.set_execution_engine('auto')        # Auto-select best engine (default)

# Shortcut methods
config.use_pandas()       # Force Pandas
config.use_clickhouse()   # Force ClickHouse
config.use_auto()         # Auto-select

# Access via DataStore.config
DataStore.config.execution_engine = 'pandas'
DataStore.config.use_pandas()
DataStore.config.use_clickhouse()
DataStore.config.use_auto()
```

### Per-Function Configuration

For fine-grained control over specific functions:

```python
from datastore import function_config, use_chdb, use_pandas

# Check current config
print(function_config.get_config_summary())

# Use Pandas for specific functions
function_config.use_pandas('upper', 'lower', 'abs')

# Use chDB for specific functions (default)
function_config.use_chdb('sum', 'avg', 'count')

# Set default engine preference
function_config.prefer_pandas()  # Default to Pandas
function_config.prefer_chdb()    # Default to chDB (default)

# Reset to defaults
function_config.reset()
```

Functions that have both chDB and Pandas implementations include:
- **String**: upper, lower, length, substring, replace, trim, etc.
- **Math**: abs, round, floor, ceil, sqrt, exp, log, etc.
- **Aggregate**: sum, avg, min, max, count, std, var, median, etc.
- **DateTime**: year, month, day, hour, minute, second, etc.

---

## Pandas-Only Functions

These functions are **automatically executed via Pandas** (no ClickHouse equivalent):

### Cumulative Functions

| Function | Description | Example |
|----------|-------------|---------|
| `cumsum` | Cumulative sum | `Function('cumsum', ds['value'])` |
| `cummax` | Cumulative maximum | `Function('cummax', ds['value'])` |
| `cummin` | Cumulative minimum | `Function('cummin', ds['value'])` |
| `cumprod` | Cumulative product | `Function('cumprod', ds['value'])` |

### Shift and Difference

| Function | Description | Example |
|----------|-------------|---------|
| `shift` | Shift values by N periods | `Function('shift', ds['value'], 1)` |
| `diff` | First discrete difference | `Function('diff', ds['value'], 1)` |
| `pct_change` | Percentage change | `Function('pct_change', ds['value'])` |

### Ranking

| Function | Description | Example |
|----------|-------------|---------|
| `rank` | Compute rank of values | `Function('rank', ds['value'])` |
| `nlargest` | N largest values | `Function('nlargest', ds['value'], 5)` |
| `nsmallest` | N smallest values | `Function('nsmallest', ds['value'], 5)` |

### Missing Value Handling

| Function | Description | Example |
|----------|-------------|---------|
| `fillna` | Fill NA/NaN values | `Function('fillna', ds['value'], 0)` |
| `ffill` | Forward fill | `Function('ffill', ds['value'])` |
| `bfill` | Backward fill | `Function('bfill', ds['value'])` |
| `interpolate` | Interpolate values | `Function('interpolate', ds['value'])` |
| `isna` / `isnull` | Check for NA | `Function('isna', ds['value'])` |
| `notna` / `notnull` | Check for not-NA | `Function('notna', ds['value'])` |

### String Functions (Pandas-only)

| Function | Description | Example |
|----------|-------------|---------|
| `title` | Title case | `Function('title', ds['name'])` |
| `capitalize` | Capitalize first letter | `Function('capitalize', ds['text'])` |
| `swapcase` | Swap case | `Function('swapcase', ds['text'])` |
| `isalpha` | Check if alphabetic | `Function('isalpha', ds['text'])` |
| `isdigit` | Check if numeric | `Function('isdigit', ds['text'])` |

### Value Operations

| Function | Description | Example |
|----------|-------------|---------|
| `unique` | Unique values | `Function('unique', ds['category'])` |
| `value_counts` | Count occurrences | `Function('value_counts', ds['category'])` |
| `duplicated` | Mark duplicates | `Function('duplicated', ds['id'])` |
| `isin` | Check membership | `Function('isin', ds['status'], ['A', 'B'])` |

### Usage Example

```python
from datastore import DataStore, Function, Field

ds = DataStore.from_file('data.csv')

# Pandas-only functions work automatically
ds['cum_total'] = Function('cumsum', Field('value'))
ds['prev_value'] = Function('shift', Field('value'), 1)
ds['change'] = Function('diff', Field('value'))
ds['title_name'] = Function('title', Field('name'))

# These execute via Pandas (no chDB fallback needed)
df = ds.to_df()
```

---

## Dynamic Pandas Method Invocation

For functions **not registered** in the function library, DataStore automatically tries to find and call the corresponding Pandas method:

```python
from datastore import DataStore, Function, Field

ds = DataStore.from_file('data.csv')

# These are NOT registered, but work via dynamic invocation:

# String methods (via Series.str accessor)
ds['no_prefix'] = Function('removeprefix', Field('name'), 'Mr. ')
ds['no_suffix'] = Function('removesuffix', Field('name'), ' Jr.')

# Series methods
ds['plus_100'] = Function('add', Field('value'), 100)
ds['times_10'] = Function('mul', Field('value'), 10)
ds['negated'] = Function('neg', Field('value'))

# DateTime methods (via Series.dt accessor)  
ds['weekday'] = Function('weekday', Field('date'))
```

### How It Works

1. **Registered functions** → Use the registered implementation
2. **Pandas-only functions** → Automatically use Pandas
3. **Unknown functions** → Try dynamic Pandas method:
   - First try `series.method_name(*args)`
   - Then try `series.str.method_name(*args)`
   - Then try `series.dt.method_name(*args)`
   - If all fail, fallback to chDB

This means you can use **any Pandas Series/str/dt method** without explicit registration!

---

## ClickHouse Function Reference

For the complete list of ClickHouse functions, see:
- [String Functions](https://clickhouse.com/docs/en/sql-reference/functions/string-functions)
- [Date/Time Functions](https://clickhouse.com/docs/en/sql-reference/functions/date-time-functions)
- [Math Functions](https://clickhouse.com/docs/en/sql-reference/functions/math-functions)
- [Aggregate Functions](https://clickhouse.com/docs/en/sql-reference/aggregate-functions)

