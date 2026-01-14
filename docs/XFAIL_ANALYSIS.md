# xfail 标记分析归类

> 生成日期: 2026-01-14
> 
> 本文档对 `tests/xfail_markers.py` 中所有活跃的 xfail 标记进行分类分析。

---

## 📊 总览

| 类别 | 标记数量 | 测试用例数 | 状态 |
|------|----------|-----------|------|
| **chdb 引擎限制** | 25 | 54 | ❌ 无法在 DataStore 层修复 |
| **DataStore Bug** | 1 | 0 | ⚠️ 应修复（定义了但未使用） |
| **DataStore 限制** | 3 | 4 | 🔧 可实现 |
| **设计决策** | 1 | 2 | ✅ 有意为之 |
| **废弃特性** | 1 | 1 | ⏳ pandas 演进 |
| **已修复 (no-op)** | 10+ | 13 | ✅ 保留用于 import 兼容 |
| **合计** | **31 活跃** | **61 + 13** | |

**测试影响**: 约 74 个测试用例被标记（61 个活跃 xfail + 13 个 no-op），分布在 32 个测试文件中。

---

## 1️⃣ chDB 引擎限制 (chdb_*) — 无法在 DataStore 层修复

这些是 chDB/ClickHouse 引擎本身的限制，DataStore 无法绕过。

### 类型支持 (4个)

| 标记 | 原因 | 备注 |
|------|------|------|
| `chdb_category_type` | chDB 不支持 CATEGORY numpy 类型 | 只读访问可以工作 |
| `chdb_timedelta_type` | chDB 不支持 TIMEDELTA numpy 类型 | 只读访问可以工作 |
| `chdb_array_nullable` | Array 类型不能在 Nullable 中 | JSON 相关函数受影响 |
| `chdb_array_string_conversion` | numpy array 在 SQL 中被转换为字符串 | 影响 array accessor |

### 函数缺失 (4个)

| 标记 | 原因 | pandas 等效 |
|------|------|-------------|
| `chdb_no_product_function` | 不支持 `product()` 聚合函数 | `df.prod()` |
| `chdb_no_normalize_utf8` | 没有 `normalizeUTF8NFD` 函数 | `str.normalize()` |
| `chdb_no_quantile_array` | `quantile` 不支持数组参数 | `quantile([0.25, 0.75])` |
| `chdb_median_in_where` | WHERE 子句中聚合函数需要子查询 | `df[df['x'] > df['x'].median()]` |

### 字符串/Unicode (2个)

| 标记 | 原因 |
|------|------|
| `chdb_unicode_filter` | SQL 过滤器中 Unicode 字符串有编码问题 |
| `chdb_strip_whitespace` | `str.strip()` 不能处理所有空白类型 |

### 日期时间 (5个)

| 标记 | 原因 | strict |
|------|------|--------|
| `chdb_datetime_range_comparison` | Python() 表函数给日期添加本地时区偏移，导致日期范围比较偏差 | True |
| `chdb_datetime_extraction_conflict` | 多个 dt 提取导致列名冲突 | True |
| `chdb_dt_month_type` | `dt.month` 在 SQL 和 DataFrame 间返回类型不一致 | True |
| `chdb_no_day_month_name` | `day_name()`/`month_name()` 未在 SQL 映射中实现 | True |
| `chdb_strftime_format_difference` | `strftime('%M')` 返回月份名而非分钟数 | True |

> **注**: `chdb_datetime_timezone` (dt.year 等日期提取) 已在 chDB 4.0.0b3 中修复。

### SQL 行为 (3个)

| 标记 | 原因 |
|------|------|
| `chdb_duplicate_column_rename` | SQL 自动重命名重复列名 |
| `chdb_case_bool_conversion` | CASE WHEN 不能在 Bool 与 Int64/String 间转换 |
| `chdb_alias_shadows_column_in_where` | 复杂 groupby 链中 SELECT alias 可能遮蔽原列名 |

### 字符串方法限制 (3个)

| 标记 | 原因 | pandas 方法 |
|------|------|-------------|
| `chdb_pad_no_side_param` | `str.pad()` 只支持左填充，无 `side` 参数 | `str.pad(side='right')` |
| `chdb_center_implementation` | `str.center()` 实现使用 rightPad 而非正确居中 | `str.center()` |
| `chdb_startswith_no_tuple` | `startswith/endswith` 不支持 tuple 参数 | `str.startswith(('a', 'b'))` |

### dtype 差异 (3个)

> **注意**: 这些情况下 **值是正确的**，仅数据类型与 pandas 不同。DataStore 返回的类型在语义上可能更正确。

| 标记 | 原因 | DataStore 返回 | pandas 返回 |
|------|------|----------------|-------------|
| `chdb_nat_returns_nullable_int` | NaT 处理 | Nullable Int32 | float64 |
| `chdb_replace_none_dtype` | `replace(None)` | Nullable Int64 | object |
| `chdb_mask_dtype_nullable` | `mask/where` 对 int | Nullable Int64 | float64 |

### chDB Bug (1个)

| 标记 | 原因 | Issue |
|------|------|-------|
| `chdb_python_table_noncontiguous_index` | Python() 表函数对非连续索引返回错误数据 | [#478](https://github.com/chdb-io/chdb/issues/478) |

---

## 2️⃣ DataStore Bug (bug_*) — 应该修复

这些是 DataStore 的 bug，应该被修复以匹配 pandas 行为。

| 标记 | 原因 | 修复建议 |
|------|------|----------|
| `bug_extractall_multiindex` | `extractall` 返回 MultiIndex DataFrame，索引信息在 lazy 执行中丢失 | 实现 MultiIndex 在 lazy 执行中的保留和传递 |

---

## 3️⃣ DataStore 限制 (limit_*) — 未实现的功能

这些是 DataStore 尚未实现的功能。

| 标记 | 原因 | 优先级 | 变通方案 |
|------|------|--------|----------|
| `limit_str_join_array` | `str.join()` 需要 Array 类型列 | 低 | 使用 pandas fallback |
| `limit_datastore_index_setter` | `index` 属性没有 setter | 中 | 使用 `set_index()` |
| `limit_groupby_series_param` | `groupby` 不支持 Series/ColumnExpr 参数 | 中 | 先 assign 到列，再 groupby 列名 |

---

## 4️⃣ 设计决策 (design_*) — 有意的行为差异

这些是有意识的设计决定，不是需要修复的 bug。

| 标记 | 原因 | 说明 |
|------|------|------|
| `design_datetime_fillna_nat` | datetime `where/mask` 使用 NaT 而非 0/-1 | pandas 用 0/-1 替代，DataStore 使用 NaT 语义更清晰 |

---

## 5️⃣ 废弃特性 (deprecated_*)

pandas 已废弃的功能。

| 标记 | 原因 | pandas 版本 |
|------|------|-------------|
| `deprecated_fillna_downcast` | `fillna(downcast=...)` 参数已废弃 | pandas 2.x |

---

## 6️⃣ Pandas 版本兼容 (pandas_version_*)

> **注意**: 这些是 `skipif` 标记，不是 `xfail`。用于处理不同 pandas 版本间的 API 差异。

| 标记 | 条件 | 说明 |
|------|------|------|
| `pandas_version_no_dataframe_map` | pandas < 2.1 | `DataFrame.map()` 在 2.1+ 添加 |
| `pandas_version_no_include_groups` | pandas < 2.1 | `groupby.apply(include_groups=...)` 在 2.1+ 添加 |
| `pandas_version_nullable_int_dtype` | pandas < 2.1 | Nullable Int64 处理在 2.1+ 改进 |
| `pandas_version_nullable_bool_sql` | pandas < 2.1 | Nullable bool SQL 处理差异 |

---

## 🎯 修复优先级建议

### 高优先级
1. **`bug_extractall_multiindex`**: MultiIndex 在 lazy 执行中的保留，影响 `str.extractall()` 的完整功能

### 中优先级
2. **`limit_groupby_series_param`**: 支持 `groupby(ds['col'].dt.year)` 语法
3. **`limit_datastore_index_setter`**: 支持 `ds.index = ...` 赋值

### 低优先级 (可考虑 pandas fallback)
4. **日期时间相关** (`chdb_datetime_*`): 问题最多的领域，可增加 fallback
5. **字符串方法** (`chdb_pad_*`, `chdb_center_*`): 使用场景较少

---

## 📁 已修复标记 (参考)

以下标记已修复，在 `xfail_markers.py` 中保留为 no-op 函数以保持 import 兼容性：

- `chdb_nullable_int64_comparison` - chDB 4.0.0b3 修复
- `chdb_null_in_groupby` - dropna 参数实现
- `chdb_nan_sum_behavior` - fillna(0) workaround
- `chdb_string_plus_operator` - 自动转换为 concat()
- `chdb_datetime_timezone` - dt.year/month/day 提取在 chDB 4.0.0b3 中修复
- `bug_groupby_first_last` - chDB any()/anyLast() 现在保序
- `bug_groupby_index` - groupby 现在正确保留 index
- `bug_index_not_preserved` - lazy 执行现在保留 index 信息
- `limit_callable_index` - callable 作为索引已支持
- `limit_query_variable_scope` - query() @variable 已支持
- `limit_loc_conditional_assignment` - loc 条件赋值已支持
- `limit_where_condition` - where() 条件已支持
- `design_unstack_column_expr` - unstack() 已实现
- `chdb_python_table_rownumber_nondeterministic` - _row_id 虚拟列解决
- `limit_datastore_no_invert` - `__invert__` 方法已添加到 PandasCompatMixin
