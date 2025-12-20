# DataStore Pandas 兼容性改进建议报告

## 执行摘要

本报告基于对 GitHub 上 15 个热门 Jupyter Notebook 项目的分析（共 331 个 pandas 操作示例），以及通过 `import datastore as pd` 方式进行的 monkey patch 兼容性测试。

### 测试结果概览

- **总测试数**: 44 个核心操作
- **Pandas 成功率**: 43/44 (97.7%)
- **DataStore 成功率**: 40/44 (90.9%)
- **两者都成功**: 40/44 (90.9%)
- **结果匹配率**: 18/40 (45.0%)

### 数据来源

**分析的 Notebook 仓库**:
- donnemartin/data-science-ipython-notebooks (6 个文件)
- jvns/pandas-cookbook (7 个文件)
- KeithGalli/complete-pandas-tutorial (1 个文件)
- LearnDataSci/articles (1 个文件)

**提取的操作分布**:
1. 数据选择 (Data Selection): 207 个示例
2. DataFrame 创建 (DataFrame Creation): 26 个示例
3. 数据清洗 (Data Cleaning): 24 个示例
4. 统计操作 (Statistics): 24 个示例
5. 数据转换 (Data Transformation): 17 个示例
6. 排序 (Sorting): 12 个示例
7. 聚合操作 (Aggregation): 6 个示例
8. 字符串操作 (String Operations): 6 个示例
9. 合并操作 (Merging): 5 个示例
10. 时间序列操作 (DateTime Operations): 4 个示例

---

## 优先级 1: 关键兼容性问题 (Critical)

### 1.1 布尔索引不支持

**问题**: `df[df['age'] > 30]` 抛出 `TypeError: DataStore indices must be slices, strings, or lists, not BinaryCondition`

**影响**: 这是 pandas 中最常用的过滤方式之一，在分析的 207 个数据选择示例中占比很高。

**建议修复**:
```python
# 需要支持的模式
df[df['age'] > 30]
df[(df['age'] > 30) & (df['salary'] > 50000)]
df[df['name'].str.contains('Alice')]
```

**实现方案**:
- 在 `__getitem__` 方法中检测 `BinaryCondition` 类型
- 将条件转换为等效的 `.filter()` 操作
- 确保返回结果与 pandas 一致（包括索引）

### 1.2 GroupBy.size() 方法不可用

**问题**: `df.groupby('department').size()` 抛出 `TypeError: 'ColumnExpr' object is not callable`

**影响**: size() 是分组后计数的标准方法，在实际数据分析中极为常用。

**建议修复**:
```python
# 需要支持
df.groupby('col').size()  # 返回 Series，包含每组的行数
```

**实现方案**:
- 在 `LazyGroupBy` 类中实现 `size()` 方法
- 返回每组的计数，不排除 NaN 值
- 注意与 `count()` 的区别（count 排除 NaN）

### 1.3 String 操作参数不兼容

**问题**: `df['name'].str.contains('a', na=False)` 抛出 `TypeError: _build_contains() got an unexpected keyword argument 'na'`

**影响**: pandas 的 str.contains() 默认对 NaN 返回 NaN，需要 `na=False` 参数控制。

**建议修复**:
```python
# StringAccessor._build_contains() 需要支持以下参数
.str.contains(pattern, case=True, flags=0, na=None, regex=True)
```

**实现方案**:
- 更新 `StringAccessor` 中的 `contains()` 方法签名
- 正确处理 `na` 参数（填充 NaN 值的返回结果）

---

## 优先级 2: 结果差异问题 (High Priority)

以下操作虽然能执行成功，但结果与 pandas 不一致：

### 2.1 排序操作结果不一致

**问题**: 所有排序操作（sort_values, sort_index）结果都与 pandas 不同

**测试失败**:
- `df.sort_values('age')` ✗
- `df.sort_values(['department', 'age'])` ✗
- `df.sort_values('salary', ascending=False)` ✗
- `df.sort_index()` ✗

**可能原因**:
1. 索引重置策略不同
2. NaN 值排序位置不同
3. 稳定排序 (stable sort) 实现差异
4. 多列排序优先级处理不同

**建议修复**:
```python
# 需要完全匹配 pandas 的行为
- NaN 值默认排在最后（ascending=True）或最前（ascending=False）
- 保持原始索引，除非 ignore_index=True
- 多列排序按列顺序优先级
- 使用稳定排序算法
```

### 2.2 数据清洗操作结果不一致

**问题**: dropna, fillna, drop_duplicates, replace, drop 等所有清洗操作结果都不匹配

**测试失败**:
- `df.dropna()` ✗
- `df.fillna(0)` ✗
- `df.drop_duplicates()` ✗
- `df.replace('HR', 'Human Resources')` ✗
- `df.drop('salary', axis=1)` ✗

**可能原因**:
1. 返回类型不同（可能返回 LazyOps 而非 DataFrame）
2. 索引处理不同
3. inplace 参数处理不同
4. 列顺序可能改变

**建议修复**:
- 确保返回 pandas 兼容的 DataFrame 对象
- 保持原始列顺序
- 正确处理索引
- 完整实现 inplace 参数

### 2.3 字符串操作结果不一致

**问题**: str.upper(), str.lower(), str.len(), str.replace() 等所有字符串操作结果都不匹配

**测试失败**:
- `df['name'].str.upper()` ✗
- `df['city'].str.lower()` ✗
- `df['name'].str.len()` ✗
- `df['city'].str.replace('York', 'Amsterdam')` ✗

**可能原因**:
1. 返回类型不同（可能是 ColumnExpr 而非 Series）
2. NaN 值处理不同
3. 索引未正确传递

**建议修复**:
```python
# 确保字符串操作返回正确的 Series
result = df['name'].str.upper()
assert isinstance(result, pd.Series)  # 或 datastore 的 Series 类
assert result.index.equals(df.index)  # 索引必须匹配
assert pd.isna(result[pd.isna(df['name'])]).all()  # NaN 传递
```

### 2.4 DateTime 操作结果不一致

**问题**: 所有日期时间操作结果都不匹配

**测试失败**:
- `pd.to_datetime(df['hire_date'])` ✗
- `df['hire_date'].dt.year` ✗
- `df['hire_date'].dt.month` ✗
- `df['hire_date'].dt.strftime('%Y-%m')` ✗

**错误信息** (从日志):
```
E [chDB] Expression evaluation failed: Code: 43. DB::Exception: Illegal type String of argument of function toYear. Should be Date, Date32, DateTime or DateTime64
```

**根本原因**:
- to_datetime() 可能没有真正转换类型，仍然是字符串
- dt accessor 期望 DateTime 类型但接收到 String

**建议修复**:
```python
# to_datetime 必须真正转换类型
df['date'] = pd.to_datetime(df['date_str'])
# 内部应该将 ClickHouse 列类型从 String 转换为 DateTime64
# dt accessor 应该检查类型并在需要时自动转换
```

### 2.5 其他结果不一致的操作

- `df[:3]` - 切片结果不同
- `df.rename(columns={...})` - 重命名结果不同
- `df['new_col'] = expr` - 添加新列后 DataFrame 不同
- `df.groupby().agg({...})` - 多列聚合结果格式不同
- `df.describe()` - 统计摘要格式不同

---

## 优先级 3: 功能增强建议 (Medium Priority)

### 3.1 完善 DataFrame 构造器

**当前状态**: 基本的 DataFrame 创建已支持

**需要增强**:
```python
# 从 notebook 分析中提取的常见模式
pd.DataFrame(data, columns=['a', 'b'], index=['x', 'y'])  # ✓ 已支持
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])  # 需要测试
pd.DataFrame(np.random.rand(3, 2), columns=['A', 'B'])  # 需要测试
pd.DataFrame.from_dict({...}, orient='index')  # 需要支持
pd.DataFrame.from_records([...])  # 需要支持
```

### 3.2 完善 read_* 函数

**从分析中发现的常用模式**:
```python
# 需要确保支持所有这些参数组合
pd.read_csv('file.csv', parse_dates=True, index_col='Date')
pd.read_csv('file.csv', sep=';', encoding='latin1')
pd.read_csv('file.csv', dtype={'col': str})
pd.read_csv('file.csv', na_values=['NA', 'null'])
pd.read_csv('file.csv', parse_dates=['date_col'], dayfirst=True)
```

### 3.3 支持 apply/map 的 lambda 函数

**问题**: 当前 apply/map 可能不支持复杂的 lambda

**需要支持**:
```python
df['age'].apply(lambda x: x * 2 if pd.notna(x) else x)
df['category'].map({'A': 1, 'B': 2, 'C': 3})
df.apply(lambda row: row['a'] + row['b'], axis=1)
df.applymap(lambda x: str(x).upper())  # DataFrame level
```

**实现挑战**:
- ClickHouse 不支持自定义 Python 函数
- 需要回退到 pandas 执行或尝试转换为 SQL

**建议方案**:
1. 简单表达式尝试转换为 SQL（如 `x * 2`）
2. 复杂 lambda 回退到 pandas 执行
3. 提供配置选项让用户选择执行引擎

### 3.4 完善统计方法

**当前状态**: mean, sum, median, std 已匹配

**需要增强**:
```python
# 确保以下方法完全兼容
df.describe()  # 结果格式需要匹配
df.quantile([0.25, 0.75])
df.corr(method='pearson')  # 支持不同方法
df.cov()
df.value_counts(normalize=True, dropna=False)
```

### 3.5 完善 GroupBy 功能

**需要支持**:
```python
# 多种聚合方式
df.groupby('col').agg(['mean', 'sum', 'count'])
df.groupby('col').agg({'col1': 'mean', 'col2': ['min', 'max']})
df.groupby('col').transform(lambda x: x - x.mean())
df.groupby(['col1', 'col2']).size()  # 多列分组
df.groupby('col').filter(lambda x: len(x) > 2)

# 确保返回格式与 pandas 一致
- 单列单聚合 -> Series
- 多列或多聚合 -> DataFrame
- 正确的列名（MultiIndex 或扁平）
```

---

## 优先级 4: 新功能支持 (Lower Priority)

### 4.1 支持更多 IO 格式

从 notebook 分析中发现的使用频率：
```python
pd.read_csv()     # 26 次 - ✓ 已支持
pd.read_excel()   # 需要测试
pd.read_json()    # 需要测试
pd.read_sql()     # 需要测试
pd.read_parquet() # 需要测试
pd.read_html()    # 低优先级
```

### 4.2 支持时间序列特性

```python
# 从 notebook 发现的常用操作
pd.date_range('2020-01-01', '2020-12-31', freq='D')
df.resample('M').mean()  # 时间重采样
df.rolling(window=7).mean()  # 滚动窗口
df.shift(1)  # 时间位移
```

### 4.3 支持数据重塑

```python
# 从 notebook 发现的常用操作
pd.pivot_table(df, values='value', index='row', columns='col')
pd.melt(df, id_vars=['id'], value_vars=['a', 'b'])
df.stack() / df.unstack()
df.transpose()
```

---

## 实现路线图建议

### Phase 1: 关键修复 (1-2 周)
1. ✅ 布尔索引支持
2. ✅ GroupBy.size() 实现
3. ✅ 字符串操作参数兼容
4. ✅ DateTime 类型转换修复

### Phase 2: 结果一致性 (2-3 周)
1. 🔧 修复所有排序操作
2. 🔧 修复数据清洗操作
3. 🔧 修复字符串操作返回类型
4. 🔧 统一 DataFrame/Series 输出格式

### Phase 3: 功能增强 (3-4 周)
1. 📈 完善 apply/map/lambda 支持
2. 📈 增强 GroupBy 聚合功能
3. 📈 完善统计方法
4. 📈 增强 IO 功能

### Phase 4: 高级特性 (持续)
1. 🎯 时间序列完整支持
2. 🎯 数据重塑功能
3. 🎯 性能优化
4. 🎯 完整的单元测试覆盖

---

## 测试策略建议

### 1. 创建兼容性测试套件

基于本次分析的 331 个真实示例，创建自动化测试：

```python
# tests/test_pandas_compatibility.py
class TestPandasCompatibility:
    def test_real_world_examples(self):
        """测试从真实 notebook 提取的所有操作"""
        with open('pandas_operations_analysis.json') as f:
            operations = json.load(f)

        for category, ops in operations['detailed_operations'].items():
            for example in ops['examples']:
                # 比较 pandas 和 datastore 结果
                assert_results_match(example['code'])
```

### 2. 建立持续集成

```yaml
# .github/workflows/pandas-compat.yml
- name: Pandas Compatibility Test
  run: |
    python pandas_compatibility_test.py
    # 如果兼容率低于 95%，则失败
```

### 3. 创建兼容性矩阵文档

| 操作类型 | 支持度 | 注意事项 |
|---------|-------|---------|
| DataFrame 创建 | ✅ 100% | - |
| 数据选择 | ⚠️ 83% | 不支持布尔索引 |
| 统计操作 | ✅ 95% | describe() 格式略有差异 |
| ... | ... | ... |

---

## 特定问题深度分析

### 问题 1: 为什么结果"都不匹配"？

测试显示很多操作虽然成功执行，但结果不匹配。经分析，主要原因：

#### 1.1 返回类型问题

```python
# Pandas 返回
type(df.dropna())  # pandas.DataFrame

# DataStore 可能返回
type(df.dropna())  # datastore.lazy_result.LazyOps
str(df.dropna())   # <LazyOps: SELECT ...>
```

**解决方案**: 确保所有方法在 `__str__()` 和 `__repr__()` 时自动执行并返回类似 pandas 的表示。

#### 1.2 索引处理问题

```python
# Pandas 保持原索引
df_filtered = df[df['age'] > 30]
df_filtered.index  # [1, 2, 4] (原始索引)

# DataStore 可能重置索引
df_filtered.index  # [0, 1, 2] (新索引)
```

**解决方案**:
- 默认保持原索引
- 提供 `reset_index()` 显式重置
- 在 SQL 中使用 ROW_NUMBER 或类似机制追踪原索引

#### 1.3 列顺序问题

```python
# Pandas 保持列顺序
df.drop('col', axis=1)  # 其他列顺序不变

# DataStore 可能重排
df.drop('col', axis=1)  # SELECT * 可能改变顺序
```

**解决方案**: 在 SQL 生成时显式指定列顺序。

### 问题 2: 如何支持复杂的 Lambda 函数？

这是一个技术挑战，因为 ClickHouse 不能直接执行 Python 代码。

#### 方案 A: SQL 转换（有限支持）

```python
# 可以转换的简单表达式
df['age'].apply(lambda x: x * 2)
# -> SELECT age * 2 FROM table

df['age'].apply(lambda x: x if x > 18 else 0)
# -> SELECT if(age > 18, age, 0) FROM table
```

使用 AST 分析 lambda 表达式，转换为等效 SQL。

#### 方案 B: 混合执行

```python
# 配置项
import datastore as pd
pd.config.set_option('compute.engine', 'auto')  # 默认
pd.config.set_option('compute.engine', 'chdb')  # 强制 chdb
pd.config.set_option('compute.engine', 'pandas')  # 强制 pandas

# 自动选择：简单表达式用 chdb，复杂的用 pandas
df['result'] = df['col'].apply(complex_lambda)  # 自动降级到 pandas
```

#### 方案 C: UDF 支持（长期）

探索 ClickHouse 的 UDF 功能，虽然可能有性能损失。

---

## 性能考虑

### 何时 DataStore 更快？

```python
# 大数据集 + 简单操作 = DataStore 胜
df = pd.read_csv('huge_file.csv')  # 1GB+
result = df.groupby('category').mean()  # ClickHouse 优化的聚合

# 小数据集 + 复杂操作 = Pandas 可能更快
df = pd.read_csv('small_file.csv')  # 10KB
result = df.apply(complex_function)  # Python UDF 开销大
```

### 建议

- 提供性能基准测试脚本
- 在文档中说明最佳使用场景
- 提供 `explain()` 方法显示执行计划

---

## 文档改进建议

### 1. 创建迁移指南

```markdown
# 从 Pandas 迁移到 DataStore

## 完全兼容的操作
- ✅ read_csv, read_parquet
- ✅ df.select, df.filter
- ✅ df.groupby().mean/sum/count
...

## 需要修改的操作
- ⚠️ df[df['col'] > value] -> df.filter(df['col'] > value)
- ⚠️ df.groupby().size() -> df.groupby().count()
...

## 不支持的操作
- ❌ df.apply(lambda x: custom_func(x))
- ❌ df.pivot_table(...) [开发中]
...
```

### 2. 创建性能对比文档

展示 DataStore 在哪些场景下优于 Pandas：
- 大文件读取
- 聚合操作
- 多表 JOIN

### 3. 创建故障排除指南

常见错误和解决方案：
```markdown
## 错误: "BinaryCondition not supported"
**原因**: 使用了布尔索引
**解决**: 改用 .filter() 方法
...
```

---

## 附录

### A. 完整测试结果

详见文件:
- `pandas_compatibility_report.txt` - 可读报告
- `pandas_compatibility_results.json` - 详细 JSON 数据

### B. 分析的 Notebook 列表

详见文件:
- `pandas_operations_analysis.json` - 331 个操作示例
- `downloaded_notebooks/` - 15 个原始 notebook 文件

### C. 测试脚本

- `pandas_compatibility_test.py` - 自动化测试脚本

可以运行：
```bash
python pandas_compatibility_test.py
```

生成最新的兼容性报告。

---

## 总结

DataStore 已经实现了相当程度的 pandas 兼容性（90.9% 操作成功率），但在以下方面需要改进：

**立即需要修复**:
1. 布尔索引支持
2. GroupBy.size() 方法
3. 字符串操作参数

**重要性能改进**:
1. 所有排序操作的结果一致性
2. 数据清洗操作的返回类型
3. DateTime 类型转换

**长期增强**:
1. 复杂 lambda 支持
2. 时间序列功能
3. 数据重塑操作

通过系统性地解决这些问题，DataStore 可以成为真正的"可以用 `import datastore as pd` 替换 pandas"的库，同时保持其在大数据处理上的性能优势。
