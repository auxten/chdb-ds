#!/usr/bin/env python3
"""
深入测试:查找 bug、边界情况和设计问题
"""
import tempfile
import os

def test_column_assignment_bug():
    """测试列赋值是否真的工作"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("BUG 测试 1: 列赋值后新列是否出现在查询中?")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test.csv")
    with open(csv_file, "w") as f:
        f.write("id,name,age\n")
        f.write("1,Alice,25\n")
        f.write("2,Bob,35\n")

    try:
        ds = DataStore.from_file(csv_file)

        # README 说可以这样做: ds['upper_name'] = ds['name'].str.upper()
        ds['upper_name'] = ds['name'].str.upper()

        # 检查 SQL
        sql = ds.select('*').to_sql()
        print(f"SQL after assignment: {sql}")

        # 问题: upper_name 列在 SQL 中吗?
        if 'upper_name' in sql or 'upper(' in sql:
            print("✓ 新列出现在 SQL 中")
        else:
            print("✗ BUG: 新列没有出现在 SQL 中!")
            print("  这意味着列赋值可能不工作")

        # 尝试执行看看结果
        try:
            result = ds.select('*').to_df()
            print(f"执行结果的列: {list(result.columns)}")
            if 'upper_name' in result.columns:
                print("✓ 新列出现在结果中")
            else:
                print("✗ BUG: 新列没有出现在结果中!")
        except Exception as e:
            print(f"✗ 执行失败: {e}")

        # 尝试明确选择新列
        try:
            result2 = ds.select('name', 'upper_name').to_df()
            print(f"✓ 可以明确选择 'upper_name' 列")
            print(f"  结果: {result2}")
        except Exception as e:
            print(f"✗ 不能明确选择 'upper_name': {e}")

    finally:
        os.unlink(csv_file)
        os.rmdir(temp_dir)

    print()

def test_uri_vs_factory_methods():
    """测试 URI 方式和工厂方法的一致性"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("一致性测试 1: URI vs 工厂方法")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test.csv")
    with open(csv_file, "w") as f:
        f.write("id,value\n")
        f.write("1,100\n")
        f.write("2,200\n")

    try:
        # 方式 1: URI
        ds1 = DataStore.uri(csv_file)
        sql1 = ds1.select('*').to_sql()

        # 方式 2: from_file
        ds2 = DataStore.from_file(csv_file)
        sql2 = ds2.select('*').to_sql()

        print(f"URI SQL:       {sql1}")
        print(f"from_file SQL: {sql2}")

        if sql1 == sql2:
            print("✓ SQL 一致")
        else:
            print("⚠ SQL 不一致 - 可能不是问题,但值得注意")

    finally:
        os.unlink(csv_file)
        os.rmdir(temp_dir)

    print()

def test_connect_confusion():
    """README 中的 connect() 混淆"""
    from datastore import DataStore

    print("="*60)
    print("文档问题 1: connect() 的必要性")
    print("="*60)

    # README 第 44 行说需要 connect()
    # 但其他示例都没有调用 connect()

    ds = DataStore.from_numbers(10)

    print("不调用 connect() 测试:")
    try:
        result1 = ds.select('*').to_df()
        print(f"✓ 不调用 connect() 可以工作, 返回 {len(result1)} 行")
    except Exception as e:
        print(f"✗ 不调用 connect() 失败: {e}")

    print("\n调用 connect() 测试:")
    ds2 = DataStore.from_numbers(10)
    try:
        ds2.connect()
        result2 = ds2.select('*').to_df()
        print(f"✓ 调用 connect() 可以工作, 返回 {len(result2)} 行")
    except Exception as e:
        print(f"✗ 调用 connect() 失败: {e}")

    print("\n结论:")
    print("⚠ README 第 43-44 行可能误导用户:")
    print("  ```python")
    print("  ds = DataStore.uri('/path/to/data.csv')")
    print("  ds.connect()  # <-- 这行可能不是必需的")
    print("  result = ds.select('*').filter(ds.age > 18).execute()")
    print("  ```")

    print()

def test_field_vs_column_access():
    """测试 ds.column_name 和 ds['column_name'] 的区别"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("API 一致性测试: Field 访问方式")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test.csv")
    with open(csv_file, "w") as f:
        f.write("id,name,value\n")
        f.write("1,Alice,100\n")

    try:
        ds = DataStore.from_file(csv_file)

        # 方式 1: ds.column_name (README 中常用)
        try:
            expr1 = ds.value
            print(f"✓ ds.value 工作: {type(expr1)}")
        except Exception as e:
            print(f"✗ ds.value 失败: {e}")

        # 方式 2: ds['column_name'] (README 中用于字符串函数)
        try:
            expr2 = ds['value']
            print(f"✓ ds['value'] 工作: {type(expr2)}")
        except Exception as e:
            print(f"✗ ds['value'] 失败: {e}")

        # 它们是同一个东西吗?
        if type(expr1) == type(expr2):
            print(f"✓ 两种方式返回相同类型")
        else:
            print(f"⚠ 两种方式返回不同类型!")

        # 生成的 SQL 相同吗?
        sql1 = ds.select(expr1).to_sql()
        sql2 = ds.select(expr2).to_sql()
        if sql1 == sql2:
            print(f"✓ 生成相同的 SQL")
        else:
            print(f"⚠ 生成不同的 SQL:")
            print(f"  ds.value: {sql1}")
            print(f"  ds['value']: {sql2}")

    finally:
        os.unlink(csv_file)
        os.rmdir(temp_dir)

    print()

def test_pandas_method_coverage():
    """测试 README 声称的 180+ pandas 方法"""
    from datastore import DataStore

    print("="*60)
    print("Pandas 兼容性验证: 180+ 方法是否存在?")
    print("="*60)

    ds = DataStore.from_numbers(10)

    # README 中明确提到的方法
    readme_methods = [
        # Properties
        'shape', 'columns', 'dtypes', 'values',
        # Statistical
        'mean', 'median', 'std', 'corr', 'describe',
        # Data manipulation
        'drop', 'rename', 'sort_values', 'fillna', 'dropna', 'drop_duplicates', 'assign',
        # Advanced
        'pivot_table', 'melt', 'merge', 'groupby',
        # Convenience
        'head', 'tail', 'sample',
        # Export
        'to_csv', 'to_json', 'to_parquet', 'to_excel'
    ]

    missing_methods = []
    for method in readme_methods:
        if hasattr(ds, method):
            print(f"✓ {method:20} 存在")
        else:
            print(f"✗ {method:20} 不存在")
            missing_methods.append(method)

    print(f"\n总结: {len(readme_methods) - len(missing_methods)}/{len(readme_methods)} 方法存在")

    if missing_methods:
        print(f"⚠ 缺失的方法: {', '.join(missing_methods)}")

    print()

def test_mixed_execution():
    """测试 README 中的混合执行示例"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("混合执行测试: SQL + Pandas 操作混合")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test.csv")
    with open(csv_file, "w") as f:
        f.write("id,price,quantity\n")
        f.write("1,150,2\n")
        f.write("2,80,5\n")
        f.write("3,200,1\n")

    try:
        ds = DataStore.from_file(csv_file)

        # README 第 251-258 行的示例:
        # result = (ds
        #     .select('*')
        #     .filter(ds.price > 100)
        #     .assign(revenue=lambda x: x['price'] * x['quantity'])
        #     .filter(ds.revenue > 1000)  # SQL on DataFrame!
        #     .add_prefix('sales_')
        #     .query('sales_revenue > 5000')
        #     .select('sales_id', 'sales_revenue'))

        print("测试简化版本:")
        try:
            # 步骤 1: SQL filter
            result1 = ds.select('*').filter(ds.price > 100)
            print(f"✓ .filter(ds.price > 100) 工作")

            # 步骤 2: Pandas assign
            result2 = result1.assign(revenue=lambda x: x['price'] * x['quantity'])
            print(f"✓ .assign() 工作")

            # 步骤 3: 再次 SQL filter?
            # README 说可以 .filter(ds.revenue > 1000)
            # 但这在 assign 之后还能工作吗?
            try:
                result3 = result2.filter(ds.revenue > 1000)
                print(f"✓ assign 后再 filter 工作")
            except Exception as e:
                print(f"✗ assign 后再 filter 失败: {e}")

        except Exception as e:
            print(f"✗ 混合执行失败: {e}")
            import traceback
            traceback.print_exc()

    finally:
        os.unlink(csv_file)
        os.rmdir(temp_dir)

    print()

def test_error_messages():
    """测试错误处理和错误消息的质量"""
    from datastore import DataStore

    print("="*60)
    print("错误处理测试: 错误消息是否有帮助?")
    print("="*60)

    # 测试 1: 不存在的列
    print("1. 访问不存在的列:")
    try:
        ds = DataStore.from_numbers(10)
        # number 是正确的列名
        result = ds.select('*').filter(ds.non_existent_column > 5).to_df()
        print("⚠ 没有报错 - 可能延迟到执行时")
    except Exception as e:
        print(f"  错误: {e}")
        if "non_existent_column" in str(e):
            print("  ✓ 错误消息提到了列名")
        else:
            print("  ⚠ 错误消息不够明确")

    # 测试 2: 无效的文件路径
    print("\n2. 无效的文件路径:")
    try:
        ds = DataStore.from_file("/nonexistent/path.csv")
        result = ds.select('*').to_df()
        print("⚠ 没有报错")
    except Exception as e:
        error_msg = str(e)
        print(f"  错误: {e}")
        if "/nonexistent/path.csv" in error_msg:
            print("  ✓ 错误消息提到了文件路径")
        else:
            print("  ⚠ 错误消息不够明确")

    # 测试 3: 语法错误 - 使用 Python 的 'and' 而不是 '&'
    print("\n3. 常见语法错误 - 使用 'and' 而不是 '&':")
    print("  (这会在 Python 层面失败,但值得测试)")
    try:
        ds = DataStore.from_numbers(10)
        # 这应该用 & 而不是 and
        # result = ds.filter((ds.number > 5) and (ds.number < 15))
        # 但这会在 Python 解释时失败,不是 DataStore 的错
        print("  (跳过 - 这是 Python 语法问题)")
    except Exception as e:
        print(f"  错误: {e}")

    print()

def test_to_sql_consistency():
    """测试 to_sql() 生成的 SQL 是否和实际执行的一致"""
    from datastore import DataStore

    print("="*60)
    print("SQL 一致性测试: to_sql() vs 实际执行")
    print("="*60)

    ds = DataStore.from_numbers(10)
    query = ds.select('*').filter(ds.number > 5).limit(3)

    sql = query.to_sql()
    print(f"to_sql() 生成: {sql}")

    # 这个测试需要检查实际执行的 SQL
    # 但我们没法直接看到执行的 SQL
    print("⚠ 注意: 无法直接验证执行的 SQL 是否和 to_sql() 一致")
    print("  建议: 添加调试模式或日志来显示实际执行的 SQL")

    print()

def test_immutability():
    """测试不可变性 - README 说 'Immutable Operations'"""
    from datastore import DataStore

    print("="*60)
    print("不可变性测试: 操作是否真的不可变?")
    print("="*60)

    ds1 = DataStore.from_numbers(100)
    sql1_before = ds1.to_sql()

    # 执行一些操作
    ds2 = ds1.filter(ds1.number > 50)
    ds3 = ds1.limit(10)

    # ds1 应该保持不变
    sql1_after = ds1.to_sql()

    print(f"ds1 before: {sql1_before}")
    print(f"ds2 (filtered): {ds2.to_sql()}")
    print(f"ds3 (limited): {ds3.to_sql()}")
    print(f"ds1 after: {sql1_after}")

    if sql1_before == sql1_after:
        print("✓ 不可变性保持 - ds1 没有被修改")
    else:
        print("✗ BUG: ds1 被修改了!")

    print()

if __name__ == "__main__":
    tests = [
        test_column_assignment_bug,
        test_uri_vs_factory_methods,
        test_connect_confusion,
        test_field_vs_column_access,
        test_pandas_method_coverage,
        test_mixed_execution,
        test_error_messages,
        test_to_sql_consistency,
        test_immutability,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗✗✗ 测试失败: {test_func.__name__}")
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("="*60)
    print("深度测试完成 - 请查看上面的问题和建议")
    print("="*60)
