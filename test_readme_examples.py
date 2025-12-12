#!/usr/bin/env python3
"""
严格测试 README 中的所有示例，找出 bug 和不一致的地方
"""
import tempfile
import os
import sys

def test_basic_operations():
    """测试基本操作"""
    from datastore import DataStore

    print("="*60)
    print("测试 1: 基本数字生成和查询")
    print("="*60)

    # 测试 from_numbers
    ds = DataStore.from_numbers(10)
    result = ds.select('*').to_df()
    print(f"✓ from_numbers(10) 返回 {len(result)} 行")
    assert len(result) == 10, f"Expected 10 rows, got {len(result)}"

    # 测试过滤和限制
    ds = DataStore.from_numbers(100)
    result = ds.select('*').filter(ds.number > 50).limit(5).to_df()
    print(f"✓ filter + limit 返回 {len(result)} 行")
    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
    print()

def test_uri_creation():
    """测试 URI 创建方式"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("测试 2: URI-based 创建 (README 推荐方式)")
    print("="*60)

    # 创建测试文件
    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test.csv")
    with open(csv_file, "w") as f:
        f.write("id,name,age\n")
        f.write("1,Alice,25\n")
        f.write("2,Bob,30\n")

    try:
        # README 说可以直接用 URI 字符串
        # 测试 1: 用路径字符串
        ds = DataStore.uri(csv_file)
        ds.connect()
        result = ds.select("*").to_df()
        print(f"✓ DataStore.uri('{csv_file}') 成功")
        print(f"  返回 {len(result)} 行, 列: {list(result.columns)}")

        # 测试 2: 用 file:// 协议
        file_uri = f"file://{csv_file}"
        ds2 = DataStore.uri(file_uri)
        ds2.connect()
        result2 = ds2.select("*").to_df()
        print(f"✓ DataStore.uri('{file_uri}') 成功")
        print()
    except Exception as e:
        print(f"✗ URI 创建失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def test_filter_operations():
    """测试过滤操作"""
    from datastore import DataStore

    print("="*60)
    print("测试 3: 过滤条件和表达式")
    print("="*60)

    ds = DataStore.from_numbers(100)

    # README 中的示例: ds.filter(ds.age > 18)
    result1 = ds.filter(ds.number > 50).to_df()
    print(f"✓ ds.filter(ds.number > 50) 返回 {len(result1)} 行")

    # README 中的示例: 复合条件
    result2 = ds.filter((ds.number > 10) & (ds.number < 20)).to_df()
    print(f"✓ 复合条件 (ds.number > 10) & (ds.number < 20) 返回 {len(result2)} 行")

    # README 中的示例: where() 是 filter() 的别名
    result3 = ds.where(ds.number > 50).to_df()
    print(f"✓ where() 别名工作正常, 返回 {len(result3)} 行")
    assert len(result1) == len(result3), "where() 应该和 filter() 返回相同结果"
    print()

def test_pandas_compatibility():
    """测试 Pandas 兼容性"""
    from datastore import DataStore

    print("="*60)
    print("测试 4: Pandas API 兼容性 (README 声称 180+ 方法)")
    print("="*60)

    ds = DataStore.from_numbers(20)

    # 测试 README 中提到的属性
    try:
        shape = ds.shape
        print(f"✓ ds.shape = {shape}")
    except Exception as e:
        print(f"✗ ds.shape 失败: {e}")

    try:
        columns = ds.columns
        print(f"✓ ds.columns = {list(columns)}")
    except Exception as e:
        print(f"✗ ds.columns 失败: {e}")

    try:
        dtypes = ds.dtypes
        print(f"✓ ds.dtypes = {dtypes}")
    except Exception as e:
        print(f"✗ ds.dtypes 失败: {e}")

    # 测试统计方法
    try:
        mean_val = ds.mean()
        print(f"✓ ds.mean() = {mean_val}")
    except Exception as e:
        print(f"✗ ds.mean() 失败: {e}")

    # 测试 head/tail
    try:
        head_result = ds.head()
        print(f"✓ ds.head() 返回 {len(head_result)} 行")
    except Exception as e:
        print(f"✗ ds.head() 失败: {e}")

    try:
        tail_result = ds.tail()
        print(f"✓ ds.tail() 返回 {len(tail_result)} 行")
    except Exception as e:
        print(f"✗ ds.tail() 失败: {e}")

    print()

def test_string_functions():
    """测试字符串函数 (README 中的 .str accessor)"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("测试 5: ClickHouse SQL 函数 - 字符串操作")
    print("="*60)

    # 创建带字符串的测试数据
    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test_strings.csv")
    with open(csv_file, "w") as f:
        f.write("id,name,email\n")
        f.write("1,alice,alice@example.com\n")
        f.write("2,bob,bob@test.com\n")

    try:
        ds = DataStore.from_file(csv_file)

        # README 中的示例: ds['name'].str.upper()
        try:
            # 测试列选择
            name_col = ds['name']
            print(f"✓ ds['name'] 返回: {type(name_col)}")

            # 测试 .str accessor
            upper_expr = ds['name'].str.upper()
            print(f"✓ ds['name'].str.upper() 创建表达式: {type(upper_expr)}")

            # 生成 SQL
            sql = ds.select(upper_expr.as_('upper_name')).to_sql()
            print(f"✓ 生成的 SQL: {sql}")

        except Exception as e:
            print(f"✗ 字符串函数失败: {e}")
            import traceback
            traceback.print_exc()

        # 测试其他字符串函数
        try:
            length_expr = ds['name'].str.length()
            print(f"✓ ds['name'].str.length() 创建成功")
        except Exception as e:
            print(f"✗ str.length() 失败: {e}")

        print()
    finally:
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def test_arithmetic_operations():
    """测试算术运算"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("测试 6: 算术运算和表达式")
    print("="*60)

    # 创建测试数据
    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test_numbers.csv")
    with open(csv_file, "w") as f:
        f.write("id,price,quantity\n")
        f.write("1,10.5,5\n")
        f.write("2,20.0,3\n")

    try:
        ds = DataStore.from_file(csv_file)

        # README 示例: ds.price * 1.1 (10% price increase)
        try:
            price_increase = ds['price'] * 1.1
            sql = ds.select(price_increase.as_('new_price')).to_sql()
            print(f"✓ 算术运算 ds['price'] * 1.1 生成 SQL:")
            print(f"  {sql}")
        except Exception as e:
            print(f"✗ 算术运算失败: {e}")
            import traceback
            traceback.print_exc()

        # README 示例: (ds.revenue - ds.cost).as_("profit")
        # 我们用 price * quantity 来测试
        try:
            revenue = ds['price'] * ds['quantity']
            sql = ds.select(revenue.as_('revenue')).to_sql()
            print(f"✓ 复杂表达式 ds['price'] * ds['quantity'] 生成 SQL:")
            print(f"  {sql}")
        except Exception as e:
            print(f"✗ 复杂表达式失败: {e}")

        print()
    finally:
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def test_column_assignment():
    """测试列赋值 (README 示例)"""
    from datastore import DataStore
    import tempfile

    print("="*60)
    print("测试 7: 列赋值 (README 中的示例)")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test_assign.csv")
    with open(csv_file, "w") as f:
        f.write("id,name,age\n")
        f.write("1,Alice,25\n")
        f.write("2,Bob,35\n")

    try:
        ds = DataStore.from_file(csv_file)

        # README 示例: ds['upper_name'] = ds['name'].str.upper()
        try:
            ds['upper_name'] = ds['name'].str.upper()
            print(f"✓ 列赋值 ds['upper_name'] = ds['name'].str.upper() 成功")
            sql = ds.select('*').to_sql()
            print(f"  生成的 SQL: {sql}")
        except Exception as e:
            print(f"✗ 列赋值失败: {e}")
            import traceback
            traceback.print_exc()

        # README 示例: ds['age_group'] = ds['age'] // 10 * 10
        try:
            ds2 = DataStore.from_file(csv_file)
            ds2['age_group'] = ds2['age'] // 10 * 10
            print(f"✓ 算术列赋值 ds['age_group'] = ds['age'] // 10 * 10 成功")
            sql = ds2.select('*').to_sql()
            print(f"  生成的 SQL: {sql}")
        except Exception as e:
            print(f"✗ 算术列赋值失败: {e}")

        print()
    finally:
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def test_exec_alias():
    """测试 exec() 是否是 execute() 的别名"""
    from datastore import DataStore

    print("="*60)
    print("测试 8: exec() 别名 (README 说 exec() 和 execute() 相同)")
    print("="*60)

    ds = DataStore.from_numbers(10)

    try:
        result1 = ds.select('*').execute()
        print(f"✓ execute() 返回: {type(result1)}")
    except Exception as e:
        print(f"✗ execute() 失败: {e}")

    try:
        result2 = ds.select('*').exec()
        print(f"✓ exec() 返回: {type(result2)}")
    except Exception as e:
        print(f"✗ exec() 失败: {e}")

    print()

def test_to_dict_to_df():
    """测试简化的 to_df() 和 to_dict() 方法"""
    from datastore import DataStore

    print("="*60)
    print("测试 9: 简化的结果获取 (README 中的 to_df() 和 to_dict())")
    print("="*60)

    ds = DataStore.from_numbers(5)

    # README 说: df = ds.select("*").to_df()
    try:
        df = ds.select("*").to_df()
        print(f"✓ to_df() 返回: {type(df)}, shape: {df.shape}")
    except Exception as e:
        print(f"✗ to_df() 失败: {e}")

    # README 说: records = ds.select("*").to_dict()
    try:
        records = ds.select("*").to_dict()
        print(f"✓ to_dict() 返回: {type(records)}, length: {len(records)}")
    except Exception as e:
        print(f"✗ to_dict() 失败: {e}")

    print()

def test_readme_inconsistencies():
    """查找 README 中的不一致之处"""
    from datastore import DataStore

    print("="*60)
    print("测试 10: README 不一致性检查")
    print("="*60)

    # README 第 43 行说: ds = DataStore.uri("/path/to/data.csv")
    # README 第 44 行说: ds.connect()
    # 但是其他地方没有提到需要 connect()

    ds = DataStore.from_numbers(10)

    # 检查是否需要 connect()
    try:
        # 不调用 connect() 直接执行
        result = ds.select('*').to_df()
        print(f"✓ 不需要 connect() 就可以执行查询")
    except Exception as e:
        print(f"? 可能需要 connect(): {e}")

    # 检查 connect() 是否存在
    if hasattr(ds, 'connect'):
        print(f"✓ DataStore 有 connect() 方法")
    else:
        print(f"✗ DataStore 没有 connect() 方法 (但 README 第 44 行提到了)")

    print()

if __name__ == "__main__":
    tests = [
        test_basic_operations,
        test_uri_creation,
        test_filter_operations,
        test_pandas_compatibility,
        test_string_functions,
        test_arithmetic_operations,
        test_column_assignment,
        test_exec_alias,
        test_to_dict_to_df,
        test_readme_inconsistencies,
    ]

    failed_tests = []

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗✗✗ 测试失败: {test_func.__name__}")
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()
            failed_tests.append(test_func.__name__)
            print()

    print("="*60)
    print("测试总结")
    print("="*60)
    print(f"总共运行: {len(tests)} 个测试")
    print(f"失败: {len(failed_tests)} 个")
    if failed_tests:
        print(f"失败的测试: {', '.join(failed_tests)}")
    else:
        print("✓ 所有测试通过!")
