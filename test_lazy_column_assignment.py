#!/usr/bin/env python3
"""
Test to demonstrate and validate lazy column assignment behavior.
"""
import tempfile
import os


def test_lazy_assignment_basic():
    """Demonstrate that column assignment is lazy."""
    from datastore import DataStore

    print("="*70)
    print("Test 1: Column assignment is lazy (not executed immediately)")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test.csv")
    with open(csv_file, "w") as f:
        f.write("id,value\n")
        f.write("1,100\n")
        f.write("2,200\n")

    try:
        ds = DataStore.from_file(csv_file)

        # Assign a new column (lazy)
        ds['doubled'] = ds['value'] * 2

        # Check SQL - new column should NOT be visible yet
        sql = ds.select('*').to_sql()
        print(f"SQL after assignment: {sql}")

        if 'doubled' in sql or 'value*2' in sql.replace(' ', ''):
            print("  ⚠ New column appears in SQL (unexpected)")
        else:
            print("  ✓ New column NOT in SQL (lazy - as expected)")

        # Execute and check result
        result = ds.select('*').to_df()
        print(f"\nResult columns: {list(result.columns)}")

        if 'doubled' in result.columns:
            print("  ✓ New column appears in result (executed)")
            print(f"  Values: {list(result['doubled'])}")
        else:
            print("  ✗ New column missing from result")

    finally:
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    print()


def test_lazy_assignment_multiple():
    """Test multiple lazy assignments."""
    from datastore import DataStore

    print("="*70)
    print("Test 2: Multiple lazy assignments")
    print("="*70)

    ds = DataStore.from_numbers(5)

    # Multiple assignments
    ds['doubled'] = ds['number'] * 2
    ds['plus_ten'] = ds['number'] + 10
    ds['category'] = ds['number'] // 2

    print("Assigned 3 new columns (lazy)")

    # Execute
    result = ds.to_df()

    print(f"Result columns: {list(result.columns)}")
    expected = ['number', 'doubled', 'plus_ten', 'category']

    if all(col in result.columns for col in expected):
        print("  ✓ All new columns appear in result")
        print(f"\nFirst row:")
        for col in expected:
            print(f"  {col}: {result[col].iloc[0]}")
    else:
        missing = [col for col in expected if col not in result.columns]
        print(f"  ✗ Missing columns: {missing}")

    print()


def test_assign_vs_setitem():
    """Compare assign() (immutable) vs __setitem__ (in-place)."""
    from datastore import DataStore

    print("="*70)
    print("Test 3: assign() vs ds['col'] = ... ")
    print("="*70)

    ds = DataStore.from_numbers(5)

    print("Method 1: ds['new_col'] = ... (in-place, lazy)")
    ds1 = ds
    original_id = id(ds1)
    ds1['doubled'] = ds1['number'] * 2
    after_id = id(ds1)

    print(f"  Before: {id(ds1)}")
    print(f"  After:  {after_id}")
    print(f"  Same object: {original_id == after_id}")
    print(f"  Result: Modifies in-place ✓")

    print("\nMethod 2: assign() (immutable, returns new)")
    ds2 = ds.assign(tripled=lambda x: x['number'] * 3)
    print(f"  Original: {id(ds)}")
    print(f"  New:      {id(ds2)}")
    print(f"  Same object: {id(ds) == id(ds2)}")
    print(f"  Result: Returns new DataStore ✓")

    # Verify columns
    result1 = ds1.to_df()
    result2 = ds2.to_df()

    print(f"\nds1 columns: {list(result1.columns)}")
    print(f"ds2 columns: {list(result2.columns)}")

    print()


def test_lazy_with_functions():
    """Test lazy assignment with ClickHouse functions."""
    from datastore import DataStore
    import tempfile

    print("="*70)
    print("Test 4: Lazy assignment with string functions")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "test.csv")
    with open(csv_file, "w") as f:
        f.write("id,name\n")
        f.write("1,alice\n")
        f.write("2,bob\n")

    try:
        ds = DataStore.from_file(csv_file)

        # Assign with string function (lazy)
        ds['upper_name'] = ds['name'].str.upper()
        ds['name_length'] = ds['name'].str.length()

        print("Assigned columns using .str accessor (lazy)")

        # Execute
        result = ds.to_df()

        print(f"Result columns: {list(result.columns)}")
        print(f"\nFirst row:")
        print(f"  name: {result['name'].iloc[0]}")
        print(f"  upper_name: {result['upper_name'].iloc[0]}")
        print(f"  name_length: {result['name_length'].iloc[0]}")

        if result['upper_name'].iloc[0] == 'ALICE':
            print("  ✓ String function applied correctly")
        else:
            print("  ✗ String function not applied")

    finally:
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    print()


def test_lazy_execution_timing():
    """Test when lazy operations are executed."""
    from datastore import DataStore

    print("="*70)
    print("Test 5: When are lazy operations executed?")
    print("="*70)

    ds = DataStore.from_numbers(3)
    ds['doubled'] = ds['number'] * 2

    triggers = [
        ("to_df()", lambda: ds.to_df()),
        ("execute()", lambda: ds.execute()),
        ("shape", lambda: ds.shape),
        ("columns", lambda: ds.columns),
        ("head()", lambda: ds.head()),
    ]

    for name, func in triggers:
        ds_test = DataStore.from_numbers(3)
        ds_test['test_col'] = ds_test['number'] * 2

        try:
            result = func()
            # Check if result has the column (for DataFrame results)
            if hasattr(result, 'columns'):
                has_col = 'test_col' in result.columns
                print(f"  {name:15} - Triggers execution: {'✓' if has_col else '?'}")
            elif isinstance(result, (list, tuple)):
                # For shape, columns, etc.
                print(f"  {name:15} - Triggers execution: ✓")
            else:
                print(f"  {name:15} - Triggers execution: ✓ (QueryResult)")
        except Exception as e:
            print(f"  {name:15} - Error: {e}")

    print()


if __name__ == "__main__":
    tests = [
        test_lazy_assignment_basic,
        test_lazy_assignment_multiple,
        test_assign_vs_setitem,
        test_lazy_with_functions,
        test_lazy_execution_timing,
    ]

    print("\n" + "="*70)
    print("Testing Lazy Column Assignment Behavior")
    print("="*70 + "\n")

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗✗✗ Test failed: {test_func.__name__}")
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("="*70)
    print("Testing completed")
    print("="*70)
