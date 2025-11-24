#!/usr/bin/env python3
"""
Demo script showcasing DataStore display improvements.

This script demonstrates:
1. head(), tail(), sample(), describe() now return DataStore instead of DataFrame
2. DataStore objects display like DataFrame in console and Jupyter
3. Method chaining works seamlessly with the new return types
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datastore import DataStore


def demo_return_types():
    """Demonstrate that methods now return DataStore."""
    print("\n" + "=" * 80)
    print("DEMO 1: Methods Return DataStore")
    print("=" * 80 + "\n")
    
    ds = DataStore.from_file("tests/dataset/users.csv")
    ds.connect()
    
    print("1. head() returns DataStore:")
    head_result = ds.head(3)
    print(f"   Type: {type(head_result).__name__}")
    print(f"   Length: {len(head_result)}")
    print()
    
    print("2. tail() returns DataStore:")
    tail_result = ds.tail(3)
    print(f"   Type: {type(tail_result).__name__}")
    print(f"   Length: {len(tail_result)}")
    print()
    
    print("3. sample() returns DataStore:")
    sample_result = ds.sample(n=2, random_state=42)
    print(f"   Type: {type(sample_result).__name__}")
    print(f"   Length: {len(sample_result)}")
    print()
    
    print("4. describe() returns DataStore:")
    describe_result = ds.describe()
    print(f"   Type: {type(describe_result).__name__}")
    print()


def demo_display():
    """Demonstrate that DataStore displays like DataFrame."""
    print("\n" + "=" * 80)
    print("DEMO 2: DataStore Display (like DataFrame)")
    print("=" * 80 + "\n")
    
    ds = DataStore.from_file("tests/dataset/users.csv")
    ds.connect()
    
    print("1. Displaying head() result using print():")
    print("-" * 80)
    head_result = ds.head(3)
    print(head_result)
    print()
    
    print("2. Displaying tail() result using str():")
    print("-" * 80)
    tail_result = ds.tail(2)
    print(str(tail_result))
    print()
    
    print("3. Displaying describe() result:")
    print("-" * 80)
    stats = ds.describe()
    print(stats)
    print()


def demo_chaining():
    """Demonstrate method chaining with new return types."""
    print("\n" + "=" * 80)
    print("DEMO 3: Method Chaining")
    print("=" * 80 + "\n")
    
    ds = DataStore.from_file("tests/dataset/users.csv")
    ds.connect()
    
    print("1. Chain: head() -> head() (materialized chaining)")
    print("-" * 80)
    result = ds.head(10).head(3)
    print(f"   Type: {type(result).__name__}")
    print(f"   Length: {len(result)}")
    print(result)
    print()
    
    print("2. Chain: tail() -> head()")
    print("-" * 80)
    result = ds.tail(8).head(3)
    print(f"   Type: {type(result).__name__}")
    print(f"   Length: {len(result)}")
    print(result)
    print()
    
    print("3. Chain: head() -> describe()")
    print("-" * 80)
    result = ds.head(5).describe()
    print(f"   Type: {type(result).__name__}")
    print(result)
    print()


def demo_html_display():
    """Demonstrate HTML display capability (for Jupyter)."""
    print("\n" + "=" * 80)
    print("DEMO 4: HTML Display (for Jupyter/IPython)")
    print("=" * 80 + "\n")
    
    ds = DataStore.from_file("tests/dataset/users.csv")
    ds.connect()
    
    head_result = ds.head(3)
    
    print("DataStore has _repr_html_() method for Jupyter display:")
    print(f"   Has _repr_html_: {hasattr(head_result, '_repr_html_')}")
    
    if hasattr(head_result, '_repr_html_'):
        html = head_result._repr_html_()
        print(f"   HTML length: {len(html)} characters")
        print(f"   Contains table tags: {'<table' in html.lower()}")
        print()
        print("   Sample HTML output (first 300 chars):")
        print("-" * 80)
        print(html[:300] + "...")
    print()


def demo_comparison():
    """Compare old vs new behavior."""
    print("\n" + "=" * 80)
    print("DEMO 5: Comparison - Old vs New Behavior")
    print("=" * 80 + "\n")
    
    ds = DataStore.from_file("tests/dataset/users.csv")
    ds.connect()
    
    print("OLD BEHAVIOR (would have been):")
    print("-" * 80)
    print("  head() returned DataFrame - chain would break")
    print("  Result couldn't be further filtered/manipulated with DataStore methods")
    print()
    
    print("NEW BEHAVIOR:")
    print("-" * 80)
    print("  head() returns DataStore - chaining works seamlessly")
    print("  Result can be filtered, sorted, or have any DataStore method applied")
    print()
    
    print("Example - Complex chain (head -> tail -> sample):")
    print("-" * 80)
    result = (
        ds.head(10)                  # Returns DataStore!
        .tail(5)                     # Can continue chaining!
        .sample(n=2, random_state=42) # And chain again!
    )
    print(f"   Final result type: {type(result).__name__}")
    print(f"   Final result length: {len(result)}")
    print(result)
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("DataStore Display Improvements - Demo")
    print("=" * 80)
    
    demo_return_types()
    demo_display()
    demo_chaining()
    demo_html_display()
    demo_comparison()
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print("\nKey Improvements:")
    print("  1. head(), tail(), sample(), describe() now return DataStore")
    print("  2. DataStore displays data content (like DataFrame) in console/Jupyter")
    print("  3. __str__, __repr__, and _repr_html_ methods work correctly")
    print("  4. Method chaining works seamlessly")
    print("  5. len() works on DataStore objects")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

