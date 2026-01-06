#!/usr/bin/env python3
"""
Script to globally replace pd.testing.assert_* calls with test_utils wrappers.

This ensures check_names=True is the default for all comparison functions.

Changes made:
1. Replace `pd.testing.assert_frame_equal` with `assert_frame_equal` from test_utils
2. Replace `pd.testing.assert_series_equal` with `assert_series_equal` from test_utils  
3. Update imports from `pandas.testing` to use test_utils wrappers
4. Remove `check_names=False` since we want to verify names are correct (default is now True)
"""

import os
import re
from pathlib import Path


def process_file(filepath: Path, remove_check_names_false: bool = True) -> tuple[bool, list[str]]:
    """
    Process a single test file and make replacements.
    
    Args:
        filepath: Path to the test file
        remove_check_names_false: If True, remove check_names=False from assert calls
    
    Returns:
        (modified, changes): Whether file was modified and list of changes made
    """
    changes = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Skip test_utils.py itself
    if filepath.name == 'test_utils.py':
        return False, []
    
    # Track what imports we need to add
    need_assert_frame_equal = False
    need_assert_series_equal = False
    
    # 1. Check if file uses pd.testing.assert_frame_equal
    if 'pd.testing.assert_frame_equal' in content:
        need_assert_frame_equal = True
        content = content.replace('pd.testing.assert_frame_equal', 'assert_frame_equal')
        changes.append('Replaced pd.testing.assert_frame_equal -> assert_frame_equal')
    
    # 2. Check if file uses pd.testing.assert_series_equal
    if 'pd.testing.assert_series_equal' in content:
        need_assert_series_equal = True
        content = content.replace('pd.testing.assert_series_equal', 'assert_series_equal')
        changes.append('Replaced pd.testing.assert_series_equal -> assert_series_equal')
    
    # 3. Handle imports from pandas.testing
    # Pattern: from pandas.testing import assert_frame_equal, assert_series_equal
    pandas_testing_import_pattern = r'from pandas\.testing import ([^\n]+)'
    match = re.search(pandas_testing_import_pattern, content)
    
    if match:
        imported_items = match.group(1)
        items = [item.strip() for item in imported_items.split(',')]
        
        # Check what's being imported
        has_frame = 'assert_frame_equal' in items
        has_series = 'assert_series_equal' in items
        
        if has_frame:
            need_assert_frame_equal = True
        if has_series:
            need_assert_series_equal = True
        
        # Remove the pandas.testing import line entirely
        content = re.sub(r'from pandas\.testing import [^\n]+\n', '', content)
        changes.append(f'Removed pandas.testing import: {imported_items}')
    
    # 4. Update or add test_utils import
    if need_assert_frame_equal or need_assert_series_equal:
        imports_to_add = []
        if need_assert_frame_equal:
            imports_to_add.append('assert_frame_equal')
        if need_assert_series_equal:
            imports_to_add.append('assert_series_equal')
        
        # Check if test_utils import already exists
        test_utils_import_pattern = r'from tests\.test_utils import ([^\n]+)'
        test_utils_match = re.search(test_utils_import_pattern, content)
        
        if test_utils_match:
            # Existing import - add our functions if not already there
            existing_imports = test_utils_match.group(1)
            existing_items = [item.strip() for item in existing_imports.split(',')]
            
            for imp in imports_to_add:
                if imp not in existing_items:
                    existing_items.append(imp)
                    changes.append(f'Added {imp} to test_utils import')
            
            # Sort and format
            existing_items.sort()
            new_import_line = f'from tests.test_utils import {", ".join(existing_items)}'
            content = re.sub(test_utils_import_pattern, new_import_line, content)
        else:
            # No existing test_utils import - need to add one
            # Find a good place to insert (after other imports)
            import_section_end = 0
            for match in re.finditer(r'^(import |from )[^\n]+\n', content, re.MULTILINE):
                import_section_end = match.end()
            
            if import_section_end > 0:
                imports_str = ', '.join(sorted(imports_to_add))
                new_import = f'from tests.test_utils import {imports_str}\n'
                content = content[:import_section_end] + new_import + content[import_section_end:]
                changes.append(f'Added test_utils import: {imports_str}')
    
    # 5. Remove check_names=False (since default is now True)
    if remove_check_names_false:
        # Pattern to match check_names=False in function calls (with optional surrounding comma/space)
        # Handle: ", check_names=False)" or "check_names=False, " or just ", check_names=False"
        patterns = [
            (r',\s*check_names=False\s*\)', ')'),      # ", check_names=False)" -> ")"
            (r'check_names=False\s*,\s*', ''),         # "check_names=False, " -> ""
            (r',\s*check_names=False', ''),            # ", check_names=False" -> ""
        ]
        
        check_names_count_before = content.count('check_names=False')
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        check_names_count_after = content.count('check_names=False')
        
        removed_count = check_names_count_before - check_names_count_after
        if removed_count > 0:
            changes.append(f'Removed {removed_count} occurrences of check_names=False')
        if check_names_count_after > 0:
            changes.append(f'WARNING: {check_names_count_after} occurrences of check_names=False remain (complex pattern)')
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    
    return False, changes


def main():
    """Main function to process all test files."""
    tests_dir = Path(__file__).parent.parent / 'tests'
    
    if not tests_dir.exists():
        print(f"Error: tests directory not found at {tests_dir}")
        return
    
    print(f"Processing test files in: {tests_dir}")
    print("=" * 60)
    
    modified_files = []
    total_changes = 0
    
    for filepath in sorted(tests_dir.glob('test_*.py')):
        modified, changes = process_file(filepath, remove_check_names_false=True)
        
        if changes:
            total_changes += len(changes)
            if modified:
                modified_files.append(filepath.name)
            print(f"\n{filepath.name}:")
            for change in changes:
                prefix = "  ✓" if not change.startswith('WARNING') else "  ⚠"
                print(f"{prefix} {change}")
    
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Files modified: {len(modified_files)}")
    print(f"  Total changes: {total_changes}")
    
    if modified_files:
        print(f"\nModified files:")
        for f in modified_files:
            print(f"  - {f}")


if __name__ == '__main__':
    main()

