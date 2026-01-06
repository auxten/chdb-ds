#!/usr/bin/env python3
"""
Script to identify and fix suspicious test patterns that may be masking bugs.

Patterns to fix:
1. .reset_index(drop=True) - may be hiding index preservation bugs
2. get_series() used for DataFrame results (dropna, fillna, etc.)

Usage:
    python scripts/fix_suspicious_tests.py --analyze  # Just analyze, don't modify
    python scripts/fix_suspicious_tests.py --fix      # Apply fixes
    python scripts/fix_suspicious_tests.py --revert   # Revert fixes (restore backups)
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

TESTS_DIR = Path(__file__).parent.parent / "tests"
BACKUP_SUFFIX = ".bak_reset_index"


# Patterns that are suspicious
SUSPICIOUS_PATTERNS = [
    # Pattern 1: reset_index(drop=True) used to mask index issues
    # But we need to be careful - some uses are legitimate (e.g., groupby with as_index=False)
    (
        r'\.reset_index\(drop=True\)',
        'reset_index(drop=True)',
        'May be masking index preservation bugs'
    ),
    # Pattern 2: get_series used for DataFrame-returning methods
    (
        r'get_series\([^)]*(?:dropna|fillna|drop_duplicates)[^)]*\)',
        'get_series() on DataFrame',
        'dropna/fillna/drop_duplicates return DataFrame, not Series'
    ),
]

# Files to analyze
TARGET_FILES = [
    "test_exploratory_batch25_boolean_indexing.py",
    "test_exploratory_batch40_eval_query_coerce.py",
    "test_pandas_compatibility.py",
    "test_exploratory_batch22_advanced_ops.py",
    "test_exploratory_discovery_2026_01_04.py",
    "test_llm_pandas_compat.py",
    "test_titanic_pandas_comparison.py",
    "test_sql_vs_pandas_filter.py",
]


def analyze_file(filepath: Path) -> List[Dict]:
    """Analyze a file for suspicious patterns."""
    issues = []
    
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # Check for reset_index(drop=True)
        if '.reset_index(drop=True)' in line:
            # Try to understand context
            context = get_context(lines, i - 1, 3)
            
            # Skip if it's in a comment
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
                
            # Check if both sides use reset_index (might be legitimate comparison)
            reset_count = line.count('.reset_index(drop=True)')
            
            issues.append({
                'file': filepath.name,
                'line': i,
                'code': line.strip(),
                'pattern': 'reset_index(drop=True)',
                'context': context,
                'reset_count': reset_count,
                'severity': 'warning' if reset_count == 2 else 'suspicious'
            })
        
        # Check for get_series on DataFrame methods
        if 'get_series(' in line:
            # Check if it's used with DataFrame-returning methods nearby
            context_str = '\n'.join(get_context(lines, i - 1, 5))
            if any(m in context_str for m in ['dropna', 'fillna', 'drop_duplicates']):
                issues.append({
                    'file': filepath.name,
                    'line': i,
                    'code': line.strip(),
                    'pattern': 'get_series() on DataFrame result',
                    'context': get_context(lines, i - 1, 3),
                    'severity': 'error'
                })
    
    return issues


def get_context(lines: List[str], center: int, radius: int) -> List[str]:
    """Get context lines around a center line."""
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    return lines[start:end]


def create_backup(filepath: Path):
    """Create a backup of the file."""
    backup_path = filepath.with_suffix(filepath.suffix + BACKUP_SUFFIX)
    shutil.copy(filepath, backup_path)
    return backup_path


def restore_backup(filepath: Path):
    """Restore file from backup."""
    backup_path = filepath.with_suffix(filepath.suffix + BACKUP_SUFFIX)
    if backup_path.exists():
        shutil.copy(backup_path, filepath)
        backup_path.unlink()
        return True
    return False


def fix_reset_index_patterns(filepath: Path) -> Tuple[int, List[str]]:
    """
    Fix reset_index(drop=True) patterns in a file.
    
    Strategy:
    - If both sides have reset_index(drop=True), remove both
    - If only one side has it, remove it
    
    Returns: (count of fixes, list of fixed lines)
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    fixes = []
    
    # Pattern: assert_xxx(a.reset_index(drop=True), b.reset_index(drop=True))
    # Remove both reset_index calls
    pattern1 = r'(assert_\w+\([^)]+)\.reset_index\(drop=True\)([^)]+)\.reset_index\(drop=True\)'
    
    def replace_both(match):
        fixes.append(f"Removed dual reset_index in: {match.group(0)[:80]}...")
        return match.group(1) + match.group(2)
    
    content = re.sub(pattern1, replace_both, content)
    
    # Pattern: single reset_index in comparison (more risky, skip for now)
    # We'll only remove when both sides use it
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
    
    return len(fixes), fixes


def fix_get_series_on_dataframe(filepath: Path) -> Tuple[int, List[str]]:
    """
    Fix get_series() used on DataFrame-returning methods.
    
    Only replace get_series(result) with get_dataframe(result) when:
    - The result is from dropna() or fillna() on a DataFrame (returns DataFrame)
    - NOT for drop_duplicates() on a Series (returns Series)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixes = []
    modified = False
    
    for i, line in enumerate(lines):
        # Look for patterns like: ds_result = get_series(ds_df.dropna(...))
        # or: ds_result = get_series(ds_result) where ds_result was from dropna/fillna
        
        # Simple case: get_series immediately after dropna/fillna assignment
        if 'get_series(' in line:
            # Check previous lines for dropna/fillna (DataFrame methods that return DataFrame)
            context_start = max(0, i - 5)
            context = ''.join(lines[context_start:i+1])
            
            # ONLY fix for DataFrame methods that return DataFrame
            # dropna(subset=...) and fillna({...}) on DataFrame return DataFrame
            # drop_duplicates on Series returns Series - DON'T fix those
            is_dataframe_method = (
                '.dropna(' in context and 'subset=' in context  # DataFrame.dropna(subset=...)
            ) or (
                '.fillna({' in context  # DataFrame.fillna({col: value})
            )
            
            # Skip if it looks like a Series operation
            is_series_method = (
                "['A']" in context or "['B']" in context or  # Column selection suggests Series
                '.drop_duplicates(' in context  # Series.drop_duplicates
            )
            
            if is_dataframe_method and not is_series_method:
                if 'get_series(ds_result)' in line or 'get_series(ds_df' in line:
                    new_line = line.replace('get_series(', 'get_dataframe(')
                    if new_line != line:
                        lines[i] = new_line
                        fixes.append(f"Line {i+1}: Changed get_series to get_dataframe")
                        modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            f.writelines(lines)
    
    return len(fixes), fixes


def print_analysis(all_issues: Dict[str, List[Dict]]):
    """Print analysis results."""
    print("\n" + "=" * 80)
    print("SUSPICIOUS TEST PATTERNS ANALYSIS")
    print("=" * 80)
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    print(f"\nFound {total_issues} suspicious patterns in {len(all_issues)} files\n")
    
    # Group by severity
    errors = []
    warnings = []
    suspicious = []
    
    for filepath, issues in all_issues.items():
        for issue in issues:
            if issue['severity'] == 'error':
                errors.append(issue)
            elif issue['severity'] == 'warning':
                warnings.append(issue)
            else:
                suspicious.append(issue)
    
    if errors:
        print("\n🚨 ERRORS (likely bugs in tests):")
        print("-" * 40)
        for issue in errors:
            print(f"  {issue['file']}:{issue['line']}")
            print(f"    Pattern: {issue['pattern']}")
            print(f"    Code: {issue['code'][:70]}...")
            print()
    
    if suspicious:
        print(f"\n⚠️  SUSPICIOUS ({len(suspicious)} instances):")
        print("-" * 40)
        # Group by file
        by_file = {}
        for issue in suspicious:
            by_file.setdefault(issue['file'], []).append(issue)
        
        for filename, file_issues in by_file.items():
            print(f"\n  {filename} ({len(file_issues)} issues):")
            for issue in file_issues[:5]:  # Show first 5
                print(f"    Line {issue['line']}: {issue['code'][:60]}...")
            if len(file_issues) > 5:
                print(f"    ... and {len(file_issues) - 5} more")
    
    if warnings:
        print(f"\n📋 WARNINGS (may be legitimate): {len(warnings)} instances")


def main():
    parser = argparse.ArgumentParser(description='Fix suspicious test patterns')
    parser.add_argument('--analyze', action='store_true', help='Only analyze, do not modify')
    parser.add_argument('--fix', action='store_true', help='Apply fixes')
    parser.add_argument('--revert', action='store_true', help='Revert from backups')
    parser.add_argument('--run-tests', action='store_true', help='Run tests after fixing')
    args = parser.parse_args()
    
    if not any([args.analyze, args.fix, args.revert]):
        args.analyze = True  # Default to analyze
    
    if args.revert:
        print("Reverting files from backups...")
        for filename in TARGET_FILES:
            filepath = TESTS_DIR / filename
            if restore_backup(filepath):
                print(f"  Restored: {filename}")
            else:
                print(f"  No backup found: {filename}")
        return
    
    # Analyze all files
    all_issues = {}
    for filename in TARGET_FILES:
        filepath = TESTS_DIR / filename
        if filepath.exists():
            issues = analyze_file(filepath)
            if issues:
                all_issues[filename] = issues
    
    print_analysis(all_issues)
    
    if args.fix:
        print("\n" + "=" * 80)
        print("APPLYING FIXES")
        print("=" * 80)
        
        total_fixes = 0
        for filename in TARGET_FILES:
            filepath = TESTS_DIR / filename
            if filepath.exists():
                # Create backup first
                create_backup(filepath)
                
                # Apply fixes
                count1, fixes1 = fix_reset_index_patterns(filepath)
                count2, fixes2 = fix_get_series_on_dataframe(filepath)
                
                if count1 + count2 > 0:
                    print(f"\n{filename}:")
                    for fix in fixes1 + fixes2:
                        print(f"  - {fix}")
                    total_fixes += count1 + count2
        
        print(f"\nTotal fixes applied: {total_fixes}")
        print("Backups created with .bak_reset_index suffix")
        print("Run with --revert to undo changes")
        
        if args.run_tests:
            print("\n" + "=" * 80)
            print("RUNNING TESTS")
            print("=" * 80)
            import subprocess
            
            test_files = ' '.join(f'tests/{f}' for f in TARGET_FILES)
            result = subprocess.run(
                f'python -m pytest {test_files} -v --tb=short 2>&1 | head -200',
                shell=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)


if __name__ == '__main__':
    main()

