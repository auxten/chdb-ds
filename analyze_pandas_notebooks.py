#!/usr/bin/env python3
"""
Analyze pandas operations from downloaded Jupyter notebooks
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict

def load_notebook(filepath):
    """Load a Jupyter notebook and extract code cells"""
    with open(filepath, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)

    return code_cells

def extract_pandas_operations(code_cells):
    """Extract pandas operations from code cells"""
    operations = defaultdict(list)

    # Define patterns for different operation types
    patterns = {
        'dataframe_creation': [
            r'pd\.DataFrame\([^)]*\)',
            r'pd\.read_csv\([^)]*\)',
            r'pd\.read_excel\([^)]*\)',
            r'pd\.read_json\([^)]*\)',
            r'pd\.read_sql\([^)]*\)',
            r'pd\.read_html\([^)]*\)',
            r'pd\.read_table\([^)]*\)',
        ],
        'data_selection': [
            r'\.loc\[',
            r'\.iloc\[',
            r'\.at\[',
            r'\.iat\[',
            r'\.query\(',
            r'\[[^\]]+\](?!\s*=)',  # Indexing but not assignment
            r'\.get\(',
            r'\.xs\(',
        ],
        'data_cleaning': [
            r'\.dropna\(',
            r'\.fillna\(',
            r'\.drop_duplicates\(',
            r'\.drop\(',
            r'\.replace\(',
            r'\.interpolate\(',
            r'\.ffill\(',
            r'\.bfill\(',
        ],
        'data_transformation': [
            r'\.apply\(',
            r'\.applymap\(',
            r'\.map\(',
            r'\.transform\(',
            r'\.astype\(',
            r'\.convert_dtypes\(',
            r'\.rename\(',
            r'\.set_index\(',
            r'\.reset_index\(',
        ],
        'aggregation': [
            r'\.groupby\(',
            r'\.agg\(',
            r'\.aggregate\(',
            r'\.pivot_table\(',
            r'\.pivot\(',
            r'\.crosstab\(',
            r'\.resample\(',
            r'\.rolling\(',
            r'\.expanding\(',
        ],
        'merging': [
            r'\.merge\(',
            r'\.join\(',
            r'pd\.merge\(',
            r'pd\.concat\(',
            r'\.append\(',
        ],
        'sorting': [
            r'\.sort_values\(',
            r'\.sort_index\(',
            r'\.rank\(',
            r'\.nlargest\(',
            r'\.nsmallest\(',
        ],
        'statistics': [
            r'\.describe\(',
            r'\.mean\(',
            r'\.median\(',
            r'\.sum\(',
            r'\.std\(',
            r'\.var\(',
            r'\.min\(',
            r'\.max\(',
            r'\.count\(',
            r'\.nunique\(',
            r'\.value_counts\(',
            r'\.corr\(',
            r'\.cov\(',
            r'\.quantile\(',
        ],
        'string_operations': [
            r'\.str\.lower\(',
            r'\.str\.upper\(',
            r'\.str\.contains\(',
            r'\.str\.replace\(',
            r'\.str\.split\(',
            r'\.str\.strip\(',
            r'\.str\.startswith\(',
            r'\.str\.endswith\(',
            r'\.str\.len\(',
            r'\.str\.cat\(',
            r'\.str\.extract\(',
            r'\.str\.match\(',
        ],
        'datetime_operations': [
            r'pd\.to_datetime\(',
            r'\.dt\.year',
            r'\.dt\.month',
            r'\.dt\.day',
            r'\.dt\.hour',
            r'\.dt\.minute',
            r'\.dt\.dayofweek',
            r'\.dt\.weekday',
            r'\.dt\.strftime\(',
            r'\.resample\(',
        ],
    }

    # Extract operations
    for code in code_cells:
        # Skip empty or comment-only cells
        if not code.strip() or code.strip().startswith('#'):
            continue

        for operation_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, code, re.MULTILINE)
                for match in matches:
                    # Get the full line containing the match
                    lines = code.split('\n')
                    for line in lines:
                        if match.group() in line:
                            # Clean up the line
                            line = line.strip()
                            if line and not line.startswith('#'):
                                operations[operation_type].append({
                                    'pattern': pattern,
                                    'code': line,
                                    'operation': match.group()
                                })
                            break

    return operations

def deduplicate_operations(operations):
    """Remove duplicate operation examples"""
    deduped = {}
    for op_type, examples in operations.items():
        seen_code = set()
        unique_examples = []
        for example in examples:
            code_normalized = example['code'].lower().strip()
            if code_normalized not in seen_code:
                seen_code.add(code_normalized)
                unique_examples.append(example)
        deduped[op_type] = unique_examples
    return deduped

def analyze_notebooks(notebook_dir):
    """Analyze all notebooks in the directory"""
    notebook_dir = Path(notebook_dir)
    notebook_files = list(notebook_dir.glob('*.ipynb'))

    print(f"Found {len(notebook_files)} notebook files")

    all_operations = defaultdict(list)

    for notebook_file in notebook_files:
        print(f"Analyzing: {notebook_file.name}")
        try:
            code_cells = load_notebook(notebook_file)
            operations = extract_pandas_operations(code_cells)

            # Merge operations
            for op_type, examples in operations.items():
                all_operations[op_type].extend(examples)
        except Exception as e:
            print(f"Error analyzing {notebook_file.name}: {e}")

    # Deduplicate operations
    all_operations = deduplicate_operations(all_operations)

    return all_operations, len(notebook_files)

def create_summary_report(operations, num_notebooks):
    """Create a summary report of the analysis"""
    summary = {
        'metadata': {
            'num_notebooks_analyzed': num_notebooks,
            'total_operation_types': len(operations),
            'total_unique_examples': sum(len(examples) for examples in operations.values())
        },
        'operation_types': {},
        'detailed_operations': {}
    }

    # Create operation type summary
    for op_type, examples in sorted(operations.items()):
        summary['operation_types'][op_type] = {
            'count': len(examples),
            'description': get_operation_description(op_type)
        }

    # Create detailed operations with examples
    for op_type, examples in sorted(operations.items()):
        # Limit to top 20 examples per operation type
        top_examples = examples[:20]
        summary['detailed_operations'][op_type] = {
            'description': get_operation_description(op_type),
            'total_examples_found': len(examples),
            'examples': [
                {
                    'code': ex['code'],
                    'operation': ex['operation']
                }
                for ex in top_examples
            ]
        }

    return summary

def get_operation_description(op_type):
    """Get description for operation type"""
    descriptions = {
        'dataframe_creation': 'Creating DataFrames and reading data from various sources (CSV, Excel, JSON, SQL, etc.)',
        'data_selection': 'Selecting data using loc, iloc, query, and indexing operations',
        'data_cleaning': 'Cleaning data by handling missing values, duplicates, and replacements',
        'data_transformation': 'Transforming data with apply, map, rename, type conversions, and index operations',
        'aggregation': 'Aggregating data with groupby, pivot tables, rolling windows, and resampling',
        'merging': 'Merging, joining, and concatenating DataFrames',
        'sorting': 'Sorting and ranking data by values or index',
        'statistics': 'Computing statistical measures like mean, median, sum, correlation, etc.',
        'string_operations': 'String manipulation operations using the .str accessor',
        'datetime_operations': 'Date and time operations using the .dt accessor and to_datetime'
    }
    return descriptions.get(op_type, 'Other pandas operations')

def main():
    notebook_dir = '/home/user/chdb-ds/downloaded_notebooks'
    output_file = '/home/user/chdb-ds/pandas_operations_analysis.json'

    print("=" * 80)
    print("Pandas Operations Analysis")
    print("=" * 80)

    # Analyze notebooks
    operations, num_notebooks = analyze_notebooks(notebook_dir)

    # Create summary report
    summary = create_summary_report(operations, num_notebooks)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nNotebooks analyzed: {num_notebooks}")
    print(f"Operation types found: {len(operations)}")
    print(f"Total unique examples: {sum(len(examples) for examples in operations.values())}")
    print(f"\nOperation type breakdown:")
    for op_type in sorted(operations.keys()):
        print(f"  - {op_type}: {len(operations[op_type])} examples")
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
