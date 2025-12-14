#!/usr/bin/env python3
"""
Fetch ClickHouse functions from system.functions and categorize them.

This script generates a JSON file with all ClickHouse functions organized by category.
Used to help build the function registry.
"""

import json
import chdb
from collections import defaultdict


def fetch_functions():
    """Fetch all functions from ClickHouse system.functions table."""
    query = """
    SELECT 
        name,
        is_aggregate,
        case_insensitive,
        alias_to,
        create_query
    FROM system.functions
    ORDER BY name
    """
    result = chdb.query(query, 'JSON')
    data = json.loads(result.bytes())
    return data.get('data', [])


def categorize_function(name: str, is_aggregate: bool) -> str:
    """
    Categorize a function based on its name prefix/suffix.
    
    Categories:
    - datetime: Date/time functions
    - string: String functions
    - array: Array functions
    - math: Mathematical functions
    - aggregate: Aggregate functions
    - hash: Hash functions
    - json: JSON functions
    - type: Type conversion functions
    - conditional: Conditional functions
    - url: URL functions
    - ip: IP address functions
    - uuid: UUID functions
    - geo: Geo functions
    - encoding: Encoding functions
    - window: Window functions
    - other: Other functions
    """
    name_lower = name.lower()
    
    if is_aggregate:
        # Check for window-like aggregates
        if any(x in name_lower for x in ['rank', 'row_number', 'ntile', 'lead', 'lag', 
                                          'first_value', 'last_value', 'nth_value']):
            return 'window'
        return 'aggregate'
    
    # DateTime functions
    datetime_prefixes = ['to', 'date', 'time', 'format', 'parse']
    datetime_keywords = ['year', 'month', 'day', 'hour', 'minute', 'second', 
                         'week', 'quarter', 'timezone', 'datetime', 'interval',
                         'monday', 'sunday', 'start_of', 'end_of', 'add', 'subtract']
    if any(name_lower.startswith(p) for p in datetime_prefixes):
        if any(k in name_lower for k in datetime_keywords + ['date', 'time']):
            return 'datetime'
    if any(k in name_lower for k in ['toYear', 'toMonth', 'toDay', 'toHour', 'toMinute', 
                                      'toSecond', 'toDate', 'toDateTime', 'now', 'today',
                                      'yesterday', 'toUnix']):
        return 'datetime'
    
    # String functions
    string_keywords = ['string', 'char', 'concat', 'substr', 'substring', 'trim', 
                       'upper', 'lower', 'length', 'replace', 'reverse', 'like',
                       'match', 'extract', 'split', 'join', 'pad', 'encode', 
                       'decode', 'base64', 'hex', 'ascii', 'format', 'ngram',
                       'token', 'lemmatize', 'stem', 'synonym']
    if any(k in name_lower for k in string_keywords):
        return 'string'
    
    # Array functions
    array_keywords = ['array', 'range', 'empty', 'notempty', 'has', 'hasall', 
                      'hasany', 'indexof', 'countequal', 'enumerate', 'flatten',
                      'reverse', 'slice', 'shuffle', 'distinct', 'intersect', 
                      'resize', 'zip', 'unzip', 'map', 'filter', 'reduce',
                      'element', 'first', 'last', 'push', 'pop']
    if any(k in name_lower for k in array_keywords):
        return 'array'
    
    # Hash functions
    hash_keywords = ['hash', 'md5', 'sha', 'crc', 'xxhash', 'siphash', 
                     'murmur', 'city', 'farm', 'metro', 'fingerprint']
    if any(k in name_lower for k in hash_keywords):
        return 'hash'
    
    # JSON functions
    json_keywords = ['json', 'simdjson', 'rapid']
    if any(k in name_lower for k in json_keywords):
        return 'json'
    
    # URL functions
    url_keywords = ['url', 'domain', 'protocol', 'port', 'path', 'query', 
                    'fragment', 'tld', 'cuturl', 'decodeurlcomponent']
    if any(k in name_lower for k in url_keywords):
        return 'url'
    
    # IP functions
    ip_keywords = ['ipv4', 'ipv6', 'ip', 'cidr', 'geohash']
    if any(k in name_lower for k in ip_keywords):
        return 'ip'
    
    # UUID functions
    uuid_keywords = ['uuid', 'generateuuid']
    if any(k in name_lower for k in uuid_keywords):
        return 'uuid'
    
    # Geo functions
    geo_keywords = ['geo', 'h3', 's2', 'polygon', 'point', 'great_circle',
                    'latitude', 'longitude', 'distance']
    if any(k in name_lower for k in geo_keywords):
        return 'geo'
    
    # Math functions
    math_keywords = ['abs', 'sqrt', 'cbrt', 'pow', 'power', 'exp', 'log', 
                     'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'pi', 'e',
                     'ceil', 'floor', 'round', 'trunc', 'sign', 'mod', 
                     'gcd', 'lcm', 'factorial', 'fibonacci', 'gamma', 
                     'erf', 'intexp', 'int_div', 'modulo']
    if any(k in name_lower for k in math_keywords):
        return 'math'
    
    # Type conversion
    type_keywords = ['cast', 'convert', 'reinterpret', 'totype', 'parse', 
                     'toint', 'touint', 'tofloat', 'tostring', 'todecimal',
                     'tonullable', 'assumenotnull', 'null', 'coalesce']
    if any(k in name_lower for k in type_keywords):
        return 'type'
    
    # Conditional functions
    cond_keywords = ['if', 'case', 'multi', 'coalesce', 'isnull', 'isnotnull',
                     'ifnull', 'nullif', 'or', 'and', 'not', 'transform']
    if any(k in name_lower for k in cond_keywords):
        return 'conditional'
    
    # Encoding functions  
    encoding_keywords = ['base58', 'base64', 'hex', 'unhex', 'bin', 'unbin',
                         'encode', 'decode', 'bitmask', 'bitcount']
    if any(k in name_lower for k in encoding_keywords):
        return 'encoding'
    
    return 'other'


def main():
    print("Fetching ClickHouse functions...")
    functions = fetch_functions()
    print(f"Found {len(functions)} functions")
    
    # Organize by category
    by_category = defaultdict(list)
    alias_mapping = {}
    
    for func in functions:
        name = func['name']
        is_aggregate = func.get('is_aggregate', 0) == 1
        alias_to = func.get('alias_to', '')
        
        if alias_to:
            alias_mapping[name] = alias_to
            continue  # Skip aliases for main list
        
        category = categorize_function(name, is_aggregate)
        by_category[category].append({
            'name': name,
            'is_aggregate': is_aggregate,
            'case_insensitive': func.get('case_insensitive', 0) == 1,
        })
    
    # Print summary
    print("\n=== Function Categories ===")
    for category, funcs in sorted(by_category.items()):
        agg_count = sum(1 for f in funcs if f['is_aggregate'])
        print(f"{category}: {len(funcs)} functions ({agg_count} aggregate)")
    
    print(f"\nTotal aliases: {len(alias_mapping)}")
    
    # Save to JSON
    output = {
        'categories': {k: v for k, v in sorted(by_category.items())},
        'aliases': alias_mapping,
        'total_functions': len(functions),
    }
    
    output_path = 'scripts/ch_functions.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    # Print some examples for each category
    print("\n=== Sample Functions by Category ===")
    for category, funcs in sorted(by_category.items()):
        samples = [f['name'] for f in funcs[:5]]
        print(f"{category}: {', '.join(samples)}")


if __name__ == '__main__':
    main()


