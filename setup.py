import ast
from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


def version():
    path = '__init__.py'
    with open(path, 'r') as file:
        t = compile(file.read(), path, 'exec', ast.PyCF_ONLY_AST)
        for node in (n for n in t.body if isinstance(n, ast.Assign)):
            if len(node.targets) == 1:
                name = node.targets[0]
                if isinstance(name, ast.Name) and name.id in ('__version__', '__version_info__', 'VERSION'):
                    v = node.value
                    if isinstance(v, ast.Str):
                        return v.s
                    if isinstance(v, ast.Constant):  # Python 3.8+
                        return v.value
                    if isinstance(v, ast.Tuple):
                        r = []
                        for e in v.elts:
                            if isinstance(e, ast.Str):
                                r.append(e.s)
                            elif isinstance(e, ast.Num):
                                r.append(str(e.n))
                            elif isinstance(e, ast.Constant):
                                r.append(str(e.value))
                        return '.'.join(r)


setup(
    # Application name:
    name="chdb-ds",

    # Version number:
    version=version(),

    # Application author details:
    author="auxten",
    author_email="auxtenwpc@gmail.com",

    # License
    license='Apache License Version 2.0',

    # Packages
    packages=find_packages(exclude=['tests', 'tests.*']),

    # Include additional files into the package
    include_package_data=True,

    # Dependencies
    install_requires=[
        'chdb>=2.0.0',  # For ClickHouse backend
    ],

    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
    },

    # Python version requirement
    python_requires='>=3.7',

    # Details
    url="https://github.com/auxten/chdb-ds",
    project_urls={
        'Documentation': 'https://github.com/auxten/chdb-ds',
        'Source': 'https://github.com/auxten/chdb-ds',
        'Tracker': 'https://github.com/auxten/chdb-ds/issues',
    },

    description="A Pandas-like data manipulation framework with automatic SQL generation",
    long_description=readme(),
    long_description_content_type='text/markdown',

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: SQL',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Database',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Operating System :: POSIX',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
    ],

    keywords=(
        'datastore pandas polars sql query builder database clickhouse '
        'data manipulation analytics dataframe chdb embedded '
        'pypika query generation immutable fluent api'
    ),

    # Test suite
    test_suite="tests",
)
