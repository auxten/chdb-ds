#!/usr/bin/env python3
"""
Script to update package version from git tag.
Extracts version from git tag (e.g., v1.0.0 -> 1.0.0) and updates datastore/__init__.py
"""
import re
import sys
import subprocess
from pathlib import Path


def get_git_tag():
    """Get the current git tag."""
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--exact-match'], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Error: Not on a tagged commit", file=sys.stderr)
        sys.exit(1)


def extract_version(tag):
    """Extract version number from tag (e.g., v1.0.0 -> 1.0.0)."""
    # Remove 'v' prefix if present
    if tag.startswith('v'):
        version = tag[1:]
    else:
        version = tag

    # Validate version format (basic semver check)
    if not re.match(r'^\d+\.\d+\.\d+', version):
        print(f"Error: Invalid version format: {version}", file=sys.stderr)
        print("Expected format: v1.2.3 or 1.2.3", file=sys.stderr)
        sys.exit(1)

    return version


def update_version_file(version, init_file='datastore/__init__.py'):
    """Update the __version__ in __init__.py."""
    init_path = Path(init_file)

    if not init_path.exists():
        print(f"Error: {init_file} not found", file=sys.stderr)
        sys.exit(1)

    # Read the file
    content = init_path.read_text(encoding='utf-8')

    # Replace the version line
    new_content = re.sub(r'__version__\s*=\s*["\'][\d.]+["\']', f'__version__ = "{version}"', content)

    if content == new_content:
        print(f"Warning: Version string not found or already set to {version}", file=sys.stderr)

    # Write back
    init_path.write_text(new_content, encoding='utf-8')
    print(f"✓ Updated version to {version} in {init_file}")


def main():
    """Main function."""
    # Allow passing version directly as argument (for testing)
    if len(sys.argv) > 1:
        version = extract_version(sys.argv[1])
    else:
        tag = get_git_tag()
        print(f"Found git tag: {tag}")
        version = extract_version(tag)

    print(f"Extracted version: {version}")
    update_version_file(version)
    print(f"✓ Successfully updated package version to {version}")


if __name__ == '__main__':
    main()
