#!/usr/bin/env python3
"""Check for non-ASCII/non-emoji characters in .md and .py files."""

import os
import re
import sys

# Allowed non-ASCII characters pattern (emoji + common typography + symbols)
ALLOWED_NON_ASCII = re.compile(
    "["
    # Emoji ranges
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026FF"  # Misc symbols (includes ✓ ✗ ⚠ ★ etc.)
    "\U00002700-\U000027BF"  # Dingbats
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U0000200D"             # Zero Width Joiner
    "\U00002300-\U000023FF"  # Misc Technical
    "\U00002B50-\U00002B55"  # Stars
    # Common typography
    "\U000000A0"             # Non-breaking space
    "\U000000D7"             # × multiplication sign
    "\U00002010-\U00002015"  # Hyphens and dashes (‐ ‑ ‒ – —)
    "\U00002018-\U0000201F"  # Curly quotes (' ' " " ‚ „ ‟)
    "\U00002022"             # • bullet
    "\U00002026"             # … ellipsis
    "\U00002032-\U00002037"  # Prime marks (′ ″ ‴ etc.)
    "\U00002039-\U0000203A"  # Single angle quotes (‹ ›)
    "\U000000AB"             # « left guillemet
    "\U000000BB"             # » right guillemet
    "\U00002190-\U000021FF"  # Arrows (← → ↑ ↓ etc.)
    "\U00002713-\U00002717"  # Check marks (✓ ✔ ✕ ✖ ✗)
    "\U0001F4A1"             # 💡 light bulb
    "\U0001F31F"             # 🌟 glowing star
    # Box drawing and mathematical symbols
    "\U00002500-\U0000257F"  # Box Drawing (─ │ ├ └ ┐ ┘ etc.)
    "\U00002200-\U000022FF"  # Mathematical Operators (≈ ≠ ≤ ≥ etc.)
    "\U000027C0-\U000027EF"  # Misc Mathematical Symbols-A (⟗ etc.)
    "\U00002980-\U000029FF"  # Misc Mathematical Symbols-B
    "\U00002A00-\U00002AFF"  # Supplemental Mathematical Operators
    "\U00002B00-\U00002BFF"  # Misc Symbols and Arrows
    "]+"
)

# Directories to skip
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "dist", "build", "chdb_ds.egg-info", ".venv", "venv"}


def is_allowed_char(char):
    """Check if a character is allowed (ASCII, emoji, or common typography)."""
    # ASCII characters (0x00-0x7F)
    if ord(char) <= 0x7F:
        return True
    # Check if it's an allowed non-ASCII character
    if ALLOWED_NON_ASCII.match(char):
        return True
    return False


def find_disallowed_chars(line):
    """Find all disallowed characters in a line."""
    return [char for char in line if not is_allowed_char(char)]


def check_file(path):
    """Check a file for non-ASCII/non-emoji characters. Returns list of (line_num, line, matches)."""
    results = []
    try:
        with open(path, "r", encoding="utf-8") as fp:
            for i, line in enumerate(fp, 1):
                matches = find_disallowed_chars(line)
                if matches:
                    results.append((i, line.rstrip(), matches))
    except Exception:
        pass
    return results


def main():
    root_dir = "."
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    found = False

    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for f in files:
            if f.endswith((".md", ".py")):
                path = os.path.join(root, f)
                results = check_file(path)
                if results:
                    for line_num, line, matches in results:
                        print(f"{path}:{line_num}: {line}")
                        print(f"  Found: {''.join(matches)}")
                    found = True

    if found:
        print("\nError: Non-ASCII/non-emoji characters found in the above files!")
        sys.exit(1)
    else:
        print("✓ All characters are ASCII or emoji")
        sys.exit(0)


if __name__ == "__main__":
    main()

