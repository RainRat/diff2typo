import sys
import os
import subprocess
import json
import csv
import io

# Add the current directory to sys.path so we can import the scripts
sys.path.insert(0, os.path.abspath(os.getcwd()))

from typostats import is_keyboard_adjacent, process_typos

def test_is_keyboard_adjacent():
    # Same row
    assert is_keyboard_adjacent('q', 'w') is True
    assert is_keyboard_adjacent('a', 's') is True
    # Different row, adjacent
    assert is_keyboard_adjacent('q', 'a') is True
    assert is_keyboard_adjacent('w', 'a') is True
    # Diagonal
    assert is_keyboard_adjacent('s', 'e') is True
    # Not adjacent
    assert is_keyboard_adjacent('q', 'p') is False
    assert is_keyboard_adjacent('a', 'l') is False
    # Case insensitive
    assert is_keyboard_adjacent('A', 's') is True
    # Same character
    assert is_keyboard_adjacent('a', 'a') is False

def test_cli_keyboard_arrow():
    # Create a temporary typo file
    typo_content = "as -> aa\nap -> aa\n"
    with open("temp_typos.txt", "w") as f:
        f.write(typo_content)

    try:
        # Run with --keyboard flag
        # We need to capture stderr because that's where the summary goes in arrow mode
        result = subprocess.run(
            [sys.executable, "typostats.py", "temp_typos.txt", "--keyboard"],
            capture_output=True, text=True
        )
        # Check stderr for keyboard adjacency summary
        assert "Keyboard Adjacency" in result.stderr
        assert "Adjacent: 1" in result.stderr
        assert "Total:    2" in result.stderr
        assert "Percent:  50.0%" in result.stderr

        # Run without --keyboard flag
        result_no_kb = subprocess.run(
            [sys.executable, "typostats.py", "temp_typos.txt"],
            capture_output=True, text=True
        )
        assert "Keyboard Adjacency" not in result_no_kb.stderr

    finally:
        if os.path.exists("temp_typos.txt"):
            os.remove("temp_typos.txt")

def test_cli_keyboard_json():
    typo_content = "as -> aa\nap -> aa\n"
    with open("temp_typos_json.txt", "w") as f:
        f.write(typo_content)

    try:
        result = subprocess.run(
            [sys.executable, "typostats.py", "temp_typos_json.txt", "--keyboard", "--format", "json"],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        assert "statistics" in data
        assert data["statistics"]["total_one_to_one"] == 2
        assert data["statistics"]["adjacent_count"] == 1
        assert data["statistics"]["adjacent_percentage"] == 50.0

    finally:
        if os.path.exists("temp_typos_json.txt"):
            os.remove("temp_typos_json.txt")

def test_cli_keyboard_csv():
    typo_content = "as -> aa\nap -> aa\n"
    with open("temp_typos_csv.txt", "w") as f:
        f.write(typo_content)

    try:
        result = subprocess.run(
            [sys.executable, "typostats.py", "temp_typos_csv.txt", "--keyboard", "--format", "csv"],
            capture_output=True, text=True
        )
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)

        assert "is_adjacent" in rows[0]
        # Sort by correct_char and typo_char to be sure
        rows.sort(key=lambda x: (x['correct_char'], x['typo_char']))
        # 'aa' is correction. replacements are ('a', 's') and ('a', 'p')
        # Wait, sorted order might be different.
        # Replacement 1: correct 'a', typo 's' -> is_adjacent: True
        # Replacement 2: correct 'a', typo 'p' -> is_adjacent: False

        row_as = next(r for r in rows if r['typo_char'] == 's')
        row_ap = next(r for r in rows if r['typo_char'] == 'p')

        assert row_as['is_adjacent'] == 'True'
        assert row_ap['is_adjacent'] == 'False'

    finally:
        if os.path.exists("temp_typos_csv.txt"):
            os.remove("temp_typos_csv.txt")
