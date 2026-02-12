import sys
import json
from pathlib import Path
from collections import defaultdict

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_get_adjacent_keys():
    adj = typostats.get_adjacent_keys(include_diagonals=True)
    assert 's' in adj['a']
    assert 'w' in adj['a']
    assert 'q' in adj['a']
    assert 'z' in adj['a']
    # 'p' is far from 'a'
    assert 'p' not in adj['a']

def test_get_adjacent_keys_no_diagonals():
    adj = typostats.get_adjacent_keys(include_diagonals=False)
    assert 's' in adj['a']
    # In the simple grid model:
    # q (0,0), w (0,1)
    # a (1,0), s (1,1)
    # w is diagonal (dr=-1, dc=1), q is vertical (dr=-1, dc=0)
    assert 'q' in adj['a']
    assert 'w' not in adj['a']
    assert 'z' in adj['a']

def test_generate_report_with_keyboard(capsys):
    # 'a' and 's' are adjacent
    # 'a' and 'p' are not
    counts = {('a', 's'): 10, ('a', 'p'): 5}

    # Run with keyboard enabled
    typostats.generate_report(counts, output_format='arrow', keyboard=True, quiet=False)

    captured = capsys.readouterr()
    # Summary should be in stderr
    assert "Keyboard Adjacency: 10/15 (66.7%)" in captured.err
    # Markers should be in stdout
    # Note: markers use ANSI colors (BOLD/RESET) so we check for the string containing [K]
    assert "[K]" in captured.out
    assert "a" in captured.out and "s" in captured.out

def test_generate_report_keyboard_no_single_char(capsys):
    # Multi-char replacements should be ignored by adjacency check
    counts = {('abc', 'abd'): 10}
    typostats.generate_report(counts, output_format='arrow', keyboard=True, quiet=False)

    captured = capsys.readouterr()
    # No Keyboard Adjacency summary if no single-char replacements
    assert "Keyboard Adjacency" not in captured.err
    assert "[K]" not in captured.out

def test_generate_report_keyboard_to_file(tmp_path):
    output_file = tmp_path / "report.txt"
    counts = {('a', 's'): 10}
    typostats.generate_report(counts, output_file=str(output_file), output_format='arrow', keyboard=True)

    content = output_file.read_text()
    assert "Keyboard Adjacency: 10/10 (100.0%)" in content
    assert "[K]" in content

def test_generate_report_keyboard_json(capsys):
    counts = {('a', 's'): 10, ('a', 'p'): 5}
    typostats.generate_report(counts, output_format='json', keyboard=True)

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    # Check that is_adjacent is present and correct
    for item in data["replacements"]:
        if item["correct"] == "a" and item["typo"] == "s":
            assert item["is_adjacent"] is True
        elif item["correct"] == "a" and item["typo"] == "p":
            assert item["is_adjacent"] is False

def test_generate_report_keyboard_case_insensitive(capsys):
    # 'A' and 'S' should be considered adjacent
    counts = {('A', 'S'): 1}
    typostats.generate_report(counts, output_format='arrow', keyboard=True, quiet=False)

    captured = capsys.readouterr()
    assert "Keyboard Adjacency: 1/1 (100.0%)" in captured.err
    assert "[K]" in captured.out
