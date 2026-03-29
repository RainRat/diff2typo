import sys
import os
import re
import pytest
from unittest.mock import patch

# Ensure the repository root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multitool

def strip_ansi(text):
    """Remove ANSI escape sequences from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def test_classify_typo_multiple_letters_gaps():
    # Line 191: Deletion-length-diff-1 but not a simple deletion
    # 'abc' (len 3) -> 'ax' (len 2)
    assert multitool.classify_typo("ax", "abc", {}) == "[M]"

    # Line 198: Insertion-length-diff-1 but not a simple insertion
    # 'abc' (len 3) -> 'axby' (len 4)
    assert multitool.classify_typo("axby", "abc", {}) == "[M]"

    # Line 201: Other length differences
    # 'abc' (len 3) -> 'abcde' (len 5)
    assert multitool.classify_typo("abcde", "abc", {}) == "[M]"

def test_search_mode_span_merging_else_branch(tmp_path):
    # Line 2824: next_start > curr_end (non-overlapping)
    input_file = tmp_path / "input.txt"
    input_file.write_text("word1 word2", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    # We need to trigger two separate spans.
    # Query 'word' will match 'word' in 'word1' and 'word' in 'word2'.
    # Mock YELLOW to cover highlighting logic in search_mode too
    with patch('multitool.YELLOW', '\033[33m'), \
         patch('multitool.RESET', '\033[0m'), \
         patch('multitool.BLUE', '\033[34m'), \
         patch('multitool.BOLD', '\033[1m'):
        multitool.search_mode(
            input_files=[str(input_file)],
            query="word",
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False,
            line_numbers=True,
            with_filename=True
        )

    content = output_file.read_text(encoding='utf-8')
    assert "\033[33mword\033[0m1" in content
    assert "\033[33mword\033[0m2" in content

def test_search_mode_literal_fallback(tmp_path):
    # Lines 2787-2788: match_found_in_word = True in the fallback
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    multitool.search_mode(
        input_files=[str(input_file)],
        query="te-h",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False
    )

    results = output_file.read_text(encoding='utf-8')
    assert "teh" in results

def test_search_mode_smart_levenshtein_subpart(tmp_path):
    # Lines 2802, 2806-2808, 2810
    input_file = tmp_path / "input.txt"
    # Word with subparts: 'MyTehWord' -> ['My', 'Teh', 'Word']
    # Add a part that results in empty sp_clean to hit line 2802
    # In search_mode, smart split is done via _smart_split.
    # _smart_split splits by [^a-zA-Z0-9]+ and CamelCase.
    # 'My-Word' -> _smart_split -> ['My', 'Word'] (non-alphanumeric are separators)
    # Wait, _smart_split implementation:
    # def _smart_split(text: str) -> List[str]:
    #     parts = re.split(r'[^a-zA-Z0-9]+', text)
    #     final_parts = []
    #     for p in parts: ...
    # So if we have '---', re.split will give empty strings or parts.
    # Actually 'My---Word' split by '[^a-zA-Z0-9]+' gives ['My', 'Word'].
    # To get an empty sp_clean from filter_to_letters(sp), we need sp to be non-alphanumeric.
    # But those are separators in re.split.
    # Let's check _smart_split again.
    input_file.write_text("MyTehWord", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    # Query 'Tehx' (similar to 'Teh' subpart)
    # Using clean_items=True (default)
    multitool.search_mode(
        input_files=[str(input_file)],
        query="Tehx",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        smart=True,
        max_dist=1,
        clean_items=True
    )

    results = output_file.read_text(encoding='utf-8')
    assert "MyTehWord" in results

def test_search_mode_smart_empty_subpart_skip(tmp_path):
    # Line 2802: if not sp_clean: continue
    # We need _smart_split to return something that filter_to_letters makes empty.
    # Looking at _smart_split:
    # parts = re.split(r'[^a-zA-Z0-9]+', text)
    # final_parts = []
    # for p in parts:
    #     sub_parts = re.findall(r'[A-Z][a-z]*|[a-z]+|[0-9]+', p)
    #     final_parts.extend(sub_parts)
    # return final_parts
    # It seems it only returns matches for those regexes, which are all alphanumeric.
    # So sp_clean will NEVER be empty if clean_items=True.
    # BUT if clean_items=False, sp_clean = sp.lower().
    # Still, it only returns alphanumeric parts.
    # Wait, if p is "123", sub_parts is ["123"].
    # If p is "ABC", sub_parts is ["A", "B", "C"]? No, [A-Z][a-z]* matches "A", "B", "C".
    # Is there ANY way to get an empty sp_clean?
    # Maybe if we mock _smart_split to return an empty string.
    with patch('multitool._smart_split', return_value=['', 'Teh']):
        input_file = tmp_path / "input.txt"
        input_file.write_text("Something", encoding='utf-8')
        output_file = tmp_path / "output.txt"
        multitool.search_mode(
            input_files=[str(input_file)],
            query="Teh",
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False,
            smart=True
        )
    assert "Something" in output_file.read_text()

def test_search_mode_fuzzy_match_direct(tmp_path):
    # Lines 2794-2795: direct fuzzy match (not smart)
    input_file = tmp_path / "input.txt"
    input_file.write_text("recieve", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    # query_clean for 'receive' is 'receive'
    # word_clean for 'recieve' is 'recieve'
    # 'receive' is not in 'recieve'
    # but levenshtein('receive', 'recieve') is 2 (c-i and v-e swap)
    # let's use a simpler one: 'teh' vs 'the' (dist 2)
    multitool.search_mode(
        input_files=[str(input_file)],
        query="receive",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        max_dist=2
    )

    results = output_file.read_text(encoding='utf-8')
    assert "recieve" in results

def test_search_mode_process_output_sorting(tmp_path):
    # Line 2858
    input_file = tmp_path / "input.txt"
    input_file.write_text("match B\nmatch A", encoding='utf-8')
    output_file = tmp_path / "output.txt"

    multitool.search_mode(
        input_files=[str(input_file)],
        query="match",
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True
    )

    results = output_file.read_text(encoding='utf-8').splitlines()
    # Sorted: match A should come before match B
    assert "match A" in results[0]
    assert "match B" in results[1]

def test_scan_mode_highlighting_and_sorting(tmp_path):
    # Lines 3679-3701 (highlighting) and 3724 (sorting)
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh", encoding='utf-8')

    input_file = tmp_path / "input.txt"
    # To hit line 3693 (smart subpart highlight) and 3698 (non-matching part)
    # also include a non-matching word for line 3700
    input_file.write_text("line B with tehVariable\nline A with teh\nother", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    # Mock YELLOW to enable highlighting logic
    with patch('multitool.YELLOW', '\033[33m'), \
         patch('multitool.RESET', '\033[0m'), \
         patch('multitool.BLUE', '\033[34m'), \
         patch('multitool.BOLD', '\033[1m'):
        multitool.scan_mode(
            input_files=[str(input_file)],
            mapping_file=str(mapping_file),
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=True,
            smart=True
        )

    content = output_file.read_text(encoding='utf-8')
    results = content.splitlines()

    # Check sorting
    assert "line A" in results[0]
    assert "line B" in results[1]

    # Check highlighting
    assert "\033[33mteh\033[0m" in content
    assert "\033[33mteh\033[0mVariable" in content

def test_scan_mode_highlighting_no_smart(tmp_path):
    # Line 3698: else branch when smart=False but use_color=True
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh", encoding='utf-8')

    input_file = tmp_path / "input.txt"
    # To hit line 3698, we need pattern.match(part) to be true,
    # but mk not in mapping, and smart to be False.
    # pattern is re.compile(r'([a-zA-Z0-9]+)')
    # mapping contains 'teh'
    input_file.write_text("teh other", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    with patch('multitool.YELLOW', '\033[33m'), \
         patch('multitool.RESET', '\033[0m'):
        multitool.scan_mode(
            input_files=[str(input_file)],
            mapping_file=str(mapping_file),
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False,
            smart=False,
            clean_items=True
        )

    content = output_file.read_text(encoding='utf-8')
    assert "\033[33mteh\033[0m" in content
    assert "other" in content

def test_resolve_mode_empty_clean_skip(tmp_path):
    # Line 3147: if not left or not right: continue
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("---,the\nteh,---", encoding='utf-8')

    output_file = tmp_path / "output.txt"

    multitool.resolve_mode(
        input_files=[str(mapping_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False
    )

    assert output_file.read_text().strip() == ""
