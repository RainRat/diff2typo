import sys
import re
from pathlib import Path
from unittest.mock import patch
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def strip_ansi(text):
    return re.sub(r'\033\[[0-9;]*m', '', text)

def test_levenshtein_distance_with_empty_second_string():
    assert typostats.levenshtein_distance("abc", "") == 3

def test_format_analysis_summary_with_fractional_retention_visual_bar():
    summary = typostats._format_analysis_summary(
        raw_count=3,
        filtered_items=["a"],
        item_label="item"
    )
    summary_text = "\n".join(summary)
    assert "33.3%" in summary_text
    assert "▋" in summary_text

def test_format_analysis_summary_with_unhashable_items_fallback():
    summary = typostats._format_analysis_summary(
        raw_count=0,
        filtered_items=[[1, 2], [1, 2], [3, 4]],
        item_label="item"
    )
    summary_text = strip_ansi("\n".join(summary))
    assert "Unique items:" in summary_text
    assert "3" in summary_text

def test_format_analysis_summary_with_non_tuple_items_string_conversion():
    summary = typostats._format_analysis_summary(
        raw_count=2,
        filtered_items=["word1", "word2"],
        item_label="word"
    )
    summary_text = strip_ansi("\n".join(summary))
    assert "Shortest word:" in summary_text
    assert "'word1' (length: 5)" in summary_text

def test_format_analysis_summary_robustness_on_mixed_types_exception():
    class ItemWithBrokenStr:
        def __str__(self):
            raise TypeError("formatting error")
        def __repr__(self):
            raise TypeError("formatting error")

    summary = typostats._format_analysis_summary(
        raw_count=0,
        filtered_items=[ItemWithBrokenStr()],
        item_label="item"
    )
    summary_text = strip_ansi("\n".join(summary))
    assert "Shortest item" not in summary_text

def test_format_analysis_summary_robustness_on_levenshtein_failure():
    with patch("typostats.levenshtein_distance", side_effect=Exception("error")):
        summary = typostats._format_analysis_summary(
            raw_count=0,
            filtered_items=[("a", "b")],
            item_label="pair"
        )
    summary_text = strip_ansi("\n".join(summary))
    assert "Min/Max/Avg changes" not in summary_text
