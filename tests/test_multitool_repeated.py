import sys
from pathlib import Path
import pytest
import io
from unittest.mock import patch

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

@pytest.fixture(autouse=True)
def reset_stdin_cache():
    """Reset the global stdin cache before and after each test."""
    multitool._STDIN_CACHE = None
    multitool._STDIN_ENCODING = None
    yield
    multitool._STDIN_CACHE = None
    multitool._STDIN_ENCODING = None

def test_extract_repeated_items_basic(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("the the quick quick brown fox")

    results = list(multitool._extract_repeated_items([str(f)]))
    assert results == [("the the", "the"), ("quick quick", "quick")]

def test_extract_repeated_items_case_insensitive_default(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("The the Quick quick")

    # clean_items=True by default in _extract_repeated_items
    results = list(multitool._extract_repeated_items([str(f)]))
    assert results == [("the the", "the"), ("quick quick", "quick")]

def test_extract_repeated_items_case_sensitive_raw(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("The the Quick Quick")

    # clean_items=False means exact matching
    results = list(multitool._extract_repeated_items([str(f)], clean_items=False))
    # "The" != "the"
    assert ("The the", "the") not in results
    assert ("Quick Quick", "Quick") in results

def test_extract_repeated_items_smart_split(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("doubled doubledWord")

    # Without smart, no repeat
    results = list(multitool._extract_repeated_items([str(f)], smart=False))
    assert results == []

    # With smart, "doubledWord" -> ["doubled", "Word"]
    results = list(multitool._extract_repeated_items([str(f)], smart=True))
    assert ("doubled doubled", "doubled") in results

def test_extract_repeated_items_min_length(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("a a the the")

    # min_length=3 (default)
    results = list(multitool._extract_repeated_items([str(f)], min_length=3))
    assert results == [("the the", "the")]

    # min_length=1
    results = list(multitool._extract_repeated_items([str(f)], min_length=1))
    assert ("a a", "a") in results
    assert ("the the", "the") in results

def test_extract_repeated_items_max_length(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("longword longword short short")

    # max_length=5
    results = list(multitool._extract_repeated_items([str(f)], max_length=5))
    assert results == [("short short", "short")]

def test_extract_repeated_items_empty_match(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("!!! !!! hello hello")

    # "!!!" becomes "" after filter_to_letters
    results = list(multitool._extract_repeated_items([str(f)], clean_items=True))
    assert results == [("hello hello", "hello")]

def test_repeated_mode_basic(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("the the\nthe the")
    out = tmp_path / "out.txt"

    # process_output=False (no deduplication)
    multitool.repeated_mode([str(f)], str(out), 1, 100, False, output_format='line')
    content = out.read_text()
    assert content.count("the the -> the") == 3

def test_repeated_mode_process_output(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("the the\nthe the")
    out = tmp_path / "out.txt"

    # process_output=True (deduplication)
    multitool.repeated_mode([str(f)], str(out), 1, 100, True, output_format='line')
    content = out.read_text()
    assert content.count("the the -> the") == 1

def test_repeated_mode_limit(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("the the quick quick brown brown")
    out = tmp_path / "out.txt"

    multitool.repeated_mode([str(f)], str(out), 1, 100, True, output_format='line', limit=2)
    content = out.read_text().splitlines()
    # Filter out empty lines or headers if any
    data_lines = [l for l in content if "->" in l]
    assert len(data_lines) == 2

def test_repeated_mode_csv(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("the the")
    out = tmp_path / "out.csv"

    multitool.repeated_mode([str(f)], str(out), 1, 100, True, output_format='csv')
    content = out.read_text()
    assert "the the,the" in content

def test_repeated_across_lines(tmp_path):
    f = tmp_path / "repeated.txt"
    f.write_text("the\nthe")
    out = tmp_path / "out.txt"

    multitool.repeated_mode([str(f)], str(out), 1, 100, False, output_format='line')
    assert "the the -> the" in out.read_text()

def test_repeated_mode_stats_output(tmp_path, caplog):
    f = tmp_path / "repeated.txt"
    f.write_text("the the quick quick")
    out = tmp_path / "out.txt"

    with caplog.at_level("INFO"):
        multitool.repeated_mode([str(f)], str(out), 1, 100, False)

    assert "ANALYSIS SUMMARY" in caplog.text
    assert "Total repeated-words encountered:   2" in caplog.text

def test_extract_repeated_items_stdin(reset_stdin_cache):
    input_text = "the the quick quick"
    with patch("sys.stdin", io.StringIO(input_text)):
        results = list(multitool._extract_repeated_items(["-"]))
        assert results == [("the the", "the"), ("quick quick", "quick")]
