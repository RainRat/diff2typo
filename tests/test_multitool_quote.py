import sys
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_extract_quote_items(tmp_path):
    """Verify that single and double quoted items are correctly extracted."""
    f = tmp_path / "quotes.txt"
    f.write_text(" 'single' and \"double\" and 'apostrophe\\'s' and \"escaped \\\" quote\" ")

    out = tmp_path / "out.txt"
    multitool.quote_mode([str(f)], str(out), 1, 100, False, clean_items=False)

    results = out.read_text().splitlines()
    assert "single" in results
    assert "double" in results
    assert "apostrophe\\'s" in results
    assert "escaped \\\" quote" in results

def test_extract_quote_items_no_apostrophes(tmp_path):
    """Verify that apostrophes in words are NOT extracted as quotes."""
    f = tmp_path / "quotes.txt"
    f.write_text("don't match this but 'match' this")

    out = tmp_path / "out.txt"
    multitool.quote_mode([str(f)], str(out), 1, 100, False, clean_items=False)

    results = out.read_text().splitlines()
    assert "match" in results
    assert "t" not in results
    assert "don" not in results

def test_extract_quote_items_mixed_and_nested(tmp_path):
    """Verify handling of mixed and nested quotes (nested as literal content)."""
    f = tmp_path / "quotes.txt"
    f.write_text("\"single ' inside double\" and 'double \" inside single' and \"outer 'inner' outer\"")

    out = tmp_path / "out.txt"
    multitool.quote_mode([str(f)], str(out), 1, 100, False, clean_items=False)

    results = out.read_text().splitlines()
    assert "single ' inside double" in results
    assert "double \" inside single" in results
    assert "outer 'inner' outer" in results
