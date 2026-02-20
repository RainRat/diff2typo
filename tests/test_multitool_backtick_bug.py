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

def test_extract_backtick_items_multi_with_marker(tmp_path):
    """
    Verify that multiple backtick items are extracted when preceded by a marker,
    even if only the first one has the marker.
    """
    f = tmp_path / "backtick.txt"
    # Current behavior: only 'foo' is extracted.
    # Desired behavior: both 'foo' and 'bar' are extracted.
    f.write_text("error: `foo` should be `bar`")

    out = tmp_path / "out.txt"
    multitool.backtick_mode([str(f)], str(out), 1, 100, False, clean_items=False)

    results = out.read_text().splitlines()
    # If this fails, it confirms the bug.
    assert "foo" in results
    assert "bar" in results

def test_extract_backtick_items_with_file_path(tmp_path):
    """
    Verify that backtick items representing file paths BEFORE a marker are ignored,
    but those AFTER a marker are kept.
    """
    f = tmp_path / "backtick.txt"
    f.write_text("`/path/to/file.c`:10: error: `foo` should be `bar`")

    out = tmp_path / "out.txt"
    multitool.backtick_mode([str(f)], str(out), 1, 100, False, clean_items=False)

    results = out.read_text().splitlines()
    assert "/path/to/file.c" not in results
    assert "foo" in results
    assert "bar" in results

def test_extract_backtick_items_no_marker(tmp_path):
    """
    Verify that all backtick items are extracted when NO marker is found.
    (Existing behavior should be preserved).
    """
    f = tmp_path / "backtick.txt"
    f.write_text("just `one` and `two` items")

    out = tmp_path / "out.txt"
    multitool.backtick_mode([str(f)], str(out), 1, 100, False, clean_items=False)

    results = out.read_text().splitlines()
    assert "one" in results
    assert "two" in results

def test_extract_backtick_items_multiple_markers(tmp_path):
    """
    Verify that multiple markers on one line still work correctly.
    """
    f = tmp_path / "backtick.txt"
    f.write_text("error: `first` and note: `second` and `third`")

    out = tmp_path / "out.txt"
    multitool.backtick_mode([str(f)], str(out), 1, 100, False, clean_items=False)

    results = out.read_text().splitlines()
    assert results == ["first", "second", "third"]
