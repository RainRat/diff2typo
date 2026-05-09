import json
import io
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path
import sys

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_diff_simple_items(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    out = tmp_path / "out.txt"

    file1.write_text("apple\nbanana\ncherry\n")
    file2.write_text("apple\ndate\ncherry\n")

    multitool.diff_mode([str(file1)], str(file2), str(out), 3, 1000, False)

    content = out.read_text()
    assert "- banana" in content
    assert "+ date" in content
    assert "apple" not in content

def test_diff_pairs(tmp_path):
    file1 = tmp_path / "pairs1.txt"
    file2 = tmp_path / "pairs2.txt"
    out = tmp_path / "out.txt"

    file1.write_text("teh -> the\nwierd -> weird\n")
    file2.write_text("teh -> the\nwierd -> wired\nnew -> newer\n")

    multitool.diff_mode([str(file1)], str(file2), str(out), 3, 1000, False, pairs=True)

    content = out.read_text()
    assert "+ new -> newer" in content
    assert "~ wierd: weird -> wired" in content
    assert "teh" not in content

def test_diff_json_output(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    out = tmp_path / "out.json"

    file1.write_text("apple\nbanana\n")
    file2.write_text("apple\ndate\n")

    multitool.diff_mode([str(file1)], str(file2), str(out), 3, 1000, False, output_format="json")

    data = json.loads(out.read_text())
    assert data["added"] == ["date"]
    assert data["removed"] == ["banana"]

def test_diff_pairs_json_output(tmp_path):
    file1 = tmp_path / "pairs1.txt"
    file2 = tmp_path / "pairs2.txt"
    out = tmp_path / "out.json"

    file1.write_text("teh -> the\nwierd -> weird\n")
    file2.write_text("teh -> the\nwierd -> wired\nnew -> newer\n")

    multitool.diff_mode([str(file1)], str(file2), str(out), 3, 1000, False, pairs=True, output_format="json")

    data = json.loads(out.read_text())
    assert data["added"] == {"new": "newer"}
    assert data["removed"] == {}
    assert data["changed"] == {"wierd": "weird -> wired"}

def test_diff_no_file2_error(caplog):
    # Testing the error logic in main() by bypassing actual sys.exit
    with patch("sys.argv", ["multitool.py", "diff"]):
        with pytest.raises(SystemExit) as e:
            multitool.main()
        assert e.value.code != 0
    assert "requires a secondary file" in caplog.text

def test_diff_multiple_inputs(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file3 = tmp_path / "file3.txt"
    out = tmp_path / "out.txt"

    file1.write_text("apple\n")
    file2.write_text("banana\n")
    file3.write_text("apple\ncherry\n")

    # diff file1 and file2 (left) vs file3 (right)
    # left = {apple, banana}, right = {apple, cherry}
    # removed: banana, added: cherry
    multitool.diff_mode([str(file1), str(file2)], str(file3), str(out), 3, 1000, False)

    content = out.read_text()
    assert "- banana" in content
    assert "+ cherry" in content
    assert "apple" not in content

def test_diff_color_output(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("apple\n")
    file2.write_text("banana\n")

    # Mock sys.stdout.isatty to True to force colors
    with patch("sys.stdout.isatty", return_value=True):
        # We also need to ensure RED/GREEN constants are set
        with patch("multitool.RED", "\033[1;31m"), \
             patch("multitool.GREEN", "\033[1;32m"), \
             patch("multitool.RESET", "\033[0m"):

            # In diff_mode: c_red = RED if out.isatty() else ""
            # so we need to mock the returned object's isatty
            mock_stdout = io.StringIO()
            mock_stdout.isatty = lambda: True

            with patch("multitool.smart_open_output") as mock_open:
                mock_open.return_value.__enter__.return_value = mock_stdout
                multitool.diff_mode([str(file1)], str(file2), "-", 3, 1000, False)

                content = mock_stdout.getvalue()
                # Check for ANSI red for removed and green for added
                assert "\033[1;31m- apple\033[0m" in content
                assert "\033[1;32m+ banana\033[0m" in content

def test_diff_limit_and_removal(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    out = tmp_path / "out.txt"

    file1.write_text("apple\nbanana\ncherry\n")
    file2.write_text("apple\n")

    # removed: banana, cherry. limit: 1
    multitool.diff_mode([str(file1)], str(file2), str(out), 3, 1000, False, limit=1)

    content = out.read_text().splitlines()
    assert len(content) == 1
    assert content[0] == "- banana"

def test_diff_pairs_removal(tmp_path):
    file1 = tmp_path / "pairs1.txt"
    file2 = tmp_path / "pairs2.txt"
    out = tmp_path / "out.txt"

    file1.write_text("teh -> the\nwierd -> weird\n")
    file2.write_text("teh -> the\n")

    # removed: wierd -> weird
    multitool.diff_mode([str(file1)], str(file2), str(out), 3, 1000, False, pairs=True)

    content = out.read_text()
    assert "- wierd -> weird" in content
