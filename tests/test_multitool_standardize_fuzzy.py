import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_standardize_baseline_casing(tmp_path):
    test_file = tmp_path / "casing.txt"
    test_file.write_text("Database\ndatabase\ndatabase\nDATABASE\n")
    output_file = tmp_path / "output.txt"

    sys.argv = [
        "multitool.py", "standardize", str(test_file),
        "--output", str(output_file),
        "--min-length", "3"
    ]
    multitool.main()
    out = output_file.read_text().splitlines()
    # database (2) vs Database (1) vs DATABASE (1) -> database wins
    assert all(line == "database" for line in out)

def test_standardize_fuzzy_logic(tmp_path):
    test_file = tmp_path / "fuzzy.txt"
    # anchors
    content = "database\n" * 20
    content += "project\n" * 20
    content += "the\n" * 20
    # typos
    content += "databaes\n" # dist 2
    content += "proyect\n"  # dist 1
    content += "teh\n"      # dist 2

    test_file.write_text(content.strip())
    output_file = tmp_path / "output.txt"

    # Fuzzy 1: only proyect should be fixed
    sys.argv = [
        "multitool.py", "standardize", str(test_file),
        "--output", str(output_file),
        "--fuzzy", "1",
        "--min-length", "3"
    ]
    multitool.main()
    out = output_file.read_text()
    assert "proyect" not in out
    assert "project" in out
    assert "databaes" in out # dist 2
    assert "teh" in out      # dist 2

    # Fuzzy 2: all should be fixed
    sys.argv = [
        "multitool.py", "standardize", str(test_file),
        "--output", str(output_file),
        "--fuzzy", "2",
        "--min-length", "3"
    ]
    multitool.main()
    out = output_file.read_text()
    assert "proyect" not in out
    assert "databaes" not in out
    assert "teh" not in out
    assert "database" in out
    assert "project" in out
    assert "the" in out

def test_standardize_threshold_logic(tmp_path):
    test_file = tmp_path / "threshold.txt"
    # apple (10) vs apply (5). Dist 1. Ratio 2.
    content = "apple\n" * 10 + "apply\n" * 5
    test_file.write_text(content.strip())
    output_file = tmp_path / "output.txt"

    # Threshold 10 (default). 10 < 5*10. No fix.
    sys.argv = [
        "multitool.py", "standardize", str(test_file),
        "--output", str(output_file),
        "--fuzzy", "1",
        "--min-length", "3"
    ]
    multitool.main()
    assert "apply" in output_file.read_text()

    # Threshold 2. 10 >= 5*2. Fix.
    sys.argv = [
        "multitool.py", "standardize", str(test_file),
        "--output", str(output_file),
        "--fuzzy", "1",
        "--threshold", "2",
        "--min-length", "3"
    ]
    multitool.main()
    assert "apply" not in output_file.read_text()
    assert "apple" in output_file.read_text()
