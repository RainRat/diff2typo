import sys
from unittest.mock import patch
import typostats


def test_is_file_excluded_include_patterns():
    # Matching include pattern
    assert not typostats._is_file_excluded("docs/test.txt", include_patterns=["*.txt"])
    assert not typostats._is_file_excluded("src/code.py", include_patterns=["*.py", "*.md"])

    # Not matching include pattern -> excluded (returns True)
    assert typostats._is_file_excluded("docs/test.json", include_patterns=["*.txt"])
    assert typostats._is_file_excluded("README.md", include_patterns=["src/*"])


def test_is_file_excluded_combined_include_and_exclude():
    # Matching exclude pattern takes priority
    assert typostats._is_file_excluded(
        "docs/ignore.txt",
        exclude_patterns=["*ignore*"],
        include_patterns=["*.txt"],
    )

    # Passes exclude pattern and matches include pattern -> allowed (returns False)
    assert not typostats._is_file_excluded(
        "docs/report.txt",
        exclude_patterns=["*ignore*"],
        include_patterns=["*.txt"],
    )


def test_typostats_include_cli_directory_scan(tmp_path, monkeypatch, capsys):
    txt_file = tmp_path / "valid.txt"
    txt_file.write_text("teh -> the\n", encoding="utf-8")

    csv_file = tmp_path / "ignored.csv"
    csv_file.write_text("wrod,word\n", encoding="utf-8")

    test_args = ["typostats.py", str(tmp_path), "-I", "*.txt", "--format", "json"]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    captured = capsys.readouterr()
    assert '"typo": "eh"' in captured.out
    assert '"correct": "he"' in captured.out
    # Confirm csv file typos were excluded by -I *.txt
    assert '"typo": "ro"' not in captured.out


def test_typostats_include_dry_run_logging(tmp_path, monkeypatch, caplog):
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("teh -> the\n", encoding="utf-8")

    test_args = ["typostats.py", str(txt_file), "-I", "*.txt", "--dry-run"]
    monkeypatch.setattr(sys, "argv", test_args)

    with caplog.at_level("INFO"):
        typostats.main()

    assert "Include Patterns: ['*.txt']" in caplog.text
    assert "--- TYPOSTATS DRY RUN ---" in caplog.text
