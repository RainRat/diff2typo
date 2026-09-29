from unittest.mock import patch
import typostats


def test_typostats_dry_run_basic(tmp_path, caplog):
    typo_file = tmp_path / "typos.txt"
    typo_file.write_text("teh -> the\nrecieve -> receive\n", encoding="utf-8")
    out_file = tmp_path / "output.json"

    with patch(
        "sys.argv",
        [
            "typostats.py",
            str(typo_file),
            "-o",
            str(out_file),
            "--dry-run",
            "-t",
            "-k",
        ],
    ):
        with patch("sys.stderr.isatty", return_value=True), caplog.at_level("INFO"):
            typostats.main()

    # Verify no output file was created
    assert not out_file.exists()

    # Verify dry run log header and execution settings
    log_text = caplog.text
    assert "--- TYPOSTATS DRY RUN ---" in log_text
    assert str(typo_file) in log_text
    assert str(out_file) in log_text
    assert "keyboard" in log_text
    assert "transposition" in log_text
    assert "Dry run complete. No files were written or exported." in log_text
    assert "Found 2 pattern candidate(s)" in log_text or "Found 1 pattern candidate(s)" in log_text


def test_typostats_dry_run_directory_input(tmp_path, caplog):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    f1 = subdir / "typos1.txt"
    f1.write_text("teh -> the\n", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "typostats.py",
            str(subdir),
            "--dry-run",
            "--1to2",
            "--2to1",
            "--include-deletions",
        ],
    ):
        with caplog.at_level("INFO"):
            typostats.main()

    log_text = caplog.text
    assert "--- TYPOSTATS DRY RUN ---" in log_text
    assert "1to2" in log_text
    assert "2to1" in log_text
    assert "deletions" in log_text
    assert "Dry run complete. No files were written or exported." in log_text


def test_typostats_dry_run_empty_input(tmp_path, caplog):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "typostats.py",
            str(empty_file),
            "--dry-run",
        ],
    ):
        with caplog.at_level("INFO"):
            typostats.main()

    log_text = caplog.text
    assert "--- TYPOSTATS DRY RUN ---" in log_text
    assert "(No typo patterns found in input)" in log_text
    assert "Dry run complete. No files were written or exported." in log_text


def test_typostats_dry_run_short_flag(tmp_path, caplog):
    typo_file = tmp_path / "typos.txt"
    typo_file.write_text("teh -> the\n", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "typostats.py",
            str(typo_file),
            "-n",
        ],
    ):
        with caplog.at_level("INFO"):
            typostats.main()

    log_text = caplog.text
    assert "--- TYPOSTATS DRY RUN ---" in log_text
    assert "Dry run complete. No files were written or exported." in log_text


def test_typostats_dry_run_sort_options(tmp_path, caplog):
    typo_file = tmp_path / "typos.txt"
    typo_file.write_text("b -> x\na -> y\n", encoding="utf-8")

    # Test sorting by typo (alphabetical by typo_char: 'a' before 'b')
    with patch("sys.argv", ["typostats.py", str(typo_file), "-n", "-s", "typo"]):
        caplog.clear()
        with caplog.at_level("INFO"):
            typostats.main()
        log_text = caplog.text
        pos_b = log_text.find("'b' -> 'x'")
        pos_a = log_text.find("'a' -> 'y'")
        assert pos_b != -1 and pos_a != -1
        assert pos_a < pos_b

    # Test sorting by typo reverse ('b' before 'a')
    with patch("sys.argv", ["typostats.py", str(typo_file), "-n", "-s", "typo", "-r"]):
        caplog.clear()
        with caplog.at_level("INFO"):
            typostats.main()
        log_text = caplog.text
        pos_b = log_text.find("'b' -> 'x'")
        pos_a = log_text.find("'a' -> 'y'")
        assert pos_b != -1 and pos_a != -1
        assert pos_b < pos_a

    # Test sorting by correct (alphabetical by correct_char: 'x' before 'y')
    with patch("sys.argv", ["typostats.py", str(typo_file), "-n", "-s", "correct"]):
        caplog.clear()
        with caplog.at_level("INFO"):
            typostats.main()
        log_text = caplog.text
        pos_b = log_text.find("'b' -> 'x'")
        pos_a = log_text.find("'a' -> 'y'")
        assert pos_b != -1 and pos_a != -1
        assert pos_b < pos_a

    # Test sorting by correct reverse ('y' before 'x')
    with patch("sys.argv", ["typostats.py", str(typo_file), "-n", "-s", "correct", "-r"]):
        caplog.clear()
        with caplog.at_level("INFO"):
            typostats.main()
        log_text = caplog.text
        pos_b = log_text.find("'b' -> 'x'")
        pos_a = log_text.find("'a' -> 'y'")
        assert pos_b != -1 and pos_a != -1
        assert pos_a < pos_b
