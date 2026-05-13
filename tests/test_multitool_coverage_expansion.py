import sys
import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_extract_pairs_markdown_table_and_colon(tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text(
        "| typo | correction |\n"
        "| :--- | :--- |\n"
        "|  teh  |  the  |\n"
        "| edge | case |\n"
        "apple: red\n"
    )

    # _extract_pairs is called by pairs_mode
    out = tmp_path / "out.txt"
    multitool.pairs_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False)

    content = out.read_text()
    assert "teh" in content and "the" in content
    assert "edge" in content and "case" in content
    assert "apple" in content and "red" in content
    assert "typo -> correction" not in content

def test_words_mode_smart_split():
    input_text = "CamelCase snake_case"
    with patch("multitool._read_file_lines_robust", return_value=[input_text]):
        mock_outfile = MagicMock()
        with patch("multitool.smart_open_output", return_value=MagicMock(__enter__=lambda s: mock_outfile)):
            multitool.words_mode(["mock"], "out", 1, 100, True, smart=True, clean_items=False)

    written_items = [call.args[0].strip() for call in mock_outfile.write.call_args_list]
    assert "Camel" in written_items
    assert "Case" in written_items
    assert "snake" in written_items
    assert "case" in written_items

def test_stats_mode_markdown_with_pairs(tmp_path):
    f = tmp_path / "input.txt"
    # Added empty lines to cover line 1114 (in multitool.py)
    f.write_text("teh -> the\n\n  \n")
    out = tmp_path / "stats.md"

    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='markdown', clean_items=True)

    content = out.read_text()
    assert "### ANALYSIS SUMMARY" in content
    assert "### PAIRED DATA STATISTICS" in content
    # Simple pair to ensure dist 1
    f.write_text("a -> b\n")
    multitool.stats_mode([str(f)], str(out), min_length=1, max_length=100, process_output=False, include_pairs=True, output_format='markdown', clean_items=True)
    content = out.read_text()
    assert "| Min character changes | 1 |" in content

def test_near_duplicates_mode_optimizations(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("cat\nhat\ncattle\n")
    out = tmp_path / "out.txt"

    multitool.near_duplicates_mode([str(f)], str(out), 1, 100, process_output=True, max_dist=1, show_dist=True)

    content = out.read_text()
    assert "cat" in content and "hat" in content and "[D:1]" in content
    assert "cattle" not in content

def test_fuzzymatch_mode_optimizations(tmp_path):
    f1 = tmp_path / "list1.txt"
    f1.write_text("cat\n")
    f2 = tmp_path / "list2.txt"
    f2.write_text("a\nbat\ndoggy\n")
    out = tmp_path / "out.txt"

    multitool.fuzzymatch_mode([str(f1)], str(f2), str(out), 1, 100, process_output=True, max_dist=1, show_dist=True)

    content = out.read_text()
    assert "cat" in content and "bat" in content and "[D:1]" in content
    assert "doggy" not in content

def test_discovery_mode_optimizations(tmp_path):
    f = tmp_path / "text.txt"
    content = "cat " + "hat "*5 + "bat "*5 + "doggy "*5 + "a "*5
    f.write_text(content)
    out = tmp_path / "out.txt"

    multitool.discovery_mode([str(f)], str(out), 1, 100, process_output=True, freq_min=5, max_dist=1, show_dist=True)

    content = out.read_text()
    assert "cat" in content and "bat" in content and "[D:1]" in content
    assert "cat" in content and "hat" in content and "[D:1]" in content
    assert "doggy" not in content
    # Check that 'a' is not a typo/correction pair on its own row
    assert not any(line.strip().startswith("a  │") or line.strip().endswith("│ a") for line in content.splitlines())

def test_md_table_mode_columns(tmp_path):
    f = tmp_path / "table.md"
    f.write_text(
        "| col0 | col1 | col2 |\n"
        "| --- | --- | --- |\n"
        "| val0 | val1 | val2 |\n"
    )
    out = tmp_path / "out.txt"
    # Test --column option to cover lines 775-777
    # Use clean_items=False to preserve 'val0'
    multitool.md_table_mode([str(f)], str(out), 1, 100, True, columns=[0, 2], clean_items=False)
    content = out.read_text().splitlines()
    assert "val0" in content
    assert "val2" in content
    assert "val1" not in content

def test_write_output_yaml_no_module(tmp_path, monkeypatch):
    # Mock ImportError for yaml to cover lines 324-326
    import builtins
    real_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)

    items = ["apple", "banana"]
    out = tmp_path / "out.yaml"
    multitool.write_output(items, str(out), output_format='yaml')
    content = out.read_text()
    assert "- apple\n- banana\n" in content

def test_standardize_no_changes_inplace(tmp_path, caplog):
    # Setup test file with consistent casing
    content = "word word word"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)

    with caplog.at_level(logging.INFO):
        multitool.standardize_mode(
            input_files=[str(test_file)],
            output_file="-",
            min_length=1,
            max_length=100,
            process_output=False,
            quiet=False,
            clean_items=True,
            in_place="",
        )

    assert "No inconsistencies found" in caplog.text
    assert test_file.read_text() == content

def test_standardize_stdin_inplace_warning(caplog):
    # To reach the stdin warning, we need some inconsistent casing first
    with caplog.at_level(logging.WARNING):
        # Pass 1 returns these lines to build the mapping
        # Pass 3 returns these lines to be processed
        with patch("multitool._read_file_lines_robust", side_effect=[
            ["word\n", "WORD\n"], # Pass 1
            ["word\n", "WORD\n"]  # Pass 3
        ]):
            multitool.standardize_mode(
                input_files=["-"],
                output_file="-",
                min_length=1,
                max_length=100,
                process_output=False,
                quiet=False,
                clean_items=True,
                in_place="",
            )

    assert "In-place modification requested for standard input; ignoring." in caplog.text

def test_standardize_inplace_with_backup(tmp_path):
    # word (2), WORD (1) -> word wins
    content = "word word WORD"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)

    multitool.standardize_mode(
        input_files=[str(test_file)],
        output_file="-",
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        in_place=".bak",
    )

    assert test_file.read_text().strip() == "word word word"
    backup_file = tmp_path / "test.txt.bak"
    assert backup_file.exists()
    assert backup_file.read_text() == content

def test_standardize_inplace_write_error(tmp_path, caplog):
    content = "word word WORD"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)

    # Pass 1: read file
    # Pass 3: read file, then open for write
    with patch("builtins.open", side_effect=[
        open(test_file, "r"), # Pass 1 read
        open(test_file, "r"), # Pass 3 read
        OSError("Disk full")  # Pass 3 write
    ]):
        with pytest.raises(SystemExit) as excinfo:
            multitool.standardize_mode(
                input_files=[str(test_file)],
                output_file="-",
                min_length=1,
                max_length=100,
                process_output=False,
                quiet=True,
                clean_items=True,
                in_place="",
            )
        assert excinfo.value.code == 1

    assert "Failed to write to" in caplog.text

def test_standardize_limit(tmp_path):
    # word (2), WORD (1) -> word wins
    # Limit to 1 line
    content = "word word WORD\nAnother line WORD"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)
    output_file = tmp_path / "output.txt"

    multitool.standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        limit=1
    )

    result_lines = output_file.read_text().strip().splitlines()
    assert len(result_lines) == 1
    assert result_lines[0] == "word word word"

def test_standardize_dry_run_no_inplace(tmp_path, caplog):
    content = "word word WORD"
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)
    output_file = tmp_path / "output.txt"

    with caplog.at_level(logging.WARNING):
        multitool.standardize_mode(
            input_files=[str(test_file)],
            output_file=str(output_file),
            min_length=1,
            max_length=100,
            process_output=False,
            quiet=False,
            clean_items=True,
            dry_run=True
        )

    assert "[Dry Run] Total replacements that would be made: 1" in caplog.text
    # Output file should NOT exist because it's dry run
    assert not output_file.exists()

# CLI Fallback Tests
def test_zip_fallback_single_arg(tmp_path):
    # zip requires file2. If only one arg given, it should be used as file2, and input from stdin.
    file2 = tmp_path / "file2.txt"
    file2.write_text("the")

    with patch("sys.argv", ["multitool.py", "zip", str(file2), "-f", "csv"]):
        with patch("sys.stdin", io.StringIO("teh\n")):
            with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                # Reset stdin cache if it was used in other tests
                multitool._STDIN_CACHE = None
                multitool.main()
                assert "teh,the" in mock_stdout.getvalue()

def test_zip_fallback_multi_arg(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("teh")
    file2 = tmp_path / "file2.txt"
    file2.write_text("the")

    with patch("sys.argv", ["multitool.py", "zip", str(file1), str(file2), "-f", "csv"]):
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            multitool._STDIN_CACHE = None
            multitool.main()
            assert "teh,the" in mock_stdout.getvalue()

def test_map_fallback_single_arg(tmp_path):
    # map requires mapping. If only one arg given, it should be used as mapping, and input from stdin.
    mapping = tmp_path / "mapping.txt"
    mapping.write_text("teh -> the")

    with patch("sys.argv", ["multitool.py", "map", str(mapping)]):
        with patch("sys.stdin", io.StringIO("teh\n")):
            with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                multitool._STDIN_CACHE = None
                multitool.main()
                assert "the" in mock_stdout.getvalue()

def test_map_fallback_multi_arg(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh")
    mapping = tmp_path / "mapping.txt"
    mapping.write_text("teh -> the")

    with patch("sys.argv", ["multitool.py", "map", str(input_file), str(mapping)]):
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            multitool._STDIN_CACHE = None
            multitool.main()
            assert "the" in mock_stdout.getvalue()

def test_search_fallback_single_arg():
    # search requires query. If only one arg given, it should be used as query, and input from stdin.
    with patch("sys.argv", ["multitool.py", "search", "teh"]):
        with patch("sys.stdin", io.StringIO("teh line\n")):
            with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                multitool._STDIN_CACHE = None
                multitool.main()
                assert "teh line" in mock_stdout.getvalue()

def test_min_length_error():
    with patch("sys.argv", ["multitool.py", "--min-length", "0", "search"]):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()
        assert excinfo.value.code == 1

def test_max_length_error():
    with patch("sys.argv", ["multitool.py", "--min-length", "10", "--max-length", "5", "search"]):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()
        assert excinfo.value.code == 1
