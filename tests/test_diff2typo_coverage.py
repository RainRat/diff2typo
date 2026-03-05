import io
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_minimal_formatter_info(caplog):
    formatter = diff2typo.MinimalFormatter()
    record = logging.LogRecord("name", logging.INFO, "path", 10, "Info message", None, None)
    assert formatter.format(record) == "Info message"

def test_minimal_formatter_warning_no_tty():
    formatter = diff2typo.MinimalFormatter()
    record = logging.LogRecord("name", logging.WARNING, "path", 10, "Warning message", None, None)
    # Mock sys.stderr.isatty to False
    with patch("sys.stderr.isatty", return_value=False):
        assert formatter.format(record) == "WARNING: Warning message"

def test_minimal_formatter_error_tty():
    # Use a subclass that ensures color constants are NOT empty for testing
    class TestFormatter(diff2typo.MinimalFormatter):
        LEVEL_COLORS = {
            logging.WARNING: "\033[1;33m",
            logging.ERROR: "\033[1;31m",
            logging.CRITICAL: "\033[1;31m",
        }

    formatter = TestFormatter()
    record = logging.LogRecord("name", logging.ERROR, "path", 10, "Error message", None, None)
    # Mock sys.stderr.isatty to True
    with patch("sys.stderr.isatty", return_value=True), \
         patch("diff2typo.RESET", "\033[0m"):
        formatted = formatter.format(record)
        assert "\033[1;31mERROR\033[0m" in formatted
        assert "Error message" in formatted

def test_read_csv_rows_required_missing(tmp_path):
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(SystemExit) as excinfo:
        diff2typo._read_csv_rows(str(missing_file), "Test file", required=True)
    assert excinfo.value.code == 1

def test_split_into_subwords_no_match():
    assert diff2typo.split_into_subwords("!!!") == ["!!!"]
    assert diff2typo.split_into_subwords(" ") == ["", ""]

def test_format_typos_malformed():
    assert diff2typo.format_typos(["eror"], "arrow") == ["eror"]
    assert diff2typo.format_typos(["eror123"], "arrow") == ["eror"]

def test_decode_with_fallback_latin1(caplog):
    latin1_data = "héllo".encode("latin-1")
    with caplog.at_level(logging.INFO):
        text = diff2typo._decode_with_fallback(latin1_data, "test data")
        assert text == "héllo"
        assert "with 'latin-1' encoding" in caplog.text

def test_read_stdin_text_str(monkeypatch, caplog):
    mock_stdin = MagicMock()
    mock_stdin.read.return_value = "text input"
    monkeypatch.setattr(sys, "stdin", mock_stdin)
    del sys.stdin.buffer

    with caplog.at_level(logging.INFO):
        text = diff2typo._read_stdin_text()
        assert text == "text input"
        assert "Successfully read input diff from standard input" in caplog.text

def test_read_diff_file_not_found():
    with pytest.raises(SystemExit) as excinfo:
        diff2typo._read_diff_file("missing.diff")
    assert excinfo.value.code == 1

def test_read_diff_sources_no_files(monkeypatch):
    monkeypatch.setattr(diff2typo, "_read_stdin_text", lambda: "stdin diff")
    assert diff2typo._read_diff_sources([]) == "stdin diff"

def test_read_diff_sources_not_a_file(tmp_path):
    (tmp_path / "subdir").mkdir()
    with pytest.raises(SystemExit) as excinfo:
        diff2typo._read_diff_sources([str(tmp_path / "subdir")])
    assert excinfo.value.code == 1

def test_filter_known_typos_write_fail(caplog):
    with patch("builtins.open", side_effect=OSError("write error")):
        with caplog.at_level(logging.ERROR):
            result = diff2typo.filter_known_typos(["eror -> error"], "typos")
            assert result == ["eror -> error"]
            assert "Error writing to temporary file" in caplog.text

def test_filter_known_typos_subprocess_fail(caplog, tmp_path):
    typos_tool = tmp_path / "typos"
    typos_tool.touch()
    with patch("subprocess.run", side_effect=FileNotFoundError("tool not found")):
        with caplog.at_level(logging.WARNING):
            result = diff2typo.filter_known_typos(["eror -> error"], str(typos_tool))
            assert result == ["eror -> error"]
            assert "Error running typos tool" in caplog.text

def test_filter_candidates_by_set_quiet_false(caplog):
    with caplog.at_level(logging.INFO):
        diff2typo._filter_candidates_by_set(["a -> b"], {"a"}, "Test", quiet=False)
        assert "Excluded 1 typo(s) based on test" in caplog.text

def test_process_new_corrections_empty_mapping(caplog):
    with caplog.at_level(logging.INFO):
        result = diff2typo.process_new_corrections(["a -> b"], {}, quiet=True)
        assert result == []
        assert "Dictionary mapping is empty" in caplog.text

def test_process_new_corrections_quiet_false():
    diff2typo.process_new_corrections(["a -> b"], {"a": {"c"}}, quiet=False)

def test_main_both_modes(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    diff_file = tmp_path / "diff.txt"
    diff_file.write_text("--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n-eror line\n+error line\n-teh line\n+thee line\n")

    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("teh,the\n")

    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        "--input", str(diff_file),
        "--output", str(output_file),
        "--dictionary", str(dictionary_file),
        "--mode", "both",
        "--allowed", "nonexistent.csv",
        "--quiet"
    ])
    # Mock subprocess.run in filter_known_typos to return empty result
    def mock_run(*args, **kwargs):
        return MagicMock(stdout="")
    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)

    with caplog.at_level(logging.INFO):
        diff2typo.main()

    content = output_file.read_text()
    assert "=== New Typos ===" in content
    assert "eror -> error" in content
    assert "=== New Corrections ===" in content
    assert "teh -> thee" in content
    # teh is already in dictionary, so it's not a NEW typo, but it could be a NEW correction if it had a different fix
    # Here teh -> the is already in dictionary, so it's not a new correction either.

def test_main_corrections_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    diff_file = tmp_path / "diff.txt"
    diff_file.write_text("--- a/f\n+++ b/f\n@@\n-teh\n+thee\n")

    dictionary_file = tmp_path / "words.csv"
    dictionary_file.write_text("teh,the\n")

    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, "argv", [
        "diff2typo.py",
        "--input", str(diff_file),
        "--output", str(output_file),
        "--dictionary", str(dictionary_file),
        "--mode", "corrections",
        "--quiet"
    ])

    diff2typo.main()
    assert output_file.read_text().strip() == "teh -> thee"

def test_main_allowed_read_fail(tmp_path, monkeypatch):
    # Covers line 644 where read_allowed_words fails
    monkeypatch.chdir(tmp_path)
    # Trigger Exception in read_allowed_words
    with patch("diff2typo.read_allowed_words", side_effect=OSError("read fail")):
        monkeypatch.setattr(sys, "argv", ["diff2typo.py", "--quiet"])
        monkeypatch.setattr(diff2typo, "_read_diff_sources", lambda _: "")
        with pytest.raises(SystemExit) as excinfo:
            diff2typo.main()
        assert excinfo.value.code == 1

def test_main_output_write_fail(tmp_path, monkeypatch):
    # Covers line 705 where smart_open_output or writing fails
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(diff2typo, "_read_diff_sources", lambda _: "")
    monkeypatch.setattr(diff2typo, "read_words_mapping", lambda *_, **__: {})
    monkeypatch.setattr(diff2typo, "read_allowed_words", lambda _: set())

    with patch("diff2typo.smart_open_output", side_effect=OSError("write fail")):
        monkeypatch.setattr(sys, "argv", ["diff2typo.py", "--output", "readonly.txt", "--quiet"])
        with pytest.raises(SystemExit) as excinfo:
            diff2typo.main()
        assert excinfo.value.code == 1
