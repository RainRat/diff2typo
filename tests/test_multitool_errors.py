import sys
import logging
import pytest
from unittest.mock import patch
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_main_min_length_error(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "file.txt", "--min-length", "0"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "--min-length must be a number of 1 or more" in caplog.text

def test_main_max_length_error(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "file.txt", "--min-length", "5", "--max-length", "3"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "--max-length must be greater than or equal to --min-length" in caplog.text

def test_main_zip_missing_file2(monkeypatch, caplog):
    # Now it should error ONLY if NO positional arguments are provided
    monkeypatch.setattr(sys, "argv", ["multitool.py", "zip"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "Zip mode requires a secondary file" in caplog.text

def test_main_map_missing_mapping(monkeypatch, caplog):
    # Now it should error ONLY if NO positional arguments are provided
    monkeypatch.setattr(sys, "argv", ["multitool.py", "map"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "Map mode requires a mapping file" in caplog.text

def test_main_filenotfound_with_filename(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "nonexistent.txt"])
    # Mock words_mode to raise FileNotFoundError with a filename
    def mock_words_mode(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "nonexistent.txt")

    with patch("multitool.words_mode", side_effect=mock_words_mode):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as cm:
                multitool.main()
    assert cm.value.code == 1
    assert "File not found: 'nonexistent.txt'" in caplog.text

def test_main_filenotfound_without_filename(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "nonexistent.txt"])
    # Mock words_mode to raise FileNotFoundError without a filename
    def mock_words_mode(*args, **kwargs):
        raise FileNotFoundError("Generic error")

    with patch("multitool.words_mode", side_effect=mock_words_mode):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as cm:
                multitool.main()
    assert cm.value.code == 1
    assert "File not found: Generic error" in caplog.text

def test_main_unexpected_exception(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "file.txt"])
    with patch("multitool.words_mode", side_effect=RuntimeError("Unexpected!")):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as cm:
                multitool.main()
    assert cm.value.code == 1
    assert "An unexpected error occurred: Unexpected!" in caplog.text

def test_main_zip_fallback(monkeypatch):
    # If we have 2 positional args and no --file2, the second one should become file2
    monkeypatch.setattr(sys, "argv", ["multitool.py", "zip", "file1.txt", "file2.txt"])

    with patch("multitool.zip_mode") as mock_zip:
        multitool.main()

    mock_zip.assert_called_once()
    args, kwargs = mock_zip.call_args
    assert kwargs['input_files'] == ['file1.txt']
    assert kwargs['file2'] == 'file2.txt'

def test_main_map_fallback(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["multitool.py", "map", "mapping.txt", "file1.txt"])

    with patch("multitool.map_mode") as mock_map:
        multitool.main()

    mock_map.assert_called_once()
    args, kwargs = mock_map.call_args
    assert kwargs['input_files'] == ['file1.txt']
    assert kwargs['mapping_file'] == 'mapping.txt'
