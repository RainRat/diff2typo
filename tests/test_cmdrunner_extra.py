import logging
import pytest
import runpy
from unittest.mock import patch
import cmdrunner
from cmdrunner import MinimalFormatter

def test_minimal_formatter_info():
    formatter = MinimalFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="info message",
        args=None,
        exc_info=None
    )
    assert formatter.format(record) == "info message"

def test_minimal_formatter_warning_no_tty():
    formatter = MinimalFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=10,
        msg="warning message",
        args=None,
        exc_info=None
    )
    # Mock sys.stderr.isatty() to return False
    with patch("sys.stderr.isatty", return_value=False):
        assert formatter.format(record) == "WARNING: warning message"

def test_minimal_formatter_error_with_tty():
    with patch("cmdrunner.RED", "\033[31m"), patch("cmdrunner.RESET", "\033[0m"):
        # Patch the class attribute
        with patch.dict(MinimalFormatter.LEVEL_COLORS, {logging.ERROR: "\033[31m"}):
            formatter = MinimalFormatter()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="error message",
                args=None,
                exc_info=None
            )
            # Mock sys.stderr.isatty() to return True
            with patch("sys.stderr.isatty", return_value=True):
                formatted = formatter.format(record)
                assert "\033[31m" in formatted
                assert "ERROR" in formatted
                assert "error message" in formatted

def test_load_config_no_yaml_available():
    with patch("cmdrunner._YAML_AVAILABLE", False):
        with patch("logging.error") as mock_log_error:
            with pytest.raises(SystemExit) as excinfo:
                cmdrunner.load_config("any.yaml")
            assert excinfo.value.code == 1
            mock_log_error.assert_called_with("PyYAML is not installed. Install via 'pip install PyYAML' to use cmdrunner.")

def test_main_block():
    with patch("sys.argv", ["cmdrunner.py", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("cmdrunner", run_name="__main__")
        assert excinfo.value.code == 0
