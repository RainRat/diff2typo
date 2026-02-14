import sys
from pathlib import Path
import logging

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_generate_report_empty(capsys):
    counts = {}
    typostats.generate_report(counts, output_format='arrow', quiet=False)
    captured = capsys.readouterr()
    assert "No replacements found matching the criteria." in captured.err
    assert "Total replacements analyzed: 0" in captured.err

def test_generate_report_summary(capsys):
    counts = {('a', 'b'): 5, ('c', 'd'): 10}
    typostats.generate_report(counts, output_format='arrow', quiet=False)
    captured = capsys.readouterr()
    assert "Total replacements analyzed: 15" in captured.err

def test_minimal_formatter():
    formatter = typostats.MinimalFormatter('%(levelname)s: %(message)s')

    # INFO level should be clean
    record_info = logging.LogRecord("name", logging.INFO, "path", 10, "message", None, None)
    assert formatter.format(record_info) == "message"

    # ERROR level should have prefix
    record_error = logging.LogRecord("name", logging.ERROR, "path", 10, "error message", None, None)
    assert formatter.format(record_error) == "ERROR: error message"
