import sys
import importlib
from pathlib import Path
from unittest.mock import patch

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import typostats

def test_generate_report_marker_multi_char(capsys):
    """Verify the [M] marker for multi-character replacements in the report."""
    # counts with a 1-to-2 replacement
    counts = {('m', 'rn'): 5}
    # We need to enable at least one multi-char flag for show_attr to be True
    # quiet=False is needed so that sys.stderr headers are printed if we wanted to check them,
    # but the markers are in stdout.
    typostats.generate_report(counts, allow_1to2=True, quiet=False)
    captured = capsys.readouterr().out
    # Check for [1:2] marker (might be colorized if TTY, but here it shouldn't be as capsys is not a TTY)
    assert "[1:2]" in captured

def test_generate_report_marker_transposition(capsys):
    """Verify the [T] marker for transpositions in the report."""
    counts = {('he', 'eh'): 5}
    typostats.generate_report(counts, allow_transposition=True, quiet=False)
    captured = capsys.readouterr().out
    # Check for [T] marker
    assert "[T]" in captured

def test_tqdm_unavailable_fallback():
    """Verify the fallback logic when tqdm is not installed."""
    # Initial state should be True in this environment if tqdm is installed
    initial_tqdm = typostats._TQDM_AVAILABLE

    try:
        # Mock sys.modules to simulate missing tqdm
        with patch.dict(sys.modules, {'tqdm': None}):
            # Reload the module to trigger the except ImportError block
            importlib.reload(typostats)
            assert typostats._TQDM_AVAILABLE is False
            assert typostats.tqdm is None
    finally:
        # Restore the module state for other tests by reloading after the patch is gone
        importlib.reload(typostats)

    # Verify it's restored
    assert typostats._TQDM_AVAILABLE == initial_tqdm
