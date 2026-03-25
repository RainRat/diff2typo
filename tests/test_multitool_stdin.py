import sys
import io
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_stdin_default_behavior(monkeypatch, capsys):
    """Test that multitool defaults to reading from stdin when no input file is provided."""

    # Prepare input data
    input_data = "typo,correction\n"
    stdin_file = io.StringIO(input_data)

    # Mock sys.stdin
    monkeypatch.setattr(sys, 'stdin', stdin_file)

    # Mock sys.argv to run 'csv' mode without input arguments
    # We add --quiet to avoid log output clutter, although capsys captures it.
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--quiet', 'csv'])

    # Run main
    multitool.main()

    # Check output
    captured = capsys.readouterr()
    output = captured.out

    # With 'csv' mode and default settings (not first column only),
    # it should extract the second column "correction"
    assert "correction" in output
    assert "typo" not in output

def test_explicit_stdin_flag(monkeypatch, capsys):
    """Test that providing '-' explicitly still works."""

    input_data = "typo,correction\n"
    stdin_file = io.StringIO(input_data)
    monkeypatch.setattr(sys, 'stdin', stdin_file)

    # Explicitly pass '-' as input file
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--quiet', 'csv', '-'])

    multitool.main()

    captured = capsys.readouterr()
    output = captured.out

    assert "correction" in output

def test_stdin_multi_pass(monkeypatch, capsys):
    """Test that stdin can be read multiple times (for example for stats --pairs)."""

    input_data = "apple -> fruit\n"

    # We need to mock sys.stdin.buffer if it exists, or just sys.stdin
    # Some environments (like pytest) might have a custom stdin object.

    class MockStdin:
        def __init__(self, data):
            self.buffer = io.BytesIO(data.encode('utf-8'))
            self.encoding = 'utf-8'
        def read(self):
            return self.buffer.read().decode(self.encoding)

    mock_stdin = MockStdin(input_data)
    monkeypatch.setattr(sys, 'stdin', mock_stdin)

    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'stats', '-', '--pairs', '--min-length', '1'])

    # Reset cache to ensure fresh start for test
    multitool._STDIN_CACHE = None
    multitool._STDIN_ENCODING = None

    multitool.main()

    captured = capsys.readouterr()
    # stats mode logs to stderr by default when writing to stdout
    # but item count and pairs count should be in the report.
    # Actually multitool.stats_mode writes the report to output_file, which defaults to '-' (stdout)

    assert "Total items encountered:            3" in captured.out
    assert "Total pairs extracted:              1" in captured.out
