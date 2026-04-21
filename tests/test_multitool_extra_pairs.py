import sys
from io import StringIO
import multitool

def test_map_extra(tmp_path):
    input_file = tmp_path / "input.txt"
    # map mode works line-by-line (whole line must match)
    # Default min-length is 3, so 'on' (length 2) is filtered out
    input_file.write_text("teh\ncat\nsat\nthe\nmat")

    # Test map mode with extra pairs
    sys.argv = ['multitool.py', 'map', str(input_file), '--add', 'teh:the']

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        multitool.main()
    finally:
        sys.stdout = sys.__stdout__

    # map mode output
    output = captured_output.getvalue()
    assert "the\ncat\nsat\nthe\nmat" in output

def test_scrub_extra(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh cat sat on teh mat.")

    # Test scrub mode with extra pairs
    sys.argv = ['multitool.py', 'scrub', str(input_file), '--add', 'teh:the']

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        multitool.main()
    finally:
        sys.stdout = sys.__stdout__

    # scrub mode outputs fixed lines
    assert "the cat sat on the mat." in captured_output.getvalue()

def test_highlight_extra(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh cat")

    # Mock colors to ensure predictable output
    monkeypatch.setattr(multitool, "YELLOW", "[Y]")
    monkeypatch.setattr(multitool, "RESET", "[R]")

    sys.argv = ['multitool.py', 'highlight', str(input_file), '--add', 'teh:the']

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        multitool.main()
    finally:
        sys.stdout = sys.__stdout__

    assert "[Y]teh[R] cat" in captured_output.getvalue()

def test_scan_extra(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh cat")

    # Mock colors to ensure predictable output
    monkeypatch.setattr(multitool, "YELLOW", "[Y]")
    monkeypatch.setattr(multitool, "RESET", "[R]")
    monkeypatch.setattr(multitool, "BOLD", "[B]")
    monkeypatch.setattr(multitool, "BLUE", "[C]")
    monkeypatch.setenv("FORCE_COLOR", "1")

    # Scan with line numbers and extra
    sys.argv = ['multitool.py', 'scan', str(input_file), '--add', 'teh:the', '--line-numbers']

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        multitool.main()
    finally:
        sys.stdout = sys.__stdout__

    # Output should include line number and highlighted match
    # Since only 1 file, filename might not be shown by default unless forced or multiple files
    assert "[B][C]1:[R] [Y]teh[R] cat" in captured_output.getvalue()

def test_map_file_and_extra(tmp_path):
    mapping_file = tmp_path / "map.csv"
    mapping_file.write_text("cat,feline")

    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\ncat")

    # Combine file and extra
    sys.argv = ['multitool.py', 'map', str(input_file), '--mapping', str(mapping_file), '--add', 'teh:the']

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        multitool.main()
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "the\nfeline" in output

def test_rename_extra(tmp_path):
    # Create a file to rename
    d = tmp_path / "subdir"
    d.mkdir()
    f = d / "teh_file.txt"
    f.write_text("content")

    # Test rename mode with extra pairs (dry-run first to avoid side effects if not needed)
    # We use --in-place to actually rename
    sys.argv = ['multitool.py', 'rename', str(f), '--add', 'teh:the', '--in-place']

    # rename_mode logs to logging.info, which we usually see on stderr in main()
    # but here it might depend on how logging is configured in the test environment.

    multitool.main()

    assert not f.exists()
    assert (d / "the_file.txt").exists()
