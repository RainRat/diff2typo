import sys
import io
import pytest
from multitool import main, _STDIN_CACHE

def test_todo_mode(tmp_path, capsys):
    # Reset STDIN cache for isolation
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.py"
    test_file.write_text("""
# TODO: Implement this feature
# FIXME: Fix this bug
# XXX: Check this
# BUG: This is a bug
# HACK: This is a hack
# Not a todo
/* TODO: multi-line todo */
<!-- FIXME: html todo -->
\"\"\" BUG: docstring todo \"\"\"
""", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "--raw"]
    main()

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")

    expected = [
        "Implement this feature",
        "Fix this bug",
        "Check this",
        "This is a bug",
        "This is a hack",
        "multi-line todo",
        "html todo",
        "docstring todo"
    ]

    for item in expected:
        assert item in output

    assert "Not a todo" not in output

def test_todo_mode_cleaning(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.c"
    test_file.write_text("/* TODO: clean me */", encoding="utf-8")

    # Default cleaning (lowercase, alphanumeric, spaces removed)
    sys.argv = ["multitool.py", "todo", str(test_file)]
    main()

    captured = capsys.readouterr()
    assert "cleanme" in captured.out

def test_todo_mode_case_insensitive(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.txt"
    test_file.write_text("todo: lowercase todo", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "--raw"]
    main()

    captured = capsys.readouterr()
    assert "lowercase todo" in captured.out

def test_todo_mode_pairs_arrow(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "app.py"
    test_file.write_text("""
# TODO: Implement feature
# FIXME: Critical bug
# HACK: Workaround here
""", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "-p", "--raw", "-f", "arrow"]
    main()

    captured = capsys.readouterr()
    output = captured.out

    assert "Location" in output
    assert "Message" in output
    assert "Marker" in output
    assert f"{test_file}:2" in output
    assert "Implement feature" in output
    assert "TODO" in output
    assert f"{test_file}:3" in output
    assert "Critical bug" in output
    assert "FIXME" in output

def test_todo_mode_pairs_json(tmp_path, capsys):
    import json
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "app.py"
    test_file.write_text("# BUG: Fix memory leak", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "-p", "--raw", "-f", "json"]
    main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    key = f"{test_file}:1"
    assert key in data
    assert "Fix memory leak BUG" in data[key]

def test_todo_mode_pairs_md_table(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "app.py"
    test_file.write_text("# XXX: Check performance", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "-p", "--raw", "-f", "md-table"]
    main()

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert "| Location | Message | Marker |" in lines[0]
    assert "Check performance" in lines[2]
    assert "XXX" in lines[2]

def test_todo_mode_pairs_process_output(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "app.py"
    test_file.write_text("""
# TODO: Alpha
# FIXME: Beta
""", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "-p", "-P", "--raw", "-f", "csv"]
    main()

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert len(lines) == 2

def test_todo_mode_marker_filter(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.py"
    test_file.write_text("""
# TODO: Implement this
# FIXME: Fix this
# BUG: Critical bug
# HACK: Fast hack
""", encoding="utf-8")

    # Single marker via -k
    sys.argv = ["multitool.py", "todo", str(test_file), "-k", "BUG", "--raw"]
    main()

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "Critical bug" in output
    assert "Implement this" not in output
    assert "Fix this" not in output
    assert "Fast hack" not in output

    # Multiple comma-separated markers via --marker
    multitool._STDIN_CACHE = None
    sys.argv = ["multitool.py", "todo", str(test_file), "--marker", "BUG,FIXME", "--raw"]
    main()

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "Critical bug" in output
    assert "Fix this" in output
    assert "Implement this" not in output
    assert "Fast hack" not in output

    # Multiple space-separated markers
    multitool._STDIN_CACHE = None
    sys.argv = ["multitool.py", "todo", str(test_file), "--marker", "TODO", "HACK", "--raw"]
    main()

    captured = capsys.readouterr()
    output = captured.out.strip().split("\n")
    assert "Implement this" in output
    assert "Fast hack" in output
    assert "Critical bug" not in output
    assert "Fix this" not in output

def test_todo_mode_pairs_with_marker_filter(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "app.py"
    test_file.write_text("""
# TODO: Normal task
# BUG: Critical flaw
# FIXME: Urgent repair
""", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "-p", "-k", "BUG", "--raw", "-f", "arrow"]
    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Critical flaw" in output
    assert "BUG" in output
    assert "Normal task" not in output
    assert "Urgent repair" not in output

def test_todo_mode_min_max_length_filter(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "app.py"
    test_file.write_text("""
# TODO: short
# TODO: medium length item
# TODO: extremely long description task that exceeds max length filter limit
""", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "-m", "10", "-M", "25", "--raw"]
    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "medium length item" in output
    assert "short" not in output
    assert "extremely long description task that exceeds max length filter limit" not in output
