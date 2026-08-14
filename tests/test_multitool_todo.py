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


def test_todo_mode_pairs_line(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.py"
    test_file.write_text("""
# TODO: implement feature X
# FIXME: bug in parser
""", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "--pairs", "--raw"]
    main()

    captured = capsys.readouterr()
    output = captured.out.strip()
    # Output format is "left -> right Attr"
    # where left = Location, right = Message, Attr = Marker
    assert f"{test_file}:2 -> implement feature X TODO" in output
    assert f"{test_file}:3 -> bug in parser FIXME" in output


def test_todo_mode_pairs_json(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.py"
    test_file.write_text("""
# BUG: broken connection
""", encoding="utf-8")

    sys.argv = ["multitool.py", "todo", str(test_file), "--pairs", "--raw", "--output-format", "json"]
    main()

    captured = capsys.readouterr()
    import json
    data = json.loads(captured.out.strip())
    # JSON paired output maps left to right + Attr
    expected_key = f"{test_file}:2"
    assert expected_key in data
    assert data[expected_key] == "broken connection BUG"


def test_todo_mode_pairs_arrow_format(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.py"
    test_file.write_text("""
# XXX: investigate leak
""", encoding="utf-8")

    # Use arrow format (not default line)
    sys.argv = ["multitool.py", "todo", str(test_file), "--pairs", "--raw", "--output-format", "arrow"]
    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Location" in output
    assert "Message" in output
    assert "Marker" in output
    assert "investigate leak" in output
    assert f"{test_file}:2" in output
    assert "XXX" in output


def test_todo_mode_pairs_process_output(tmp_path, capsys):
    import multitool
    multitool._STDIN_CACHE = None

    test_file = tmp_path / "test.py"
    test_file.write_text("""
# TODO: implement feature B
# TODO: implement feature A
""", encoding="utf-8")

    # Use --process-output to sort the pairs
    sys.argv = ["multitool.py", "todo", str(test_file), "--pairs", "--raw", "--process-output"]
    main()

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    # sorted order of tuples: (location, text, marker)
    # Both are from the same file, line 2 vs line 3.
    # f"{test_file}:2" comes before f"{test_file}:3"
    assert "implement feature B" in lines[0]
    assert "implement feature A" in lines[1]
