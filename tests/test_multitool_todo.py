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
