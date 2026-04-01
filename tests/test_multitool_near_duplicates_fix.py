import pytest
from unittest.mock import patch
import multitool

def test_near_duplicates_cli_crash_fix(tmp_path):
    words_file = tmp_path / "words.txt"
    words_file.write_text("hello\nhellp\n", encoding="utf-8")

    output_file = tmp_path / "output.txt"

    test_args = [
        "multitool.py",
        "near_duplicates",
        str(words_file),
        "-o", str(output_file),
        "--quiet"
    ]

    with patch("sys.argv", test_args):
        multitool.main()

    assert output_file.exists()
    content = output_file.read_text()
    assert "hello -> hellp" in content or "hellp -> hello" in content
