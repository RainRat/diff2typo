import sys
from io import StringIO
from unittest.mock import patch
import multitool

def test_align_mode_direct_basic(tmp_path):
    """Verify align mode with default separator using direct main() call."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("teh,the\nabcde,abc\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(input_file)]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    expected = "teh   -> the\nabcde -> abc\n"
    assert fake_out.getvalue() == expected

def test_align_mode_direct_custom_separator(tmp_path):
    """Verify align mode with custom separator via --sep."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("teh,the\nabcde,abc\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(input_file), "--sep", " | "]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    expected = "teh   | the\nabcde | abc\n"
    assert fake_out.getvalue() == expected

def test_align_mode_direct_cleaning_and_empty(tmp_path):
    """Verify cleaning and skipping empty sides (branch 3476-3477)."""
    input_file = tmp_path / "typos.csv"
    # 'teh1' becomes 'teh', '123' becomes empty and should be skipped
    input_file.write_text("teh1,the\n123,abc\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(input_file)]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    expected = "teh -> the\n"
    assert fake_out.getvalue() == expected

def test_align_mode_direct_raw(tmp_path):
    """Verify --raw flag disables cleaning."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("teh1,the\n123,abc\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(input_file), "--raw"]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    # With --raw, 'teh1' and '123' are preserved
    expected = "teh1 -> the\n123  -> abc\n"
    assert fake_out.getvalue() == expected

def test_align_mode_direct_length_filtering(tmp_path):
    """Verify length filtering (branch 3480-3481)."""
    input_file = tmp_path / "typos.csv"
    # apple(5), banana(6), cherry(6), date(4)
    input_file.write_text("apple,banana\ncherry,date\n", encoding="utf-8")

    # Min length 5, max length 5 -> only apple should pass (but banana is 6, so pair fails)
    # Wait, the logic is: min_length <= len(left) <= max_length AND min_length <= len(right) <= max_length

    with patch("sys.argv", ["multitool.py", "align", str(input_file), "--min-length", "5", "--max-length", "6"]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    # apple(5)->banana(6) passes because 5 and 6 are within [5, 6]
    # cherry(6)->date(4) fails because 4 < 5
    expected = "apple -> banana\n"
    assert fake_out.getvalue() == expected

def test_align_mode_direct_process_output(tmp_path):
    """Verify --process-output (-P) sorting and deduplication."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("banana,fruit\napple,fruit\nbanana,fruit\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(input_file), "-P"]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    # Should be sorted: apple then banana, and deduplicated
    expected = "apple  -> fruit\nbanana -> fruit\n"
    assert fake_out.getvalue() == expected

def test_align_mode_direct_limit(tmp_path):
    """Verify --limit (-L) restricts output."""
    input_file = tmp_path / "typos.csv"
    input_file.write_text("a,b\nc,d\ne,f\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(input_file), "-L", "2"]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    # Should only show first 2 pairs
    expected = "a -> b\nc -> d\n"
    assert fake_out.getvalue() == expected

def test_align_mode_multi_file(tmp_path):
    """Verify alignment across multiple input files."""
    file1 = tmp_path / "1.csv"
    file1.write_text("short,s\n", encoding="utf-8")
    file2 = tmp_path / "2.csv"
    file2.write_text("verylong,l\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(file1), str(file2)]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    # Max width of left is 'verylong' (8)
    expected = "short    -> s\nverylong -> l\n"
    assert fake_out.getvalue() == expected

def test_align_mode_arrow_input(tmp_path):
    """Verify align mode works with arrow format input via _extract_pairs."""
    input_file = tmp_path / "typos.txt"
    input_file.write_text("teh -> the\n", encoding="utf-8")

    with patch("sys.argv", ["multitool.py", "align", str(input_file)]), \
         patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            multitool.main()
        except SystemExit:
            pass

    assert "teh -> the" in fake_out.getvalue()
