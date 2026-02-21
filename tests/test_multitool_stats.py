import json
import pytest
from multitool import main
import sys
from io import StringIO

def test_stats_items(tmp_path):
    input_file = tmp_path / "items.txt"
    input_file.write_text("apple\nbanana\ncherry\napple\n")

    output_file = tmp_path / "stats.json"

    # Mock sys.argv
    sys.argv = ["multitool.py", "stats", str(input_file), "-o", str(output_file), "-f", "json"]
    main()

    with open(output_file, "r") as f:
        stats = json.load(f)

    assert stats["items"]["total_encountered"] == 4
    assert stats["items"]["unique_count"] == 3
    assert stats["items"]["min_length"] == 5 # apple
    assert stats["items"]["max_length"] == 6 # banana

def test_stats_pairs(tmp_path):
    input_file = tmp_path / "pairs.txt"
    # apple -> apple (overlap)
    # teh -> the (dist 2)
    # teh -> tha (conflict)
    input_file.write_text("apple -> apple\nteh -> the\nteh -> tha\n")

    output_file = tmp_path / "stats.json"

    sys.argv = ["multitool.py", "stats", str(input_file), "-o", str(output_file), "-f", "json", "--pairs"]
    main()

    with open(output_file, "r") as f:
        stats = json.load(f)

    assert stats["pairs"]["total_extracted"] == 3
    assert stats["pairs"]["unique_typos"] == 2 # apple, teh
    assert stats["pairs"]["conflicts"] == 1 # teh
    assert stats["pairs"]["overlaps"] == 1 # apple
    assert stats["pairs"]["min_dist"] == 0 # apple -> apple
    assert stats["pairs"]["max_dist"] == 2 # teh -> the (t-e-h vs t-h-e) -> e replaced by h, h replaced by e (dist 2 in this impl)

def test_stats_human_readable(tmp_path, capsys):
    input_file = tmp_path / "items.txt"
    input_file.write_text("apple\n")

    # Output to stdout
    sys.argv = ["multitool.py", "stats", str(input_file), "--pairs"]
    main()

    captured = capsys.readouterr()
    assert "ANALYSIS STATISTICS" in captured.out
    assert "PAIRED DATA STATISTICS" in captured.out
    # The new formatting uses wider labels and aligned values
    assert "Total items encountered:" in captured.out
    assert "1" in captured.out
