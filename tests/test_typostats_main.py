
import sys
import runpy
from unittest.mock import patch

def test_typostats_main(tmp_path):
    # We need to provide arguments that won't make it fail or take too long
    f1 = tmp_path / "left.txt"
    f1.write_text("apple\n")
    f2 = tmp_path / "right.txt"
    f2.write_text("aple\n")

    # Mocking sys.argv and running as __main__
    with patch.object(sys, 'argv', ["typostats.py", str(f1), str(f2), "--quiet"]):
        runpy.run_path("typostats.py", run_name="__main__")
