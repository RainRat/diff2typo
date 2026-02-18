
import subprocess
import os
import pytest

def test_near_duplicates_logic():
    test_file = "test_words_nd.txt"
    with open(test_file, "w") as f:
        f.write("apple\n")
        f.write("aple\n")
        f.write("banana\n")
        f.write("bananna\n")
        f.write("cherry\n")
        f.write("berry\n")

    try:
        # Run near_duplicates with max-dist 1
        result = subprocess.run(
            ["python", "multitool.py", "near_duplicates", test_file, "--max-dist", "1"],
            capture_output=True, text=True
        )
        assert "aple -> apple" in result.stdout
        assert "banana -> bananna" in result.stdout
        assert "cherry -> berry" not in result.stdout

        # Run near_duplicates with max-dist 2
        result = subprocess.run(
            ["python", "multitool.py", "near_duplicates", test_file, "--max-dist", "2"],
            capture_output=True, text=True
        )
        assert "berry -> cherry" in result.stdout

        # Test show-dist
        result = subprocess.run(
            ["python", "multitool.py", "near_duplicates", test_file, "--max-dist", "1", "--show-dist"],
            capture_output=True, text=True
        )
        assert "aple -> apple (dist: 1)" in result.stdout

        # Test output format JSON
        result = subprocess.run(
            ["python", "multitool.py", "near_duplicates", test_file, "--max-dist", "1", "--format", "json"],
            capture_output=True, text=True
        )
        assert '"aple": "apple"' in result.stdout

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
