import sys
import os

# Ensure the repository root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diff2typo import _compare_word_lists, find_typos

def test_misaligned_word_lists():
    # Case 1: Simple aligned replacement (Should work)
    before_1 = ["the", "teh", "house"]
    after_1 = ["the", "the", "house"]
    result_1 = _compare_word_lists(before_1, after_1, min_length=2)
    print(f"Test 1 (Aligned): {result_1}")
    assert "teh -> the" in result_1

    # Case 2: Misaligned by adding a word (Current diff2typo fails here)
    # "the teh house" -> "the big the house"
    # 'teh' is corrected to 'the', but 'big' is added.
    before_2 = ["the", "teh", "house"]
    after_2 = ["the", "big", "the", "house"]
    result_2 = _compare_word_lists(before_2, after_2, min_length=2)
    print(f"Test 2 (Misaligned/Added Word): {result_2}")
    # This should find 'teh -> the'

    # Case 3: Full diff test
    diff = """--- a/file.txt
+++ b/file.txt
-the teh house
+the big the house
"""
    result_3 = find_typos(diff, min_length=2)
    print(f"Test 3 (Full Diff/Added Word): {result_3}")
    assert "teh -> the" in result_3

    # Test filters
    # min_length=4 should skip 'teh' (length 3)
    result_4 = _compare_word_lists(before_2, after_2, min_length=4)
    print(f"Test 4 (min_length=4): {result_4}")
    assert len(result_4) == 0

    # max_dist=0 should skip 'teh' (dist 1)
    result_5 = _compare_word_lists(before_2, after_2, min_length=2, max_dist=0)
    print(f"Test 5 (max_dist=0): {result_5}")
    assert len(result_5) == 0

if __name__ == "__main__":
    try:
        test_misaligned_word_lists()
        print("\nSUCCESS: All alignment and filter tests passed.")
    except AssertionError as e:
        print(f"\nFAILURE: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
