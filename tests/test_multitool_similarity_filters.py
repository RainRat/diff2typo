import sys
import unittest
from io import StringIO
from unittest.mock import patch
import multitool
import os

class TestSimilarityFilters(unittest.TestCase):

    def setUp(self):
        # Create some test data
        self.test_dir = "temp_test_similarity_filters"
        os.makedirs(self.test_dir, exist_ok=True)

        self.words_file = os.path.join(self.test_dir, "words.txt")
        with open(self.words_file, "w") as f:
            # teh -> the [T] dist 2
            # thw -> the [K] dist 1
            # thz -> the [R] dist 1 (z is not near e)
            f.write("teh\n")
            f.write("the\n")
            f.write("thw\n")
            f.write("thz\n")

        self.pairs_file = os.path.join(self.test_dir, "pairs.txt")
        with open(self.pairs_file, "w") as f:
            f.write("teh -> the\n")
            f.write("thw -> the\n")
            f.write("thz -> the\n")

        self.dict_file = os.path.join(self.test_dir, "dict.txt")
        with open(self.dict_file, "w") as f:
            f.write("the\n")
            f.write("other\n")

    def tearDown(self):
        # Clean up
        for f in [self.words_file, self.pairs_file, self.dict_file]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def run_multitool(self, args):
        with patch('sys.stdout', new=StringIO()) as fake_out, \
             patch('sys.stderr', new=StringIO()) as fake_err, \
             patch('sys.argv', ['multitool.py'] + args):
            try:
                multitool.main()
            except SystemExit as e:
                if e.code != 0:
                    # In case of error, show what happened
                    sys.stderr.write(fake_err.getvalue())
                    raise
            return fake_out.getvalue(), fake_err.getvalue()

    def test_similarity_keyboard(self):
        # thw -> the is Keyboard
        # teh -> the is Transposition
        # thz -> the is Replacement (non-adjacent)
        out, _ = self.run_multitool(['similarity', self.pairs_file, '--keyboard'])
        self.assertIn("thw -> the", out)
        self.assertNotIn("teh -> the", out)
        self.assertNotIn("thz -> the", out)

    def test_similarity_transposition(self):
        out, _ = self.run_multitool(['similarity', self.pairs_file, '--transposition'])
        self.assertIn("teh -> the", out)
        self.assertNotIn("thw -> the", out)
        self.assertNotIn("thz -> the", out)

    def test_near_duplicates_keyboard(self):
        # the and thw are near duplicates (Keyboard)
        # the and thz are near duplicates (Replacement, dist 1)
        # the and teh are near duplicates (Transposition, dist 2)
        # near_duplicates default max-dist is 1
        out, _ = self.run_multitool(['near_duplicates', self.words_file, '--keyboard', '--max-dist', '1'])
        self.assertIn("the", out)
        self.assertIn("thw", out)
        self.assertNotIn("thz", out)
        self.assertNotIn("teh", out)

    def test_fuzzymatch_transposition(self):
        # list1: teh, thw, thz
        # list2: the
        # teh -> the is transposition (dist 2)
        out, _ = self.run_multitool(['fuzzymatch', self.words_file, '--file2', self.dict_file, '--transposition', '--max-dist', '2'])
        self.assertIn("teh", out)
        self.assertIn("the", out)
        self.assertNotIn("thw", out)
        self.assertNotIn("thz", out)

    def test_discovery_keyboard(self):
        discovery_file = os.path.join(self.test_dir, "discovery.txt")
        with open(discovery_file, "w") as f:
            for _ in range(10):
                f.write("the\n")
            f.write("thw\n")
            f.write("thz\n")
            f.write("teh\n")

        out, _ = self.run_multitool(['discovery', discovery_file, '--keyboard', '--freq-min', '5', '--rare-max', '1'])
        self.assertIn("thw", out)
        self.assertIn("the", out)
        self.assertNotIn("thz", out)
        self.assertNotIn("teh", out)

        if os.path.exists(discovery_file):
            os.remove(discovery_file)

if __name__ == "__main__":
    unittest.main()
