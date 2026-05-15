import os
import unittest
from unittest.mock import patch
import io
import multitool

class TestMultitoolDistBoosting(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data_dist_boosting"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def run_multitool(self, args):
        with patch("sys.argv", ["multitool.py"] + args), \
             patch("sys.stdout", new=io.StringIO()) as fake_out, \
             patch("sys.stderr", new=io.StringIO()):
            try:
                multitool.main()
            except SystemExit as e:
                if e.code != 0:
                    raise
            return fake_out.getvalue()

    def test_similarity_boosting(self):
        input_file = os.path.join(self.test_dir, "similarity.txt")
        with open(input_file, "w") as f:
            f.write("the -> teh\n")  # Transposition [T], dist 2
            f.write("the -> rhe\n")  # Keyboard [K], dist 1

        # Test transposition with max-dist 0 (BUG: currently won't boost)
        out = self.run_multitool(["similarity", input_file, "--transposition", "--max-dist", "0", "-f", "csv", "-R"])
        # Expected: teh found. Current: nothing found.
        self.assertIn("the,teh", out)

        # Test keyboard with max-dist 0 (BUG: currently won't boost)
        out = self.run_multitool(["similarity", input_file, "--keyboard", "--max-dist", "0", "-f", "csv", "-R"])
        # Expected: rhe found. Current: nothing found.
        self.assertIn("the,rhe", out)

        # Test both with max-dist 1 (BUG: currently only boosts to 1 due to elif, so teh is missed)
        out = self.run_multitool(["similarity", input_file, "--keyboard", "--transposition", "--max-dist", "1", "-f", "csv", "-R"])
        self.assertIn("the,rhe", out)
        self.assertIn("the,teh", out)

    def test_near_duplicates_boosting(self):
        input_file = os.path.join(self.test_dir, "nd.txt")
        with open(input_file, "w") as f:
            f.write("the\nteh\n")

        # Test transposition with max-dist 0
        out = self.run_multitool(["near_duplicates", input_file, "--transposition", "--max-dist", "0", "-f", "csv", "-R"])
        self.assertIn("teh,the", out)

    def test_fuzzymatch_boosting(self):
        file1 = os.path.join(self.test_dir, "fm1.txt")
        file2 = os.path.join(self.test_dir, "fm2.txt")
        with open(file1, "w") as f:
            f.write("the\n")
        with open(file2, "w") as f:
            f.write("teh\n")

        # Test transposition with max-dist 0
        out = self.run_multitool(["fuzzymatch", file1, file2, "--transposition", "--max-dist", "0", "-f", "csv", "-R"])
        self.assertIn("the,teh", out)

    def test_discovery_boosting(self):
        input_file = os.path.join(self.test_dir, "discovery.txt")
        with open(input_file, "w") as f:
            f.write("the\n" * 10)
            f.write("teh\n")

        # Test transposition with max-dist 0
        out = self.run_multitool(["discovery", input_file, "--transposition", "--max-dist", "0", "-f", "csv", "-R", "--freq-min", "5"])
        self.assertIn("teh,the", out)

if __name__ == "__main__":
    unittest.main()
