import os
import unittest
from unittest.mock import patch
import io
import multitool

class TestMultitoolTypoFilters(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data_typo_filters"
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

    def test_similarity_filters(self):
        input_file = os.path.join(self.test_dir, "similarity.txt")
        with open(input_file, "w") as f:
            f.write("the -> rhe\n")  # Keyboard [K]
            f.write("the -> teh\n")  # Transposition [T]
            f.write("the -> tha\n")  # Replacement [R]

        # Test keyboard filter
        out = self.run_multitool(["similarity", input_file, "--keyboard", "-f", "csv", "-R"])
        self.assertIn("the,rhe", out)
        self.assertNotIn("teh", out)
        self.assertNotIn("tha", out)

        # Test transposition filter
        out = self.run_multitool(["similarity", input_file, "--transposition", "-f", "csv", "-R"])
        self.assertIn("the,teh", out)
        self.assertNotIn("rhe", out)
        self.assertNotIn("tha", out)

    def test_near_duplicates_filters(self):
        input_file = os.path.join(self.test_dir, "nd.txt")
        with open(input_file, "w") as f:
            f.write("the\nrhe\nteh\ntha\n")

        # Test keyboard filter
        out = self.run_multitool(["near_duplicates", input_file, "--keyboard", "-f", "csv", "-R"])
        self.assertIn("rhe,the", out)
        self.assertNotIn("teh", out)
        self.assertNotIn("tha", out)

        # Test transposition filter
        out = self.run_multitool(["near_duplicates", input_file, "--transposition", "-f", "csv", "-R"])
        self.assertIn("teh,the", out)
        self.assertNotIn("rhe", out)
        self.assertNotIn("tha", out)

    def test_fuzzymatch_filters(self):
        file1 = os.path.join(self.test_dir, "fm1.txt")
        file2 = os.path.join(self.test_dir, "fm2.txt")
        with open(file1, "w") as f:
            f.write("the\n")
        with open(file2, "w") as f:
            f.write("rhe\nteh\ntha\n")

        # Test keyboard filter
        out = self.run_multitool(["fuzzymatch", file1, file2, "--keyboard", "-f", "csv", "-R"])
        self.assertIn("the,rhe", out)
        self.assertNotIn("teh", out)
        self.assertNotIn("tha", out)

        # Test transposition filter
        out = self.run_multitool(["fuzzymatch", file1, file2, "--transposition", "-f", "csv", "-R"])
        self.assertIn("the,teh", out)
        self.assertNotIn("rhe", out)
        self.assertNotIn("tha", out)

    def test_discovery_filters(self):
        input_file = os.path.join(self.test_dir, "discovery.txt")
        with open(input_file, "w") as f:
            # 'the' is frequent, others are rare
            f.write("the\n" * 10)
            f.write("rhe\nteh\ntha\n")

        # Test keyboard filter
        out = self.run_multitool(["discovery", input_file, "--keyboard", "-f", "csv", "-R", "--freq-min", "5"])
        self.assertIn("rhe,the", out)
        self.assertNotIn("teh", out)
        self.assertNotIn("tha", out)

        # Test transposition filter
        out = self.run_multitool(["discovery", input_file, "--transposition", "-f", "csv", "-R", "--freq-min", "5"])
        self.assertIn("teh,the", out)
        self.assertNotIn("rhe", out)
        self.assertNotIn("tha", out)

    def test_search_filters(self):
        input_file = os.path.join(self.test_dir, "search.txt")
        with open(input_file, "w") as f:
            f.write("rhe\n")
            f.write("teh\n")
            f.write("tha\n")

        # Test keyboard filter
        out = self.run_multitool(["search", "the", input_file, "--keyboard", "-R"])
        self.assertIn("rhe", out)
        self.assertNotIn("teh", out)
        self.assertNotIn("tha", out)

        # Test transposition filter
        out = self.run_multitool(["search", "the", input_file, "--transposition", "-R"])
        self.assertIn("teh", out)
        self.assertNotIn("rhe", out)
        self.assertNotIn("tha", out)

if __name__ == "__main__":
    unittest.main()
