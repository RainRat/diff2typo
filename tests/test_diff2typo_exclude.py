import pytest
from diff2typo import _is_file_excluded, find_typos, main
from unittest.mock import patch, MagicMock

def test_is_file_excluded():
    assert _is_file_excluded("tests/test_math.py", ["*.py"])
    assert _is_file_excluded("config.json", ["*.json"])
    assert _is_file_excluded("src/lib.py", ["src/*"])
    assert _is_file_excluded("package-lock.json", ["package-lock.json"])
    assert not _is_file_excluded("src/lib.py", ["*.json"])
    assert not _is_file_excluded("src/lib.py", None)
    assert not _is_file_excluded("", ["*.py"])


def test_find_typos_exclusion():
    diff_text = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,2 +1,2 @@
-def hello():
-    print("hllo")
+def hello():
+    print("hello")
diff --git a/config.json b/config.json
--- a/config.json
+++ b/config.json
@@ -1,2 +1,2 @@
- "vrsion": "1.0.0"
+ "version": "1.0.0"
"""
    # Exclude config.json
    results = find_typos(diff_text, exclude_patterns=["*.json"])
    assert "hllo -> hello" in results
    assert "vrsion -> version" not in results

    # Exclude src/*
    results = find_typos(diff_text, exclude_patterns=["src/*"])
    assert "hllo -> hello" not in results
    assert "vrsion -> version" in results

    # No exclusions
    results = find_typos(diff_text)
    assert "hllo -> hello" in results
    assert "vrsion -> version" in results


def test_find_typos_exclusion_rename():
    diff_rename = """diff --git a/old_path.py b/new_path.py
similarity index 100%
rename from old_path.py
rename to new_path.py
"""
    diff_rename_with_content = """diff --git a/old_dir/registre.py b/new_dir/register.py
similarity index 85%
rename from old_dir/registre.py
rename to new_dir/register.py
--- a/old_dir/registre.py
+++ b/new_dir/register.py
@@ -1,2 +1,2 @@
-class Registre:
+class Register:
"""
    # Exclude new_dir/*
    results = find_typos(diff_rename_with_content, exclude_patterns=["new_dir/*"])
    assert "registre -> register" not in results

    # No exclusions
    results = find_typos(diff_rename_with_content)
    assert "registre -> register" in results


def test_find_typos_exclusion_unified_header():
    diff_text = """--- a/src/math.py
+++ b/src/math.py
@@ -1,2 +1,2 @@
-x = y + 1 # formulaa
+x = y + 1 # formula
"""
    # Exclude math.py
    results = find_typos(diff_text, exclude_patterns=["math.py"])
    assert "formulaa -> formula" not in results

    # No exclusion
    results = find_typos(diff_text)
    assert "formulaa -> formula" in results


def test_find_typos_quotes_and_spaces():
    diff_text = """diff --git a/"my dir/spaced file.py" b/"my dir/spaced file.py"
--- a/"my dir/spaced file.py"
+++ b/"my dir/spaced file.py"
@@ -1,2 +1,2 @@
-def f(): # typoo
+def f(): # typo
"""
    # Exclude files in "my dir"
    results = find_typos(diff_text, exclude_patterns=["my dir/*"])
    assert "typoo -> typo" not in results

    # No exclusion
    results = find_typos(diff_text)
    assert "typoo -> typo" in results


def test_cli_exclude():
    # Verify command line argument parsing with mock
    with patch("sys.argv", ["diff2typo.py", "some.diff", "--exclude", "*.json", "tests/*", "-o", "-"]), \
         patch("diff2typo._read_diff_sources", return_value=""), \
         patch("diff2typo.find_typos", return_value=[]) as mock_find:
        try:
            main()
        except SystemExit:
            pass
        mock_find.assert_called_once()
        args, kwargs = mock_find.call_args
        assert kwargs["exclude_patterns"] == ["*.json", "tests/*"]


def test_find_typos_exclusion_edge_cases():
    # Test diff --git with no a/ or b/ prefixes
    diff_no_prefix = """diff --git main.py main.py
--- main.py
+++ main.py
@@ -1,2 +1,2 @@
-def hello():
-    print("hllo")
+def hello():
+    print("hello")
"""
    results = find_typos(diff_no_prefix, exclude_patterns=["main.py"])
    assert "hllo -> hello" not in results

    results2 = find_typos(diff_no_prefix)
    assert "hllo -> hello" in results2

    # Test file renames with quoted paths
    diff_rename_quoted = """diff --git "a/old file.py" "b/new file.py"
similarity index 100%
rename from "old file.py"
rename to "new file.py"
"""
    # Exclude the new file.py path
    results_rename = find_typos(diff_rename_quoted, exclude_patterns=["new file.py"])
    assert results_rename == []

    # Test ValueError fallback for shlex.split with unmatched quote
    diff_unmatched_quote = """diff --git a/unmatched" b/unmatched
--- a/unmatched
+++ b/unmatched
@@ -1,2 +1,2 @@
-def f(): # typoo
+def f(): # typo
"""
    results_unmatched = find_typos(diff_unmatched_quote, exclude_patterns=["unmatched"])
    assert "typoo -> typo" not in results_unmatched

    # Test fallback unquoting when shlex splits failed and filename has quotes
    diff_fallback_quotes = """diff --git "file.py" "file.py"
--- "file.py"
+++ "file.py"
@@ -1,2 +1,2 @@
-def f(): # typoo
+def f(): # typo
"""
    with patch("shlex.split", side_effect=ValueError):
        results_fallback = find_typos(diff_fallback_quotes, exclude_patterns=["file.py"])
        assert "typoo -> typo" not in results_fallback
