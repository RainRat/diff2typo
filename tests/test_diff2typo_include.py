import pytest
from diff2typo import _match_pattern, find_typos, main
from unittest.mock import patch

def test_match_pattern_include():
    assert _match_pattern("tests/test_math.py", ["*.py"])
    assert _match_pattern("config.json", ["*.json"])
    assert _match_pattern("src/lib.py", ["src/*"])
    assert _match_pattern("package-lock.json", ["package-lock.json"])
    assert not _match_pattern("src/lib.py", ["*.json"])
    assert not _match_pattern("src/lib.py", None)
    assert not _match_pattern("", ["*.py"])


def test_find_typos_inclusion():
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
    # Include only src/*
    results = find_typos(diff_text, include_patterns=["src/*"])
    assert "hllo -> hello" in results
    assert "vrsion -> version" not in results

    # Include only config.json
    results = find_typos(diff_text, include_patterns=["*.json"])
    assert "hllo -> hello" not in results
    assert "vrsion -> version" in results

    # No inclusion patterns (defaults to all)
    results = find_typos(diff_text)
    assert "hllo -> hello" in results
    assert "vrsion -> version" in results


def test_find_typos_inclusion_rename():
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
    # Include only old_dir/* (should match rename destination or skip_current_file context)
    results = find_typos(diff_rename_with_content, include_patterns=["new_dir/*"])
    assert "registre -> register" in results

    results_none = find_typos(diff_rename_with_content, include_patterns=["other_dir/*"])
    assert "registre -> register" not in results_none


def test_find_typos_inclusion_unified_header():
    diff_text = """--- a/src/math.py
+++ b/src/math.py
@@ -1,2 +1,2 @@
-x = y + 1 # formulaa
+x = y + 1 # formula
"""
    # Include other files
    results = find_typos(diff_text, include_patterns=["config.json"])
    assert "formulaa -> formula" not in results

    # Include math.py
    results = find_typos(diff_text, include_patterns=["math.py"])
    assert "formulaa -> formula" in results


def test_find_typos_inclusion_and_exclusion():
    diff_text = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,2 +1,2 @@
-def hello():
-    print("hllo")
+def hello():
+    print("hello")
diff --git a/src/test.py b/src/test.py
--- a/src/test.py
+++ b/src/test.py
@@ -1,2 +1,2 @@
-def test_val(): # formulaa
+def test_val(): # formula
"""
    # Include src/* but exclude test.py
    results = find_typos(diff_text, exclude_patterns=["*test.py"], include_patterns=["src/*"])
    assert "hllo -> hello" in results
    assert "formulaa -> formula" not in results


def test_cli_include():
    # Verify command line argument parsing with mock
    with patch("sys.argv", ["diff2typo.py", "some.diff", "--include", "*.py", "src/*", "-o", "-"]), \
         patch("diff2typo._read_diff_sources", return_value=""), \
         patch("diff2typo.find_typos", return_value=[]) as mock_find:
        try:
            main()
        except SystemExit:
            pass
        mock_find.assert_called_once()
        args, kwargs = mock_find.call_args
        assert kwargs["include_patterns"] == ["*.py", "src/*"]
