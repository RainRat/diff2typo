import pytest
from diff2typo import _is_file_included, find_typos, main
from unittest.mock import patch

def test_is_file_included():
    assert _is_file_included("tests/test_math.py", ["*.py"])
    assert _is_file_included("config.json", ["*.json"])
    assert _is_file_included("src/lib.py", ["src/*"])
    assert _is_file_included("package-lock.json", ["package-lock.json"])
    assert not _is_file_included("src/lib.py", ["*.json"])
    assert _is_file_included("src/lib.py", None)
    assert not _is_file_included("", ["*.py"])


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
    results = find_typos(diff_text, include_patterns=["*.py"])
    assert "hllo -> hello" in results
    assert "vrsion -> version" not in results

    results = find_typos(diff_text, include_patterns=["*.json"])
    assert "hllo -> hello" not in results
    assert "vrsion -> version" in results

    results = find_typos(diff_text, include_patterns=["*.py", "*.json"])
    assert "hllo -> hello" in results
    assert "vrsion -> version" in results

    results = find_typos(diff_text, include_patterns=["*.py"], exclude_patterns=["src/*"])
    assert "hllo -> hello" not in results
    assert "vrsion -> version" not in results


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
    results = find_typos(diff_rename_with_content, include_patterns=["new_dir/*"])
    assert "registre -> register" in results

    results = find_typos(diff_rename_with_content, include_patterns=["other/*"])
    assert "registre -> register" not in results


def test_find_typos_inclusion_unified_header():
    diff_text = """--- a/src/math.py
+++ b/src/math.py
@@ -1,2 +1,2 @@
-x = y + 1 # formulaa
+x = y + 1 # formula
"""
    results = find_typos(diff_text, include_patterns=["math.py"])
    assert "formulaa -> formula" in results

    results = find_typos(diff_text, include_patterns=["other.py"])
    assert "formulaa -> formula" not in results


def test_find_typos_inclusion_quotes_and_spaces():
    diff_text = """diff --git a/"my dir/spaced file.py" b/"my dir/spaced file.py"
--- a/"my dir/spaced file.py"
+++ b/"my dir/spaced file.py"
@@ -1,2 +1,2 @@
-def f(): # typoo
+def f(): # typo
"""
    results = find_typos(diff_text, include_patterns=["my dir/*"])
    assert "typoo -> typo" in results

    results = find_typos(diff_text, include_patterns=["other dir/*"])
    assert "typoo -> typo" not in results


def test_cli_include():
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


def test_find_typos_inclusion_edge_cases():
    diff_no_prefix = """diff --git main.py main.py
--- main.py
+++ main.py
@@ -1,2 +1,2 @@
-def hello():
-    print("hllo")
+def hello():
+    print("hello")
"""
    results = find_typos(diff_no_prefix, include_patterns=["main.py"])
    assert "hllo -> hello" in results

    results2 = find_typos(diff_no_prefix, include_patterns=["other.py"])
    assert "hllo -> hello" not in results2

    diff_rename_quoted = """diff --git "a/old file.py" "b/new file.py"
similarity index 100%
rename from "old file.py"
rename to "new file.py"
"""
    results_rename = find_typos(diff_rename_quoted, include_patterns=["new file.py"])
    assert results_rename == []

    diff_unmatched_quote = """diff --git a/unmatched" b/unmatched
--- a/unmatched
+++ b/unmatched
@@ -1,2 +1,2 @@
-def f(): # typoo
+def f(): # typo
"""
    results_unmatched = find_typos(diff_unmatched_quote, include_patterns=["unmatched"])
    assert "typoo -> typo" in results_unmatched

    diff_fallback_quotes = """diff --git "file.py" "file.py"
--- "file.py"
+++ "file.py"
@@ -1,2 +1,2 @@
-def f(): # typoo
+def f(): # typo
"""
    with patch("shlex.split", side_effect=ValueError):
        results_fallback = find_typos(diff_fallback_quotes, include_patterns=["file.py"])
        assert "typoo -> typo" in results_fallback
