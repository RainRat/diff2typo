import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_parse_markdown_table_row_short():
    """Covers line 67: if len(parts) < 2: return None"""
    assert multitool._parse_markdown_table_row("| onlyone |") is None

def test_apply_smart_case_empty_original():
    """Covers line 92: if not original: return replacement"""
    assert multitool._apply_smart_case("", "replacement") == "replacement"

def test_apply_smart_case_variations():
    """Covers lines 93-99: smart case variations"""
    assert multitool._apply_smart_case("Original", "replacement") == "Replacement"
    assert multitool._apply_smart_case("ORIGINAL", "replacement") == "REPLACEMENT"
    assert multitool._apply_smart_case("lower", "Replacement") == "replacement"

def test_parse_markdown_table_row_skips():
    """Covers lines 70-76: skips for dividers and headers"""
    assert multitool._parse_markdown_table_row("| --- | --- |") is None
    assert multitool._parse_markdown_table_row("| typo | correction |") is None

def test_load_and_clean_file_empty_lines(tmp_path):
    """Covers line 254: if not line_content: continue"""
    f = tmp_path / "empty_lines.txt"
    f.write_text("worda\n\n   \nwordb")
    raw, cleaned, unique = multitool._load_and_clean_file(str(f), 1, 100)
    assert cleaned == ["worda", "wordb"]

def test_count_mode_limit_and_md_table(tmp_path):
    """Covers line 1081 (limit) and lines 1099-1102 (md-table format)"""
    f = tmp_path / "input.txt"
    f.write_text("a a b b c c")
    out = tmp_path / "out.md"

    multitool.count_mode([str(f)], str(out), 1, 100, False, output_format='md-table', limit=2)

    content = out.read_text()
    assert "| Item | Count |" in content
    assert "| :--- | :--- |" in content
    assert "| a | 2 |" in content
    assert "| b | 2 |" in content
    assert "| c | 2 |" not in content

def test_extract_json_items_error(tmp_path, caplog):
    """Covers lines 674-676: json.JSONDecodeError logging"""
    f = tmp_path / "bad.json"
    f.write_text("{ invalid }")

    with caplog.at_level(logging.ERROR):
        items = list(multitool._extract_json_items(str(f), "key"))
        assert items == []
        assert "Failed to parse JSON" in caplog.text

def test_conflict_mode_sort(tmp_path):
    """Covers line 1383: if process_output: conflicts.sort()"""
    f = tmp_path / "input.txt"
    # b comes before a, so sorting should flip them
    f.write_text("banana -> fruit\nbanana -> yellow\napple -> fruit\napple -> red\n")
    out = tmp_path / "out.txt"

    multitool.conflict_mode([str(f)], str(out), 1, 100, process_output=True)

    content = out.read_text().splitlines()
    # arrow format: apple -> fruit, red should be first
    assert "apple -> fruit, red" in content[0]
    assert "banana -> fruit, yellow" in content[1]

def test_extract_table_items_no_closing_quote(tmp_path):
    """Covers line 620: value extraction when no closing quote is present"""
    f = tmp_path / "table.toml"
    f.write_text('key = "no closing quote')

    items = list(multitool._extract_table_items(str(f), right_side=True))
    assert items == ["no closing quote"]

def test_levenshtein_distance_optimizations():
    """Covers line 105, 107: Levenshtein distance optimizations"""
    assert multitool.levenshtein_distance("abc", "") == 3
    assert multitool.levenshtein_distance("", "abc") == 3

def test_scrub_mode_stdin_inplace_warning(caplog, monkeypatch):
    """Covers line 1942: warning when in-place requested for stdin"""
    # Mock sys.stdin to avoid OSError from pytest capture
    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = b"some content\n"
    monkeypatch.setattr(sys, "stdin", mock_stdin)
    # Reset cache
    multitool._STDIN_CACHE = None

    with caplog.at_level(logging.WARNING):
        # We need to provide a mapping file that exists
        with patch("multitool._load_mapping_file", return_value={}):
            multitool.scrub_mode(["-"], "map.csv", "out.txt", 1, 100, False, in_place=".bak")
            assert "In-place modification requested for standard input; ignoring." in caplog.text

def test_scrub_mode_dry_run_and_limit(tmp_path, caplog):
    """Covers dry-run logging and limit in scrub mode"""
    mapping = tmp_path / "map.csv"
    mapping.write_text("teh,the")

    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\nteh\n")

    out = tmp_path / "out.txt"

    with caplog.at_level(logging.WARNING):
        # Test dry-run
        multitool.scrub_mode([str(input_file)], str(mapping), str(out), 1, 100, False, dry_run=True)
        assert "[Dry Run] Total replacements that would be made: 2" in caplog.text

        # Test limit
        multitool.scrub_mode([str(input_file)], str(mapping), str(out), 1, 100, False, limit=1)
        assert out.read_text() == "the\n"

def test_normalize_mode_args_errors():
    """Covers lines 2826-2836: error cases in _normalize_mode_args"""
    parser = MagicMock()
    parser.error.side_effect = SystemExit

    # Multiple --mode flags
    with pytest.raises(SystemExit):
        multitool._normalize_mode_args(["--mode", "csv", "--mode", "words"], parser)
    parser.error.assert_any_call("Only one --mode flag may be provided.")

    # Missing value
    with pytest.raises(SystemExit):
        multitool._normalize_mode_args(["--mode"], parser)
    parser.error.assert_any_call("--mode requires a value.")

    # Conflict with positional
    with pytest.raises(SystemExit):
        multitool._normalize_mode_args(["csv", "--mode", "words"], parser)
    parser.error.assert_any_call("--mode 'words' conflicts with positional mode 'csv'.")

@pytest.fixture(autouse=True)
def reset_stdin_cache():
    """Ensures stdin cache is cleared before each test."""
    multitool._STDIN_CACHE = None
    multitool._STDIN_ENCODING = None
    yield
    multitool._STDIN_CACHE = None
    multitool._STDIN_ENCODING = None

def test_main_min_length_validation(monkeypatch, caplog):
    """Covers line 2857: min-length < 1 validation"""
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "file.txt", "--min-length", "0"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as cm:
            multitool.main()
    assert cm.value.code == 1
    assert "--min-length must be a number of 1 or more" in caplog.text

def test_extract_pairs_csv_error(tmp_path):
    """Covers lines 454-455: csv.Error handling in _extract_pairs"""
    f = tmp_path / "bad.csv"
    # Create a line that might trigger csv.Error.
    # Actually, csv.reader is quite robust. Let's mock it.
    f.write_text("a,b,c")

    with patch("csv.reader", side_effect=multitool.csv.Error("Mock error")):
        pairs = list(multitool._extract_pairs([str(f)]))
        assert pairs == []

def test_write_paired_output_md_table_headers(tmp_path):
    """Covers lines 515-522: different md-table headers"""
    out = tmp_path / "out.md"
    pairs = [("t", "c")]

    multitool._write_paired_output(pairs, str(out), "md-table", "Similarity")
    assert "| Typo | Correction |" in out.read_text()

    multitool._write_paired_output(pairs, str(out), "md-table", "NearDuplicates")
    assert "| Word 1 | Word 2 |" in out.read_text()

def test_extract_csv_items_columns(tmp_path):
    """Covers lines 716-718: column extraction in _extract_csv_items"""
    f = tmp_path / "test.csv"
    f.write_text("a,b,c\n1,2,3")

    items = list(multitool._extract_csv_items(str(f), False, columns=[0, 2]))
    assert items == ["a", "c", "1", "3"]

def test_extract_markdown_items_empty_content(tmp_path):
    """Covers line 819-820: empty content after bullet"""
    f = tmp_path / "test.md"
    f.write_text("- \n- item")

    items = list(multitool._extract_markdown_items(str(f)))
    assert items == ["item"]

def test_extract_regex_items_groups(tmp_path):
    """Covers lines 819-820: multiple groups in regex yield separate items"""
    f = tmp_path / "test.txt"
    f.write_text("apple banana\ncherry date")

    items = list(multitool._extract_regex_items(str(f), r"(\w+) (\w+)"))
    assert items == ["apple", "banana", "cherry", "date"]

def test_backtick_mode_empty_backticks(tmp_path):
    """Covers line 620: empty string between backticks is skipped"""
    f = tmp_path / "test.txt"
    f.write_text("`` `filled` ``")

    items = list(multitool._extract_backtick_items(str(f)))
    assert items == ["filled"]

def test_minimal_formatter_color(monkeypatch):
    """Covers lines 2831-2834: color in MinimalFormatter"""
    formatter = multitool.MinimalFormatter()

    # INFO record (line 2827)
    info_record = logging.LogRecord("test", logging.INFO, "path", 10, "info message", None, None)
    assert formatter.format(info_record) == "info message"

    # ERROR record with color
    error_record = logging.LogRecord("test", logging.ERROR, "pathname", 10, "error message", None, None)

    # Force isatty to True
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    formatted = formatter.format(error_record)
    assert multitool.RED in formatted
    assert "ERROR" in formatted
    assert "error message" in formatted

def test_write_paired_output_conflict_header(tmp_path):
    """Covers lines 515-516: Conflict header in md-table"""
    out = tmp_path / "out.md"
    pairs = [("t", "c1, c2")]
    multitool._write_paired_output(pairs, str(out), "md-table", "Conflict")
    assert "| Typo | Corrections |" in out.read_text()

def test_similarity_mode_dist_filters(tmp_path):
    """Covers lines 1423, 1427, 1432, 1434, 1438, 1443: dist filtering in similarity_mode"""
    f = tmp_path / "input.txt"
    # lines:
    # 1. valid (dist 1)
    # 2. same (dist 0)
    # 3. too different (dist 3)
    # 4. empty side (hit line 1423)
    # 5. too short (hit line 1427)
    f.write_text("cat -> bat\ncat -> cat\ncat -> doggy\ncat -> \na -> b\n")
    out = tmp_path / "out.txt"

    # show_dist=True (line 1438)
    multitool.similarity_mode([str(f)], str(out), 3, 100, False, min_dist=1, show_dist=True)
    content = out.read_text()
    assert "cat -> bat (changes: 1)" in content
    assert "cat -> cat" not in content

    # max_dist 1
    multitool.similarity_mode([str(f)], str(out), 3, 100, False, max_dist=1)
    content = out.read_text()
    assert "cat -> bat" in content
    assert "doggy" not in content

def test_set_operation_intersection(tmp_path):
    """Covers lines 2501-2502: intersection in set_operation_mode"""
    f1 = tmp_path / "f1.txt"
    f1.write_text("a\nb\nc")
    f2 = tmp_path / "f2.txt"
    f2.write_text("b\nc\nd")
    out = tmp_path / "out.txt"

    multitool.set_operation_mode([str(f1)], str(f2), str(out), 1, 100, False, "intersection")
    assert out.read_text().splitlines() == ["b", "c"]

def test_extract_yaml_items_doc_none_and_error(tmp_path, caplog):
    """Covers line 697 (doc is None) and lines 699-701 (YAMLError)"""
    f = tmp_path / "test.yaml"
    f.write_text("---\n---\nkey: val")

    items = list(multitool._extract_yaml_items(str(f), "key"))
    assert items == ["val"]

    f.write_text("!!invalid")
    with caplog.at_level(logging.ERROR):
        items = list(multitool._extract_yaml_items(str(f), "key"))
        assert items == []
        assert "Failed to parse YAML" in caplog.text

def test_main_secondary_file_errors(monkeypatch, caplog):
    """Covers lines 2329-2331 and 2341-2345: missing file2 error"""
    # filterfragments
    monkeypatch.setattr(sys, "argv", ["multitool.py", "filterfragments"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            multitool.main()
    assert "Filterfragments mode requires a secondary file" in caplog.text

    # set_operation
    monkeypatch.setattr(sys, "argv", ["multitool.py", "set_operation", "--operation", "union"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            multitool.main()
    assert "Set_operation mode requires a secondary file" in caplog.text

def test_extract_pairs_yaml_no_module(tmp_path, caplog, monkeypatch):
    """Covers line 419: PyYAML not installed in _extract_pairs"""
    import builtins
    real_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError
        return real_import(name, *args, **kwargs)

    f = tmp_path / "test.yaml"
    f.write_text("key: value")

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mocked_import)
        with caplog.at_level(logging.ERROR):
            pairs = list(multitool._extract_pairs([str(f)]))
            assert pairs == []
            assert "PyYAML not installed" in caplog.text

def test_extract_yaml_items_no_module(tmp_path, caplog, monkeypatch):
    """Covers lines 685-687: PyYAML not installed in _extract_yaml_items"""
    import builtins
    real_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError
        return real_import(name, *args, **kwargs)

    f = tmp_path / "test.yaml"
    f.write_text("key: value")

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mocked_import)
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                list(multitool._extract_yaml_items(str(f), "key"))
            assert "PyYAML is not installed" in caplog.text

def test_write_paired_output_yaml_no_module(tmp_path, caplog, monkeypatch):
    """Covers lines 497-500: PyYAML not installed in _write_paired_output"""
    import builtins
    real_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError
        return real_import(name, *args, **kwargs)

    out = tmp_path / "out.yaml"
    pairs = [("k", "v")]

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mocked_import)
        multitool._write_paired_output(pairs, str(out), "yaml", "Pairs")
        assert "k: v\n" in out.read_text()

def test_scrub_mode_inplace_backup_and_no_changes(tmp_path, caplog):
    """Covers line 1949 (backup), line 1985 (write), and line 1992 (no changes)"""
    mapping = tmp_path / "map.csv"
    mapping.write_text("teh,the")

    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\n")

    # In-place with backup
    multitool.scrub_mode([str(input_file)], str(mapping), "-", 1, 100, False, in_place=".bak")
    assert (tmp_path / "input.txt.bak").exists()
    assert input_file.read_text() == "the\n"

    # No changes needed
    with caplog.at_level(logging.INFO):
        multitool.scrub_mode([str(input_file)], str(mapping), "-", 1, 100, False, in_place="")
        assert "No changes needed for" in caplog.text

def test_filter_fragments_empty_list1(tmp_path):
    """Covers line 2091: empty all_cleaned_list1 optimization"""
    f1 = tmp_path / "f1.txt"
    f1.write_text("")
    f2 = tmp_path / "f2.txt"
    f2.write_text("anything")
    out = tmp_path / "out.txt"

    multitool.filter_fragments_mode([str(f1)], str(f2), str(out), 1, 100, False)
    assert out.read_text() == ""

def test_set_operation_invalid_op():
    """Covers line 2269: ValueError for invalid operation"""
    with pytest.raises(ValueError, match="Invalid operation 'invalid'"):
        multitool.set_operation_mode(["f1"], "f2", "out", 1, 100, False, "invalid")

def test_fuzzymatch_mode_len_optimization(tmp_path):
    """Covers line 1636: length optimization in fuzzymatch"""
    f1 = tmp_path / "f1.txt"
    f1.write_text("apple")
    f2 = tmp_path / "f2.txt"
    f2.write_text("a\nbanana")
    out = tmp_path / "out.txt"
    multitool.fuzzymatch_mode([str(f1)], str(f2), str(out), 1, 100, False, max_dist=1)
    assert out.read_text() == ""

def test_filter_fragments_no_ahocorasick(tmp_path, monkeypatch, caplog):
    """Covers lines 2501-2502: ahocorasick not available"""
    monkeypatch.setattr(multitool, "_AHOCORASICK_AVAILABLE", False)
    f1 = tmp_path / "f1.txt"
    f1.write_text("a")
    f2 = tmp_path / "f2.txt"
    f2.write_text("b")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            multitool.filter_fragments_mode([str(f1)], str(f2), "out", 1, 100, False)
        assert "pyahocorasick' to use this mode." in caplog.text

def test_similarity_mode_process_output(tmp_path):
    """Covers line 1443: sorted(set()) in similarity_mode"""
    f = tmp_path / "input.txt"
    f.write_text("cat -> bat\ncat -> bat\n")
    out = tmp_path / "out.txt"
    multitool.similarity_mode([str(f)], str(out), 1, 100, True)
    assert out.read_text().strip() == "cat -> bat"

def test_zip_mode_process_output(tmp_path):
    """Covers line 1949: sorted(set()) in zip_mode"""
    f1 = tmp_path / "f1.txt"
    f1.write_text("a\na")
    f2 = tmp_path / "f2.txt"
    f2.write_text("1\n1")
    out = tmp_path / "out.txt"
    multitool.zip_mode([str(f1)], str(f2), str(out), 1, 100, True, clean_items=False)
    assert out.read_text().strip() == "a -> 1"

def test_pairs_mode_empty_and_process(tmp_path):
    """Covers line 1985 (empty) and 1992 (process) in pairs_mode"""
    f = tmp_path / "input.txt"
    f.write_text("apple -> \n -> banana\ncherry -> date\ncherry -> date")
    out = tmp_path / "out.txt"
    multitool.pairs_mode([str(f)], str(out), 3, 100, True)
    assert out.read_text().strip() == "cherry -> date"

def test_main_filenotfound_generic(monkeypatch, caplog):
    """Covers lines 3923-3924: FileNotFoundError without filename"""
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "nonexistent"])
    with patch("multitool.words_mode", side_effect=FileNotFoundError("Generic error")):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                multitool.main()
            assert "File not found: Generic error" in caplog.text

def test_main_unexpected_error(monkeypatch, caplog):
    """Covers lines 3927-3928: unexpected Exception in main"""
    monkeypatch.setattr(sys, "argv", ["multitool.py", "words", "file.txt"])
    with patch("multitool.words_mode", side_effect=RuntimeError("Oops")):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                multitool.main()
            assert "An unexpected error occurred: Oops" in caplog.text

def test_zip_mode_empty_sides(tmp_path):
    """Covers line 1942: if not left or not right: continue in zip_mode"""
    f1 = tmp_path / "f1.txt"
    f1.write_text("apple\n\ncherry")
    f2 = tmp_path / "f2.txt"
    # Note: \n\n in f1 will result in an empty string in left_items.
    # We need f2 to have enough lines to pair with 'cherry'
    f2.write_text("11111\n22222\n33333\n")
    out = tmp_path / "out.txt"
    # 'apple' (f1 line 1) pairs with '11111' (f2 line 1)
    # '' (f1 line 2) pairs with '22222' (f2 line 2) -> should be skipped
    # 'cherry' (f1 line 3) pairs with '33333' (f2 line 3)
    multitool.zip_mode([str(f1)], str(f2), str(out), 1, 100, False, clean_items=False)
    content = out.read_text().splitlines()
    assert "apple -> 11111" in content
    assert "cherry -> 33333" in content
    assert len(content) == 2

def test_swap_mode_empty_sides(tmp_path):
    """Covers line 2031: if not new_left or not new_right: continue in swap_mode"""
    f = tmp_path / "input.txt"
    # Note: swap reverses, so "apple -> " becomes " -> apple" which is then skipped by cleaning if min_length is set
    f.write_text("apple -> \n -> banana\ncherry -> date")
    out = tmp_path / "out.txt"
    multitool.swap_mode([str(f)], str(out), 3, 100, False)
    assert out.read_text().strip() == "date -> cherry"

def test_sample_mode_empty_input(tmp_path, caplog):
    """Covers line 2089: if not raw_items: ... return in sample_mode"""
    f = tmp_path / "empty.txt"
    f.write_text("")
    out = tmp_path / "out.txt"
    with caplog.at_level(logging.WARNING):
        multitool.sample_mode([str(f)], str(out), 1, 100, False, sample_count=5)
        assert "Input is empty or no lines found." in caplog.text

def test_scrub_mode_backup_error(tmp_path):
    """Covers lines 2329-2331: failed backup in scrub_mode"""
    mapping = tmp_path / "map.csv"
    mapping.write_text("a,b")
    input_file = tmp_path / "input.txt"
    input_file.write_text("a")

    with patch("shutil.copy2", side_effect=Exception("Mock backup error")):
        with pytest.raises(SystemExit):
            multitool.scrub_mode([str(input_file)], str(mapping), "-", 1, 100, False, in_place=".bak")

def test_scrub_mode_write_error(tmp_path):
    """Covers lines 2341-2343: failed write in scrub_mode"""
    mapping = tmp_path / "map.csv"
    mapping.write_text("a,b")
    input_file = tmp_path / "input.txt"
    input_file.write_text("a")

    # We want to patch only the open(..., 'w') call
    real_open = open
    def mocked_open(path, mode='r', *args, **kwargs):
        if mode == 'w' and str(path) == str(input_file):
            raise Exception("Mock write error")
        return real_open(path, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=mocked_open):
        with pytest.raises(SystemExit):
            multitool.scrub_mode([str(input_file)], str(mapping), "-", 1, 100, False, in_place="")
