import logging
import sys
import types
from pathlib import Path
from io import StringIO
import pytest
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_minimal_formatter():
    formatter = gentypos.MinimalFormatter()

    # Test INFO level
    record_info = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="info message", args=None, exc_info=None
    )
    assert formatter.format(record_info) == "info message"

    # Test WARNING level
    record_warning = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="", lineno=0,
        msg="warning message", args=None, exc_info=None
    )
    assert formatter.format(record_warning) == "WARNING: warning message"

def test_get_adjacent_keys_no_diagonals():
    # Test with diagonals (default)
    adj_with = gentypos.get_adjacent_keys(include_diagonals=True)
    # 's' is surrounded by 'qwe', 'adz', 'x' (approx)
    # In 'qwertyuiop', 'asdfghjkl', 'zxcvbnm'
    # 's' is at (1, 1). Neighbors: (0,0),(0,1),(0,2), (1,0),(1,2), (2,0),(2,1),(2,2)
    # (0,0)=q, (0,1)=w, (0,2)=e, (1,0)=a, (1,2)=d, (2,0)=z, (2,1)=x, (2,2)=c
    assert 'w' in adj_with['s']
    assert 'q' in adj_with['s'] # diagonal

    # Test without diagonals
    adj_without = gentypos.get_adjacent_keys(include_diagonals=False)
    # Non-diagonals of 's' (1, 1): (0,1)=w, (1,0)=a, (1,2)=d, (2,1)=x
    assert 'w' in adj_without['s']
    assert 'a' in adj_without['s']
    assert 'd' in adj_without['s']
    assert 'x' in adj_without['s']
    assert 'q' not in adj_without['s'] # diagonal
    assert 'z' not in adj_without['s'] # diagonal

def test_load_custom_substitutions_variants():
    # v is None
    custom_none = {'a': None}
    assert gentypos.load_custom_substitutions(custom_none) == {}

    # v is bytes
    custom_bytes = {'a': b'b'}
    # str(b'b') is "b'b'"
    assert gentypos.load_custom_substitutions(custom_bytes) == {'a': {"b'b'"}}

    # v is not iterable (and not str/bytes)
    custom_non_iterable = {'a': 123}
    assert gentypos.load_custom_substitutions(custom_non_iterable) == {'a': {'123'}}

def test_generate_typos_by_transposition_identical_chars():
    # 'aa' should produce no transpositions because chars are identical
    result = gentypos.generate_typos_by_transposition('aa', distance=1)
    assert result == set()

def test_load_file_none():
    assert gentypos.load_file(None) == set()

def test_load_file_stdin(monkeypatch):
    input_data = "hello\nworld\n"
    monkeypatch.setattr(sys, 'stdin', StringIO(input_data))
    result = gentypos.load_file('-')
    assert result == {'hello', 'world'}

def test_load_file_exception(tmp_path):
    # Pass a directory instead of a file to trigger an exception
    dir_path = tmp_path / "a_directory"
    dir_path.mkdir()
    with pytest.raises(SystemExit):
        gentypos.load_file(str(dir_path))

def test_validate_config_cli():
    config = {}
    # Should not raise SystemExit if cli_mode=True even if required fields are missing
    gentypos.validate_config(config, cli_mode=True)
    assert 'typo_types' in config # merged from defaults

def test_extract_config_settings_invalid_format(caplog):
    config = {'output_format': 'invalid_format'}
    settings = gentypos._extract_config_settings(config)
    assert settings.output_format == 'arrow'
    assert "Unknown output format 'invalid_format'. Defaulting to 'arrow'." in caplog.text

def test_extract_config_settings_table_header():
    config = {'output_format': 'table'}
    settings = gentypos._extract_config_settings(config)
    assert settings.output_header == "[default.extend-words]"

def test_setup_generation_tools_merge(tmp_path):
    subs_file = tmp_path / "subs.json"
    # a: existing is list
    # b: existing is None (line 640)
    # c: not in custom_subs_raw (line 646)
    subs_file.write_text('{"a": ["list_val"], "b": ["none_val"], "c": ["new_val"]}')

    settings = types.SimpleNamespace(
        custom_substitutions_config={'a': ['orig_list'], 'b': None},
        substitutions_file=str(subs_file),
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False
    )

    adj, custom = gentypos._setup_generation_tools(settings)
    assert custom['a'] == {'orig_list', 'list_val'}
    assert custom['b'] == {'none_val'}
    assert custom['c'] == {'new_val'}

    # Test merging when existing is a single value (not a list) (line 644)
    subs_file_single = tmp_path / "subs_single.json"
    subs_file_single.write_text('{"a": ["val2"]}')
    settings_single = types.SimpleNamespace(
        custom_substitutions_config={'a': 'val1'},
        substitutions_file=str(subs_file_single),
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False
    )
    adj, custom = gentypos._setup_generation_tools(settings_single)
    assert custom['a'] == {'val1', 'val2'}

def test_parse_yaml_config_no_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(gentypos, "_YAML_AVAILABLE", False)
    with pytest.raises(SystemExit):
        gentypos.parse_yaml_config("any.yaml")

def test_parse_yaml_config_exception(tmp_path, monkeypatch):
    # Mock open to raise an exception other than FileNotFoundError
    def mock_open(*args, **kwargs):
        raise Exception("unexpected error")

    with patch("builtins.open", mock_open):
        with pytest.raises(SystemExit):
            gentypos.parse_yaml_config("any.yaml")

def test_setup_generation_tools_disabled():
    settings = types.SimpleNamespace(
        custom_substitutions_config={'a': ['b']},
        substitutions_file=None,
        enable_custom_substitutions=False,
        enable_adjacent_substitutions=False
    )
    adj, custom = gentypos._setup_generation_tools(settings)
    assert adj == {}
    assert custom == {}

def test_run_typo_generation_length_filtering():
    settings = types.SimpleNamespace(
        min_length=5,
        max_length=10,
        repeat_modifications=1,
        typo_types={'deletion': True},
        transposition_distance=1,
        enable_adjacent_substitutions=False,
        enable_custom_substitutions=False
    )

    # 'cat' (3) is too short, 'internationally' (15) is too long
    words = ['cat', 'hello', 'internationally']
    result = gentypos._run_typo_generation(words, set(), settings, {}, {}, quiet=True)

    # Only 'hello' should have typos
    for typo, correct in result.items():
        assert correct == 'hello'
    assert len(result) > 0

def test_run_typo_generation_no_dictionary():
    settings = types.SimpleNamespace(
        min_length=0,
        max_length=None,
        repeat_modifications=1,
        typo_types={'deletion': True},
        transposition_distance=1,
        enable_adjacent_substitutions=False,
        enable_custom_substitutions=False
    )

    result = gentypos._run_typo_generation(['a'], set(), settings, {}, {}, quiet=True)
    # deletion of 'a' gives ''
    assert '' in result
    assert result[''] == 'a'

def test_main_verbose_quiet(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "--verbose", "word", "--no-filter"])
    # main() calls basicConfig, which might be tricky if already configured
    # but let's try.
    with caplog.at_level(logging.DEBUG):
        with patch("gentypos._run_typo_generation", return_value={}):
            with patch("gentypos.format_typos", return_value=[]):
                gentypos.main()
    assert "Verbose mode enabled." in caplog.text

    caplog.clear()
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "--quiet", "word", "--no-filter"])
    with caplog.at_level(logging.DEBUG):
        with patch("gentypos._run_typo_generation", return_value={}):
            with patch("gentypos.format_typos", return_value=[]):
                gentypos.main()
    assert "Quiet mode enabled." in caplog.text
    # "Quiet mode enabled" is logged at DEBUG level, so it might not show up
    # if log level is WARNING.
    # Actually line 865 is logging.debug("Quiet mode enabled.")
    # And line 857: log_level = logging.WARNING if args.quiet else ...
    # So DEBUG won't show.

def test_main_missing_config_exit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "--config", "nonexistent.yaml"])
    with pytest.raises(SystemExit):
        gentypos.main()

def test_main_cli_overrides(monkeypatch, tmp_path):
    # Test --word (legacy), cli_mode defaults, and various overrides
    monkeypatch.setattr(sys, "argv", [
        "gentypos.py", "--word", "hello",
        "--output", str(tmp_path / "out.txt"),
        "--format", "csv",
        "--substitutions", str(tmp_path / "subs.json"),
        "--no-filter",
        "--min-length", "2",
        "--max-length", "10"
    ])

    (tmp_path / "subs.json").write_text("{}")

    with patch("gentypos._run_typo_generation", return_value={"hallo": "hello"}) as mock_gen:
        gentypos.main()
        # Verify settings passed to _run_typo_generation if possible,
        # or just verify it runs.
        assert mock_gen.called
        args, kwargs = mock_gen.call_args
        settings = args[2]
        assert settings.min_length == 2
        assert settings.max_length == 10
        assert settings.output_format == "csv"

def test_main_exception_on_write(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--output", str(tmp_path / "readonly.txt")])
    (tmp_path / "readonly.txt").write_text("content")

    # Make file readonly
    (tmp_path / "readonly.txt").chmod(0o444)

    with patch("gentypos._run_typo_generation", return_value={"wrd": "word"}):
        with pytest.raises(SystemExit):
            gentypos.main()

    # Restore permissions for cleanup
    (tmp_path / "readonly.txt").chmod(0o644)

def test_main_cli_defaults_stdout(monkeypatch, capsys):
    # Tests lines 882, 891, 897, 898
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--no-filter"])
    # Ensure gentypos.yaml doesn't exist in current dir if we want to hit line 882
    # But wait, it might exist in the repo root.

    with patch("os.path.exists", return_value=False):
        with patch("gentypos._run_typo_generation", return_value={"wrd": "word"}):
            gentypos.main()
            captured = capsys.readouterr()
            assert "wrd -> word" in captured.out

def test_main_output_header_stdout(monkeypatch, capsys):
    # Test stdout with header (lines 975-978)
    # Force output_file to '-' to ensure we hit the stdout branch
    # Line 897 usually sets it to '-' in CLI mode if not provided,
    # but let's be explicit via argv if possible, or just mock it.
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--no-filter", "--format", "table", "--output", "-"])
    with patch("gentypos._run_typo_generation", return_value={"wrd": "word"}):
        gentypos.main()
        captured = capsys.readouterr()
        assert "[default.extend-words]" in captured.out
        assert 'wrd = "word"' in captured.out

def test_main_word_length_config_not_present(monkeypatch):
    # Test line 912: if 'word_length' not in config
    monkeypatch.setattr(sys, "argv", ["gentypos.py", "word", "--no-filter", "--min-length", "2"])
    # Mocking os.path.exists to ensure no config file is loaded
    with patch("os.path.exists", return_value=False):
        with patch("gentypos._run_typo_generation", return_value={}):
            gentypos.main()
            # If it reaches here without error, it's fine.
            # Coverage will tell if line 912 was hit.

def test_generate_typos_by_replacement_multi_char():
    # Multi-character substring replacements
    adjacent = {}
    custom = {'ph': {'f'}}
    result = gentypos.generate_typos_by_replacement('phone', adjacent, custom)
    assert 'fone' in result

    # Test multiple occurrences
    result_multi = gentypos.generate_typos_by_replacement('phph', adjacent, custom)
    assert 'fph' in result_multi
    assert 'phf' in result_multi
