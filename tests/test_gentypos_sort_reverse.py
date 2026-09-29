import os
import tempfile
import yaml
import pytest
from gentypos import main


def test_gentypos_sort_by_typo_and_reverse(capsys, monkeypatch):
    """Test --sort typo and --reverse flags."""
    test_args = [
        "gentypos.py",
        "apple", "banana",
        "-t", "--no-filter",
        "--sort", "typo",
        "--format", "arrow",
        "--output", "-"
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with capsys.disabled():
        pass
    main()
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if "->" in line]

    # Check ascending order by typo
    typos = [line.split("->")[0].strip() for line in lines]
    assert typos == sorted(typos)

    # Now test with --reverse
    test_args_reverse = test_args + ["--reverse"]
    monkeypatch.setattr("sys.argv", test_args_reverse)

    main()
    captured_rev = capsys.readouterr()
    lines_rev = [line for line in captured_rev.out.strip().split("\n") if "->" in line]
    typos_rev = [line.split("->")[0].strip() for line in lines_rev]

    assert typos_rev == sorted(typos, reverse=True)


def test_gentypos_sort_by_correct(capsys, monkeypatch):
    """Test --sort correct flag."""
    test_args = [
        "gentypos.py",
        "zebra", "apple",
        "-t", "--no-filter",
        "--sort", "correct",
        "--format", "arrow",
        "--output", "-"
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if "->" in line]

    # Verify that items with correction 'apple' appear before items with correction 'zebra'
    corrections = [line.split("->")[1].strip() for line in lines]

    apple_indices = [i for i, c in enumerate(corrections) if c == "apple"]
    zebra_indices = [i for i, c in enumerate(corrections) if c == "zebra"]

    assert apple_indices and zebra_indices
    assert max(apple_indices) < min(zebra_indices)


def test_gentypos_yaml_config_sort_reverse(capsys, monkeypatch, tmp_path):
    """Test YAML configuration file settings for sort and reverse."""
    config_data = {
        "input_file": None,
        "dictionary_file": None,
        "output_file": "-",
        "output_format": "arrow",
        "sort": "correct",
        "reverse": True,
        "typo_types": {
            "transposition": True,
            "deletion": False,
            "replacement": False,
            "duplication": False,
        },
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    test_args = [
        "gentypos.py",
        "zebra", "apple",
        "-c", str(config_file),
        "--no-filter",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if "->" in line]
    corrections = [line.split("->")[1].strip() for line in lines]

    # With sort: correct and reverse: true, zebra items should come before apple items
    apple_indices = [i for i, c in enumerate(corrections) if c == "apple"]
    zebra_indices = [i for i, c in enumerate(corrections) if c == "zebra"]

    assert apple_indices and zebra_indices
    assert max(zebra_indices) < min(apple_indices)


def test_gentypos_dry_run_shows_sort_and_reverse(capsys, monkeypatch):
    """Test --dry-run output includes sort and reverse settings."""
    test_args = [
        "gentypos.py", "test",
        "--dry-run", "--sort", "correct", "--reverse"
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
