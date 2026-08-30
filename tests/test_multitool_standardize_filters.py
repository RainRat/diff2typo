import sys
import pytest
import multitool


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text(
        "above above above above above above above above above above\n"
        "abovf\n"
        "the the the the the the the the the the\n"
        "teh\n"
    )
    return str(f)


@pytest.mark.parametrize(
    "flags,expected_present,expected_absent",
    [
        (["--fuzzy", "2"], ["above", "the"], ["abovf", "teh"]),
        (["--fuzzy", "2", "--keyboard"], ["above", "teh"], ["abovf"]),
        (["--fuzzy", "2", "--transposition"], ["the", "abovf"], ["teh"]),
        (["--fuzzy", "2", "--keyboard", "--transposition"], [], ["abovf", "teh"]),
        (["--transposition"], ["abovf"], ["teh"]),
    ],
)
def test_standardize_typo_filters(
    sample_file, monkeypatch, capsys, flags, expected_present, expected_absent
):
    monkeypatch.setattr(
        sys, "argv", ["multitool.py", "standardize", sample_file] + flags
    )
    try:
        multitool.main()
    except SystemExit:
        pass
    captured = capsys.readouterr()
    for word in expected_present:
        assert word in captured.out
    for word in expected_absent:
        assert word not in captured.out
