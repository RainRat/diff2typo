import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_md_table_mode_basic(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "| Typo | Correction |\n"
        "| --- | --- |\n"
        "| teh | the |\n"
        "|  wod  |  word  |\n"
    )
    output_file = tmp_path / "output.txt"

    # Column 1 (default)
    multitool.md_table_mode([str(input_file)], str(output_file), 1, 20, False, right_side=False, clean_items=False)
    assert sorted(output_file.read_text().splitlines()) == ["teh", "wod"]

def test_md_table_mode_right_side(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "| Typo | Correction |\n"
        "| --- | --- |\n"
        "| teh | the |\n"
        "| wod | word |\n"
    )
    output_file = tmp_path / "output.txt"

    # Column 2
    multitool.md_table_mode([str(input_file)], str(output_file), 1, 20, False, right_side=True, clean_items=False)
    assert sorted(output_file.read_text().splitlines()) == ["the", "word"]

def test_md_table_mode_multi_column(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "| Typo | Correction | Extra |\n"
        "|---|---|---|\n"
        "| val1 | val2 | val3 |\n"
    )
    output_file = tmp_path / "output.txt"

    # Should still pick column 1
    multitool.md_table_mode([str(input_file)], str(output_file), 1, 20, False, right_side=False, clean_items=False)
    assert output_file.read_text().strip() == "val1"

    # Should pick column 2 if right_side=True
    multitool.md_table_mode([str(input_file)], str(output_file), 1, 20, False, right_side=True, clean_items=False)
    assert output_file.read_text().strip() == "val2"

def test_md_table_mode_skipping(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "| item | count |\n"
        "| --- | --- |\n"
        "| apple | 1 |\n"
        "| typo | correction |\n"
        "| word 1 | word 2 |\n"
        "| left | right |\n"
    )
    output_file = tmp_path / "output.txt"
    multitool.md_table_mode([str(input_file)], str(output_file), 1, 20, False, clean_items=False)
    # Only "apple" should remain, others are headers or dividers
    assert output_file.read_text().strip() == "apple"

def test_extract_pairs_from_md_table(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "| Typo | Correction |\n"
        "| --- | --- |\n"
        "| teh | the |\n"
        "| wod | word |\n"
    )
    output_file = tmp_path / "output.txt"

    # pairs mode uses _extract_pairs
    sys.argv = ["multitool.py", "pairs", str(input_file), "--output", str(output_file)]
    multitool.main()

    content = output_file.read_text()
    assert "teh -> the" in content
    assert "wod -> word" in content

def test_swap_mode_with_md_table_input(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "| Typo | Correction |\n"
        "| --- | --- |\n"
        "| teh | the |\n"
    )
    output_file = tmp_path / "output.txt"

    # swap mode uses _extract_pairs
    sys.argv = ["multitool.py", "swap", str(input_file), "--output", str(output_file)]
    multitool.main()

    assert output_file.read_text().strip() == "the -> teh"

def test_zip_mode_empty_after_cleaning(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("!!!\napple")
    file2 = tmp_path / "file2.txt"
    file2.write_text("???\nbanana")
    output_file = tmp_path / "output.txt"

    # Zip them. !!! and ??? should be empty after character cleaning and thus skipped.
    multitool.zip_mode([str(file1)], str(file2), str(output_file), 1, 20, False, clean_items=True)

    assert output_file.read_text().strip() == "apple -> banana"

def test_swap_mode_empty_after_cleaning(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! -> ???\nteh -> the")
    output_file = tmp_path / "output.txt"

    # Swap them. !!! -> ??? should be skipped after cleaning.
    multitool.swap_mode([str(input_file)], str(output_file), 1, 20, False, clean_items=True)

    assert output_file.read_text().strip() == "the -> teh"

def test_pairs_mode_empty_after_cleaning(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! -> ???\nteh -> the")
    output_file = tmp_path / "output.txt"

    # Pairs mode. !!! -> ??? should be skipped after cleaning.
    multitool.pairs_mode([str(input_file)], str(output_file), 1, 20, False, clean_items=True)

    assert output_file.read_text().strip() == "teh -> the"

def test_md_table_parsing_no_leading_trailing_pipes(tmp_path):
    input_file = tmp_path / "input.md"
    # Technically Markdown tables need leading/trailing pipes for some parsers,
    # but the current logic in multitool requires startswith('|').
    # Let's verify what happens if there are no leading pipes.
    input_file.write_text(
        "Typo | Correction\n"
        "--- | ---\n"
        "teh | the\n"
    )
    output_file = tmp_path / "output.txt"

    multitool.md_table_mode([str(input_file)], str(output_file), 1, 20, False, clean_items=False)
    # Based on content.startswith('|'), these should NOT be picked up.
    assert output_file.read_text().strip() == ""

def test_md_table_parsing_with_bullet_points(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "- | teh | the |\n"
        "* | wod | word |\n"
    )
    output_file = tmp_path / "output.txt"

    # _extract_pairs strips bullet points before checking for |
    sys.argv = ["multitool.py", "pairs", str(input_file), "--output", str(output_file)]
    multitool.main()

    content = output_file.read_text()
    assert "teh -> the" in content
    assert "wod -> word" in content
