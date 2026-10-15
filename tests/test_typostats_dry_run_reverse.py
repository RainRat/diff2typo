from unittest.mock import patch
import typostats


def test_dry_run_reverse_count_sorting(tmp_path, caplog):
    typo_file = tmp_path / "typos.txt"
    # Provide data where frequencies differ
    typo_file.write_text("teh -> the\nteh -> the\nrecieve -> receive\n", encoding="utf-8")

    with patch(
        "sys.argv",
        [
            "typostats.py",
            str(typo_file),
            "--dry-run",
            "--reverse",
        ],
    ):
        with caplog.at_level("INFO"):
            typostats.main()

    log_text = caplog.text
    assert "--- TYPOSTATS DRY RUN ---" in log_text
    assert "Dry run complete. No files were written or exported." in log_text
    # Check that sample preview items were rendered in ascending frequency order
    pos_ie = log_text.find("'ie' -> 'ei'")  # count 1
    pos_eh = log_text.find("'eh' -> 'he'")  # count 2
    assert pos_ie != -1 and pos_eh != -1
    assert pos_ie < pos_eh
