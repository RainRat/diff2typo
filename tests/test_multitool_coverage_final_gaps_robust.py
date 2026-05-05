import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_count_mode_quiet_stdout_no_stderr(tmp_path, capsys):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\napple\nbanana\n")

    multitool.count_mode(
        input_files=[str(input_file)],
        output_file='-',
        min_length=0,
        max_length=100,
        process_output=False,
        output_format='arrow',
        quiet=True
    )

    captured = capsys.readouterr()
    assert captured.err == ""

def test_get_mode_summary_text_truncation(monkeypatch):
    long_summary = "This is a very long summary that definitely exceeds the thirty-three character limit."

    orig_details = multitool.MODE_DETAILS.copy()
    new_details = orig_details.copy()
    new_details["arrow"] = orig_details["arrow"].copy()
    new_details["arrow"]["summary"] = long_summary

    monkeypatch.setattr(multitool, "MODE_DETAILS", new_details)

    output = multitool.get_mode_summary_text()

    expected_truncated = long_summary[:33-3] + "..."
    assert expected_truncated in output
    assert long_summary not in output
