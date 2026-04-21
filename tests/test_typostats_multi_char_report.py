import typostats

def test_generate_report_multi_char_summary_stderr(capsys):
    counts = {('a', 'aa'): 1, ('bb', 'b'): 1}
    typostats.generate_report(counts, include_deletions=True, quiet=False)
    captured = capsys.readouterr()
    assert "Insertions [Ins]:" in captured.err
    assert "Deletions [Del]:" in captured.err
    assert "1/2" in captured.err

def test_generate_report_multi_char_summary_file(tmp_path):
    output_file = tmp_path / "report.txt"
    counts = {('abc', 'ab'): 5}

    typostats.generate_report(
        counts,
        output_file=str(output_file),
        allow_2to1=True
    )

    content = output_file.read_text()
    assert "Deletions [Del]:" in content
    assert "5/5" in content

def test_generate_report_multi_char_summary_mixed(capsys):
    counts = {('a', 'b'): 3, ('c', 'cc'): 2}

    typostats.generate_report(counts, allow_1to2=True, quiet=False)
    captured = capsys.readouterr()
    assert "Insertions [Ins]:" in captured.err
    assert "2/5" in captured.err
