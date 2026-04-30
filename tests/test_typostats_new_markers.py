import typostats

def test_new_markers(capsys):
    """Specifically verify the logic for [Ins], [Del], [1:2], and [2:1] markers."""
    counts = {
        ('a', 'ab'): 1,   # Insertion
        ('bc', 'b'): 1,   # Deletion
        ('m', 'rn'): 1,   # 1-to-2 replacement
        ('ph', 'f'): 1    # 2-to-1 replacement
    }
    typostats.generate_report(counts, all=True)
    captured = capsys.readouterr().out

    assert "[Ins]" in captured
    assert "[Del]" in captured
    assert "[1:2]" in captured
    assert "[2:1]" in captured

def test_summary_labels(capsys):
    """Verify improved labels in the analysis summary."""
    counts = {('teh', 'the'): 1}
    # generate_report prints summary to stderr when no output_file
    typostats.generate_report(counts, quiet=False)
    captured = capsys.readouterr().err

    assert "Total patterns found:" in captured
    assert "Total patterns kept:" in captured
    assert "Unique patterns found:" in captured
