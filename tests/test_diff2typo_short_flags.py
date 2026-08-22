import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_short_flags_parsing(monkeypatch):
    """Verify that short flags -g, -M, -D, -c, -s parse identically to long flags."""

    import argparse
    original_parse = argparse.ArgumentParser.parse_args
    captured_args = []
    def mock_parse(self, *args, **kwargs):
        res = original_parse(self, *args, **kwargs)
        captured_args.append(res)
        return res

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', mock_parse)
    monkeypatch.setattr(diff2typo, '_read_diff_sources', lambda _: "--- a/f\n+++ b/f\n-teh\n+the")
    monkeypatch.setattr(diff2typo, 'read_words_mapping', lambda *a, **kw: {})
    monkeypatch.setattr(diff2typo, 'read_allowed_words', lambda *a, **kw: set())

    # Test short flags: -g, -M, -D, -c, -s
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'diff2typo.py',
            '-g', 'HEAD~1',
            '-M', 'both',
            '-D', '3',
            '-c', '2',
            '-s', 'count',
            '--quiet'
        ]
    )

    try:
        diff2typo.main()
    except SystemExit:
        pass

    assert len(captured_args) > 0
    args = captured_args[-1]
    assert args.git == 'HEAD~1'
    assert args.mode == 'both'
    assert args.max_dist == 3
    assert args.min_count == 2
    assert args.sort == 'count'
