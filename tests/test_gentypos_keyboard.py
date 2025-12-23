import sys
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_get_adjacent_keys_center():
    """
    Test adjacent keys for a central key 'g'.
    'g' (1,4).
    Neighbors:
    (0,3)='r', (0,4)='t', (0,5)='y'
    (1,3)='f',          , (1,5)='h'
    (2,3)='v', (2,4)='b', (2,5)='n'
    """
    adj = gentypos.get_adjacent_keys(include_diagonals=True)
    expected = {'r', 't', 'y', 'f', 'h', 'v', 'b', 'n'}
    assert adj['g'] == expected

    adj_no_diag = gentypos.get_adjacent_keys(include_diagonals=False)
    # No diagonals: t, f, h, b
    expected_no_diag = {'t', 'f', 'h', 'b'}
    assert adj_no_diag['g'] == expected_no_diag

def test_get_adjacent_keys_corners():
    """
    Test adjacent keys for corner keys: 'q', 'p', 'z', 'm'.
    """
    adj = gentypos.get_adjacent_keys(include_diagonals=True)

    # q (0,0) -> w(0,1), a(1,0), s(1,1)
    assert adj['q'] == {'w', 'a', 's'}

    # p (0,9) -> o(0,8), l(1,8). (1,9) is out of bounds.
    assert adj['p'] == {'o', 'l'}

    # z (2,0) -> a(1,0), s(1,1), x(2,1)
    assert adj['z'] == {'a', 's', 'x'}

    # m (2,6) -> n(2,5), h(1,5), j(1,6), k(1,7)
    assert adj['m'] == {'h', 'j', 'n', 'k'}

def test_get_adjacent_keys_edges():
    """
    Test adjacent keys for edge keys (not corners): 'a', 'l'.
    """
    adj = gentypos.get_adjacent_keys(include_diagonals=True)

    # a (1,0) -> q(0,0), w(0,1), s(1,1), z(2,0), x(2,1)
    assert adj['a'] == {'q', 'w', 's', 'z', 'x'}

    # l (1,8) -> i(0,7), o(0,8), p(0,9), k(1,7). Row 2 indices 7,8,9 are out of bounds.
    assert adj['l'] == {'i', 'o', 'p', 'k'}
