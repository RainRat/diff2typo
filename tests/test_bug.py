import typostats

def test_is_one_letter_replacement_bug():
    assert typostats.is_one_letter_replacement("caat", "cat", allow_two_char=True) == ('a', 'aa')
    assert typostats.is_one_letter_replacement("catt", "cat", allow_two_char=True) == ('t', 'tt')
