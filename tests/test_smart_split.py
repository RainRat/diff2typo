from multitool import _smart_split

def test_smart_split_camel_case():
    assert _smart_split("myVariable") == ["my", "Variable"]

def test_smart_split_snake_case():
    assert _smart_split("some_other_word") == ["some", "other", "word"]

def test_smart_split_alphanumeric():
    assert _smart_split("123Numbers") == ["123", "Numbers"]

def test_smart_split_mixed():
    assert _smart_split("myVariable some_other_word 123Numbers") == ["my", "Variable", "some", "other", "word", "123", "Numbers"]

def test_smart_split_special_chars():
    # Only special characters should return empty list (since they are separators)
    assert _smart_split("!!!") == []
    assert _smart_split("   ") == []

def test_smart_split_empty():
    assert _smart_split("") == []

def test_smart_split_single_char():
    assert _smart_split("A") == ["A"]
    assert _smart_split("a") == ["a"]
    assert _smart_split("1") == ["1"]

def test_smart_split_complex_casing():
    # ABC123def -> ["ABC", "123", "def"]
    assert _smart_split("ABC123def") == ["ABC", "123", "def"]
    # XMLHTTPRequest -> ["XMLHTTP", "Request"]
    # Note: The current regex [A-Z]+(?![a-z]) matches "XMLHTTP" because only "T" is followed by lowercase "r"?
    # Wait, XMLHTTP... the 'P' is NOT followed by a lowercase letter.
    # [A-Z]?[a-z]+ | [A-Z]+(?![a-z]) | [0-9]+
    # For XMLHTTPRequest:
    # 1. XMLHTTP matches [A-Z]+(?![a-z]) because 'P' is not followed by lowercase.
    # 2. Request matches [A-Z]?[a-z]+
    assert _smart_split("XMLHTTPRequest") == ["XMLHTTP", "Request"]
    # lowercaseThenUPPERCASE -> ["lowercase", "Then", "UPPERCASE"]
    assert _smart_split("lowercaseThenUPPERCASE") == ["lowercase", "Then", "UPPERCASE"]
