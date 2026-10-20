import gentypos

def test_load_file_latin1_fallback(tmp_path):
    p = tmp_path / "latin1.txt"
    p.write_bytes('hello caf\xe9 world'.encode('latin-1'))

    words = gentypos.load_file(str(p))
    assert 'hello' in words
    assert 'world' in words

def test_load_file_empty_content(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n\t  ", encoding='utf-8')

    words = gentypos.load_file(str(p))
    assert words == set()

def test_load_file_json_invalid_fallback(tmp_path):
    p = tmp_path / "invalid.json"
    p.write_text("{invalid json content word1 word2}", encoding='utf-8')

    words = gentypos.load_file(str(p))
    assert "word1" in words
    assert "word2" in words

def test_load_file_yaml_invalid_fallback(tmp_path):
    p = tmp_path / "invalid.yaml"
    p.write_text(":\n  - : [invalid yaml", encoding='utf-8')

    words = gentypos.load_file(str(p))
    assert isinstance(words, set)

def test_load_file_toml_toml_package_fallback(tmp_path, monkeypatch):
    p = tmp_path / "test.toml"
    p.write_text("word1 = 'hello'\nword2 = 'world'", encoding='utf-8')

    monkeypatch.setattr(gentypos, '_TOMLLIB_AVAILABLE', False)
    monkeypatch.setattr(gentypos, '_TOML_AVAILABLE', True)
    words = gentypos.load_file(str(p))
    assert "hello" in words or "world" in words

def test_load_file_toml_invalid_fallback(tmp_path):
    p = tmp_path / "invalid.toml"
    p.write_text("[unclosed table\nkey = invalid", encoding='utf-8')

    words = gentypos.load_file(str(p))
    assert "key" in words or "invalid" in words

def test_load_file_csv_header_and_exception(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("typo,correction\nteh,the\n", encoding='utf-8')

    words = gentypos.load_file(str(p))
    assert "typo" not in words
    assert "correction" not in words
    assert "teh" in words
    assert "the" in words
