from sdk.toon import from_toon, to_toon


def test_roundtrip_basic():
    objs = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    toon = to_toon(objs)
    parsed = from_toon(toon)
    # values from basic toon are strings when using toon.py (no schema)
    assert parsed[0]["a"] == "1"
    assert parsed[1]["b"] == "y"


def test_escape_unescape():
    objs = [{"col": "Line1\nLine2"}, {"col": "Pipe | here"}]
    toon = to_toon(objs)
    parsed = from_toon(toon)
    assert parsed[0]["col"] == "Line1\nLine2"
    assert parsed[1]["col"] == "Pipe | here"
