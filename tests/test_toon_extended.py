from sdk.toon_extended import chunk_toon, from_toon, to_toon


def test_schema_and_casting():
    objs = [{"id": 1, "flag": True, "val": 3.14}, {"id": 2, "flag": False, "val": 2.0}]
    toon = to_toon(objs, include_schema=True)
    parsed = from_toon(toon)
    assert isinstance(parsed[0]["id"], int)
    assert isinstance(parsed[0]["flag"], bool)
    assert isinstance(parsed[0]["val"], float)


def test_chunking():
    objs = [{"id": i} for i in range(7)]
    chunks = chunk_toon(objs, chunk_size=3, include_schema=False)
    assert len(chunks) == 3
    assert chunks[0]["start"] == 0 and chunks[0]["end"] == 3
    assert chunks[-1]["end"] == 7
