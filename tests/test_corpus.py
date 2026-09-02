import json

import pytest

from counterstrat.corpus import register_demo


@pytest.mark.demo
def test_register_demo_idempotent(demo_path, tmp_path):
    manifest = tmp_path / "corpus.jsonl"
    rec1 = register_demo(demo_path, manifest)
    rec2 = register_demo(demo_path, manifest)
    assert rec1.match_id == rec2.match_id
    assert rec1.map_name == "de_anubis"
    assert rec1.patch_version == "14178"
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["match_id"] == rec1.match_id
