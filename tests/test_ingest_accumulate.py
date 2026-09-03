"""Task 0: teambooks must accumulate scripts across demos of the same (team, map)."""

import json
from pathlib import Path

from conftest import SYNTHETIC_TEAM

from counterstrat.mining.tendencies import build_teambook
from counterstrat.web.ingest import _scripts_for_team


def _write_scripts(data_root: Path, scripts) -> None:
    for s in scripts:
        p = data_root / "scripts" / s.match_id / f"round_{s.round_num}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s.to_json(), encoding="utf-8")


def _manifest_line(match_id: str, map_name: str) -> str:
    return json.dumps(
        {
            "match_id": match_id,
            "path": f"demos/{match_id}.dem",
            "map_name": map_name,
            "patch_version": "14178",
            "demo_version_guid": "guid",
            "server_name": "srv",
            "registered_at": "2026-09-03T00:00:00+00:00",
        }
    )


def _write_manifest(data_root: Path, entries: list[tuple[str, str]]) -> None:
    lines = [_manifest_line(mid, map_name) for mid, map_name in entries]
    (data_root / "corpus.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_scripts_for_team_accumulates_prior_matches(tmp_path, synthetic_scripts):
    prior = [s.model_copy(update={"match_id": "m0"}) for s in synthetic_scripts]
    _write_scripts(tmp_path, prior)
    _write_manifest(tmp_path, [("m0", "de_anubis"), ("m1", "de_anubis")])

    merged = _scripts_for_team(tmp_path, "de_anubis", SYNTHETIC_TEAM, synthetic_scripts)

    assert {s.match_id for s in merged} == {"m0", "m1"}
    assert len(merged) == 2 * len(synthetic_scripts)


def test_scripts_for_team_skips_other_maps_and_teams(tmp_path, synthetic_scripts):
    other_map = [
        s.model_copy(update={"match_id": "m2", "map_name": "de_ancient"}) for s in synthetic_scripts
    ]
    other_team = [
        s.model_copy(update={"match_id": "m3", "t_team_key": "zzz", "ct_team_key": "yyy"})
        for s in synthetic_scripts
    ]
    _write_scripts(tmp_path, other_map + other_team)
    _write_manifest(tmp_path, [("m2", "de_ancient"), ("m3", "de_anubis")])

    merged = _scripts_for_team(tmp_path, "de_anubis", SYNTHETIC_TEAM, synthetic_scripts)

    assert {s.match_id for s in merged} == {"m1"}
    assert len(merged) == len(synthetic_scripts)


def test_teambook_from_accumulated_scripts_spans_matches(tmp_path, synthetic_scripts):
    prior = [s.model_copy(update={"match_id": "m0"}) for s in synthetic_scripts]
    _write_scripts(tmp_path, prior)
    _write_manifest(tmp_path, [("m0", "de_anubis")])

    merged = _scripts_for_team(tmp_path, "de_anubis", SYNTHETIC_TEAM, synthetic_scripts)
    tb = build_teambook(merged, SYNTHETIC_TEAM)

    assert set(tb.generated_from) == {"m0", "m1"}
    # Every tendency's n doubles relative to a single-match book.
    single = build_teambook(synthetic_scripts, SYNTHETIC_TEAM)
    merged_total = sum(t.n for t in tb.tendencies)
    assert merged_total == 2 * sum(t.n for t in single.tendencies)


def test_scripts_for_team_survives_unreadable_script(tmp_path, synthetic_scripts):
    _write_manifest(tmp_path, [("m4", "de_anubis")])
    bad = tmp_path / "scripts" / "m4" / "round_1.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json", encoding="utf-8")

    merged = _scripts_for_team(tmp_path, "de_anubis", SYNTHETIC_TEAM, synthetic_scripts)

    assert {s.match_id for s in merged} == {"m1"}
