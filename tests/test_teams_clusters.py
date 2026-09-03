"""Team cluster tests: lineups sharing a player core are one team."""

import json

import polars as pl

from counterstrat.teams import (
    OVERLAP_MIN,
    TeamCluster,
    build_team_clusters,
    load_or_build_clusters,
)

# Five-player cores: B shares 4 players with A (stand-in swap); C is disjoint.
SQUAD_A = [101, 102, 103, 104, 105]
SQUAD_B = [101, 102, 103, 104, 106]
SQUAD_C = [201, 202, 203, 204, 205]
SQUAD_D = [301, 302, 303, 304, 305]


def _lineup_key(sids: list[int]) -> str:
    import hashlib

    return hashlib.sha1(",".join(str(s) for s in sorted(sids)).encode()).hexdigest()[:12]


def _write_match(
    data_root, match_id: str, map_name: str, teams: list[tuple[list[int], str]], rounds: int = 4
) -> None:
    rows = []
    for rn in range(1, rounds + 1):
        for i, (sids, clan) in enumerate(teams):
            rows.append(
                {
                    "round_num": rn,
                    "side": "CT" if i == 0 else "TERRORIST",
                    "team_key": _lineup_key(sids),
                    "steamids": sorted(sids),
                    "clan_name": clan,
                }
            )
    lake_dir = data_root / "lake" / match_id
    lake_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(lake_dir / "rosters.parquet")

    manifest_path = data_root / "corpus.jsonl"
    line = json.dumps(
        {
            "match_id": match_id,
            "path": f"demos/{match_id}.dem",
            "map_name": map_name,
            "patch_version": "1",
            "demo_version_guid": "g",
            "server_name": "s",
            "registered_at": "2026-09-03T00:00:00+00:00",
        }
    )
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def test_stand_in_lineups_merge_into_one_team(tmp_path):
    _write_match(tmp_path, "m_one", "de_ancient", [(SQUAD_A, "team_7yrant"), (SQUAD_C, "team_ost")])
    _write_match(tmp_path, "m_two", "de_ancient", [(SQUAD_B, "team_7yrant"), (SQUAD_D, "team_duk")])

    clusters = build_team_clusters(tmp_path)
    unique = {c.team_id: c for c in clusters.values()}
    assert len(unique) == 3  # 7yrant merged; two distinct opponents

    seven = clusters[_lineup_key(SQUAD_A)]
    assert seven is clusters[_lineup_key(SQUAD_B)]
    assert seven.team_id == min(_lineup_key(SQUAD_A), _lineup_key(SQUAD_B))
    assert seven.name == "team_7yrant"
    assert set(seven.lineup_keys) == {_lineup_key(SQUAD_A), _lineup_key(SQUAD_B)}
    assert set(seven.steamids) == set(SQUAD_A) | set(SQUAD_B)
    assert seven.matches["m_one"].map_name == "de_ancient"
    assert seven.matches["m_one"].rounds == 4
    assert set(seven.matches) == {"m_one", "m_two"}


def test_disjoint_teams_stay_separate(tmp_path):
    _write_match(tmp_path, "m_one", "de_anubis", [(SQUAD_A, "alpha"), (SQUAD_C, "gamma")])
    clusters = build_team_clusters(tmp_path)
    assert clusters[_lineup_key(SQUAD_A)].team_id != clusters[_lineup_key(SQUAD_C)].team_id


def test_transitive_merge_through_shared_core(tmp_path):
    # A ~ B (4 shared) and B ~ E (3 shared with B, only 2 with A): one team.
    squad_e = [103, 104, 106, 901, 902]
    assert len(set(SQUAD_A) & set(squad_e)) < OVERLAP_MIN  # not directly linked to A
    assert len(set(SQUAD_B) & set(squad_e)) >= OVERLAP_MIN
    _write_match(tmp_path, "m1", "de_ancient", [(SQUAD_A, "x"), (SQUAD_C, "c")])
    _write_match(tmp_path, "m2", "de_ancient", [(SQUAD_B, "x"), (SQUAD_C, "c")])
    _write_match(tmp_path, "m3", "de_ancient", [(squad_e, "x"), (SQUAD_C, "c")])
    clusters = build_team_clusters(tmp_path)
    assert clusters[_lineup_key(SQUAD_A)] is clusters[_lineup_key(squad_e)]


def test_cluster_index_resolves_team_id_and_lineups(tmp_path):
    _write_match(tmp_path, "m_one", "de_ancient", [(SQUAD_A, "a"), (SQUAD_C, "c")])
    clusters = build_team_clusters(tmp_path)
    c = clusters[_lineup_key(SQUAD_A)]
    assert clusters[c.team_id] is c  # resolvable by team_id too


def test_load_or_build_writes_and_reuses_cache(tmp_path):
    _write_match(tmp_path, "m_one", "de_ancient", [(SQUAD_A, "a"), (SQUAD_C, "c")])
    assert _lineup_key(SQUAD_A) in load_or_build_clusters(tmp_path)
    teams_path = tmp_path / "teams.json"
    assert teams_path.exists()
    blob = json.loads(teams_path.read_text(encoding="utf-8"))
    assert blob["built_from"] == ["m_one"]

    # A new match invalidates the cache and triggers a rebuild.
    _write_match(tmp_path, "m_two", "de_ancient", [(SQUAD_B, "a"), (SQUAD_D, "d")])
    clusters2 = load_or_build_clusters(tmp_path)
    assert _lineup_key(SQUAD_B) in clusters2
    blob2 = json.loads(teams_path.read_text(encoding="utf-8"))
    assert blob2["built_from"] == ["m_one", "m_two"]


def test_empty_data_root_yields_no_clusters(tmp_path):
    assert build_team_clusters(tmp_path) == {}
    assert isinstance(TeamCluster.model_json_schema(), dict)
