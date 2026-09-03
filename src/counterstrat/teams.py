"""Team identity clustering: lineups sharing a player core are one team.

The lake keys every round by an exact-roster hash (5 steamids), so a single
stand-in splits a team into two identities and analysis stops accumulating.
This module clusters lineup keys into teams by steamid overlap (>= OVERLAP_MIN
of 5 shared, transitively), persists the result to ``data/teams.json``, and
lets every consumer resolve either a team id or any lineup key to the cluster.
Per-round lineup keys in the lake stay untouched as ground truth.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from counterstrat.corpus import load_manifest

logger = logging.getLogger(__name__)

# Lineups sharing at least this many players belong to the same team. Three of
# five is a majority core: it merges one- and two-stand-in lineups without
# gluing genuinely different squads together.
OVERLAP_MIN = 3


class TeamMatch(BaseModel):
    map_name: str
    rounds: int


class TeamCluster(BaseModel):
    team_id: str  # deterministic: smallest lineup key in the cluster
    name: str  # display label: most common clan name
    lineup_keys: list[str]
    steamids: list[int] = Field(default_factory=list)  # union across lineups
    matches: dict[str, TeamMatch] = Field(default_factory=dict)

    def all_keys(self) -> set[str]:
        return {self.team_id, *self.lineup_keys}


class _Lineup(BaseModel):
    key: str
    steamids: set[int] = Field(default_factory=set)
    clan_names: list[str] = Field(default_factory=list)
    matches: dict[str, TeamMatch] = Field(default_factory=dict)


def _find(parent: dict[str, str], k: str) -> str:
    while parent[k] != k:
        parent[k] = parent[parent[k]]
        k = parent[k]
    return k


def _union(parent: dict[str, str], a: str, b: str) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def build_team_clusters(data_root: Path) -> dict[str, TeamCluster]:
    """Cluster every lineup in the lake; index by team_id AND every lineup key."""
    manifest = load_manifest(data_root / "corpus.jsonl")
    lineups: dict[str, _Lineup] = {}

    for match_id in sorted(manifest):
        rec = manifest[match_id]
        rosters_path = data_root / "lake" / match_id / "rosters.parquet"
        if not rosters_path.exists():
            continue
        try:
            df = pl.read_parquet(rosters_path)
        except Exception as exc:  # noqa: BLE001 - one bad lake table must not kill identity
            logger.warning("Unreadable rosters %s: %s", rosters_path, exc)
            continue
        for key_tuple, g in df.partition_by("team_key", as_dict=True).items():
            key = key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple
            key = str(key)
            if not key or key in ("T", "CT", "None"):
                continue
            lu = lineups.setdefault(key, _Lineup(key=key))
            for row in g.iter_rows(named=True):
                lu.steamids.update(int(s) for s in (row.get("steamids") or []))
                clan = str(row.get("clan_name") or "").strip()
                if clan:
                    lu.clan_names.append(clan)
            lu.matches[match_id] = TeamMatch(map_name=rec.map_name, rounds=g.height)

    keys = sorted(lineups)
    parent = {k: k for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if len(lineups[a].steamids & lineups[b].steamids) >= OVERLAP_MIN:
                _union(parent, a, b)

    grouped: dict[str, list[_Lineup]] = {}
    for k in keys:
        grouped.setdefault(_find(parent, k), []).append(lineups[k])

    index: dict[str, TeamCluster] = {}
    for root, members in grouped.items():
        names = Counter(n for m in members for n in m.clan_names)
        matches: dict[str, TeamMatch] = {}
        steamids: set[int] = set()
        for m in members:
            steamids.update(m.steamids)
            matches.update(m.matches)
        cluster = TeamCluster(
            team_id=root,
            name=min((n for n, c in names.items() if c == max(names.values())), default=root)
            if names
            else root,
            lineup_keys=sorted(m.key for m in members),
            steamids=sorted(steamids),
            matches=dict(sorted(matches.items())),
        )
        for k in cluster.all_keys():
            index[k] = cluster
    return index


def write_team_clusters(data_root: Path, index: dict[str, TeamCluster]) -> None:
    unique = {c.team_id: c for c in index.values()}
    manifest = load_manifest(data_root / "corpus.jsonl")
    payload = {
        "built_from": sorted(manifest),
        "clusters": [unique[tid].model_dump() for tid in sorted(unique)],
    }
    (data_root / "teams.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_or_build_clusters(data_root: Path) -> dict[str, TeamCluster]:
    """Serve the cached index when it covers the whole manifest, else rebuild."""
    teams_path = data_root / "teams.json"
    manifest_ids = sorted(load_manifest(data_root / "corpus.jsonl"))
    if teams_path.exists():
        try:
            blob = json.loads(teams_path.read_text(encoding="utf-8"))
            if blob.get("built_from") == manifest_ids:
                index: dict[str, TeamCluster] = {}
                for raw in blob.get("clusters", []):
                    cluster = TeamCluster.model_validate(raw)
                    for k in cluster.all_keys():
                        index[k] = cluster
                return index
        except Exception as exc:  # noqa: BLE001 - torn cache rebuilds below
            logger.warning("Unreadable teams.json, rebuilding: %s", exc)
    index = build_team_clusters(data_root)
    write_team_clusters(data_root, index)
    return index


def resolve_team_id(data_root: Path, team_key: str) -> str:
    """Canonical team id for any team id or lineup key (identity for unknowns)."""
    cluster = load_or_build_clusters(data_root).get(team_key)
    return cluster.team_id if cluster else team_key
