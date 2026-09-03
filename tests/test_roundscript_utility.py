import json
from pathlib import Path

import polars as pl
import pytest

from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.roundscript.models import UtilEvent
from counterstrat.roundscript.utility import (
    cluster_lineups,
    subdivision_report,
    utility_events,
    utility_events_with_xyz,
)

REPO = Path(__file__).resolve().parents[1]
SUBDIVISION_JSON = REPO / "data" / "mapcards" / "de_anubis" / "subdivision_report.json"


def _two_synthetic_lineups(
    n_each: int,
    separation: float,
    nade: str = "smoke",
    to_zone: str = "Window",
) -> tuple[list[UtilEvent], pl.DataFrame]:
    """Two tight (origin, landing) lineups `separation` units apart, landing in one place."""
    events: list[UtilEvent] = []
    rows: list[dict[str, float]] = []
    for lineup in (0, 1):
        shift = lineup * separation
        for i in range(n_each):
            jitter = i * 5.0  # <= 20 units of spread: same cluster at eps=150
            events.append(
                UtilEvent(
                    t=float(10 * lineup + i),
                    thrower=f"p{lineup}",
                    side="T",
                    nade=nade,
                    from_zone="TSpawn" if lineup == 0 else "Middle",
                    to_zone=to_zone,
                )
            )
            rows.append(
                {
                    "origin_x": shift + jitter,
                    "origin_y": shift + jitter,
                    "origin_z": 0.0,
                    "landing_x": 2000.0 + shift + jitter,
                    "landing_y": 2000.0 + shift + jitter,
                    "landing_z": 0.0,
                }
            )
    return events, pl.DataFrame(rows)


def test_cluster_lineups_two_smokes():
    evs, xyz = _two_synthetic_lineups(n_each=5, separation=800.0)
    names = cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    labelled = {e.lineup_id for e in evs if e.lineup_id}
    assert len(labelled) == 2  # two distinct lineup ids
    assert all("-" in x for x in labelled)
    # same landing place -> ordinals disambiguate, size tie broken by centroid order
    assert labelled == {"Window-S1", "Window-S2"}
    assert set(names.values()) == labelled
    assert [e.lineup_id for e in evs[:5]] == ["Window-S1"] * 5
    assert [e.lineup_id for e in evs[5:]] == ["Window-S2"] * 5


def test_cluster_lineups_noise_unlabelled():
    evs, xyz = _two_synthetic_lineups(n_each=1, separation=800.0)
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    assert all(e.lineup_id is None for e in evs)


def test_cluster_lineups_separates_nade_types():
    smokes, smoke_xyz = _two_synthetic_lineups(n_each=3, separation=0.0)
    flashes, flash_xyz = _two_synthetic_lineups(n_each=3, separation=0.0, nade="flash")
    evs = smokes + flashes
    names = cluster_lineups(evs, pl.concat([smoke_xyz, flash_xyz]), eps=150.0, min_samples=3)
    # identical coordinates, different nade -> two clusters, distinct initials
    assert set(names.values()) == {"Window-S1", "Window-F1"}


def test_cluster_lineups_empty_and_mismatched():
    assert cluster_lineups([], pl.DataFrame()) == {}
    evs, xyz = _two_synthetic_lineups(n_each=3, separation=800.0)
    with pytest.raises(ValueError, match="rows"):
        cluster_lineups(evs, xyz.head(2))


def test_cluster_lineups_resets_previous_ids():
    evs, xyz = _two_synthetic_lineups(n_each=1, separation=800.0)
    for e in evs:
        e.lineup_id = "stale"
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    assert all(e.lineup_id is None for e in evs)


def test_subdivision_report_flags_collapsed_place(tmp_path):
    evs, xyz = _two_synthetic_lineups(n_each=5, separation=800.0)
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    mapper = ZoneMapper.fit(
        pl.DataFrame(
            {
                "X": [0.0] * 50 + [1000.0] * 50,
                "Y": [0.0] * 100,
                "Z": [0.0] * 100,
                "last_place_name": ["Window"] * 50 + ["Middle"] * 50,
            }
        )
    )
    out = tmp_path / "subdivision_report.json"
    report = subdivision_report(evs, mapper, out_path=out)
    assert report == {"Window": 2}
    assert json.loads(out.read_text()) == {"Window": 2}


def test_subdivision_report_single_cluster_place_not_flagged():
    evs, xyz = _two_synthetic_lineups(n_each=3, separation=0.0)
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    mapper = ZoneMapper.fit(
        pl.DataFrame(
            {
                "X": [0.0] * 50,
                "Y": [0.0] * 50,
                "Z": [0.0] * 50,
                "last_place_name": ["Window"] * 50,
            }
        )
    )
    assert subdivision_report(evs, mapper) == {}


@pytest.fixture(scope="module")
def anubis_utility(anubis_lake, anubis_assets):
    ticks = pl.read_parquet(anubis_lake.ticks)
    mapper = ZoneMapper.fit(ticks)
    lex = build_lexicon(
        "de_anubis",
        unique_places(parse_places(anubis_assets.vents)),
        get_default_overlay_path("de_anubis"),
    )
    rounds = pl.read_parquet(anubis_lake.rounds)["round_num"].to_list()
    events: list[UtilEvent] = []
    frames: list[pl.DataFrame] = []
    for rn in rounds:
        evs, xyz = utility_events_with_xyz(anubis_lake, mapper, int(rn))
        assert [e.t for e in evs] == sorted(e.t for e in evs)
        events.extend(evs)
        frames.append(xyz)
    return {"events": events, "raw_xyz": pl.concat(frames), "mapper": mapper, "lex": lex}


@pytest.mark.demo
def test_utility_events_speak_lexicon(anubis_utility):
    events = anubis_utility["events"]
    lex = anubis_utility["lex"]
    # 173 smokes + 131 infernos verified in the lake, plus flashes/HEs
    assert len(events) >= 304
    assert {e.nade for e in events} <= {"smoke", "molly", "flash", "he"}
    assert all(lex.is_valid_zone(e.to_zone) for e in events)
    assert all(lex.is_valid_zone(e.from_zone) for e in events)
    assert {e.side for e in events} == {"T", "CT"}
    assert all(e.thrower for e in events)
    # blind durations land on flashes only
    flashes = [e for e in events if e.nade == "flash"]
    assert any(e.blinded for e in flashes)
    assert all(not e.blinded for e in events if e.nade != "flash")
    assert all(d > 0.0 for e in flashes for _, d in e.blinded)


@pytest.mark.demo
def test_utility_events_single_round_matches_helper(anubis_lake, anubis_utility):
    mapper = anubis_utility["mapper"]
    events, xyz = utility_events_with_xyz(anubis_lake, mapper, 1)
    assert [e.model_dump() for e in utility_events(anubis_lake, mapper, 1)] == [
        e.model_dump() for e in events
    ]
    assert xyz.height == len(events)
    assert any(e.nade == "smoke" for e in events)


@pytest.mark.demo
def test_cluster_lineups_and_subdivision_on_demo(anubis_utility):
    events = anubis_utility["events"]
    names = cluster_lineups(events, anubis_utility["raw_xyz"], min_samples=3)
    assert len(names) >= 1
    assert all(n.count("-") == 1 for n in names.values())
    assert any(e.lineup_id is not None for e in events)

    report = subdivision_report(events, anubis_utility["mapper"], out_path=SUBDIVISION_JSON)
    assert SUBDIVISION_JSON.exists()
    assert json.loads(SUBDIVISION_JSON.read_text()) == report
    assert all(v >= 2 for v in report.values())
