"""Map health checks: every silent-gap class found on de_inferno (2026-09-04)
must be mechanically detectable on any map."""

from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.health import check_card_health


def _card(**overrides) -> MapCard:
    card = MapCard(
        map="de_test",
        card_version="1.0",
        game_version="14178",
        nav_source="nav",
        frame={},
        zones={
            "BombsiteA": {"id": "BombsiteA", "aliases": [], "tags": ["site"], "quadrant": "east"},
            "BombsiteB": {"id": "BombsiteB", "aliases": [], "tags": ["site"], "quadrant": "west"},
            "Middle": {"id": "Middle", "aliases": [], "tags": [], "quadrant": "center"},
            "Lonely": {"id": "Lonely", "aliases": [], "tags": []},  # no edges, no quadrant
        },
        topology={
            "BombsiteA": {"Middle": 5.0},
            "Middle": {"BombsiteA": 5.0, "BombsiteB": 6.0},
            "BombsiteB": {"Middle": 6.0},
        },
        rotates=[{"from": "BombsiteA", "to": "BombsiteB", "via": ["Middle"], "run_s": 11.0}],
        timings={"CT": {"BombsiteA": 5.0}, "T": {"BombsiteA": 12.0}},
        objectives={"sites": ["BombsiteA", "BombsiteB"]},
        sightlines=[],
        checksum="chk",
    )
    return card.model_copy(update=overrides) if overrides else card


ANCHORS = {
    "BombsiteA": (0.8, 0.3, "default"),
    "BombsiteB": (0.2, 0.3, "default"),
    "Middle": (0.5, 0.5, "default"),
    "Lonely": (0.9, 0.9, "default"),
}


def _codes(card: MapCard, anchors: dict) -> set[str]:
    return {f["code"] for f in check_card_health(card, anchors)}


def test_healthy_card_reports_only_known_infos():
    codes = _codes(_card(), ANCHORS)
    assert "no_sites" not in codes
    assert "no_rotates" not in codes
    assert "missing_anchor" not in codes
    assert "implausible_edge" not in codes
    # Lonely has no edges and no quadrant: surfaced as info, not silence
    assert "isolated_zones" in codes
    assert "no_sightlines" in codes  # known global gap, always visible


def test_missing_sites_detected():
    card = _card(objectives={"sites": []})
    findings = check_card_health(card, ANCHORS)
    f = next(x for x in findings if x["code"] == "no_sites")
    assert f["level"] == "warn"
    assert "rotates" in f["msg"]  # explains the cascade


def test_sites_without_rotates_detected():
    codes = _codes(_card(rotates=[]), ANCHORS)
    assert "no_rotates" in codes


def test_empty_side_timings_detected():
    codes = _codes(_card(timings={"CT": {}, "T": {"BombsiteA": 12.0}}), ANCHORS)
    assert "no_timings" in codes


def test_missing_anchor_detected():
    anchors = {k: v for k, v in ANCHORS.items() if k != "Middle"}
    codes = _codes(_card(), anchors)
    assert "missing_anchor" in codes


def test_anchor_overlap_detected():
    anchors = dict(ANCHORS)
    anchors["Middle"] = (0.802, 0.301, "default")  # on top of BombsiteA
    codes = _codes(_card(), anchors)
    assert "anchor_overlap" in codes


def test_implausible_edge_detected():
    # 0.6 of the map crossed in under 2 seconds: teleport-class residue
    card = _card(topology={"BombsiteA": {"BombsiteB": 1.0}})
    codes = _codes(card, ANCHORS)
    assert "implausible_edge" in codes
