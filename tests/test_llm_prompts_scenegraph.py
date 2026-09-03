"""Pins for format_map_scene_graph - the measured spatial representation.

The lab (docs/plans/2026-09-04-llm-spatiotemporal-representation.md) showed
topology + per-edge seconds + compass bearings + (u,v) anchors beats the raw
card.yaml dump by +19 easy / +36 hard points; these tests pin that exact shape.
"""

from counterstrat.llm.prompts import build_chat_system, format_map_scene_graph
from counterstrat.mapcard.compile import MapCard
from counterstrat.mining.tendencies import TeamBook


def _card(**overrides) -> MapCard:
    card = MapCard(
        map="de_test",
        card_version="1.0",
        game_version="14178",
        nav_source="nav",
        frame={"bbox": {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100}},
        zones={
            "Hub": {"id": "Hub", "aliases": ["middle"], "tags": ["mid_control"]},
            "North": {"id": "North", "aliases": [], "tags": ["site"]},
            "East": {"id": "East", "aliases": [], "tags": []},
            "FarWest": {"id": "FarWest", "aliases": [], "tags": []},
        },
        topology={
            "Hub": {"North": 4.8, "East": 2.0},
            "East": {"Hub": 2.2},
            "FarWest": {"Hub": 9.0},
        },
        rotates=[{"from": "North", "to": "East", "via": ["Hub"], "run_s": 7.0}],
        timings={"T": {"North": 12.5}, "CT": {"North": 5.0}},
        objectives={"sites": ["North"], "round_seconds": 115, "bomb_seconds": 40},
        sightlines=[],
        checksum="abc123def456",
    )
    return card.model_copy(update=overrides) if overrides else card


ANCHORS = {
    "Hub": (0.50, 0.50, "default"),
    "North": (0.50, 0.20, "default"),
    "East": (0.80, 0.50, "default"),
    "FarWest": (0.10, 0.80, "default"),
}


def test_scene_graph_block_shape():
    text = format_map_scene_graph(_card(), ANCHORS)
    assert "u: 0=west edge -> 1=east edge" in text
    assert "v: 0=NORTH edge -> 1=SOUTH edge" in text
    assert "`Hub` (u=0.50, v=0.50)" in text
    assert "-> `North`: 4.8s N" in text
    # raw yaml internals must not leak
    assert "card_version" not in text
    assert "frame:" not in text
    assert "checksum" not in text


def test_bearing_8way():
    text = format_map_scene_graph(_card(), ANCHORS)
    assert "-> `North`: 4.8s N" in text  # due north of Hub
    assert "-> `East`: 2.0s E" in text  # due east of Hub
    # FarWest -> Hub points north-east
    assert "-> `Hub`: 9.0s NE" in text


def test_edges_symmetrized_min():
    # Hub->East 2.0 and East->Hub 2.2: undirected edge takes the min, both sides
    text = format_map_scene_graph(_card(), ANCHORS)
    assert "-> `East`: 2.0s E" in text
    assert "-> `Hub`: 2.0s W" in text


def test_multi_level_suffix():
    anchors = dict(ANCHORS)
    anchors["East"] = (0.80, 0.50, "lower")
    text = format_map_scene_graph(_card(), anchors)
    assert "`East` (u=0.80, v=0.50, lower level)" in text
    assert "`Hub` (u=0.50, v=0.50, upper level)" in text


def test_rotates_and_timings_sections():
    text = format_map_scene_graph(_card(), ANCHORS)
    assert "`North` -> `East`: 7.0s via `Hub`" in text
    assert "earliest_reach" in text
    assert "T: North 12.5s" in text.replace("`", "")
    empty = format_map_scene_graph(_card(rotates=[], timings={}), ANCHORS)
    assert "rotates" not in empty
    assert "earliest_reach" not in empty


def test_tags_and_objectives_render():
    text = format_map_scene_graph(_card(), ANCHORS)
    assert "tags=[site]" in text
    assert "sites=['North']" in text or "sites=[North]" in text


def test_chat_system_embeds_scene_graph():
    tb = TeamBook(
        team_key="tk",
        map_name="de_test",
        card_checksum="chk",
        generated_from=[],
        tendencies=[],
        roles=[],
    )
    block = format_map_scene_graph(_card(), ANCHORS)
    system = build_chat_system(block, tb)
    assert "u: 0=west edge -> 1=east edge" in system
    assert "<map_card>" in system
    assert "topology:" not in system  # no raw yaml
