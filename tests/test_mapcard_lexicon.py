from pathlib import Path

import pytest

from counterstrat.mapcard.lexicon import build_lexicon


def test_lexicon_engine_base_plus_overlay(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("map: m\nzones:\n  A: {aliases: [a site], tags: [site]}\n")
    lex = build_lexicon("m", ["A", "B"], overlay)
    assert set(lex.zones) == {"A", "B"}
    assert lex.zones["A"].tags == ["site"]
    assert lex.zones["B"].aliases == []
    assert lex.is_valid_zone("A") and not lex.is_valid_zone("Ghost")
    assert len(lex.checksum) == 12
    # determinism: same inputs → same checksum
    assert build_lexicon("m", ["B", "A"], overlay).checksum == lex.checksum


def test_lexicon_overlay_unknown_zone_rejected(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("map: m\nzones:\n  Ghost: {tags: [site]}\n")
    with pytest.raises(ValueError, match="Ghost"):
        build_lexicon("m", ["A"], overlay)


def test_lexicon_overlay_map_mismatch(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("map: wrong_map\nzones:\n  A: {tags: [site]}\n")
    with pytest.raises(ValueError, match="wrong_map"):
        build_lexicon("m", ["A"], overlay)


def test_lexicon_overlay_subdivision_rejected(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("map: m\nzones:\n  A: {parent: B}\n")
    with pytest.raises(ValueError, match="parent"):
        build_lexicon("m", ["A", "B"], overlay)


def test_lexicon_no_overlay():
    lex = build_lexicon("m", ["A", "B"])
    assert set(lex.zones) == {"A", "B"}
    assert lex.zones["A"].aliases == []
    assert lex.zones["A"].tags == []
    assert lex.zones["A"].engine_place == "A"
    assert len(lex.checksum) == 12


def test_anubis_lexicon_real():
    # Test with real de_anubis overlay and 28 verified names
    overlay_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "counterstrat"
        / "mapcard"
        / "overlays"
        / "de_anubis.yaml"
    )
    places = [
        "Alley",
        "BackofB",
        "BombsiteA",
        "BombsiteB",
        "Bricks",
        "Bridge",
        "CTSideUpper",
        "CTSpawn",
        "Canal",
        "Connector",
        "Fountain",
        "Heaven",
        "LowerTunnel",
        "Main",
        "MidDoors",
        "Middle",
        "OutsideLong",
        "PalaceInterior",
        "Ruins",
        "SnipersNest",
        "Street",
        "TSideUpper",
        "TSpawn",
        "TStairs",
        "Tunnel",
        "TunnelStairs",
        "Walkway",
        "Water",
    ]
    lex = build_lexicon("de_anubis", places, overlay_path)
    assert len(lex.zones) == 28
    assert lex.zones["BombsiteA"].aliases == ["A site"]
    assert "site" in lex.zones["BombsiteA"].tags


def test_ancient_lexicon_real():
    overlay_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "counterstrat"
        / "mapcard"
        / "overlays"
        / "de_ancient.yaml"
    )
    places = ["BombsiteA", "BombsiteB", "CTSpawn", "TSpawn", "Middle"]
    lex = build_lexicon("de_ancient", places, overlay_path)
    assert len(lex.zones) == 5
    assert lex.zones["BombsiteA"].aliases == ["A site"]
    assert "site" in lex.zones["BombsiteA"].tags
    assert lex.zones["Middle"].aliases == []
