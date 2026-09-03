"""Callout alias tests: one vocabulary at the LLM boundary (feature B)."""

import pytest

from counterstrat.aliases import (
    Renamer,
    alias_fingerprint,
    load_aliases,
    save_aliases,
)

ALIASES = {"Middle": "Mid", "Water": "Banana Low"}


def test_renamer_resolves_and_renames_text():
    r = Renamer(ALIASES)
    assert r.resolve("Middle") == "Mid"
    assert r.resolve("BombsiteA") == "BombsiteA"
    text = "Attack `Middle` then `Water`; TSpawn > Water > Middle. Not middleman."
    out = r.rename_text(text)
    assert out == "Attack `Mid` then `Banana Low`; TSpawn > Banana Low > Mid. Not middleman."


def test_renamer_word_boundaries_respect_composites():
    r = Renamer({"Middle": "Mid"})
    # Underscore composites (identifiers) stay; hyphenated lineup ids and
    # formation signatures ("3xMiddle") rename so the model's vocabulary is
    # consistent everywhere.
    assert r.rename_text("team_Middle Middle-H1 Middle") == "team_Middle Mid-H1 Mid"
    assert r.rename_text("3xMiddle 2xTSpawn") == "3xMid 2xTSpawn"


def test_renamer_sql_literals_round_trip():
    r = Renamer(ALIASES)
    sql = "SELECT * FROM ticks WHERE last_place_name = 'Mid' OR zone = 'Banana Low'"
    assert (
        r.unalias_sql(sql)
        == "SELECT * FROM ticks WHERE last_place_name = 'Middle' OR zone = 'Water'"
    )
    # Untouched literals stay untouched; double quotes handled too.
    assert r.unalias_sql('SELECT "Mid"') == 'SELECT "Middle"'
    assert r.unalias_sql("WHERE x = 'CTSpawn'") == "WHERE x = 'CTSpawn'"


def test_identity_renamer_is_noop():
    r = Renamer({})
    assert r.rename_text("`Middle` stays") == "`Middle` stays"
    assert r.unalias_sql("WHERE a='Mid'") == "WHERE a='Mid'"


def test_save_and_load_aliases_roundtrip(tmp_path):
    valid = {"Middle", "Water", "BombsiteA"}
    saved = save_aliases(tmp_path, "de_anubis", {"Middle": " Mid ", "Water": ""}, valid)
    assert saved == {"Middle": "Mid"}  # stripped; empty = removal
    assert load_aliases(tmp_path, "de_anubis") == {"Middle": "Mid"}
    # Fingerprint changes with content, empty maps share the empty print.
    assert alias_fingerprint({"Middle": "Mid"}) != alias_fingerprint({})
    assert alias_fingerprint({}) == alias_fingerprint({})


def test_save_aliases_rejects_bad_input(tmp_path):
    valid = {"Middle", "Water"}
    with pytest.raises(ValueError, match="Unknown zone"):
        save_aliases(tmp_path, "de_anubis", {"Ghost": "X"}, valid)
    with pytest.raises(ValueError, match="collides"):
        save_aliases(tmp_path, "de_anubis", {"Middle": "Water"}, valid)
    with pytest.raises(ValueError, match="Duplicate"):
        save_aliases(tmp_path, "de_anubis", {"Middle": "Mid", "Water": "Mid"}, valid)
    with pytest.raises(ValueError, match="invalid characters"):
        save_aliases(tmp_path, "de_anubis", {"Middle": "Mid`!"}, valid)
