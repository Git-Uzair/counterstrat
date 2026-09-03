"""Grammar and lexicon linter for RoundScript."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from counterstrat.mapcard.lexicon import Lexicon
    from counterstrat.roundscript.models import RoundScript


def extract_movement_zones(sentence: str) -> list[str]:
    """Extract zone identifiers from a movement sentence."""
    parts = sentence.split(": ", 1)
    if len(parts) < 2:
        return []
    zones: list[str] = []
    tokens = parts[1].split(" > ")
    for tok in tokens:
        tok = tok.strip().lstrip("~")
        head = tok.split()[0] if tok.split() else ""
        zone = head.split("(")[0]
        if zone:
            zones.append(zone)
    return zones


def lint_script(script: "RoundScript", lex: "Lexicon") -> list[str]:
    """Validate a RoundScript against closed lexicon and grammatical contracts.

    Returns a list of problem strings (empty list if clean).
    """
    problems: list[str] = []

    # Beats zones
    for beat in script.beats:
        for _, z in beat.t_form.zones:
            if not lex.is_valid_zone(z):
                problems.append(f"Invalid zone: {z}")
        for _, z in beat.ct_form.zones:
            if not lex.is_valid_zone(z):
                problems.append(f"Invalid zone: {z}")

    # Kills zones
    for k in script.kills:
        if not lex.is_valid_zone(k.zone):
            problems.append(f"Invalid zone: {k.zone}")

    # Utility zones
    for u in script.utility:
        if not lex.is_valid_zone(u.from_zone):
            problems.append(f"Invalid zone: {u.from_zone}")
        if not lex.is_valid_zone(u.to_zone):
            problems.append(f"Invalid zone: {u.to_zone}")

    # Movements zones
    for m in script.movements:
        for z in extract_movement_zones(m.sentence):
            if not lex.is_valid_zone(z):
                problems.append(f"Invalid zone: {z}")

    # First contact check
    if script.kills:
        if script.first_contact is None:
            problems.append("Missing first contact when kills exist")
        elif script.first_contact != script.kills[0]:
            problems.append(
                f"First contact does not match earliest kill: {script.first_contact} != {script.kills[0]}"
            )
    else:
        if script.first_contact is not None:
            problems.append("First contact present but no kills exist")

    return problems
