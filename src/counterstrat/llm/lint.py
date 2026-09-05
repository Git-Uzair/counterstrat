"""Anti-hallucination lint for generated scouting text (First Read).

Names keep their historical "dossier" form: the lint predates the dossier
feature's removal and is shared by every LLM text generator.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from counterstrat.mapcard.lexicon import Lexicon
from counterstrat.mining.tendencies import TeamBook


class DossierLint(BaseModel):
    """Anti-hallucination lint findings for generated scouting text."""

    unknown_zones: list[str] = Field(default_factory=list)
    bad_citations: list[str] = Field(default_factory=list)
    freq_mismatches: list[str] = Field(default_factory=list)
    ok: bool = True


def lint_dossier(
    text: str,
    teambook: TeamBook,
    lex: Lexicon,
    valid_evidence: set[str] | Sequence[str],
    aliases: dict[str, str] | None = None,
) -> DossierLint:
    """Audit generated scouting text for fabricated zones, citations, or frequencies.

    With ``aliases`` (canonical -> user callout) the valid vocabulary flips: the
    user's callout is accepted and the renamed canonical is rejected, because
    the model was never shown it (single-vocabulary rule).
    """
    aliases = aliases or {}
    alias_values = set(aliases.values())
    valid_evidence_set = set(valid_evidence)
    unknown_zones: list[str] = []
    bad_citations: list[str] = []
    freq_mismatches: list[str] = []

    # 1. Zone verification: find all backtick tokens
    # Reject TitleCase/CamelCase tokens not in lexicon, player names, or lineup IDs
    player_names = {r.player for r in teambook.roles}
    backtick_tokens = re.findall(r"`([A-Za-z0-9_ -]+)`", text)
    for raw_tok in backtick_tokens:
        tok = raw_tok.strip()
        # Check if single titlecase/PascalCase word of length >= 2
        if re.match(r"^[A-Z][A-Za-z0-9]+$", tok):
            # Allowed if valid zone, known player, lineup id pattern, team key, or side
            is_zone = tok not in aliases and (tok in lex.zones or lex.is_valid_zone(tok))
            is_alias = tok in alias_values
            is_player = tok in player_names
            is_lineup = any(tag in tok for tag in ("-S", "-F", "-M", "-H"))
            is_team_or_side = tok in (teambook.team_key, "CT", "T")
            if (
                not (is_zone or is_alias or is_player or is_lineup or is_team_or_side)
                and tok not in unknown_zones
            ):
                unknown_zones.append(tok)

    # 2. Citation verification: match_id:round_num
    # Match patterns like deadbeefcafe:7, ffffffffffff:99, m1:1, or 1-7065...:7
    citation_candidates = re.findall(r"\b([a-zA-Z0-9_-]+:\d+)\b", text)
    for cit in citation_candidates:
        if cit in valid_evidence_set:
            continue
        # Avoid false positives on clock times like 1:45 or @0:50 unless it contains letters or length > 2
        prefix, _ = cit.split(":", 1)
        is_clock_time = prefix.isdigit() and len(prefix) <= 2
        if is_clock_time:
            continue
        if cit not in bad_citations:
            bad_citations.append(cit)

    # 3. Frequency verification: check NN% adjacent to tendency names within 40 chars
    # Collect expected percentages from TeamBook
    tendency_keywords: dict[str, set[int]] = {}
    all_teambook_pcts: set[int] = set()

    for t in teambook.tendencies:
        b_class = t.key.buy_class.lower()
        b_class_space = b_class.replace("_", " ")
        t_pcts: set[int] = set()

        for z, p in t.first_contact_zone.items():
            pct = round(p * 100)
            t_pcts.add(pct)
            all_teambook_pcts.add(pct)
            tendency_keywords.setdefault(z.lower(), set()).add(pct)

        for s, p in t.site_committed.items():
            pct = round(p * 100)
            t_pcts.add(pct)
            all_teambook_pcts.add(pct)
            tendency_keywords.setdefault(s.lower(), set()).add(pct)

        for p in t.opening_formation.values():
            pct = round(p * 100)
            t_pcts.add(pct)
            all_teambook_pcts.add(pct)

        tendency_keywords.setdefault(b_class, set()).update(t_pcts)
        tendency_keywords.setdefault(b_class_space, set()).update(t_pcts)
        tendency_keywords.setdefault(t.key.score_bucket.lower(), set()).update(t_pcts)

    for r in teambook.roles:
        d_pct = round(r.opening_duel_rate * 100)
        l_pct = round(r.lurk_rate * 100)
        all_teambook_pcts.add(d_pct)
        all_teambook_pcts.add(l_pct)
        tendency_keywords.setdefault(r.player.lower(), set()).update({d_pct, l_pct})

    for m in re.finditer(r"(\d+)%", text):
        val = int(m.group(1))
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        window = text[start:end].lower()

        # Check which tendency keywords appear in this window
        matched_expected: set[int] = set()
        matched_kw: str | None = None
        for kw, expected_set in tendency_keywords.items():
            # Check whole word or phrase
            if re.search(rf"\b{re.escape(kw)}\b", window):
                matched_expected.update(expected_set)
                if matched_kw is None:
                    matched_kw = kw

        if (
            matched_kw is not None
            and matched_expected
            and not any(abs(val - exp) <= 1 for exp in matched_expected)
        ):
            msg = f"{val}% near '{matched_kw}' does not match TeamBook frequencies"
            if msg not in freq_mismatches:
                freq_mismatches.append(msg)

    ok = len(unknown_zones) == 0 and len(bad_citations) == 0 and len(freq_mismatches) == 0
    return DossierLint(
        unknown_zones=unknown_zones,
        bad_citations=bad_citations,
        freq_mismatches=freq_mismatches,
        ok=ok,
    )
