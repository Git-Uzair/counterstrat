"""Deterministic Level-2 tendency mining and TeamBook profile generation."""

import re
from collections import Counter, defaultdict
from statistics import median

from pydantic import BaseModel, ConfigDict

from counterstrat.roundscript.models import RoundScript

# A distribution is a "read" only when it is concentrated and sampled: below
# either bar the row is noise and must not be quoted as signal (plan Task 1).
SIGNAL_MIN_N = 3
SIGNAL_MIN_CONCENTRATION = 0.5


class TendencyKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    map_name: str
    side: str  # "T" | "CT"
    buy_class: str  # "full_eco" | "semi_eco" | "semi_buy" | "full_buy" | "any"
    score_bucket: str  # "behind" | "even" | "ahead" | "any"
    prev_outcome: str  # "won" | "lost" | "first" | "any"


class Tendency(BaseModel):
    key: TendencyKey
    first_contact_zone: dict[str, float]  # P per zone
    opening_formation: dict[str, float]  # P per formation signature
    site_committed: dict[str, float]  # "A" | "B" | "none" -> P
    lineup_sets: dict[str, float]  # str representation of frozenset -> P
    median_first_contact_s: float | None
    n: int
    evidence: list[str]  # "match_id:round_num"
    low_n: bool = False
    # Aggregation level: 0 = side only, 1 = side+buy, 2 = full situational key.
    # Defaults keep pre-upgrade teambook.json artifacts loadable.
    level: int = 2
    fc_concentration: float = 0.0  # max share in first_contact_zone
    site_concentration: float = 0.0  # max share in site_committed
    signal: bool = False  # n and concentration clear the bars above


class RoleCard(BaseModel):
    player: str
    steamid: str
    modal_zone_fe15: dict[str, str]  # side -> modal zone at B+15
    opening_duel_rate: float  # share of team first-contacts involving them
    lurk_rate: float  # share of rounds with lurk role_hint
    # Extended profile (plan Task 5); defaults keep pre-upgrade JSON loadable.
    opening_kill_rate: float = 0.0  # of their FC involvements, share won
    opening_zones: dict[str, int] = {}  # zone -> FC involvement count
    awp_rounds: int = 0  # rounds with an awp kill by this player
    trade_discipline: float = 0.0  # share of their deaths traded within 4s
    median_fc_t: float | None = None  # median FC time when involved


class TeamBook(BaseModel):
    team_key: str
    map_name: str
    card_checksum: str
    tendencies: list[Tendency]
    roles: list[RoleCard]
    generated_from: list[str]  # match_ids, date-ordered / sorted

    def to_table_text(self) -> str:
        lines = [
            f"# TeamBook: {self.team_key} ({self.map_name})",
            "",
            "## Tendencies",
            "Levels: 0 = side rollup, 1 = side+buy, 2 = full situation. Level-2 rows",
            "are shown only when they carry signal (n >= 3 and a >= 50% concentration).",
            "Opening Formation is a single-tick 15s snapshot signature and fragments",
            "by nature - never present one as THE setup; setup claims come from",
            "per-player movement data (setup posts / role cards), not this column.",
            "",
            "| Lvl | Side | Buy | Score | Prev | FC Zone | Opening Formation | Site | Median FC | N | Signal |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in self.tendencies:
            if t.level == 2 and not t.signal:
                continue  # fine-grained noise stays out of prompts; tools can still fetch it
            fc_str = ", ".join(f"{z} ({p:.0%})" for z, p in t.first_contact_zone.items()) or "-"
            form_str = (
                ", ".join(f"{sig} ({p:.0%})" for sig, p in list(t.opening_formation.items())[:2])
                or "-"
            )
            site_str = ", ".join(f"{s} ({p:.0%})" for s, p in t.site_committed.items()) or "-"
            med_fc = (
                f"{t.median_first_contact_s:.1f}s" if t.median_first_contact_s is not None else "-"
            )
            signal_str = "Yes" if t.signal else ("low_n" if t.low_n else "No")
            lines.append(
                f"| {t.level} | {t.key.side} | {t.key.buy_class} | {t.key.score_bucket} | "
                f"{t.key.prev_outcome} | {fc_str} | {form_str} | {site_str} | {med_fc} | "
                f"{t.n} | {signal_str} |"
            )
        if self.roles:
            lines.extend(
                [
                    "",
                    "## Roles",
                    "| Player | SteamID | Modal Zone (B+15) | Opening Duel Rate | Lurk Rate |",
                    "|---|---|---|---|---|",
                ]
            )
            for r in self.roles:
                fe15_str = (
                    ", ".join(f"{side}: {z}" for side, z in sorted(r.modal_zone_fe15.items()))
                    or "-"
                )
                duel_str = f"{r.opening_duel_rate:.0%}"
                lurk_str = f"{r.lurk_rate:.0%}"
                lines.append(f"| {r.player} | {r.steamid} | {fe15_str} | {duel_str} | {lurk_str} |")
        return "\n".join(lines)

    def to_sentences(self) -> list[str]:
        """One sentence per read: most specific signal rows win, noise is named as such."""

        def covers(specific: Tendency, coarse: Tendency) -> bool:
            if specific.level <= coarse.level or specific.key.side != coarse.key.side:
                return False
            return coarse.key.buy_class in ("any", specific.key.buy_class)

        emitted: list[Tendency] = []
        for level in (2, 1, 0):
            for t in self.tendencies:
                if t.level != level or not t.signal:
                    continue
                if any(covers(e, t) for e in emitted):
                    continue
                emitted.append(t)

        sentences = [self._sentence(t) for t in emitted]
        for side in ("T", "CT"):
            if any(t.key.side == side for t in emitted):
                continue
            l0 = next((t for t in self.tendencies if t.level == 0 and t.key.side == side), None)
            if l0 is not None:
                sentences.append(
                    f"No concentrated first-contact read on {side} side "
                    f"(n={l0.n}) - treat their openings as mixed."
                )
        return sentences

    def _sentence(self, t: Tendency) -> str:
        top_fc, fc_p = ("none", 0.0)
        if t.first_contact_zone:
            top_fc, fc_p = min(t.first_contact_zone.items(), key=lambda x: (-x[1], x[0]))
        pct_str = f"{round(fc_p * 100)}%"
        action = "attacks" if t.key.side == "T" else "contests"
        parts = []
        if t.key.side == "CT":
            parts.append("CT")
        if t.key.buy_class != "any":
            parts.append(t.key.buy_class)
        prefix = f"On {' '.join(parts)}" if parts else "Overall"
        if t.key.score_bucket != "any":
            prefix += f" when {t.key.score_bucket}"
        if top_fc != "none":
            return f"{prefix}, team {action} {top_fc} {pct_str} of the time (n={t.n})."
        return f"{prefix}, team had no first contact (n={t.n})."


def _extract_fe15_zone(sentence: str) -> str:
    """Extract player's zone at freeze_end + 15s (B+15) from a movement sentence."""
    if not sentence:
        return "unknown"
    tok_str = sentence.split(": ", 1)[1] if ": " in sentence else sentence
    raw_tokens = [tok.strip() for tok in tok_str.split(" > ") if tok.strip()]
    if not raw_tokens:
        return "unknown"

    cleaned: list[tuple[str, bool]] = []
    for tok in raw_tokens:
        t = tok.lstrip("~")
        first_word = t.split()[0] if t else ""
        has_long_dwell = bool(re.search(r"\(\d+s\)", first_word))
        zone = re.sub(r"\(\d+s\)", "", first_word).strip()
        cleaned.append((zone, has_long_dwell))

    if len(cleaned) == 1:
        return cleaned[0][0] or "unknown"

    first_zone, first_long_dwell = cleaned[0]
    if "spawn" in first_zone.lower() and not first_long_dwell:
        return cleaned[1][0] or "unknown"
    return first_zone or "unknown"


def _normalize_site(site_str: str | None) -> str:
    """Normalize site string to 'A', 'B', or 'none'."""
    if not site_str:
        return "none"
    s = site_str.strip().upper()
    if "BOMBSITE" in s:
        s = s.replace("BOMBSITE", "")
    if s in ("A", "SITE_A", "SITEA"):
        return "A"
    if s in ("B", "SITE_B", "SITEB"):
        return "B"
    if "A" in s and "B" not in s:
        return "A"
    if "B" in s and "A" not in s:
        return "B"
    return "none"


def _mine_bucket(level: int, key: TendencyKey, g_scripts: list[RoundScript]) -> Tendency:
    """Aggregate one bucket of rounds into a Tendency at the given level."""
    side = key.side
    n = len(g_scripts)

    # first_contact_zone
    fc_counts: Counter[str] = Counter()
    for s in g_scripts:
        z = s.first_contact.zone if s.first_contact else "none"
        fc_counts[z] += 1
    fc_dist = {z: count / n for z, count in fc_counts.items()}
    first_contact_zone = dict(sorted(fc_dist.items(), key=lambda x: (-x[1], x[0])))

    # opening_formation: find beat B+15
    form_counts: Counter[str] = Counter()
    for s in g_scripts:
        beat_b15 = next(
            (
                b
                for b in s.beats
                if b.label == "B+15" or abs(b.t - 15.0) < 1.0 or b.label.startswith("B+15")
            ),
            None,
        )
        if beat_b15:
            form = beat_b15.t_form if side == "T" else beat_b15.ct_form
            sig = " ".join(f"{cnt}x{zone}" for cnt, zone in form.zones) if form.zones else "none"
        else:
            sig = "none"
        form_counts[sig] += 1
    form_dist = {sig: count / n for sig, count in form_counts.items()}
    opening_formation = dict(sorted(form_dist.items(), key=lambda x: (-x[1], x[0])))

    # site_committed
    site_counts: Counter[str] = Counter()
    for s in g_scripts:
        site = _normalize_site(s.plant.site) if s.plant else "none"
        site_counts[site] += 1
    site_dist = {site: count / n for site, count in site_counts.items()}
    site_committed = dict(sorted(site_dist.items(), key=lambda x: (-x[1], x[0])))

    # lineup_sets
    lineup_counts: Counter[str] = Counter()
    for s in g_scripts:
        lineups = {u.lineup_id for u in s.utility if u.side == side and u.lineup_id}
        l_str = ",".join(sorted(lineups))
        lineup_counts[l_str] += 1
    lineup_dist = {l_str: count / n for l_str, count in lineup_counts.items()}
    lineup_sets = dict(sorted(lineup_dist.items(), key=lambda x: (-x[1], x[0])))

    # median_first_contact_s
    fc_times = [s.first_contact.t for s in g_scripts if s.first_contact is not None]
    med_fc = float(median(fc_times)) if fc_times else None

    # evidence
    evidence = sorted(
        [f"{s.match_id}:{s.round_num}" for s in g_scripts],
        key=lambda x: (x.split(":")[0], int(x.split(":")[1])),
    )

    fc_concentration = max(first_contact_zone.values(), default=0.0)
    site_concentration = max(site_committed.values(), default=0.0)
    return Tendency(
        key=key,
        first_contact_zone=first_contact_zone,
        opening_formation=opening_formation,
        site_committed=site_committed,
        lineup_sets=lineup_sets,
        median_first_contact_s=med_fc,
        n=n,
        evidence=evidence,
        low_n=n < SIGNAL_MIN_N,
        level=level,
        fc_concentration=fc_concentration,
        site_concentration=site_concentration,
        signal=n >= SIGNAL_MIN_N and fc_concentration >= SIGNAL_MIN_CONCENTRATION,
    )


class RoundState(BaseModel):
    """One participating round with its situational key, shared by all miners."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: RoundScript
    side: str  # side team_key played this round
    buy_class: str
    score_bucket: str  # "behind" | "even" | "ahead"
    prev_outcome: str  # "won" | "lost" | "first"


def iter_round_states(scripts: list[RoundScript], team_key: str) -> list[RoundState]:
    """Situational state per round team_key played, ordered by (match, round)."""
    participating = [s for s in scripts if team_key in (s.t_team_key, s.ct_team_key)]
    scripts_by_match: dict[str, list[RoundScript]] = defaultdict(list)
    for s in participating:
        scripts_by_match[s.match_id].append(s)

    states: list[RoundState] = []
    for match_id in sorted(scripts_by_match):
        m_scripts = sorted(scripts_by_match[match_id], key=lambda s: s.round_num)
        rounds_map = {s.round_num: s for s in m_scripts}
        for s in m_scripts:
            side = "T" if s.t_team_key == team_key else "CT"

            is_ot_start = s.round_num >= 25 and (s.round_num - 25) % 3 == 0
            if (
                s.round_num == 1
                or s.round_num == 13
                or is_ot_start
                or (s.round_num - 1) not in rounds_map
            ):
                prev_outcome = "first"
            else:
                prev_s = rounds_map[s.round_num - 1]
                prev_side = "T" if prev_s.t_team_key == team_key else "CT"
                won = (prev_s.winner == prev_side) or (prev_s.winner == team_key)
                prev_outcome = "won" if won else "lost"

            team_score = s.score_t if side == "T" else s.score_ct
            opp_score = s.score_ct if side == "T" else s.score_t
            if team_score > opp_score:
                score_bucket = "ahead"
            elif team_score < opp_score:
                score_bucket = "behind"
            else:
                score_bucket = "even"

            econ = s.economy.get(side)
            buy_class = econ.buy_type if econ and hasattr(econ, "buy_type") else "full_buy"

            states.append(
                RoundState(
                    script=s,
                    side=side,
                    buy_class=buy_class,
                    score_bucket=score_bucket,
                    prev_outcome=prev_outcome,
                )
            )
    return states


def build_teambook(scripts: list[RoundScript], team_key: str) -> TeamBook:
    """Mine deterministic Level-2 TeamBook profile for team_key from RoundScripts."""
    participating = [s for s in scripts if s.t_team_key == team_key or s.ct_team_key == team_key]
    if not participating:
        map_name = scripts[0].map_name if scripts else ""
        card_checksum = scripts[0].card_checksum if scripts else ""
        return TeamBook(
            team_key=team_key,
            map_name=map_name,
            card_checksum=card_checksum,
            tendencies=[],
            roles=[],
            generated_from=[],
        )

    # (level, map, side, buy, score, prev) -> list[RoundScript]
    grouped_scripts: dict[tuple[int, str, str, str, str, str], list[RoundScript]] = defaultdict(
        list
    )
    for st in iter_round_states(participating, team_key):
        s = st.script
        grouped_scripts[
            (2, s.map_name, st.side, st.buy_class, st.score_bucket, st.prev_outcome)
        ].append(s)
        grouped_scripts[(1, s.map_name, st.side, st.buy_class, "any", "any")].append(s)
        grouped_scripts[(0, s.map_name, st.side, "any", "any", "any")].append(s)

    tendencies: list[Tendency] = []
    for (level, map_name, side, buy_class, score_bucket, prev_outcome), g in sorted(
        grouped_scripts.items()
    ):
        key = TendencyKey(
            map_name=map_name,
            side=side,
            buy_class=buy_class,
            score_bucket=score_bucket,
            prev_outcome=prev_outcome,
        )
        tendencies.append(_mine_bucket(level, key, g))

    # Sort deterministically: rollups first, then by side and sample size.
    tendencies.sort(
        key=lambda t: (
            t.level,
            t.key.side,
            -t.n,
            t.key.buy_class,
            t.key.score_bucket,
            t.key.prev_outcome,
        )
    )

    # Calculate RoleCards
    player_rounds: Counter[str] = Counter()
    player_lurks: Counter[str] = Counter()
    fe15_zones: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for s in participating:
        team_side = "T" if s.t_team_key == team_key else "CT"
        for m in s.movements:
            if m.side == team_side:
                player_rounds[m.player] += 1
                if m.role_hint == "lurk":
                    player_lurks[m.player] += 1
                z = _extract_fe15_zone(m.sentence)
                fe15_zones[m.player][team_side].append(z)

    # Opening duel rate
    team_fc_rounds = [s for s in participating if s.first_contact is not None]
    total_fc = len(team_fc_rounds)

    # Per-player kill-log aggregates (plan Task 5).
    awp_rounds: dict[str, set[str]] = defaultdict(set)
    deaths: Counter[str] = Counter()
    deaths_traded: Counter[str] = Counter()
    for s in participating:
        round_id = f"{s.match_id}:{s.round_num}"
        for k in s.kills:
            if "awp" in k.weapon.lower():
                awp_rounds[k.killer].add(round_id)
            deaths[k.victim] += 1
            if k.traded_within_4s:
                deaths_traded[k.victim] += 1

    roles: list[RoleCard] = []
    for player in sorted(player_rounds.keys()):
        rounds_cnt = player_rounds[player]
        lurk_rate = player_lurks[player] / rounds_cnt if rounds_cnt > 0 else 0.0

        fc_hits = [
            s.first_contact
            for s in team_fc_rounds
            if s.first_contact is not None
            and player in (s.first_contact.killer, s.first_contact.victim)
        ]
        fc_involved = len(fc_hits)
        opening_duel_rate = fc_involved / total_fc if total_fc > 0 else 0.0
        opening_kill_rate = (
            sum(1 for fc in fc_hits if fc.killer == player) / fc_involved if fc_involved else 0.0
        )
        opening_zones = dict(Counter(fc.zone for fc in fc_hits if fc.zone).most_common())
        median_fc_t = float(median(fc.t for fc in fc_hits)) if fc_hits else None
        trade_discipline = deaths_traded[player] / deaths[player] if deaths[player] else 0.0

        modal_fe15: dict[str, str] = {}
        for side in ("T", "CT"):
            z_list = fe15_zones[player][side]
            if z_list:
                z_counts = Counter(z_list)
                mode_z = min(z_counts.items(), key=lambda x: (-x[1], x[0]))[0]
                modal_fe15[side] = mode_z

        steamid = player if player.isdigit() else ""
        roles.append(
            RoleCard(
                player=player,
                steamid=steamid,
                modal_zone_fe15=modal_fe15,
                opening_duel_rate=opening_duel_rate,
                lurk_rate=lurk_rate,
                opening_kill_rate=opening_kill_rate,
                opening_zones=opening_zones,
                awp_rounds=len(awp_rounds[player]),
                trade_discipline=trade_discipline,
                median_fc_t=median_fc_t,
            )
        )

    map_name = participating[0].map_name if participating else ""
    card_checksum = participating[0].card_checksum if participating else ""
    generated_from = sorted({s.match_id for s in participating})

    return TeamBook(
        team_key=team_key,
        map_name=map_name,
        card_checksum=card_checksum,
        tendencies=tendencies,
        roles=roles,
        generated_from=generated_from,
    )
