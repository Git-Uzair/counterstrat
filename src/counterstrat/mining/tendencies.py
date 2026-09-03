"""Deterministic Level-2 tendency mining and TeamBook profile generation."""

import re
from collections import Counter, defaultdict
from statistics import median

from pydantic import BaseModel, ConfigDict

from counterstrat.roundscript.models import RoundScript


class TendencyKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    map_name: str
    side: str  # "T" | "CT"
    buy_class: str  # "full_eco" | "semi_eco" | "semi_buy" | "full_buy"
    score_bucket: str  # "behind" | "even" | "ahead"
    prev_outcome: str  # "won" | "lost" | "first"


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


class RoleCard(BaseModel):
    player: str
    steamid: str
    modal_zone_fe15: dict[str, str]  # side -> modal zone at B+15
    opening_duel_rate: float  # share of team first-contacts involving them
    lurk_rate: float  # share of rounds with lurk role_hint


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
            "| Side | Buy | Score | Prev | FC Zone | Opening Formation | Site | Median FC | N | Low N |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in self.tendencies:
            fc_str = ", ".join(f"{z} ({p:.0%})" for z, p in t.first_contact_zone.items()) or "-"
            form_str = (
                ", ".join(f"{sig} ({p:.0%})" for sig, p in list(t.opening_formation.items())[:2])
                or "-"
            )
            site_str = ", ".join(f"{s} ({p:.0%})" for s, p in t.site_committed.items()) or "-"
            med_fc = (
                f"{t.median_first_contact_s:.1f}s" if t.median_first_contact_s is not None else "-"
            )
            low_n_str = "Yes" if t.low_n else "No"
            lines.append(
                f"| {t.key.side} | {t.key.buy_class} | {t.key.score_bucket} | {t.key.prev_outcome} | "
                f"{fc_str} | {form_str} | {site_str} | {med_fc} | {t.n} | {low_n_str} |"
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
        sentences: list[str] = []
        for t in self.tendencies:
            top_fc, fc_p = ("none", 0.0)
            if t.first_contact_zone:
                top_fc, fc_p = min(t.first_contact_zone.items(), key=lambda x: (-x[1], x[0]))
            pct_str = f"{round(fc_p * 100)}%"
            action = "attacks" if t.key.side == "T" else "contests"
            side_str = "CT " if t.key.side == "CT" else ""
            if top_fc != "none":
                s = (
                    f"On {side_str}{t.key.buy_class} when {t.key.score_bucket}, team {action} {top_fc} "
                    f"{pct_str} of the time (n={t.n})."
                )
            else:
                s = (
                    f"On {side_str}{t.key.buy_class} when {t.key.score_bucket}, team had no first contact "
                    f"(n={t.n})."
                )
            sentences.append(s)
        return sentences


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

    # Group by match_id to compute prev_outcome sequentially
    scripts_by_match: dict[str, list[RoundScript]] = defaultdict(list)
    for s in participating:
        scripts_by_match[s.match_id].append(s)

    # Key tuple -> list[RoundScript]
    grouped_scripts: dict[tuple[str, str, str, str, str], list[RoundScript]] = defaultdict(list)

    for m_scripts in scripts_by_match.values():
        m_scripts.sort(key=lambda s: s.round_num)
        rounds_map = {s.round_num: s for s in m_scripts}

        for s in m_scripts:
            side = "T" if s.t_team_key == team_key else "CT"

            # prev_outcome
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

            # score_bucket
            team_score = s.score_t if side == "T" else s.score_ct
            opp_score = s.score_ct if side == "T" else s.score_t
            if team_score > opp_score:
                score_bucket = "ahead"
            elif team_score < opp_score:
                score_bucket = "behind"
            else:
                score_bucket = "even"

            # buy_class
            econ = s.economy.get(side)
            buy_class = econ.buy_type if econ and hasattr(econ, "buy_type") else "full_buy"

            k_tuple = (s.map_name, side, buy_class, score_bucket, prev_outcome)
            grouped_scripts[k_tuple].append(s)

    tendencies: list[Tendency] = []
    for k_tuple, g_scripts in grouped_scripts.items():
        map_name, side, buy_class, score_bucket, prev_outcome = k_tuple
        n = len(g_scripts)
        key = TendencyKey(
            map_name=map_name,
            side=side,
            buy_class=buy_class,
            score_bucket=score_bucket,
            prev_outcome=prev_outcome,
        )

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
                sig = (
                    " ".join(f"{cnt}x{zone}" for cnt, zone in form.zones) if form.zones else "none"
                )
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

        tendencies.append(
            Tendency(
                key=key,
                first_contact_zone=first_contact_zone,
                opening_formation=opening_formation,
                site_committed=site_committed,
                lineup_sets=lineup_sets,
                median_first_contact_s=med_fc,
                n=n,
                evidence=evidence,
                low_n=n < 3,
            )
        )

    # Sort tendencies deterministically
    tendencies.sort(
        key=lambda t: (
            -t.n,
            t.key.side,
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

    roles: list[RoleCard] = []
    for player in sorted(player_rounds.keys()):
        rounds_cnt = player_rounds[player]
        lurk_rate = player_lurks[player] / rounds_cnt if rounds_cnt > 0 else 0.0

        fc_involved = sum(
            1
            for s in team_fc_rounds
            if s.first_contact is not None
            and (s.first_contact.killer == player or s.first_contact.victim == player)
        )
        opening_duel_rate = fc_involved / total_fc if total_fc > 0 else 0.0

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
