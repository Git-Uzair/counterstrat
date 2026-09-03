"""Utility book mining: recurring nade patterns and dump-timing windows (plan Task 2)."""

from collections import Counter, defaultdict
from statistics import median

from pydantic import BaseModel

from counterstrat.roundscript.models import RoundScript, UtilEvent

MAX_PATTERNS = 40
MAX_EVIDENCE = 6
EARLY_THROW_S = 25.0  # throws before this clock mark count as the opening package
MAX_DUMP_KTH = 4


class UtilityPattern(BaseModel):
    side: str  # "T" | "CT"
    nade: str  # smoke | flash | he | molotov | decoy
    to_zone: str
    from_zones: dict[str, int]  # origin zone -> throw count
    lineup_id: str | None  # dominant cluster id, if any throws carried one
    count: int  # rounds containing this pattern at least once
    rounds_seen: int  # rounds where the side threw >= 1 nade of this type
    share: float  # count / rounds_seen
    median_t: float  # median throw clock_s across all matching throws
    early_share: float  # share of matching throws before EARLY_THROW_S
    evidence: list[str]  # "match_id:round_num", capped at MAX_EVIDENCE


class UtilityBook(BaseModel):
    team_key: str
    map_name: str
    patterns: list[UtilityPattern]  # sorted by count desc, capped at MAX_PATTERNS
    # {"side": "T", "kth": 2, "median_t": 21.0, "n": 7}: when the k-th nade of a
    # round is typically spent - the "their utility is gone by ~Xs" read.
    dump_windows: list[dict]

    def top_patterns(self, side: str | None = None, limit: int = 10) -> list[UtilityPattern]:
        rows = [p for p in self.patterns if side is None or p.side == side]
        return rows[:limit]


def _team_side(script: RoundScript, team_key: str) -> str | None:
    if script.t_team_key == team_key:
        return "T"
    if script.ct_team_key == team_key:
        return "CT"
    return None


def build_utility_book(scripts: list[RoundScript], team_key: str) -> UtilityBook:
    """Mine the team's recurring utility usage from RoundScripts."""
    map_name = scripts[0].map_name if scripts else ""

    # (side, nade, to_zone) -> per-round throw lists
    throws: dict[tuple[str, str, str], list[tuple[str, UtilEvent]]] = defaultdict(list)
    rounds_with_nade: dict[tuple[str, str], set[str]] = defaultdict(set)
    per_round_ts: dict[tuple[str, str], list[float]] = defaultdict(list)  # (side, round_id) -> ts

    for s in scripts:
        side = _team_side(s, team_key)
        if side is None:
            continue
        round_id = f"{s.match_id}:{s.round_num}"
        for u in s.utility:
            if u.side != side or not u.to_zone:
                continue
            throws[(side, u.nade, u.to_zone)].append((round_id, u))
            rounds_with_nade[(side, u.nade)].add(round_id)
            per_round_ts[(side, round_id)].append(u.t)

    patterns: list[UtilityPattern] = []
    for (side, nade, to_zone), events in throws.items():
        round_ids = sorted(
            {rid for rid, _ in events},
            key=lambda x: (x.split(":")[0], int(x.split(":")[1])),
        )
        rounds_seen = len(rounds_with_nade[(side, nade)])
        ts = [u.t for _, u in events]
        from_zones = Counter(u.from_zone for _, u in events if u.from_zone)
        lineup_counts = Counter(u.lineup_id for _, u in events if u.lineup_id)
        patterns.append(
            UtilityPattern(
                side=side,
                nade=nade,
                to_zone=to_zone,
                from_zones=dict(from_zones.most_common()),
                lineup_id=lineup_counts.most_common(1)[0][0] if lineup_counts else None,
                count=len(round_ids),
                rounds_seen=rounds_seen,
                share=len(round_ids) / rounds_seen if rounds_seen else 0.0,
                median_t=float(median(ts)),
                early_share=sum(1 for t in ts if t < EARLY_THROW_S) / len(ts),
                evidence=round_ids[:MAX_EVIDENCE],
            )
        )
    patterns.sort(key=lambda p: (-p.count, p.side, p.nade, p.to_zone))

    dump_windows: list[dict] = []
    for side in ("T", "CT"):
        kth_ts: dict[int, list[float]] = defaultdict(list)
        for (s_side, _), ts in per_round_ts.items():
            if s_side != side:
                continue
            for k, t in enumerate(sorted(ts)[:MAX_DUMP_KTH], start=1):
                kth_ts[k].append(t)
        for k in sorted(kth_ts):
            dump_windows.append(
                {
                    "side": side,
                    "kth": k,
                    "median_t": float(median(kth_ts[k])),
                    "n": len(kth_ts[k]),
                }
            )

    return UtilityBook(
        team_key=team_key,
        map_name=map_name,
        patterns=patterns[:MAX_PATTERNS],
        dump_windows=dump_windows,
    )
