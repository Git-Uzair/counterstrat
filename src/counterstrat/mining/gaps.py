"""Site-hold gap mining: where a team leaves key zones uncovered, and on what trigger.

A key zone (bombsite) counts as HELD when any player stands in its hold
complex: the zone itself or any zone within ``SITE_HOLD_RADIUS_S`` seconds of
it over the map card topology (Dijkstra on measured move times). Players hold
sites from Heaven/Main/Jungle-style positions without standing on the plant
zone, so literal polygon vacancy misread covered sites as conceded ones.
Complex vacancies are conditioned on the research-derived trigger vocabulary:
previous-round outcome, score state, a won/lost first contact before the beat,
and an early utility dump. Windows are the 15s beat grid plus the
first-contact and plant anchors.
"""

import heapq
from collections import Counter, defaultdict

from pydantic import BaseModel

from counterstrat.mining.tendencies import iter_round_states
from counterstrat.roundscript.models import Beat, RoundScript

MIN_N = 3
MIN_VACANCY = 0.6
MIN_LIFT = 0.15
MAX_FINDINGS = 30
MAX_EVIDENCE = 6
UTIL_DUMP_COUNT = 3
BASE_WINDOWS = ("B+00", "B+15", "B+30", "B+45")
# Calibrated over the 7 shipped map cards (2026-09-06 sweep of 5/6/7/8s): 5s
# keeps every complex tactically tight (Anubis A = site + Heaven, Walkway,
# Main, Fountain, TunnelStairs); 6s and up start swallowing mid on dense cards
# (Cache A, Anubis B).
SITE_HOLD_RADIUS_S = 5.0
MAX_HOLDS = 4


class GapFinding(BaseModel):
    side: str  # the observed team's side in these rounds
    zone: str  # the uncovered key zone
    window: str  # "B+15" ... or "post-FC" / "post-PL"
    trigger: str  # "base" or one of the trigger vocabulary
    vacancy_rate: float  # rounds with the hold complex EMPTY / rounds matching trigger
    n: int  # rounds matching trigger with this window present
    baseline_rate: float  # unconditioned vacancy rate for (side, zone, window)
    lift: float  # vacancy_rate - baseline_rate (0.0 for base rows)
    evidence: list[str]  # uncovered round ids, capped at MAX_EVIDENCE
    # In the covered rounds, which complex members provided the hold: zone ->
    # number of rounds it was occupied, largest first, capped at MAX_HOLDS.
    # Shows HOW the site is held (on-site vs from Heaven/Main) - the read.
    top_holds: dict[str, int] = {}


class GapReport(BaseModel):
    team_key: str
    map_name: str
    key_zones: list[str]
    findings: list[GapFinding]  # triggered rows filtered by thresholds, sorted by lift
    # key zone -> sorted members of its hold complex (just the zone itself when
    # no topology was available).
    site_complexes: dict[str, list[str]] = {}


def _symmetric_graph(topology: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Undirected weighted graph from card topology (edges can be recorded one-way)."""
    g: dict[str, dict[str, float]] = defaultdict(dict)
    for u, nbrs in topology.items():
        for v, w in nbrs.items():
            w = float(w)
            g[u][v] = min(g[u].get(v, w), w)
            g[v][u] = min(g[v].get(u, w), w)
    return g


def hold_complex(
    zone: str,
    topology: dict[str, dict[str, float]] | None,
    cutoff_s: float = SITE_HOLD_RADIUS_S,
) -> set[str]:
    """Zones from which `zone` is contestable within cutoff_s seconds (incl. itself)."""
    if not topology:
        return {zone}
    g = _symmetric_graph(topology)
    dist = {zone: 0.0}
    heap = [(0.0, zone)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in g.get(u, {}).items():
            nd = d + w
            if nd <= cutoff_s and nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return set(dist)


def _window(label: str) -> str | None:
    if label.startswith("FC"):
        return "post-FC"
    if label.startswith("PL"):
        return "post-PL"
    if label in BASE_WINDOWS:
        return label
    return None


def _triggers_at_beat(script: RoundScript, side: str, beat: Beat, state) -> set[str]:
    out: set[str] = set()
    if state.score_bucket in ("behind", "ahead"):
        out.add(state.score_bucket)
    if state.prev_outcome == "lost":
        out.add("after_loss")
    elif state.prev_outcome == "won":
        out.add("after_win")
    fc = script.first_contact
    if fc is not None and fc.t <= beat.t:
        out.add("after_fc_win" if fc.killer_side == side else "after_fc_loss")
    dumped = sum(1 for u in script.utility if u.side == side and u.t <= beat.t)
    if dumped >= UTIL_DUMP_COUNT:
        out.add("after_util_dump")
    return out


def _default_key_zones(scripts: list[RoundScript]) -> list[str]:
    zones = {s.plant.site for s in scripts if s.plant and s.plant.site}
    return sorted(zones)


def build_gap_report(
    scripts: list[RoundScript],
    team_key: str,
    key_zones: list[str] | None = None,
    topology: dict[str, dict[str, float]] | None = None,
) -> GapReport:
    """Mine systematic key-zone hold gaps for team_key, conditioned on triggers.

    `topology` (a map card ``topology``) derives each key zone's hold complex;
    without it the complex degrades to the literal zone.
    """
    map_name = scripts[0].map_name if scripts else ""
    states = iter_round_states(scripts, team_key)
    zones = key_zones if key_zones is not None else _default_key_zones(scripts)
    if not states or not zones:
        return GapReport(team_key=team_key, map_name=map_name, key_zones=zones, findings=[])

    complexes = {z: hold_complex(z, topology) for z in zones}

    # (side, window, zone) -> [(round_id, holder_zones, active_triggers)]
    obs: dict[tuple[str, str, str], list[tuple[str, frozenset[str], set[str]]]] = defaultdict(list)
    for st in states:
        s = st.script
        round_id = f"{s.match_id}:{s.round_num}"
        seen_windows: set[str] = set()
        for beat in s.beats:
            window = _window(beat.label)
            if window is None or window in seen_windows:
                continue
            seen_windows.add(window)
            form = beat.t_form if st.side == "T" else beat.ct_form
            occupied = {z for _, z in form.zones}
            triggers = _triggers_at_beat(s, st.side, beat, st)
            if window == "post-PL":
                # Post-plant, leaving the NON-planted site is normal retake
                # rotation, not a gap: only the planted site is a read there.
                beat_zones = [z for z in zones if s.plant is not None and z == s.plant.site]
            else:
                beat_zones = zones
            for zone in beat_zones:
                holders = frozenset(occupied & complexes[zone])
                obs[(st.side, window, zone)].append((round_id, holders, triggers))

    def top_holds_of(rows: list[tuple[str, frozenset[str]]]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for _, holders in rows:
            counts.update(holders)
        return dict(counts.most_common(MAX_HOLDS))

    findings: list[GapFinding] = []
    for (side, window, zone), rows in sorted(obs.items()):
        n = len(rows)
        if n < MIN_N:
            continue
        vacant_ids = [rid for rid, holders, _ in rows if not holders]
        base_rate = len(vacant_ids) / n
        findings.append(
            GapFinding(
                side=side,
                zone=zone,
                window=window,
                trigger="base",
                vacancy_rate=base_rate,
                n=n,
                baseline_rate=base_rate,
                lift=0.0,
                evidence=vacant_ids[:MAX_EVIDENCE],
                top_holds=top_holds_of([(rid, h) for rid, h, _ in rows]),
            )
        )
        trigger_names = sorted({t for _, _, trigs in rows for t in trigs})
        for trigger in trigger_names:
            t_rows = [(rid, holders) for rid, holders, trigs in rows if trigger in trigs]
            t_n = len(t_rows)
            if t_n < MIN_N:
                continue
            t_vacant = [rid for rid, holders in t_rows if not holders]
            rate = len(t_vacant) / t_n
            lift = rate - base_rate
            if rate < MIN_VACANCY or lift < MIN_LIFT:
                continue
            findings.append(
                GapFinding(
                    side=side,
                    zone=zone,
                    window=window,
                    trigger=trigger,
                    vacancy_rate=rate,
                    n=t_n,
                    baseline_rate=base_rate,
                    lift=lift,
                    evidence=t_vacant[:MAX_EVIDENCE],
                    top_holds=top_holds_of(t_rows),
                )
            )

    findings.sort(key=lambda f: (-f.lift, -f.vacancy_rate, f.side, f.zone, f.window, f.trigger))
    return GapReport(
        team_key=team_key,
        map_name=map_name,
        key_zones=zones,
        findings=findings[:MAX_FINDINGS],
        site_complexes={z: sorted(members) for z, members in complexes.items()},
    )
