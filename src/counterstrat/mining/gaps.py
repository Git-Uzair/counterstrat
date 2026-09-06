"""Site-hold gap mining: where a team leaves key zones uncovered, and on what trigger.

A key zone (bombsite) counts as HELD when any player stands in its hold
complex: the zone itself or any zone within ``SITE_HOLD_RADIUS_S`` seconds of
it over the map card topology (Dijkstra on measured move times). Players hold
sites from Heaven/Main/Jungle-style positions without standing on the plant
zone, so literal polygon vacancy misread covered sites as conceded ones.

Vacancy is a SETUP read, so pre-plant windows only count while a setup exists:
the defending (CT) side, the bomb not yet down, and at least
``SETUP_ALIVE_MIN`` players alive - post-plant collapses and man-down chaos
said "they abandon A in 45% of rounds" about a team whose intact setups left
it open in 10%. Cover comes from the beat snapshot UNION the movement tracks
(a one-tick snapshot misses a holder mid-strafe), and vacant rounds are
annotated with eyes-on watchers (a stint watching into the complex from
outside it) and opponent pressure (enemies inside the complex or utility
landing there), so an abandoned site is never confused with a contested one.
Complex vacancies are conditioned on the research-derived trigger vocabulary:
previous-round outcome, score state, a won/lost first contact before the beat,
and an early utility dump. Windows are the 15s beat grid (B+00 excluded -
spawn-time vacancy is a tautology) plus the first-contact and plant anchors.
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
BASE_WINDOWS = ("B+15", "B+30", "B+45")
# Calibrated over the 7 shipped map cards (2026-09-06 sweep of 5/6/7/8s): 5s
# keeps every complex tactically tight (Anubis A = site + Heaven, Walkway,
# Main, Fountain, TunnelStairs); 6s and up start swallowing mid on dense cards
# (Cache A, Anubis B).
SITE_HOLD_RADIUS_S = 5.0
MAX_HOLDS = 4
# A defensive setup stops being readable below this many players alive.
SETUP_ALIVE_MIN = 4
# A grenade landing in the complex within this many seconds of the beat marks
# the site contested.
PRESSURE_UTIL_S = 5.0


class GapFinding(BaseModel):
    side: str  # the observed team's side in these rounds
    zone: str  # the uncovered key zone
    window: str  # "B+15" ... or "post-FC" / "post-PL"
    trigger: str  # "base" or one of the trigger vocabulary
    vacancy_rate: float  # rounds with the hold complex EMPTY / rounds matching trigger
    n: int  # setup-intact rounds matching trigger with this window present
    baseline_rate: float  # unconditioned vacancy rate for (side, zone, window)
    lift: float  # vacancy_rate - baseline_rate (0.0 for base rows)
    evidence: list[str]  # uncovered round ids, capped at MAX_EVIDENCE
    # In the covered rounds, which complex members provided the hold: zone ->
    # number of rounds it was occupied, largest first, capped at MAX_HOLDS.
    # Shows HOW the site is held (on-site vs from Heaven/Main) - the read.
    top_holds: dict[str, int] = {}
    # Of the vacant rounds: how many still had a same-side player watching into
    # the complex from outside it (eyes-on, not boots-on), where those
    # watchers stood, and how many had opponents inside the complex or utility
    # landing there (contested - conceded under pressure, not open by design).
    watched_n: int = 0
    top_watch_zones: dict[str, int] = {}
    pressured_n: int = 0


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


def _stint_state(
    script: RoundScript, side: str, t: float
) -> tuple[set[str], list[tuple[str, str]]]:
    """(zones occupied, [(standing zone, watched zone)]) from ``side``'s tracks at ``t``.

    Tracks are alive-only by construction, so a dead player never covers.
    Empty on pre-v2 scripts - callers fall back to the beat snapshot alone.
    """
    zones: set[str] = set()
    watch_pairs: list[tuple[str, str]] = []
    for player, stints in script.tracks.items():
        if script.sides.get(player) != side:
            continue
        for stint in stints:
            if stint.t0 <= t <= stint.t1:
                zones.add(stint.zone)
                if stint.watched:
                    watch_pairs.append((stint.zone, stint.watched))
    return zones, watch_pairs


class _Obs(BaseModel):
    """One round's state for one (side, window, zone) cell."""

    round_id: str
    holders: frozenset[str]  # complex members covered (snapshot UNION tracks)
    triggers: set[str]
    watch_zones: frozenset[str]  # where same-side players watching INTO the complex stood
    pressured: bool  # opponents inside the complex, or their utility landing there


def build_gap_report(
    scripts: list[RoundScript],
    team_key: str,
    key_zones: list[str] | None = None,
    topology: dict[str, dict[str, float]] | None = None,
) -> GapReport:
    """Mine systematic key-zone hold gaps for team_key, conditioned on triggers.

    `topology` (a map card ``topology``) derives each key zone's hold complex;
    without it the complex degrades to the literal zone. Pre-plant windows are
    setup reads: defending side only, bomb not down, >= SETUP_ALIVE_MIN alive.
    """
    map_name = scripts[0].map_name if scripts else ""
    states = iter_round_states(scripts, team_key)
    zones = key_zones if key_zones is not None else _default_key_zones(scripts)
    if not states or not zones:
        return GapReport(team_key=team_key, map_name=map_name, key_zones=zones, findings=[])

    complexes = {z: hold_complex(z, topology) for z in zones}

    obs: dict[tuple[str, str, str], list[_Obs]] = defaultdict(list)
    for st in states:
        s = st.script
        round_id = f"{s.match_id}:{s.round_num}"
        seen_windows: set[str] = set()
        for beat in s.beats:
            window = _window(beat.label)
            if window is None or window in seen_windows:
                continue
            seen_windows.add(window)
            own_form = beat.t_form if st.side == "T" else beat.ct_form
            opp_form = beat.ct_form if st.side == "T" else beat.t_form
            if window == "post-PL":
                # Post-plant, leaving the NON-planted site is normal retake
                # rotation, not a gap: only the planted site is a read there.
                beat_zones = [z for z in zones if s.plant is not None and z == s.plant.site]
            else:
                # Setup windows exist only while a defensive setup does: the
                # defending side, the bomb not yet down, enough players alive.
                # Attackers not standing on sites, post-plant collapses, and
                # man-down scrambles are normal play, not gaps.
                if st.side != "CT":
                    continue
                if s.plant is not None and s.plant.t <= beat.t:
                    continue
                if sum(c for c, _ in own_form.zones) < SETUP_ALIVE_MIN:
                    continue
                beat_zones = zones
            occupied = {z for _, z in own_form.zones}
            stint_zones, watch_pairs = _stint_state(s, st.side, beat.t)
            opp_zones = {z for _, z in opp_form.zones}
            triggers = _triggers_at_beat(s, st.side, beat, st)
            for zone in beat_zones:
                cx = complexes[zone]
                pressured = bool(opp_zones & cx) or any(
                    u.side != st.side and u.to_zone in cx and abs(u.t - beat.t) <= PRESSURE_UTIL_S
                    for u in s.utility
                )
                obs[(st.side, window, zone)].append(
                    _Obs(
                        round_id=round_id,
                        holders=frozenset((occupied | stint_zones) & cx),
                        triggers=triggers,
                        watch_zones=frozenset(
                            stand for stand, watched in watch_pairs if watched in cx
                        ),
                        pressured=pressured,
                    )
                )

    def top_holds_of(rows: list[_Obs]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for o in rows:
            counts.update(o.holders)
        return dict(counts.most_common(MAX_HOLDS))

    def finding_of(
        side: str, window: str, zone: str, trigger: str, rows: list[_Obs], base_rate: float
    ) -> GapFinding:
        vacant = [o for o in rows if not o.holders]
        rate = len(vacant) / len(rows)
        watch_counts: Counter[str] = Counter()
        for o in vacant:
            watch_counts.update(o.watch_zones)
        return GapFinding(
            side=side,
            zone=zone,
            window=window,
            trigger=trigger,
            vacancy_rate=rate,
            n=len(rows),
            baseline_rate=base_rate,
            lift=0.0 if trigger == "base" else rate - base_rate,
            evidence=[o.round_id for o in vacant][:MAX_EVIDENCE],
            top_holds=top_holds_of(rows),
            watched_n=sum(1 for o in vacant if o.watch_zones),
            top_watch_zones=dict(watch_counts.most_common(MAX_HOLDS)),
            pressured_n=sum(1 for o in vacant if o.pressured),
        )

    findings: list[GapFinding] = []
    for (side, window, zone), rows in sorted(obs.items()):
        n = len(rows)
        if n < MIN_N:
            continue
        base_rate = sum(1 for o in rows if not o.holders) / n
        findings.append(finding_of(side, window, zone, "base", rows, base_rate))
        trigger_names = sorted({t for o in rows for t in o.triggers})
        for trigger in trigger_names:
            t_rows = [o for o in rows if trigger in o.triggers]
            if len(t_rows) < MIN_N:
                continue
            f = finding_of(side, window, zone, trigger, t_rows, base_rate)
            if f.vacancy_rate < MIN_VACANCY or f.lift < MIN_LIFT:
                continue
            findings.append(f)

    findings.sort(key=lambda f: (-f.lift, -f.vacancy_rate, f.side, f.zone, f.window, f.trigger))
    return GapReport(
        team_key=team_key,
        map_name=map_name,
        key_zones=zones,
        findings=findings[:MAX_FINDINGS],
        site_complexes={z: sorted(members) for z, members in complexes.items()},
    )
