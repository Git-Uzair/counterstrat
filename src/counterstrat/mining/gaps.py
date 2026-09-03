"""Zone vacancy mining: where a team leaves gaps, when, and on what trigger (plan Task 3).

Findings condition zone vacancies (a key zone absent from the side's beat
formation) on the research-derived trigger vocabulary: previous-round outcome,
score state, a won/lost first contact before the beat, and an early utility
dump. Windows are the 15s beat grid plus the first-contact and plant anchors.
"""

from collections import defaultdict

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


class GapFinding(BaseModel):
    side: str  # the observed team's side in these rounds
    zone: str  # the vacated key zone
    window: str  # "B+15" ... or "post-FC" / "post-PL"
    trigger: str  # "base" or one of the trigger vocabulary
    vacancy_rate: float  # vacant rounds / rounds matching trigger
    n: int  # rounds matching trigger with this window present
    baseline_rate: float  # unconditioned vacancy rate for (side, zone, window)
    lift: float  # vacancy_rate - baseline_rate (0.0 for base rows)
    evidence: list[str]  # vacant round ids, capped at MAX_EVIDENCE


class GapReport(BaseModel):
    team_key: str
    map_name: str
    key_zones: list[str]
    findings: list[GapFinding]  # triggered rows filtered by thresholds, sorted by lift


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
) -> GapReport:
    """Mine systematic key-zone vacancies for team_key, conditioned on triggers."""
    map_name = scripts[0].map_name if scripts else ""
    states = iter_round_states(scripts, team_key)
    zones = key_zones if key_zones is not None else _default_key_zones(scripts)
    if not states or not zones:
        return GapReport(team_key=team_key, map_name=map_name, key_zones=zones, findings=[])

    # (side, window, zone) -> [(round_id, vacant, active_triggers)]
    obs: dict[tuple[str, str, str], list[tuple[str, bool, set[str]]]] = defaultdict(list)
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
                # Post-plant, vacating the NON-planted site is normal retake
                # rotation, not a gap: only the planted site is a read there.
                beat_zones = [z for z in zones if s.plant is not None and z == s.plant.site]
            else:
                beat_zones = zones
            for zone in beat_zones:
                obs[(st.side, window, zone)].append((round_id, zone not in occupied, triggers))

    findings: list[GapFinding] = []
    for (side, window, zone), rows in sorted(obs.items()):
        n = len(rows)
        if n < MIN_N:
            continue
        vacant_ids = [rid for rid, vacant, _ in rows if vacant]
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
            )
        )
        trigger_names = sorted({t for _, _, trigs in rows for t in trigs})
        for trigger in trigger_names:
            t_rows = [(rid, vacant) for rid, vacant, trigs in rows if trigger in trigs]
            t_n = len(t_rows)
            if t_n < MIN_N:
                continue
            t_vacant = [rid for rid, vacant in t_rows if vacant]
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
                )
            )

    findings.sort(key=lambda f: (-f.lift, -f.vacancy_rate, f.side, f.zone, f.window, f.trigger))
    return GapReport(
        team_key=team_key,
        map_name=map_name,
        key_zones=zones,
        findings=findings[:MAX_FINDINGS],
    )
