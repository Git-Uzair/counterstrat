"""Rotation report miner (2026-09-05 advanced-analytics plan Task 2).

Aggregates the serialize-time RotationEvents into per-(player, trigger)
latency medians and an over-rotation rate: the share of responses whose
trigger turned out to be a fake (no follow-up contact at the trigger zone
within 10s). Everything is correlation over hold-breaks, never causation -
the report words feed sentences like "B4RT3Q breaks B anchor 2.1s after
A-side utility (n=9, fake-follow rate 44%)".
"""

from statistics import median

from pydantic import BaseModel

from counterstrat.roundscript.models import RotationEvent, RoundScript

FOLLOWUP_WINDOW_S = 10.0  # a trigger with no contact at its zone within this = fake
MAX_EVIDENCE = 6
TRIGGER_MATCH_TOL_S = 0.5  # matching a rotation back to its utility event


class RotationRow(BaseModel):
    side: str  # the side the team played when these rotations happened
    player: str
    trigger: str
    n: int
    median_latency_s: float
    fake_rate: float | None  # None when no trigger in the group has a locatable zone
    fakes_n: int
    evidence: list[str]  # round ids, capped


class RotationReport(BaseModel):
    team_key: str
    map_name: str
    rows: list[RotationRow]  # n desc


def _team_side(script: RoundScript, team_key: str) -> str | None:
    if script.t_team_key == team_key:
        return "T"
    if script.ct_team_key == team_key:
        return "CT"
    return None


def _trigger_zone(script: RoundScript, rot: RotationEvent) -> str | None:
    """The zone where this rotation's trigger happened, when the script knows it."""
    if rot.trigger == "utility_near":
        candidates = [
            u
            for u in script.utility
            if u.side != rot.side and abs(u.t - rot.t_trigger) <= TRIGGER_MATCH_TOL_S
        ]
        if candidates:
            return min(candidates, key=lambda u: abs(u.t - rot.t_trigger)).to_zone
        return None
    if rot.trigger == "first_blood" and script.kills:
        return min(script.kills, key=lambda k: k.t).zone
    if rot.trigger == "plant" and script.plant is not None:
        return script.plant.site
    return None  # shots / visible_contact carry no zone in the script


def _is_fake(script: RoundScript, rot: RotationEvent) -> bool | None:
    """True when nothing happened at the trigger zone within the follow-up window."""
    zone = _trigger_zone(script, rot)
    if zone is None:
        return None
    followed = any(
        k.zone == zone and rot.t_trigger < k.t <= rot.t_trigger + FOLLOWUP_WINDOW_S
        for k in script.kills
    )
    return not followed


def build_rotation_report(scripts: list[RoundScript], team_key: str) -> RotationReport:
    """Mine per-(player, trigger) rotation latencies and fake-follow rates."""
    map_name = scripts[0].map_name if scripts else ""
    groups: dict[tuple[str, str, str], list[tuple[float, str, bool | None]]] = {}
    for s in scripts:
        side = _team_side(s, team_key)
        if side is None:
            continue
        round_id = f"{s.match_id}:{s.round_num}"
        for rot in s.rotations:
            if rot.side != side:
                continue
            groups.setdefault((side, rot.player, rot.trigger), []).append(
                (rot.latency_s, round_id, _is_fake(s, rot))
            )

    rows: list[RotationRow] = []
    for (side, player, trigger), obs in sorted(groups.items()):
        located = [fake for _, _, fake in obs if fake is not None]
        fakes_n = sum(1 for f in located if f)
        rows.append(
            RotationRow(
                side=side,
                player=player,
                trigger=trigger,
                n=len(obs),
                median_latency_s=round(float(median(lat for lat, _, _ in obs)), 1),
                fake_rate=round(fakes_n / len(located), 2) if located else None,
                fakes_n=fakes_n,
                evidence=[rid for _, rid, _ in obs][:MAX_EVIDENCE],
            )
        )
    rows.sort(key=lambda r: (-r.n, r.player, r.trigger))
    return RotationReport(team_key=team_key, map_name=map_name, rows=rows)
