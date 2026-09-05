"""Utility ROI miner (2026-09-05 advanced-analytics plan Task 2).

Per recurring throw pattern (the lineup cluster when one exists, else the
from>to zone pair): average enemy/team blind seconds, damage per HE/molly,
kills through the smoke, and the dollar cost per point of value. Cost is the
in-game grenade price (the per-nade spend; team econ cannot attribute a
single grenade). Averages skip events without a measured effect (old lakes) -
None means unmeasured, never zero. Dollar verdicts are withheld below n=5:
the numbers are emitted and the model hedges.
"""

from pydantic import BaseModel

from counterstrat.roundscript.models import RoundScript, UtilEvent

MAX_EVIDENCE = 6
MIN_VERDICT_N = 5  # below this, no cost-per-value verdicts
# In-game prices; the CT molly is the 600$ incendiary.
NADE_COST = {"flash": 200, "smoke": 300, "he": 300, "molly": 400}
CT_MOLLY_COST = 600


class UtilityROIRow(BaseModel):
    side: str
    nade: str
    pattern: str  # lineup id when clustered, else "from>to"
    n: int
    avg_enemy_blind_s: float | None  # over events with a measured split
    avg_team_blind_s: float | None
    avg_damage: float | None  # HE/molly, over measured events
    kills_through: int | None  # smokes: total across measured events
    cost: int  # unit price of this nade for this side
    cost_per_enemy_blind_s: float | None  # verdicts only at n >= MIN_VERDICT_N
    cost_per_damage: float | None
    evidence: list[str]


class UtilityROI(BaseModel):
    team_key: str
    map_name: str
    rows: list[UtilityROIRow]  # n desc


def _team_side(script: RoundScript, team_key: str) -> str | None:
    if script.t_team_key == team_key:
        return "T"
    if script.ct_team_key == team_key:
        return "CT"
    return None


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _cost(nade: str, side: str) -> int:
    if nade == "molly" and side == "CT":
        return CT_MOLLY_COST
    return NADE_COST.get(nade, 0)


def build_utility_roi(scripts: list[RoundScript], team_key: str) -> UtilityROI:
    """Mine the team's per-pattern utility value from enriched UtilEvents."""
    map_name = scripts[0].map_name if scripts else ""
    groups: dict[tuple[str, str, str], list[tuple[UtilEvent, str]]] = {}
    for s in scripts:
        side = _team_side(s, team_key)
        if side is None:
            continue
        round_id = f"{s.match_id}:{s.round_num}"
        for u in s.utility:
            if u.side != side:
                continue
            pattern = u.lineup_id or f"{u.from_zone}>{u.to_zone}"
            groups.setdefault((side, u.nade, pattern), []).append((u, round_id))

    rows: list[UtilityROIRow] = []
    for (side, nade, pattern), obs in sorted(groups.items()):
        events = [u for u, _ in obs]
        n = len(events)
        enemy_vals = [u.enemy_blind_s for u in events if u.enemy_blind_s is not None]
        damage_vals = [float(u.damage) for u in events if u.damage is not None]
        avg_enemy = _avg(enemy_vals)
        avg_team = _avg([u.team_blind_s for u in events if u.team_blind_s is not None])
        avg_damage = _avg(damage_vals)
        kt_measured = [u.kills_through for u in events if u.kills_through is not None]
        kills_through = sum(kt_measured) if kt_measured else None
        cost = _cost(nade, side)
        # Verdicts gate on the MEASURED sub-sample: five throws with one
        # measured effect is one data point, not five.
        rows.append(
            UtilityROIRow(
                side=side,
                nade=nade,
                pattern=pattern,
                n=n,
                avg_enemy_blind_s=avg_enemy,
                avg_team_blind_s=avg_team,
                avg_damage=avg_damage,
                kills_through=kills_through,
                cost=cost,
                cost_per_enemy_blind_s=(
                    round(cost / avg_enemy, 1)
                    if len(enemy_vals) >= MIN_VERDICT_N and avg_enemy
                    else None
                ),
                cost_per_damage=(
                    round(cost / avg_damage, 1)
                    if len(damage_vals) >= MIN_VERDICT_N and avg_damage
                    else None
                ),
                evidence=sorted({rid for _, rid in obs})[:MAX_EVIDENCE],
            )
        )
    rows.sort(key=lambda r: (-r.n, r.side, r.nade, r.pattern))
    return UtilityROI(team_key=team_key, map_name=map_name, rows=rows)
