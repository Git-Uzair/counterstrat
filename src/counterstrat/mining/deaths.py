"""Death profile miner (2026-09-05 advanced-analytics plan Task 2).

Per player: how often they die on the move, how far off the killer their
crosshair sat, and what they were holding - from the death contexts folded
into KillEvents at serialize time. Fields measured on a subset (old scripts
carry None) report their own sub-sample n; the range split uses the shared
constants.range_band bands over kill distance.
"""

from statistics import median

from pydantic import BaseModel

from counterstrat.constants import range_band
from counterstrat.roundscript.models import KillEvent, RoundScript

MAX_EVIDENCE = 6
BAND_ORDER = {"close": 0, "medium": 1, "long": 2}


class DeathBand(BaseModel):
    band: str  # close / medium / long
    n: int
    moving_rate: float | None  # over measured deaths in this band


class DeathProfile(BaseModel):
    player: str
    n: int  # deaths observed
    moving_n: int  # deaths where victim_moving was measured
    moving_rate: float | None
    preaim_n: int
    median_preaim_off_deg: float | None
    weapons: dict[str, int]  # weapon-out-at-death counts (None-tolerant)
    by_range: list[DeathBand]
    evidence: list[str]


class DeathProfiles(BaseModel):
    team_key: str
    map_name: str
    players: list[DeathProfile]  # n desc


def _team_side(script: RoundScript, team_key: str) -> str | None:
    if script.t_team_key == team_key:
        return "T"
    if script.ct_team_key == team_key:
        return "CT"
    return None


def _is_our_death(script: RoundScript, kill: KillEvent, side: str) -> bool:
    victim_side = script.sides.get(kill.victim)
    if victim_side is not None:
        return victim_side == side
    # Pre-v2 scripts carry no sides map: an enemy-side killer implies our
    # victim (misses team-kills, the standard heuristic).
    return kill.killer_side != side


def build_death_profiles(scripts: list[RoundScript], team_key: str) -> DeathProfiles:
    """Mine per-player death contexts for team_key from RoundScript kills."""
    map_name = scripts[0].map_name if scripts else ""
    deaths: dict[str, list[tuple[KillEvent, str]]] = {}
    for s in scripts:
        side = _team_side(s, team_key)
        if side is None:
            continue
        round_id = f"{s.match_id}:{s.round_num}"
        for k in s.kills:
            if k.victim and _is_our_death(s, k, side):
                deaths.setdefault(k.victim, []).append((k, round_id))

    players: list[DeathProfile] = []
    for player, obs in sorted(deaths.items()):
        kills = [k for k, _ in obs]
        moving = [k.victim_moving for k in kills if k.victim_moving is not None]
        preaim = [k.victim_preaim_off_deg for k in kills if k.victim_preaim_off_deg is not None]
        weapons: dict[str, int] = {}
        for k in kills:
            if k.victim_weapon:
                weapons[k.victim_weapon] = weapons.get(k.victim_weapon, 0) + 1

        bands: dict[str, list[bool | None]] = {}
        for k in kills:
            if k.distance is None:
                continue
            bands.setdefault(range_band(k.distance), []).append(k.victim_moving)
        by_range = []
        for band, vals in sorted(bands.items(), key=lambda kv: BAND_ORDER.get(kv[0], 9)):
            measured = [v for v in vals if v is not None]
            by_range.append(
                DeathBand(
                    band=band,
                    n=len(vals),
                    moving_rate=(round(sum(measured) / len(measured), 2) if measured else None),
                )
            )

        players.append(
            DeathProfile(
                player=player,
                n=len(kills),
                moving_n=len(moving),
                moving_rate=round(sum(moving) / len(moving), 2) if moving else None,
                preaim_n=len(preaim),
                median_preaim_off_deg=round(float(median(preaim)), 1) if preaim else None,
                weapons=dict(sorted(weapons.items(), key=lambda kv: (-kv[1], kv[0]))),
                by_range=by_range,
                evidence=[rid for _, rid in obs][:MAX_EVIDENCE],
            )
        )
    players.sort(key=lambda p: (-p.n, p.player))
    return DeathProfiles(team_key=team_key, map_name=map_name, players=players)
