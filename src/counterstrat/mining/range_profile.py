"""Engagement-range profile miner (space-vision research item 2).

A team's kill-distance distribution is a read: kills clustered close mean
tight SMG-style positions that must close space; kills clustered long mean
static long-line holds. Mined from RoundScript kills like every other miner;
kills without a distance (scripts serialized before the KillEvent upgrade)
are skipped, never guessed.
"""

from statistics import median

from pydantic import BaseModel

from counterstrat.constants import range_band
from counterstrat.roundscript.models import KillEvent, RoundScript

LOW_BUYS = {"full_eco", "semi_eco", "semi_buy"}
MIN_PLAYER_KILLS = 3
MAX_PLAYERS = 10


class RangeStats(BaseModel):
    n: int
    median_dist: float
    close_share: float
    medium_share: float
    long_share: float
    smoke_share: float  # kills through smoke
    wallbang_share: float  # penetrated kills


class PlayerRange(BaseModel):
    player: str
    n: int
    median_dist: float
    close_share: float
    long_share: float


class RangeProfile(BaseModel):
    team_key: str
    map_name: str
    sides: dict[str, RangeStats]  # only sides the team actually played
    low_buy: RangeStats | None  # kills in full_eco/semi_eco/semi_buy rounds
    full_buy: RangeStats | None
    players: list[PlayerRange]  # n desc, capped
    kills_total: int
    kills_with_distance: int

    def to_prompt_lines(self) -> list[str]:
        """Prompt-ready bullets; empty when there is nothing measured."""

        def _fmt(s: RangeStats) -> str:
            base = (
                f"{s.close_share:.0%} close / {s.medium_share:.0%} medium / "
                f"{s.long_share:.0%} long (n={s.n}, median {s.median_dist:.0f}u)"
            )
            extras = []
            if s.smoke_share > 0:
                extras.append(f"{s.smoke_share:.0%} through smoke")
            if s.wallbang_share > 0:
                extras.append(f"{s.wallbang_share:.0%} wallbangs")
            return base + (f"; {', '.join(extras)}" if extras else "")

        lines = [f"- {side}: {_fmt(st)}" for side, st in sorted(self.sides.items())]
        if self.full_buy:
            lines.append(f"- full buys: {_fmt(self.full_buy)}")
        if self.low_buy:
            lines.append(f"- low buys (eco/force): {_fmt(self.low_buy)}")
        for p in self.players:
            lines.append(
                f"- {p.player}: median {p.median_dist:.0f}u, {p.close_share:.0%} close, "
                f"{p.long_share:.0%} long (n={p.n})"
            )
        return lines


def _team_side(script: RoundScript, team_key: str) -> str | None:
    if script.t_team_key == team_key:
        return "T"
    if script.ct_team_key == team_key:
        return "CT"
    return None


def _stats(kills: list[KillEvent]) -> RangeStats | None:
    if not kills:
        return None
    dists = [k.distance for k in kills if k.distance is not None]
    if not dists:
        return None
    bands = [range_band(d) for d in dists]
    n = len(dists)
    return RangeStats(
        n=n,
        median_dist=float(median(dists)),
        close_share=bands.count("close") / n,
        medium_share=bands.count("medium") / n,
        long_share=bands.count("long") / n,
        smoke_share=sum(1 for k in kills if k.distance is not None and k.thrusmoke) / n,
        wallbang_share=sum(1 for k in kills if k.distance is not None and k.penetrated) / n,
    )


def build_range_profile(scripts: list[RoundScript], team_key: str) -> RangeProfile:
    """Mine the team's engagement-range tendencies from RoundScript kills."""
    map_name = scripts[0].map_name if scripts else ""
    by_side: dict[str, list[KillEvent]] = {}
    by_buy: dict[str, list[KillEvent]] = {"low": [], "full": []}
    by_player: dict[str, list[KillEvent]] = {}
    kills_total = 0

    for s in scripts:
        side = _team_side(s, team_key)
        if side is None:
            continue
        econ = s.economy.get(side)
        buy = "low" if econ is not None and econ.buy_type in LOW_BUYS else "full"
        for k in s.kills:
            if k.killer_side != side:
                continue
            kills_total += 1
            by_side.setdefault(side, []).append(k)
            by_buy[buy].append(k)
            if k.killer:
                by_player.setdefault(k.killer, []).append(k)

    sides = {side: st for side, ks in by_side.items() if (st := _stats(ks)) is not None}

    players: list[PlayerRange] = []
    for player, ks in by_player.items():
        st = _stats(ks)
        if st is None or st.n < MIN_PLAYER_KILLS:
            continue
        players.append(
            PlayerRange(
                player=player,
                n=st.n,
                median_dist=st.median_dist,
                close_share=st.close_share,
                long_share=st.long_share,
            )
        )
    players.sort(key=lambda p: (-p.n, p.player))

    return RangeProfile(
        team_key=team_key,
        map_name=map_name,
        sides=sides,
        low_buy=_stats(by_buy["low"]),
        full_buy=_stats(by_buy["full"]),
        players=players[:MAX_PLAYERS],
        kills_total=kills_total,
        kills_with_distance=sum(st.n for st in sides.values()),
    )
