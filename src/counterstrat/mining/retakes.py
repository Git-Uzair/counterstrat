"""Retake report miner (2026-09-05 advanced-analytics plan Task 2).

Post-plant conversion from both chairs: when the team is CT the row is a
retake (win = CT round win), when the team is T it is a post-plant hold
(win = T round win). Man-diff is the acting side's advantage at the plant.
Approach vectors come from the tracks: the zones CT players entered the
planted site from after the plant - the team's own retake package on CT
rows, the enemy vectors it held against on T rows. Six-demo corpora will
mostly say n<=3; the sample-size doctrine handles the hedging.
"""

from pydantic import BaseModel

from counterstrat.roundscript.models import RoundScript

MAX_EVIDENCE = 6


class RetakeRow(BaseModel):
    side: str  # "CT" = retake, "T" = post-plant hold
    site: str
    man_diff: str  # acting side's advantage at plant: "-2".."+2" or "even"
    n: int
    win_rate: float
    evidence: list[str]


class ApproachRow(BaseModel):
    side: str  # the TEAM's side that round (CT: our vectors; T: theirs)
    site: str
    approaches: list[str]  # sorted zones the retakers entered the site from
    n: int
    win_rate: float  # from the team's perspective
    evidence: list[str]


class RetakeReport(BaseModel):
    team_key: str
    map_name: str
    rows: list[RetakeRow]  # n desc
    approaches: list[ApproachRow]  # n desc


def _team_side(script: RoundScript, team_key: str) -> str | None:
    if script.t_team_key == team_key:
        return "T"
    if script.ct_team_key == team_key:
        return "CT"
    return None


def _man_diff_label(diff: int) -> str:
    if diff == 0:
        return "even"
    return f"{diff:+d}"


def _retake_vectors(script: RoundScript) -> list[str]:
    """Zones CT players entered the planted site from, after the plant."""
    plant = script.plant
    if plant is None or not script.tracks:
        return []
    vectors: set[str] = set()
    for player, stints in script.tracks.items():
        if script.sides.get(player) != "CT":
            continue
        for i in range(1, len(stints)):
            st = stints[i]
            if st.zone == plant.site and float(st.t0) > plant.t:
                vectors.add(stints[i - 1].zone)
                break
    return sorted(vectors)


def build_retake_report(scripts: list[RoundScript], team_key: str) -> RetakeReport:
    """Mine post-plant conversion and approach vectors for team_key."""
    map_name = scripts[0].map_name if scripts else ""
    rate_obs: dict[tuple[str, str, str], list[tuple[bool, str]]] = {}
    vector_obs: dict[tuple[str, str, tuple[str, ...]], list[tuple[bool, str]]] = {}

    for s in scripts:
        side = _team_side(s, team_key)
        if side is None or s.plant is None:
            continue
        round_id = f"{s.match_id}:{s.round_num}"
        won = s.winner == side or s.winner == team_key
        diff = (
            s.plant.alive_ct - s.plant.alive_t
            if side == "CT"
            else s.plant.alive_t - s.plant.alive_ct
        )
        rate_obs.setdefault((side, s.plant.site, _man_diff_label(diff)), []).append((won, round_id))
        vectors = _retake_vectors(s)
        if vectors:
            vector_obs.setdefault((side, s.plant.site, tuple(vectors)), []).append((won, round_id))

    rows = [
        RetakeRow(
            side=side,
            site=site,
            man_diff=diff,
            n=len(obs),
            win_rate=round(sum(w for w, _ in obs) / len(obs), 2),
            evidence=[rid for _, rid in obs][:MAX_EVIDENCE],
        )
        for (side, site, diff), obs in sorted(rate_obs.items())
    ]
    rows.sort(key=lambda r: (-r.n, r.side, r.site, r.man_diff))

    approaches = [
        ApproachRow(
            side=side,
            site=site,
            approaches=list(vectors),
            n=len(obs),
            win_rate=round(sum(w for w, _ in obs) / len(obs), 2),
            evidence=[rid for _, rid in obs][:MAX_EVIDENCE],
        )
        for (side, site, vectors), obs in sorted(vector_obs.items())
    ]
    approaches.sort(key=lambda a: (-a.n, a.side, a.site, a.approaches))

    return RetakeReport(team_key=team_key, map_name=map_name, rows=rows, approaches=approaches)
