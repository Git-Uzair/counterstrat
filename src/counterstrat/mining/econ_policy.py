"""Economy policy mining: how the team buys in each money state (plan Task 4)."""

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from counterstrat.mining.tendencies import _normalize_site, iter_round_states
from counterstrat.roundscript.models import RoundScript

MAX_EVIDENCE = 4
PISTOL_ROUNDS = (1, 13)


class EconPolicy(BaseModel):
    team_key: str
    map_name: str
    # state -> {buy_class -> share}. States: pistol, after_win, after_loss_1,
    # after_loss_2, after_loss_3plus.
    policy: dict[str, dict[str, float]] = Field(default_factory=dict)
    ns: dict[str, int] = Field(default_factory=dict)
    # Site committed on T-side pistols ("A"/"B"/"none" shares); n in ns["pistol_t"].
    pistol_round_sites: dict[str, float] = Field(default_factory=dict)
    # Buy shares in rounds 2/14 after losing the pistol; n in ns["post_pistol_loss"].
    post_pistol_loss_buy: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, list[str]] = Field(default_factory=dict)


def _shares(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    dist = {k: v / total for k, v in counts.items()}
    return dict(sorted(dist.items(), key=lambda x: (-x[1], x[0])))


def build_econ_policy(scripts: list[RoundScript], team_key: str) -> EconPolicy:
    """Mine buy behavior per money state from RoundScripts."""
    map_name = scripts[0].map_name if scripts else ""
    states = iter_round_states(scripts, team_key)

    by_match: dict[str, list] = defaultdict(list)
    for st in states:
        by_match[st.script.match_id].append(st)

    policy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    evidence: dict[str, list[str]] = defaultdict(list)
    pistol_sites: Counter[str] = Counter()
    pistol_t_n = 0
    post_pistol_loss: Counter[str] = Counter()
    post_pistol_loss_n = 0

    for match_states in by_match.values():
        match_states.sort(key=lambda st: st.script.round_num)
        streak = 0  # losses immediately before the current round
        for st in match_states:
            s = st.script
            rid = f"{s.match_id}:{s.round_num}"
            if st.prev_outcome == "first":
                streak = 0

            if s.round_num in PISTOL_ROUNDS:
                state_name: str | None = "pistol"
            elif st.prev_outcome == "won":
                state_name = "after_win"
            elif st.prev_outcome == "lost":
                if streak <= 1:
                    state_name = "after_loss_1"
                elif streak == 2:
                    state_name = "after_loss_2"
                else:
                    state_name = "after_loss_3plus"
            else:
                state_name = None  # OT starts / gaps: no meaningful money state

            if state_name is not None:
                policy_counts[state_name][st.buy_class] += 1
                if len(evidence[state_name]) < MAX_EVIDENCE:
                    evidence[state_name].append(rid)

            if s.round_num in PISTOL_ROUNDS and st.side == "T":
                pistol_t_n += 1
                pistol_sites[_normalize_site(s.plant.site) if s.plant else "none"] += 1

            if s.round_num in (2, 14) and st.prev_outcome == "lost":
                post_pistol_loss_n += 1
                post_pistol_loss[st.buy_class] += 1

            won_this = s.winner in (st.side, team_key)
            streak = 0 if won_this else streak + 1

    ns = {state: sum(counts.values()) for state, counts in policy_counts.items()}
    if pistol_t_n:
        ns["pistol_t"] = pistol_t_n
    if post_pistol_loss_n:
        ns["post_pistol_loss"] = post_pistol_loss_n

    return EconPolicy(
        team_key=team_key,
        map_name=map_name,
        policy={state: _shares(counts) for state, counts in sorted(policy_counts.items())},
        ns=ns,
        pistol_round_sites=_shares(pistol_sites),
        post_pistol_loss_buy=_shares(post_pistol_loss),
        evidence=dict(evidence),
    )
