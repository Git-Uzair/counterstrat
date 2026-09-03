"""Per-round predictor (spec Phase 5): next-round call from TeamBook + match state."""

from typing import Literal

from pydantic import BaseModel

from counterstrat.llm.base import LLMClient, LLMResult
from counterstrat.mapcard.compile import MapCard
from counterstrat.mining.tendencies import TeamBook


class PredictedRound(BaseModel):
    buy_type: Literal["full_eco", "semi_eco", "semi_buy", "full_buy"]
    first_contact_zone: str
    site: Literal["A", "B", "none"]
    execute: bool
    fast: bool
    p_site: dict[str, float] = {"A": 0.333, "B": 0.333, "none": 0.334}


class MatchState(BaseModel):
    score_t: int
    score_ct: int
    side: str  # "T" | "CT"
    prev_round_summaries: list[str] = []
    economy_estimate: str = ""


def predict_round(
    client: LLMClient,
    card: MapCard,
    teambook: TeamBook,
    match_state: MatchState,
) -> tuple[PredictedRound, LLMResult]:
    """Predict the next round's shape; structured output via ``complete_json``."""
    system = (
        f"You are a CS2 tactical analyst predicting the next round for side {match_state.side}.\n"
        f"Map Card:\n{card.to_yaml()}"
    )
    user = (
        f"TeamBook:\n{teambook.to_table_text()}\n\n"
        f"Match State: score T {match_state.score_t} - CT {match_state.score_ct}, "
        f"predicting for {match_state.side}.\n"
        f"Economy: {match_state.economy_estimate}\n"
        f"Recent rounds: {'; '.join(match_state.prev_round_summaries[-3:])}\n"
        "Predict: buy_type, first_contact_zone (valid map zone), site (A/B/none), "
        "execute (bool), fast (bool: first contact < 25s), and the p_site distribution."
    )
    return client.complete_json(system=system, user=user, schema=PredictedRound)
