"""RoundScript data models for CS2 round representation."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from counterstrat.roundscript.econ import EconSummary


class Formation(BaseModel):  # one side at one beat
    zones: list[tuple[int, str]]  # [(count, zone_id)], count-desc then alpha order


class Beat(BaseModel):
    label: str  # "B+00", "B+15", "FC+34", "PL+52"
    t: float  # clock_s
    t_form: Formation
    ct_form: Formation


class KillEvent(BaseModel):
    t: float
    killer: str
    victim: str
    killer_side: str
    zone: str
    weapon: str
    headshot: bool
    traded_within_4s: bool


class UtilEvent(BaseModel):  # filled by Task 15
    t: float
    thrower: str
    side: str
    nade: str
    from_zone: str
    to_zone: str
    lineup_id: str | None = None
    blinded: list[tuple[str, float]] = []


class PlantEvent(BaseModel):
    t: float
    site: str
    planter: str
    alive_t: int
    alive_ct: int


class MovementLine(BaseModel):  # filled by Task 14
    player: str
    side: str
    role_hint: str
    sentence: str


class RoundScript(BaseModel):
    match_id: str
    map_name: str
    card_checksum: str
    round_num: int
    score_t: int
    score_ct: int
    t_team_key: str
    ct_team_key: str
    economy: dict[str, "EconSummary"]
    beats: list[Beat]
    kills: list[KillEvent]
    utility: list[UtilEvent]
    plant: PlantEvent | None
    first_contact: KillEvent | None
    winner: str
    reason: str
    clock_used_s: float
    movements: list[MovementLine] = []

    def to_text(self) -> str:
        return ""  # Task 16 will implement full text serialization


try:
    from counterstrat.roundscript.econ import EconSummary

    RoundScript.model_rebuild()
except ImportError:
    pass
