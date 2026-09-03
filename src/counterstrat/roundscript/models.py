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

    def to_text(self, include_movements: bool = False, max_utility: int = 16) -> str:
        t_econ = self.economy.get("T")
        ct_econ = self.economy.get("CT")
        t_buy = t_econ.buy_type if t_econ else "unknown"
        t_spend = t_econ.spend if t_econ else 0
        ct_buy = ct_econ.buy_type if ct_econ else "unknown"
        ct_spend = ct_econ.spend if ct_econ else 0
        lines = [
            f"R{self.round_num} [T {t_buy}(${t_spend / 1000:.1f}k) | CT {ct_buy}(${ct_spend / 1000:.1f}k)] score {self.score_t}-{self.score_ct}"
        ]
        for beat in self.beats:
            t_zones = ", ".join(f"{c}x{z}" for c, z in beat.t_form.zones)
            ct_zones = ", ".join(f"{c}x{z}" for c, z in beat.ct_form.zones)
            lines.append(f"{beat.label}: T: {t_zones} | CT: {ct_zones}")
        if self.first_contact:
            fc = self.first_contact
            traded = " [traded]" if fc.traded_within_4s else ""
            lines.append(
                f"FC: {fc.killer}({fc.killer_side}) killed {fc.victim} @{fc.zone} [{fc.weapon}]{traded}"
            )
        if self.plant:
            lines.append(f"PL: {self.plant.planter} planted @{self.plant.site}")
        u_events = self.utility[:max_utility] if max_utility is not None else self.utility
        for u in u_events:
            line = f"{round(u.t)}s: {u.thrower}({u.side}) {u.nade} {u.from_zone}>{u.to_zone}"
            if u.lineup_id:
                line += f" [{u.lineup_id}]"
            if u.blinded:
                b_str = ", ".join(f"{v} {d:.1f}s" for v, d in u.blinded)
                line += f" (blinded: {b_str})"
            lines.append(line)
        if include_movements and self.movements:
            for m in self.movements:
                lines.append(m.sentence)
        m = int(self.clock_used_s // 60)
        s = int(self.clock_used_s % 60)
        lines.append(f"END {self.winner} {self.reason} @{m}:{s:02d}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return self.model_dump_json()


try:
    from counterstrat.roundscript.econ import EconSummary

    RoundScript.model_rebuild()
except ImportError:
    pass
