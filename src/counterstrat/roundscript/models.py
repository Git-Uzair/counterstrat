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
    # Range fields (space-vision research item 2); None/False on scripts
    # serialized before the upgrade - consumers must skip, not crash.
    distance: float | None = None  # attacker->victim in METERS (awpy unit)
    thrusmoke: bool = False
    penetrated: bool = False  # wallbang


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


class ZoneStint(BaseModel):
    """One contiguous alive stay in a zone, whole seconds since freeze end.

    Gaze fields (space-vision research item 4) are set only on hold-stints
    (>= 8s) serialized after the upgrade; None means "not annotated", never
    "confirmed nothing".
    """

    t0: int
    t1: int
    zone: str
    watched: str | None = None  # zone the stint's view direction centered on
    locked: bool | None = None  # True: held one line; False: scanning sweep
    support_m: float | None = None  # median distance to nearest living teammate


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
    # v2 (2026-09-04 plan Task 3): per-player zone stints + sides; empty on
    # scripts serialized before the upgrade (renderers must degrade gracefully).
    tracks: dict[str, list[ZoneStint]] = {}
    sides: dict[str, str] = {}

    def to_text(self, include_movements: bool = False, max_utility: int = 16) -> str:
        lines = [self._header_line()]
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

    def _header_line(self) -> str:
        t_econ = self.economy.get("T")
        ct_econ = self.economy.get("CT")
        t_buy = t_econ.buy_type if t_econ else "unknown"
        t_spend = t_econ.spend if t_econ else 0
        ct_buy = ct_econ.buy_type if ct_econ else "unknown"
        ct_spend = ct_econ.spend if ct_econ else 0
        return (
            f"R{self.round_num} [T {t_buy}(${t_spend / 1000:.1f}k) | "
            f"CT {ct_buy}(${ct_spend / 1000:.1f}k)] score {self.score_t}-{self.score_ct}"
        )

    def _side_of(self, player: str) -> str:
        return self.sides.get(player, "?")

    def to_timeline_text(self, lite: bool = False, include_holds: bool = False) -> str:
        """The complete chronological round timeline (2026-09-04 plan Task 3).

        Everything that happened with absolute seconds: anchors header
        (first contact / plant / end with precomputed deltas), 15s state
        snapshots from the beats, then the merged event stream - SPAWNS and
        MOVE from ``tracks`` (skipped when ``lite`` or on pre-v2 scripts),
        EVERY kill (``to_text`` renders only first contact), uncapped utility,
        PLANT with timestamp and alive counts. The lab measured today's text
        at 83%/62% (easy/hard) vs 100%/96% for this stream.

        ``lite`` + ``include_holds``: the budget mode keeps the gaze - HOLD
        lines for every stint with a resolved watch (who held which angle)
        stay in, only the plain MOVE/SPAWNS traffic is dropped. This is what
        the First Read embeds so it can tell an info lurk from a duel lurk.
        """
        lines = [self._header_line()]

        anchors: list[str] = []
        if self.first_contact:
            fc = self.first_contact
            anchors.append(
                f"first_contact={fc.t:.0f}s ({fc.killer} {fc.killer_side} kills "
                f"{fc.victim} in `{fc.zone}`)"
            )
        if self.plant:
            rel = (
                f" (+{self.plant.t - self.first_contact.t:.0f}s after first contact)"
                if self.first_contact
                else ""
            )
            anchors.append(f"plant={self.plant.t:.0f}s at `{self.plant.site}`{rel}")
        anchors.append(f"end={self.clock_used_s:.0f}s ({self.winner} wins, {self.reason})")
        lines.append("anchors: " + "; ".join(anchors))

        for beat in self.beats:
            if beat.t <= 0:
                continue
            t_zones = ", ".join(f"{c}x`{z}`" for c, z in beat.t_form.zones)
            ct_zones = ", ".join(f"{c}x`{z}`" for c, z in beat.ct_form.zones)
            lines.append(f"state@{beat.t:.0f}s: T[{t_zones}] CT[{ct_zones}]")

        events: list[tuple[float, int, str]] = []  # (t, tiebreak, line)
        if self.tracks and (not lite or include_holds):

            def _gaze(st: ZoneStint) -> str:
                if not st.watched:
                    return ""
                parts = [f"{'locked' if st.locked else 'scanning'}"]
                if st.support_m is not None:
                    parts.append(f"nearest mate {st.support_m:.0f}m")
                return f" watching `{st.watched}` ({', '.join(parts)})"

            if not lite:
                spawn = ", ".join(
                    f"{p}:`{st[0].zone}`" for p, st in sorted(self.tracks.items()) if st
                )
                if spawn:
                    lines.append(f"t=0s SPAWNS {spawn}")

            for player, stints in sorted(self.tracks.items()):
                for i, st in enumerate(stints):
                    gaze = _gaze(st)
                    if lite:
                        # Budget mode: only the tactically loaded lines - a
                        # stint whose view direction resolved to a zone.
                        if gaze:
                            events.append(
                                (
                                    float(st.t0),
                                    0,
                                    f"HOLD {player} ({self._side_of(player)}) in `{st.zone}`{gaze}",
                                )
                            )
                        continue
                    if i == 0:
                        # spawn stint: only noteworthy when a watch resolved
                        if gaze:
                            events.append(
                                (
                                    float(st.t0),
                                    0,
                                    f"HOLD {player} ({self._side_of(player)}) in `{st.zone}`{gaze}",
                                )
                            )
                        continue
                    events.append(
                        (
                            float(st.t0),
                            0,
                            f"MOVE {player} ({self._side_of(player)}) enters `{st.zone}`"
                            + (f",{gaze}" if gaze else ""),
                        )
                    )
        for k in self.kills:
            traded = " [traded]" if k.traded_within_4s else ""
            kill_line = (
                f"KILL {k.killer} ({k.killer_side}) kills {k.victim} in "
                f"`{k.zone}` [{k.weapon}]{traded}"
            )
            events.append((k.t, 1, kill_line))
        for u in self.utility:
            line = f"UTIL {u.thrower} ({u.side}) {u.nade} from `{u.from_zone}` lands `{u.to_zone}`"
            if u.lineup_id:
                line += f" [{u.lineup_id}]"
            if u.blinded:
                line += " (blinds " + ", ".join(f"{v} {d:.1f}s" for v, d in u.blinded) + ")"
            events.append((u.t, 1, line))
        if self.plant:
            p = self.plant
            events.append(
                (
                    p.t,
                    1,
                    f"PLANT {p.planter} plants at `{p.site}` ({p.alive_t}v{p.alive_ct} alive)",
                )
            )
        events.sort(key=lambda e: (e[0], e[1]))
        lines += [f"t={t:.0f}s {text}" for t, _, text in events]
        lines.append(f"END: {self.winner} wins ({self.reason}) at {self.clock_used_s:.0f}s")
        return "\n".join(lines)

    def to_json(self) -> str:
        return self.model_dump_json()


try:
    from counterstrat.roundscript.econ import EconSummary

    RoundScript.model_rebuild()
except ImportError:
    pass
