"""Economy policy miner tests (plan Task 4)."""

from conftest import SYNTHETIC_TEAM

from counterstrat.mining.econ_policy import build_econ_policy
from counterstrat.roundscript.econ import EconSummary
from counterstrat.roundscript.models import PlantEvent, RoundScript

TEAM = "abc"
OPP = "xyz"


def _round(
    round_num: int,
    *,
    buy: str,
    winner: str,
    side: str = "T",
    plant_site: str | None = None,
) -> RoundScript:
    return RoundScript(
        match_id="em1",
        map_name="de_anubis",
        card_checksum="chk123",
        round_num=round_num,
        score_t=0,
        score_ct=0,
        t_team_key=TEAM if side == "T" else OPP,
        ct_team_key=TEAM if side == "CT" else OPP,
        economy={
            side: EconSummary(buy_type=buy, spend=1000, equip=2000, awps=0, loss_streak=0),
            ("CT" if side == "T" else "T"): EconSummary(
                buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0
            ),
        },
        beats=[],
        kills=[],
        utility=[],
        plant=(
            PlantEvent(t=40.0, site=plant_site, planter="p1", alive_t=4, alive_ct=3)
            if plant_site
            else None
        ),
        first_contact=None,
        winner=winner,
        reason="t_killed",
        clock_used_s=60.0,
        movements=[],
    )


def test_econ_policy_loss_streak_states():
    scripts = [
        _round(1, buy="semi_eco", winner="CT", plant_site="BombsiteA"),  # pistol, lost
        _round(2, buy="full_eco", winner="CT"),  # after_loss_1
        _round(3, buy="semi_buy", winner="CT"),  # after_loss_2
        _round(4, buy="full_buy", winner="T"),  # after_loss_3plus
        _round(5, buy="full_buy", winner="T"),  # after_win
    ]
    pol = build_econ_policy(scripts, TEAM)

    assert pol.policy["pistol"]["semi_eco"] == 1.0 and pol.ns["pistol"] == 1
    assert pol.policy["after_loss_1"]["full_eco"] == 1.0 and pol.ns["after_loss_1"] == 1
    assert pol.policy["after_loss_2"]["semi_buy"] == 1.0 and pol.ns["after_loss_2"] == 1
    assert pol.policy["after_loss_3plus"]["full_buy"] == 1.0
    assert pol.ns["after_loss_3plus"] == 1
    assert pol.policy["after_win"]["full_buy"] == 1.0 and pol.ns["after_win"] == 1

    assert pol.post_pistol_loss_buy == {"full_eco": 1.0}
    assert pol.pistol_round_sites == {"A": 1.0}
    assert pol.evidence["after_loss_3plus"] == ["em1:4"]


def test_econ_policy_halftime_resets_streak():
    scripts = [
        _round(11, buy="full_buy", winner="CT"),
        _round(12, buy="full_buy", winner="CT"),  # after_loss_1 (r11 lost)
        _round(13, buy="semi_eco", winner="T", side="CT"),  # second pistol: streak reset
    ]
    pol = build_econ_policy(scripts, TEAM)
    assert pol.ns["pistol"] == 1
    assert pol.policy["pistol"]["semi_eco"] == 1.0
    assert "after_loss_2" not in pol.policy


def test_econ_policy_synthetic_smoke(synthetic_scripts):
    pol = build_econ_policy(synthetic_scripts, SYNTHETIC_TEAM)
    assert pol.team_key == SYNTHETIC_TEAM
    for state, dist in pol.policy.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert pol.ns[state] >= 1


def test_econ_policy_empty_for_unknown_team(synthetic_scripts):
    pol = build_econ_policy(synthetic_scripts, "nobody")
    assert pol.policy == {} and pol.ns == {}
