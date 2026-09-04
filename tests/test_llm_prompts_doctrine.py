"""Item 1 (space-vision research): tactical doctrine blocks in chat + insights prompts.

The First Read misread a lone info-gathering player as duel-seeking because the
prompts carried no doctrine separating the two readings. These tests pin the
doctrine block into both LLM-facing system prompts.
"""

from conftest import SYNTHETIC_TEAM, build_synthetic_scripts

from counterstrat.llm.insights import build_insights_system
from counterstrat.llm.prompts import TACTICAL_DOCTRINE, build_chat_system
from counterstrat.mining.tendencies import build_teambook


def _chat_system() -> str:
    teambook = build_teambook(build_synthetic_scripts(), SYNTHETIC_TEAM)
    return build_chat_system("map block", teambook)


def test_doctrine_separates_info_lurk_from_duel_lurk():
    assert "info lurk" in TACTICAL_DOCTRINE.lower()
    assert "duel" in TACTICAL_DOCTRINE.lower()
    # the anti-lurk counters: deny the information, punish the lurk-up timing
    assert "sightline" in TACTICAL_DOCTRINE.lower()
    assert "2-3s" in TACTICAL_DOCTRINE


def test_doctrine_carries_engagement_range_bands():
    # bands must quote the shared constants so text and miners cannot drift
    from counterstrat.constants import RANGE_CLOSE_U, RANGE_LONG_U

    assert f"{RANGE_CLOSE_U:.0f}u" in TACTICAL_DOCTRINE
    assert f"{RANGE_LONG_U:.0f}u" in TACTICAL_DOCTRINE
    assert "force" in TACTICAL_DOCTRINE.lower()  # force the other range


def test_doctrine_carries_control_conversion_language():
    low = TACTICAL_DOCTRINE.lower()
    assert "conceded" in low or "concede" in low
    assert "convert" in low  # control only pays when converted


def test_chat_system_embeds_doctrine():
    assert TACTICAL_DOCTRINE in _chat_system()


def test_insights_system_embeds_doctrine():
    assert TACTICAL_DOCTRINE in build_insights_system("map block")
