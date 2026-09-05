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
    from counterstrat.constants import RANGE_CLOSE_M, RANGE_LONG_M

    assert f"{RANGE_CLOSE_M:.0f}m" in TACTICAL_DOCTRINE
    assert f"{RANGE_LONG_M:.0f}m" in TACTICAL_DOCTRINE
    assert "force" in TACTICAL_DOCTRINE.lower()  # force the other range


def test_doctrine_carries_control_conversion_language():
    low = TACTICAL_DOCTRINE.lower()
    assert "conceded" in low or "concede" in low
    assert "convert" in low  # control only pays when converted


def test_chat_system_embeds_doctrine():
    assert TACTICAL_DOCTRINE in _chat_system()


def test_insights_system_embeds_doctrine():
    assert TACTICAL_DOCTRINE in build_insights_system("map block")


# --- Utility / contested-space doctrine (2026-09-05 strategy research) ---
# The First Read told teams to "wait out utility" and treated post-dump zones
# as free. These pins force the corrected physics: smokes cut vision both
# ways, spent utility does not un-man an angle, and some space must be taken
# even at HP cost.


def test_doctrine_smoke_blocks_vision_both_ways():
    low = TACTICAL_DOCTRINE.lower()
    assert "both ways" in low
    assert "not a corridor" in low  # pushing through a smoke into a crossfire
    assert "one-way" not in low[: low.find("both ways")]  # the claim is about walls


def test_doctrine_spent_utility_is_not_open_space():
    low = TACTICAL_DOCTRINE.lower()
    assert "denial" in low  # what expires is the denial layer
    assert "still hold" in low or "stay manned" in low or "stays manned" in low
    assert "never dry" in low  # attack the window with your own utility


def test_doctrine_mandatory_space_and_hp_exchange():
    low = TACTICAL_DOCTRINE.lower()
    assert "mandatory" in low
    assert "exchange rate" in low  # wait-vs-force is a trade, not a default
    assert "clock" in low  # waiting costs time: resets, rotations, info decay


def test_chat_prompt_never_calls_spent_zones_naked():
    s = _chat_system()
    assert "go naked" not in s


def test_map_priors_block_for_pool_maps():
    from counterstrat.llm.prompts import map_priors_block

    block = map_priors_block("de_ancient")
    assert "<map_priors>" in block
    assert "override" in block.lower()  # data beats priors, stated in the block
    assert "donut" in block.lower()
    anubis = map_priors_block("de_anubis")
    assert "mid" in anubis.lower() and "connector" in anubis.lower()
    assert map_priors_block("de_unknownmap") == ""
    assert map_priors_block("") == ""


def test_chat_and_insights_prompts_embed_map_priors():
    # The synthetic teambook is de_anubis: its priors ride into the chat prompt.
    assert "<map_priors>" in _chat_system()
    sys_prompt = build_insights_system("map block", map_name="de_ancient")
    assert "<map_priors>" in sys_prompt and "donut" in sys_prompt.lower()
    assert "<map_priors>" not in build_insights_system("map block")
