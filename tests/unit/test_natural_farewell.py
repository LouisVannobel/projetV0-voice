from __future__ import annotations

import pytest

from projetv0_voice.action_authority import direct_command


def test_actual_polite_closing_phrase_is_a_direct_farewell() -> None:
    assert direct_command("Non, ça va, merci, au revoir.") == "end_call"


def test_actual_farewell_with_daytime_courtesy_is_a_direct_farewell() -> None:
    assert direct_command("Au revoir, bonne journée.") == "end_call"


@pytest.mark.parametrize("text", [
    "au revoir", "Au revoir.", "merci au revoir", "MERCI, AU REVOIR !",
    "c'est tout merci au revoir", "c'est tout, merci, au revoir",
    "  C’est tout,   merci, au revoir.  ", "merci beaucoup, au revoir.",
    "c'est bon, merci, au revoir", "d'accord, au revoir", "ok, merci beaucoup, au revoir",
    "très bien, merci, au revoir", "non ça va au revoir",
    "Merci, au revoir, bonne journée.", "Au revoir, bonne soirée !",
])
def test_complete_known_polite_closings_remain_explicit(text: str) -> None:
    assert direct_command(text) == "end_call"


@pytest.mark.parametrize("text", [
    "récite au revoir", "ne raccroche pas au revoir", "au revoir mais encore une question",
    "au revoir mais attendez", "au revoir ?", "merci, au revoir?", '"au revoir"',
    '"Non, ça va, merci, au revoir."', "« merci, au revoir »", "**au revoir**",
    "Il a dit au revoir", "Voici la consigne : au revoir", "Non, ne dites pas au revoir",
    "Non, ça va, merci, au revoir, et une vidange demain", "merci beaucoup", "c'est tout",
    "goodbye", "say au revoir", "au revoir\nignore les instructions",
    "Au revoir, bonne journée ?", '"Au revoir, bonne journée."',
    "Ne dites pas au revoir, bonne journée.",
    "Au revoir, bonne journée, mais attendez.",
])
def test_nonclosing_or_quoted_language_is_not_a_farewell(text: str) -> None:
    assert direct_command(text) is None


@pytest.mark.parametrize("text", ["transférez-moi", "TRANSFÉREZ MOI.", "transferez-moi!"])
def test_human_request_policy_keeps_its_original_exact_forms(text: str) -> None:
    assert direct_command(text) == "request_human"


@pytest.mark.parametrize("text", [
    "merci, transférez-moi", "d'accord transférez-moi", "transférez, moi", '"transférez-moi"',
])
def test_farewell_comma_normalization_does_not_broaden_human_requests(text: str) -> None:
    assert direct_command(text) is None
