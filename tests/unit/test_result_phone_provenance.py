"""Caller coordinates require complete supported occurrences in cited caller evidence."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import pytest

from projetv0_voice.models import MessageResultV1
from projetv0_voice.persistence.business_result import (
    RetainedCall,
    RetainedTurn,
    validate_result_provenance,
)

PHONE = "+33102030407"
CALLER = UUID(int=101)
SECOND = UUID(int=102)


@dataclass(frozen=True, repr=False)
class _FrenchRetainedCall(RetainedCall):
    default_phone_region: Literal["FR"] = "FR"


def result(phone=PHONE, evidence=None):
    return MessageResultV1.model_validate({
        "schema_version": 1, "quality": "partial", "category": "callback",
        "summary": "Rappel demandé.", "next_action": "Rappeler.",
        "contact": {"name": None, "callback_e164": phone, "preference": None,
                    "callback_source": "caller", "callback_confirmed": False},
        "evidence": evidence or [{"turn_id": str(CALLER), "role": "user"}],
        "request_confirmed": False,
    })


@pytest.mark.parametrize("text", [
    PHONE, "+33 1 02 03 04 07", "+33\u00a01\u00a002\u00a003\u00a004\u00a007",
    "+33-1-02-03-04-07", "+33.1.02.03.04.07", "(+33 1 02 03 04 07)",
    "Rappelez au +33 1 02 03 04 07. Merci.",
])
def test_explicit_international_format_preserves_exact_caller_digits(text):
    retained = RetainedCall((RetainedTurn(CALLER, 1, "user", text, False),), 0)
    validate_result_provenance(result(), retained, None)


def test_explicit_non_french_international_number_needs_no_country_inference():
    retained = RetainedCall((RetainedTurn(
        CALLER, 1, "user", "Call me at +44 20 7123 4567.", False),), 0)
    validate_result_provenance(result("+442071234567"), retained, None)


@pytest.mark.parametrize("text,phone", [
    ("01 02 03 04 07", PHONE),
    ("0 6. 1 2 34 56 78.", "+33612345678"),
    ("Rappelez au 07-70-12-34-56. Merci.", "+33770123456"),
])
def test_configured_fr_region_accepts_one_complete_national_caller_token(text, phone):
    retained = _FrenchRetainedCall((RetainedTurn(CALLER, 1, "user", text, False),), 0)
    validate_result_provenance(result(phone), retained, None)


@pytest.mark.parametrize("text,phone", [
    ("0262123456", "+33262123456"),
    ("0692123456", "+33692123456"),
    ("01 02 03 04 07 08", PHONE),
    ("01 02 03 04", PHONE),
    ("22 01 02 03 04 07", PHONE),
    ("x0102030407", PHONE),
    ("0102030407x", PHONE),
])
def test_fr_region_does_not_repair_or_guess_unsupported_national_tokens(text, phone):
    retained = _FrenchRetainedCall((RetainedTurn(CALLER, 1, "user", text, False),), 0)
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(phone), retained, None)


@pytest.mark.parametrize("text", [
    "0102030407", "01 02 03 04 07", "0033102030407",
    "+331020304070", "+33102030407+442071234567", "++33102030407", "1+33102030407",
    "x+33102030407", "+33102030407x", "é+33102030407", "+33102030407é",
    "\u0661+33102030407", "+33102030407\u0661", "+33102030407_",
    "22 +33102030407 33", "+33102030407 08", "+33 1 02 03 04 07 08",
    "+33/1/02/03/04/07", "+33\t1\t02\t03\t04\t07", "+33\u202f1 02 03 04 07",
    "plus trente-trois un zéro deux zéro trois zéro quatre zéro sept",
])
def test_unsupported_or_partial_phone_tokens_never_authorize_coordinates(text):
    retained = RetainedCall((RetainedTurn(CALLER, 1, "user", text, False),), 0)
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(), retained, None)


@pytest.mark.parametrize("call_type", [RetainedCall, _FrenchRetainedCall])
def test_international_phone_cannot_be_joined_across_two_cited_caller_turns(call_type):
    retained = call_type((RetainedTurn(CALLER, 1, "user", "+33102", False),
                         RetainedTurn(SECOND, 2, "user", "030407", False)), 0)
    evidence = [{"turn_id": str(item.turn_id), "role": "user"} for item in retained.turns]
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(evidence=evidence), retained, None)


@pytest.mark.parametrize("role,cited", [("assistant", True), ("user", False)])
@pytest.mark.parametrize("call_type", [RetainedCall, _FrenchRetainedCall])
def test_only_cited_caller_turn_can_supply_the_phone(role, cited, call_type):
    phone = "01 02 03 04 07" if call_type is _FrenchRetainedCall else PHONE
    retained = call_type((RetainedTurn(CALLER, 1, "user", "Rappelez-moi.", False),
                         RetainedTurn(SECOND, 2, role, phone, False)), 0)
    evidence = [{"turn_id": str(CALLER), "role": "user"}]
    if cited:
        evidence.append({"turn_id": str(SECOND), "role": role})
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(evidence=evidence), retained, None)
