"""FR dictation spans preserve the actual numeric continuity and all contributor IDs."""

import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from projetv0_voice.persistence.business_result import (
    ResultProvenanceRejected,
    RetainedCall,
    RetainedTurn,
    infer_partial_result,
    validate_result_provenance,
)
from tests.unit.test_result_phone_provenance import result

PHONE = "+33612345678"


def call(*texts, region="FR", loss=0):
    return RetainedCall(tuple(
        RetainedTurn(UUID(int=101 + index), index + 1, "user", text, False)
        for index, text in enumerate(texts)
    ), loss, default_phone_region=region)


def references(retained, indexes=None):
    selected = retained.turns if indexes is None else [retained.turns[i] for i in indexes]
    return [{"turn_id": str(turn.turn_id), "role": turn.role} for turn in selected]


@pytest.mark.parametrize("texts", [
    ("Je suis Camille, rappelez au 06 12 34 56", "78"),
    ("06", "12", "34", "56", "78"),
    ("0", "6", "1", "2", "3", "4", "5", "6", "7", "8"),
    ("06.12.34.56.", "78"),
    ("06\u00a012\u00a034\u00a056", "78"),
])
def test_fr_numeric_dictation_uses_all_actual_contributors_without_mutating_turns(texts):
    retained = call(*texts)
    before = retained.turns
    validate_result_provenance(result(PHONE, references(retained)), retained, None)
    assert retained.turns is before
    assert tuple(t.text for t in retained.turns) == texts


@pytest.mark.parametrize("indexes", [[0], [1]])
def test_fragmented_phone_requires_every_contributing_caller_reference(indexes):
    retained = call("06 12 34 56", "78")
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained, indexes)), retained, None)


@pytest.mark.parametrize("middle", ["", " ", "non, corrigez", "donc", "78 puis"])
def test_numeric_fragments_are_not_joined_across_empty_words_or_corrections(middle):
    retained = call("06 12 34 56", middle, "78")
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained)), retained, None)


def test_assistant_recap_cannot_replace_a_numeric_caller_fragment():
    retained = call("06 12 34 56", "78", "D'accord : 06 12 34 56 78")
    retained = RetainedCall((retained.turns[0], RetainedTurn(
        retained.turns[1].turn_id, 2, "assistant", "78", False), RetainedTurn(
        retained.turns[2].turn_id, 3, "assistant", "D'accord : 06 12 34 56 78", False)),
        0, default_phone_region="FR")
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained)), retained, None)


def test_numeric_caller_fragments_are_not_filtered_past_an_assistant_turn():
    retained = call("06 12 34 56", "Je vous écoute.", "78")
    middle = retained.turns[1]
    retained = RetainedCall((retained.turns[0], RetainedTurn(
        middle.turn_id, middle.turn_no, "assistant", middle.text, False), retained.turns[2]),
        0, default_phone_region="FR")
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained)), retained, None)


@pytest.mark.parametrize("loss,gap", [(1, False), (0, True)])
def test_known_loss_or_missing_turn_number_prevents_fragment_join(loss, gap):
    retained = call("06 12 34 56", "78", loss=loss)
    if gap:
        second = retained.turns[1]
        retained = RetainedCall((retained.turns[0], RetainedTurn(
            second.turn_id, 3, second.role, second.text, second.interrupted)), 0,
            default_phone_region="FR")
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained)), retained, None)


@pytest.mark.parametrize("texts", [
    ("06 12 34 56", "78", "90"),
    ("00", "06 12 34 56", "78"),
    ("06", "12", "34", "56", "78", "90"),
    ("x06 12 34 56", "78"),
    ("06 12 34 56x", "78"),
    ("06/12/34/56", "78"),
    ("06\t12\t34\t56", "78"),
    ("06\u202f12\u202f34\u202f56", "78"),
    ("zéro six douze trente-quatre cinquante-six", "78"),
    ("06 12 34 56", "soixante-dix-huit"),
    ("06 12 34 56", "78 merci"),
    ("06 12 34 56", "78x"),
])
def test_join_never_truncates_repairs_or_harvests_digits(texts):
    retained = call(*texts)
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained)), retained, None)


def test_national_fragment_join_requires_explicit_fr_region():
    retained = call("06 12 34 56", "78", region=None)
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained)), retained, None)


def test_joined_national_digits_still_require_existing_metropolitan_fr_validity():
    retained = call("0262 12 34", "56")
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result("+33262123456", references(retained)), retained, None)


@pytest.mark.parametrize("indexes", [[0, 1], [3, 4]])
def test_each_complete_occurrence_of_same_number_can_authorize_contact(indexes):
    retained = call("06 12 34 56", "78", "Je répète.", "06 12 34 56", "78")
    validate_result_provenance(result(PHONE, references(retained, indexes)), retained, None)


def test_evidence_cannot_mix_fragments_from_two_occurrences():
    retained = call("06 12 34 56", "78", "Je répète.", "06 12 34 56", "78")
    with pytest.raises(ValueError, match="^result_caller_coordinate_missing$"):
        validate_result_provenance(result(PHONE, references(retained, [0, 4])), retained, None)


@pytest.mark.parametrize("texts,phone", [
    (("06 12 34 56 78", "01 02 03 04 07"), PHONE),
    (("06 12 34 56 78", "90"), PHONE),
    (("+44 20 7123 4567",), "+442071234567"),
])
def test_existing_complete_occurrences_still_work_with_known_transcript_loss(texts, phone):
    retained = call(*texts, loss=1)
    validate_result_provenance(result(phone, references(retained, [0])), retained, None)


@pytest.mark.parametrize("provider", [PHONE, "+33102030407"])
def test_provider_source_still_requires_exact_observed_coordinate(provider):
    retained = call("Merci de me rappeler.")
    observed_result = result(PHONE)
    observed_result = observed_result.model_copy(update={
        "contact": observed_result.contact.model_copy(update={"callback_source": "provider"})
    })
    if provider == PHONE:
        validate_result_provenance(observed_result, retained, provider)
    else:
        with pytest.raises(ValueError, match="^result_provider_coordinate_mismatch$"):
            validate_result_provenance(observed_result, retained, provider)


@pytest.mark.parametrize("kind", ["missing_id", "forged_role", "erased"])
def test_span_support_does_not_weaken_global_evidence_checks(kind):
    retained = call("06 12 34 56", "78")
    evidence = references(retained)
    if kind == "missing_id":
        evidence.append({"turn_id": str(UUID(int=999)), "role": "user"})
    elif kind == "forged_role":
        evidence[1]["role"] = "assistant"
    else:
        retained = RetainedCall(retained.turns, 0, erased=True, default_phone_region="FR")
    with pytest.raises(ValueError):
        validate_result_provenance(result(PHONE, evidence), retained, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("omit_support", [False, True])
async def test_inference_catalogue_exposes_whole_occurrence_and_invalid_response_stays_rejected(
    omit_support,
):
    retained = call("Je suis Camille, rappelez au 06 12 34 56", "78")
    observed = []

    async def infer(context, **kwargs):
        observed.append(json.loads(context.get_messages()[0]["content"]))
        response = result(PHONE, references(retained, [0] if omit_support else None))
        return json.dumps({"result": response.model_dump(mode="json")})

    service = SimpleNamespace(run_inference=infer)
    if omit_support:
        with pytest.raises(ResultProvenanceRejected, match="^result_provenance_rejected$"):
            await infer_partial_result(service, retained, None)
    else:
        actual = await infer_partial_result(service, retained, None)
        assert actual.contact.callback_e164 == PHONE
        assert [item.turn_id for item in actual.evidence] == [t.turn_id for t in retained.turns]
    assert len(observed) == 1
    assert observed[0]["caller_phone_candidates"] == [{
        "callback_e164": PHONE,
        "turn_ids": [str(UUID(int=101)), str(UUID(int=102))],
    }]
    assert [t["text"] for t in observed[0]["turns"]] == [t.text for t in retained.turns]
