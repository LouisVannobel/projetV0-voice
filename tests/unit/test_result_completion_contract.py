"""An attempted result request yields a strict envelope or a closed local rejection."""

from types import SimpleNamespace
from uuid import UUID

import pytest

from projetv0_voice.persistence import business_result

CALL = business_result.RetainedCall((business_result.RetainedTurn(
    UUID(int=101), 1, "user", "Pouvez-vous me rappeler ?", False),), 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, "", " ", "null", " null "])
async def test_attempted_result_requires_envelope_even_for_empty_response(response):
    requests = []

    async def infer(context, **kwargs):
        requests.append(kwargs)
        return response

    with pytest.raises(business_result.ResultSchemaRejected, match="^result_schema_rejected$"):
        await business_result.infer_partial_result(SimpleNamespace(run_inference=infer), CALL, None)
    assert len(requests) == 1 and requests[0]["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_validated_null_envelope_is_the_only_inferred_empty_result():
    async def infer(context, **kwargs):
        return '{"result":null}'

    assert await business_result.infer_partial_result(
        SimpleNamespace(run_inference=infer), CALL, None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", [business_result.RetainedCall((), 0),
                                     business_result.RetainedCall(CALL.turns, 0, erased=True)])
async def test_erased_or_no_caller_still_returns_without_inference(retained):
    async def infer(context, **kwargs):
        pytest.fail("no request for absent caller or erased content")

    assert await business_result.infer_partial_result(
        SimpleNamespace(run_inference=infer), retained, None) is None


@pytest.mark.parametrize("reason,expected", [("length", "length"), ("empty", "empty"),
                                             ("private-untrusted-reason", "unknown"),
                                             (None, "unknown")])
def test_completion_rejection_retains_only_closed_reason_and_constant_message(reason, expected):
    assert hasattr(business_result, "ResultCompletionRejected")
    error = business_result.ResultCompletionRejected(reason)
    assert error.reason == expected
    assert str(error) == "result_completion_rejected"
    assert vars(error) == {"reason": expected}
    assert "private-untrusted-reason" not in repr(error)
