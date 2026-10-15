"""The provider-facing schema must enforce the same phone format as storage."""

import pytest
from jsonschema import Draft202012Validator

from projetv0_voice.models import MessageContactV1


@pytest.mark.parametrize("number", ["01 02 03 04 07", "0102030407", "+330102030407x"])
def test_provider_schema_refuses_numbers_rejected_by_local_contact(number):
    value = {
        "name": None,
        "callback_e164": number,
        "preference": None,
        "callback_source": "caller",
        "callback_confirmed": False,
    }
    assert not Draft202012Validator(MessageContactV1.model_json_schema()).is_valid(value)
    with pytest.raises(ValueError):
        MessageContactV1.model_validate(value)


@pytest.mark.parametrize("number,source", [("+33102030407", "caller"), (None, "missing")])
def test_provider_schema_and_local_contact_accept_supported_coordinates(number, source):
    value = {
        "name": None,
        "callback_e164": number,
        "preference": None,
        "callback_source": source,
        "callback_confirmed": False,
    }
    Draft202012Validator(MessageContactV1.model_json_schema()).validate(value)
    assert MessageContactV1.model_validate(value).callback_e164 == number
