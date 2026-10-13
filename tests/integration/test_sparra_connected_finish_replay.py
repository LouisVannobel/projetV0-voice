"""Finish replay must address the active admitted fixture call before media close."""

from inspect import signature

import pytest

from tests.integration.sparra_connected_scenario import Scenario


@pytest.mark.asyncio
async def test_finish_replay_targets_active_call_distinct_from_event_default(monkeypatch):
    scenario = Scenario.__new__(Scenario)
    scenario.control = "active-oct14-fixture"
    event_signature = signature(Scenario.event)
    assert scenario.control != event_signature.parameters["control"].default
    observed = []

    class EventObserved(Exception):
        pass

    async def observe_event(*args, **kwargs):
        # Bind against the actual event interface, including its real default.
        # Stop before transport/SQL: this regression qualifies routing only.
        bound = event_signature.bind(scenario, *args, **kwargs)
        bound.apply_defaults()
        observed.append((bound.arguments["kind"], bound.arguments["control"]))
        raise EventObserved

    monkeypatch.setattr(scenario, "event", observe_event)
    with pytest.raises(EventObserved):
        await scenario.finish_replay()
    assert observed == [("call.hangup", scenario.control)]
