from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D
from uuid import UUID

import pytest

from projetv0_voice.budget_observation import BudgetObservation, BudgetObservationIdentity
from tests.integration.test_fixed_v2_process import NOW


def budget_policy(**changes):
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPolicy

    values = dict(
        policy_id=UUID(int=1),
        identity=BudgetObservationIdentity(
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "+33102030405",
            "fixture-connection",
            NOW - timedelta(days=1),
            NOW + timedelta(days=1),
        ),
        openrouter_anchor_usd=D("0"),
        telnyx_anchor_usd=D("0"),
        paid_until=NOW + timedelta(days=1),
        qualified_profile_sha256="5" * 64,
        runtime_contract_sha256="6" * 64,
        agent_bundle_sha256="7" * 64,
        inference_profile_sha256="8" * 64,
        image_digest="fixture@sha256:" + "9" * 64,
        deployment_id="fixture",
    )
    return OwnedPilotBudgetPolicy(**(values | changes))


def observation(**changes):
    value = BudgetObservation(
        NOW,
        NOW,
        budget_policy().identity,
        D("0"),
        D("5"),
        D("5"),
        D("5"),
        None,
        NOW + timedelta(days=1),
        D("0"),
        D("5"),
        frozenset(),
        "a" * 64,
    )
    return replace(value, **changes)


@pytest.mark.parametrize(
    "provider,limit",
    [
        ("openrouter", "1.219592"),
        ("telnyx", "3.78"),
    ],
)
def test_independent_exact_whole_reserve_threshold(provider, limit):
    from projetv0_voice.owned_pilot_budget import affordable

    field = provider + "_reported_spend_usd"
    assert affordable(budget_policy(), observation(**{field: D(limit)}), 0,
                      reported_floors=(D("0"), D("0")))
    assert not affordable(budget_policy(), observation(**{field: D(limit) + D("0.000001")}), 0,
                          reported_floors=(D("0"), D("0")))


def test_forecast_gap_cash_and_last_permit_count_are_distinct():
    from projetv0_voice.owned_pilot_budget import affordable

    p = budget_policy()
    # With zero historical anchor the full unchanged envelope permits n=0,1,2,
    # then conservatively refuses. The lifetime5 is a maximum, not five promises.
    assert affordable(p, observation(), 2, reported_floors=(D("0"), D("0")))
    assert not affordable(p, observation(), 3, reported_floors=(D("0"), D("0")))
    assert not affordable(p, observation(openrouter_available_cash_usd=D("4.790543")), 2,
                          reported_floors=(D("0"), D("0")))
    assert affordable(p, observation(openrouter_available_cash_usd=D("4.790544")), 2,
                      reported_floors=(D("0"), D("0")))


def test_reported_floors_cannot_be_implicitly_zero():
    from projetv0_voice.owned_pilot_budget import affordable

    with pytest.raises(TypeError):
        affordable(budget_policy(), observation(), 0)


@pytest.mark.parametrize(
    "floors",
    [None, (), (D("0"),), (D("0"), D("0"), D("0")), [D("0"), D("0")],
     (0, D("0")), ("0", D("0")), (D("NaN"), D("0")), (D("sNaN"), D("0")),
     (D("Infinity"), D("0")), (D("-0.1"), D("0")), (D("1e-25"), D("0")),
     (D("1e25"), D("0")), (D("0." + "1" * 65), D("0")),
     (D("0"), D("NaN")), (D("0"), D("-0.1"))],
)
def test_invalid_reported_floors_refuse_without_money_arithmetic(floors):
    from projetv0_voice.owned_pilot_budget import affordable

    assert affordable(budget_policy(), observation(), 0, reported_floors=floors) is False


@pytest.mark.parametrize(
    "floors,count,raw,allowed",
    [
        (("1.219592", "3.78"), 0, ("0.2", "0.5"), True),
        (("1.219593", "0"), 0, ("0.2", "0.5"), False),
        (("0", "3.780001"), 0, ("0.2", "0.5"), False),
        (("0.1", "0.1"), 2, ("0.2", "0.5"), True),
        (("0.1", "0.1"), 3, ("0.2", "0.5"), False),
        (("0.1", "0.1"), 0, ("1.219593", "0.5"), False),
    ],
)
def test_floor_raw_and_forecast_each_preserve_the_independent_provider_cap(
    floors, count, raw, allowed,
):
    from projetv0_voice.owned_pilot_budget import affordable

    fact = observation(openrouter_reported_spend_usd=D(raw[0]),
                       telnyx_reported_spend_usd=D(raw[1]))
    assert affordable(budget_policy(), fact, count,
                      reported_floors=(D(floors[0]), D(floors[1]))) is allowed


@pytest.mark.parametrize("provider", ["openrouter", "telnyx"])
def test_original_anchor_remains_a_raw_observation_fence(provider):
    from projetv0_voice.owned_pilot_budget import observation_valid

    policy = budget_policy(**{provider + "_anchor_usd": D("0.2")})
    assert observation_valid(policy, observation(**{provider + "_reported_spend_usd": D("0.2")}),
                             NOW)
    assert not observation_valid(
        policy, observation(**{provider + "_reported_spend_usd": D("0.199999")}), NOW,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_calls", 6),
        ("cost_scope", "different"),
        ("openrouter_anchor_usd", D("NaN")),
        ("telnyx_anchor_usd", D("-1")),
    ],
)
def test_policy_rejects_changed_envelope(field, value):
    with pytest.raises(ValueError):
        budget_policy(**{field: value})


def test_operator_command_requires_retirement_before_reading_runtime(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    result = subprocess.run(
        [sys.executable, "-m", "projetv0_voice.owned_pilot_budget_cli", "enroll"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={"SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:/Windows"),
             "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
    )
    assert result.returncode == 2
    assert "enroll requires --predecessor-retired" in result.stderr
    assert not list(tmp_path.iterdir())
