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
    "count,openrouter_limit,telnyx_limit",
    [
        (0, "1.219592", "3.78"),
        (1, "1.724660", "3.90"),
        (2, "2.229728", "4.02"),
        (3, "2.734796", "4.14"),
        (4, "3.239864", "4.26"),
    ],
)
@pytest.mark.parametrize("provider", ["openrouter", "telnyx"])
@pytest.mark.parametrize("dominant", ["raw", "floor"])
def test_independent_exact_remaining_reserve_cap(
    count, openrouter_limit, telnyx_limit, provider, dominant,
):
    from projetv0_voice.owned_pilot_budget import affordable

    field = provider + "_reported_spend_usd"
    limit = D(openrouter_limit if provider == "openrouter" else telnyx_limit)
    for excess, allowed in ((D("0"), True), (D("0.000001"), False)):
        floors = [D("0"), D("0")]
        if dominant == "floor":
            floors[provider == "telnyx"] = limit + excess
        fact = observation(**{field: limit + excess if dominant == "raw" else D("0.2")})
        assert affordable(budget_policy(), fact, count, reported_floors=tuple(floors)) is allowed


def test_forecast_gap_cash_and_last_permit_count_are_distinct():
    from projetv0_voice.owned_pilot_budget import affordable

    p = budget_policy()
    for count in range(5):
        assert affordable(p, observation(), count, reported_floors=(D("0"), D("0")))
    assert not affordable(p, observation(), 5, reported_floors=(D("0"), D("0")))
    assert not affordable(p, observation(openrouter_available_cash_usd=D("3.780407")), 2,
                          reported_floors=(D("0"), D("0")))
    assert affordable(p, observation(openrouter_available_cash_usd=D("3.780408")), 2,
                      reported_floors=(D("0"), D("0")))


@pytest.mark.parametrize(
    "count,openrouter_threshold,telnyx_threshold",
    [(0, "4.380408", "1.62"), (1, "3.875340", "1.50"),
     (2, "3.580408", "1.38"), (3, "3.580408", "1.26"), (4, "3.580408", "1.14")],
)
@pytest.mark.parametrize(
    "field", ["openrouter_available_cash_usd", "openrouter_limit_remaining_usd",
              "telnyx_available_cash_usd"],
)
def test_remaining_reserve_funding_keeps_raw_gap(
    count, openrouter_threshold, telnyx_threshold, field,
):
    from projetv0_voice.owned_pilot_budget import affordable

    threshold = D(telnyx_threshold if field.startswith("telnyx") else openrouter_threshold)
    for shortfall, allowed in ((D("0"), True), (D("0.000001"), False)):
        fact = observation(openrouter_reported_spend_usd=D("0.2"),
                           telnyx_reported_spend_usd=D("0.1"),
                           **{field: threshold - shortfall})
        assert affordable(budget_policy(), fact, count,
                          reported_floors=(D("0.8"), D("0.5"))) is allowed


@pytest.mark.parametrize("count", [True, False, -1, 5, 6, 1.0, "1", None])
def test_invalid_or_exhausted_count_cannot_release_reserve(count):
    from projetv0_voice.owned_pilot_budget import affordable

    assert not affordable(budget_policy(), observation(), count,
                          reported_floors=(D("0"), D("0")))


@pytest.mark.parametrize(
    "field", ["openrouter_available_cash_usd", "openrouter_limit_remaining_usd"],
)
@pytest.mark.parametrize("raw", ["0.33", "0.30"])
def test_original_anchor_example_requires_full_forecast_gap(field, raw):
    from projetv0_voice.owned_pilot_budget import affordable

    # A=.30,m=2,E=1.310136. A .03 restatement requires .03 more funding.
    threshold = D("3.750408") if raw == "0.33" else D("3.780408")
    for shortfall, allowed in ((D("0"), True), (D("0.000001"), False)):
        fact = observation(openrouter_reported_spend_usd=D(raw),
                           **{field: threshold - shortfall})
        assert affordable(budget_policy(openrouter_anchor_usd=D("0.30")), fact, 2,
                          reported_floors=(D("0.33"), D("0"))) is allowed


@pytest.mark.parametrize(
    "raw,floor,threshold",
    [("0.33", "1.310135", "3.750408"), ("0.33", "1.310136", "3.750408"),
     ("0.33", "1.310137", "3.750409"), ("1.310135", "0.33", "2.770273"),
     ("1.310136", "0.33", "2.770272"), ("1.310137", "0.33", "2.770272")],
)
def test_forecast_report_transition_keeps_exact_remaining_envelope(raw, floor, threshold):
    from projetv0_voice.owned_pilot_budget import affordable

    for shortfall, allowed in ((D("0"), True), (D("0.000001"), False)):
        fact = observation(openrouter_reported_spend_usd=D(raw),
                           openrouter_available_cash_usd=D(threshold) - shortfall)
        assert affordable(budget_policy(openrouter_anchor_usd=D("0.30")), fact, 2,
                          reported_floors=(D(floor), D("0"))) is allowed


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
        (("0.1", "0.1"), 3, ("0.2", "0.5"), True),
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
