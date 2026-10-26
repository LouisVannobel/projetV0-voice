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
        ("total_calls", 8),
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


def budgeted_policy(**changes):
    return budget_policy(**{
        "total_calls": None, "openrouter_anchor_usd": D("0.30"),
        "telnyx_anchor_usd": D("3.28"), **changes,
    })


def budgeted_observation(**changes):
    return observation(**{
        "openrouter_reported_spend_usd": D("0.33"),
        "telnyx_reported_spend_usd": D("3.30"), **changes,
    })


def test_explicit_null_budget_retains_forecast_after_a_cheap_eighth_call():
    from projetv0_voice.owned_pilot_budget import affordable

    policy = budgeted_policy()
    assert '"total_calls":null' in policy.model_dump_json()
    assert affordable(policy, budgeted_observation(), 7, reported_floors=(D("0.33"), D("3.30")))
    assert not affordable(policy, budgeted_observation(), 8,
                          reported_floors=(D("0.33"), D("3.30")))
    assert budget_policy().total_calls == 5


@pytest.mark.parametrize("provider,guard", [
    ("openrouter", "cash"), ("openrouter", "cap"), ("openrouter", "key"),
    ("telnyx", "cash"), ("telnyx", "cap"),
])
@pytest.mark.parametrize("dominant", ["forecast", "raw", "floor"])
def test_budgeted_reserve_floor_exact_boundaries_use_raw_spend(provider, dominant, guard):
    from projetv0_voice.owned_pilot_budget import affordable

    index = provider == "telnyx"
    reserve = D("0.50" if index else "0.75")
    raw = D("3.30" if index else "0.33")
    effective = D("4.12" if index else "3.835476") if dominant == "forecast" else D("4.20")
    if guard == "cap":
        effective = 5 - reserve
    for shortfall, allowed in ((D("0"), True), (D("0.000001"), False)):
        changes = {}
        floors = [D("0.33"), D("3.30")]
        if dominant == "raw" or guard == "cap" and dominant == "forecast":
            raw = effective + (shortfall if guard == "cap" else 0)
            changes[provider + "_reported_spend_usd"] = raw
        if dominant == "floor":
            floors[index] = effective + (shortfall if guard == "cap" else 0)
        if guard != "cap":
            field = provider + ("_available_cash_usd" if guard == "cash"
                                else "_limit_remaining_usd")
            changes[field] = effective - raw + reserve - shortfall
        assert affordable(budgeted_policy(), budgeted_observation(**changes), 7,
                          reported_floors=tuple(floors)) is allowed


def test_budgeted_openrouter_key_rejects_old_declining_reserve():
    from projetv0_voice.owned_pilot_budget import affordable

    fact = budgeted_observation(openrouter_limit_remaining_usd=D("4.005476"))
    # E - rawS = 3.505476; only .50 remains, below the new .75 minimum.
    assert not affordable(budgeted_policy(), fact, 7,
                          reported_floors=(D("0.33"), D("3.30")))


@pytest.mark.parametrize("count", [True, False, -1, 1.0, "7", None, 2**63 - 1, 2**63])
def test_budgeted_invalid_count_or_maximum_never_admits(count):
    from projetv0_voice.owned_pilot_budget import affordable

    assert not affordable(budgeted_policy(), budgeted_observation(), count,
                          reported_floors=(D("0.33"), D("3.30")))


@pytest.mark.parametrize("ordinal", [1, 7, 8, 2**63 - 1])
def test_budget_permit_strict_sqlite_ordinal_range(ordinal):
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPermit

    permit = OwnedPilotBudgetPermit(policy_sha256="a" * 64, ordinal=ordinal,
        call_id=UUID(int=1), call_control_id="offline", call_leg_id=None,
        call_session_id=None, generation=UUID(int=2), paid_until=budget_policy().paid_until)
    assert permit.ordinal == ordinal


@pytest.mark.parametrize("ordinal", [True, False, 0, -1, 1.0, "8", 2**63])
def test_budget_permit_rejects_invalid_ordinal(ordinal):
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPermit

    with pytest.raises(ValueError):
        OwnedPilotBudgetPermit(policy_sha256="a" * 64, ordinal=ordinal,
            call_id=UUID(int=1), call_control_id="offline", call_leg_id=None,
            call_session_id=None, generation=UUID(int=2), paid_until=budget_policy().paid_until)


@pytest.mark.parametrize("limit,digest", [
    (5, "bdfb9e53f74bcb51e84e8d4341e7234d58b6a4dac9b034cdd30bd9b4b1884673"),
    (6, "04df8ec5e2c0dea5979c8795402912344435bb930cd0213fa71e3fecb38026f4"),
    (7, "02ac7a62e10ee271ba838011b7484226ae448f8314f2158ca63bdd92d5a000ae"),
])
def test_legacy_policy_canonical_json_hashes_and_finite_arithmetic_stay_fixed(limit, digest):
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPolicy, affordable

    policy = budget_policy(total_calls=limit)
    assert policy.sha256 == digest
    assert OwnedPilotBudgetPolicy.model_validate_json(policy.model_dump_json()) == policy
    assert f'"total_calls":{limit}' in policy.model_dump_json()
    for count in range(limit):
        for cash in (D("3.780407"), D("3.780408"), D("5")):
            fact = observation(openrouter_available_cash_usd=cash)
            assert affordable(policy, fact, count, reported_floors=(D("0"), D("0"))) is (
                cash >= D("3.780408"))
            assert affordable(policy, fact, count, reported_floors=(D("0"), D("0"))) == (
                affordable(budget_policy(total_calls=None), fact, count,
                           reported_floors=(D("0"), D("0"))))
    assert not affordable(policy, observation(), limit, reported_floors=(D("0"), D("0")))
    without_limit = policy.model_dump()
    without_limit.pop("total_calls")
    assert OwnedPilotBudgetPolicy.model_validate(without_limit).total_calls == 5
