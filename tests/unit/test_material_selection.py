"""Synthetic offline Compose selection; no physical or authority qualification."""

import copy
import hashlib
import importlib.util
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

OLD = "sha256:4a0704afbfbee02aaaefb4fcaffa8201b9ead3e696ca83cc8a851cf5cd428c26"
NEW = "sha256:a444705d6ba207f542d4cb8691ac7303c605de3ec5bcb22e7cdc879ffcd238f4"
REPO = "ghcr.io/louisvannobel/projetv0-voice@"
EXPIRY = datetime(2030, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)
BUILD = "/opt/studio/sparra-dokploy/material-handover-1465/build"
OPERATOR = "/opt/studio/sparra-dokploy/material-handover-1465-source/material_handover_1465.py"
CONTROL = "/opt/studio/sparra-dokploy/material-handover-1465/control"
# Synthetic UUID shapes, never an actual qualification identity.
OLD_RUN = "11111111-1111-1111-1111-111111111111"
NEW_RUN = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def operator():
    path = Path(__file__).resolve().parents[2] / "scripts/material_handover_1465.py"
    spec = importlib.util.spec_from_file_location("synthetic_material_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pure_selection_callable_exists(operator):
    assert callable(getattr(operator, "derive_held_compose", None)), (
        "pure selection callable absent"
    )


@pytest.fixture
def before():
    return {
        "x-synthetic": "Labelled test source only: café",
        "volumes": {"synthetic-spool": {"external": True}},
        "networks": {"sparra-pilot": {"external": True}},
        "services": {"voice": {
            "image": REPO + OLD, "restart": "no", "user": "10001:10001",
            "read_only": True,
            "environment": {
                "VOICE_IMAGE_DIGEST": REPO + OLD,
                "VOICE_QUALIFICATION_RUN_ID": OLD_RUN,
                "VOICE_AGENT_BUNDLE_SHA256": "b" * 64,
                "VOICE_INFERENCE_PROFILE_SHA256": "f" * 64,
                "VOICE_QUALIFICATION_CANDIDATE_PATH": "/run/sparra-v2/candidate.json",
                "SYNTHETIC_PRIVATE_SETTING": "synthetic-unchanged",
            },
            "volumes": [
                {"type": "bind", "source": "/synthetic/original", "target": "/run/sparra-v2",
                 "read_only": True, "bind": {"create_host_path": False}},
                {"type": "volume", "source": "synthetic-spool", "target": "/srv/data"},
                {"type": "bind", "source": "/synthetic/private-keys", "target": "/run/keys",
                 "read_only": True},
            ],
            "healthcheck": {"test": ["CMD", "python", "-m", "synthetic_probe"],
                            "interval": "30s", "retries": 3},
            "networks": ["sparra-pilot"], "ports": ["127.0.0.1:8080:8080"],
            "mem_limit": "1g", "cpus": 1, "pids_limit": 256,
            "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"],
        }},
    }


def invoke(operator, before, **changes):
    import yaml

    raw = changes.pop("before_spec", yaml.safe_dump(before, sort_keys=False))
    env = copy.deepcopy(before["services"]["voice"]["environment"])
    env.update(VOICE_IMAGE_DIGEST=REPO + NEW, VOICE_QUALIFICATION_RUN_ID=NEW_RUN,
               VOICE_AGENT_BUNDLE_SHA256="e" * 64, VOICE_INFERENCE_PROFILE_SHA256="d" * 64)
    arguments = dict(before_spec=raw,
                     expected_preimage_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                     original_expiry=EXPIRY, candidate_expiry=EXPIRY, new_environment=env,
                     attempt_sha256="a" * 64, candidate_sha256="c" * 64, nonce_sha256="d" * 64)
    arguments.update(changes)
    return operator.derive_held_compose(**arguments)


def parse_selection(value):
    import yaml

    from projetv0_voice.config import _UniqueKeySafeLoader

    return yaml.load(value, Loader=_UniqueKeySafeLoader)


@pytest.mark.parametrize("expiry", [EXPIRY, EXPIRY.replace(microsecond=0)])
def test_expiry_argument_stays_string_after_yaml12_removes_quotes(operator, before, expiry):
    import yaml

    selected = parse_selection(invoke(operator, before, original_expiry=expiry,
                                      candidate_expiry=expiry))
    argument = selected["services"]["voice"]["command"][-1]
    assert datetime.fromisoformat(argument) == expiry
    # Dokploy's YAML1.2 serializer may emit a timestamp-looking string unquoted.
    # Compose's YAML1.1 reader must still see a command string, not a datetime.
    downstream = yaml.safe_load("value: " + argument)["value"]
    assert type(downstream) is str
    assert downstream == argument


def test_exact_permitted_delta_preserves_spool_keys_health_and_security(operator, before, capsys):
    untouched = copy.deepcopy(before)
    result = invoke(operator, before)
    assert isinstance(result, str)
    actual = parse_selection(result)
    expected = copy.deepcopy(before)
    service = expected["services"]["voice"]
    service["image"] = REPO + NEW
    service["environment"].update(VOICE_IMAGE_DIGEST=REPO + NEW,
                                  VOICE_QUALIFICATION_RUN_ID=NEW_RUN,
                                  VOICE_AGENT_BUNDLE_SHA256="e" * 64,
                                  VOICE_INFERENCE_PROFILE_SHA256="d" * 64)
    service["volumes"][0]["source"] = BUILD
    service["volumes"] += [
        {"type": "bind", "source": OPERATOR,
         "target": "/run/sparra-material/material_handover_1465.py",
         "read_only": True},
        {"type": "bind", "source": CONTROL,
         "target": "/run/sparra-material/control", "read_only": True},
    ]
    service["entrypoint"] = ["python", "/run/sparra-material/material_handover_1465.py"]
    service["command"] = ["native-hold", "--release-directory", "/run/sparra-material/control",
                          "--attempt-sha256", "a" * 64, "--candidate-sha256", "c" * 64,
                          "--nonce-sha256", "d" * 64,
                          "--candidate-expiry", "20300101T120000.123456+0000"]
    assert actual == expected
    assert before == untouched
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("field,value", [
    ("image", REPO + NEW), ("restart", "always"), ("restart", False),
    ("user", "0:0"), ("read_only", False), ("read_only", "true"),
    ("command", []), ("command", "python -m projetv0_voice.server"),
    ("entrypoint", ["sh", "-c"]),
])
def test_unsafe_native_service_refuses(operator, before, field, value):
    before["services"]["voice"][field] = value
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before)


@pytest.mark.parametrize("field", ["user", "command", "entrypoint"])
def test_supported_absent_or_null_native_fields(operator, before, field):
    if field == "user":
        del before["services"]["voice"][field]
    else:
        before["services"]["voice"][field] = None
    assert parse_selection(invoke(operator, before))["services"]["voice"]["restart"] == "no"


def test_exact_native_entrypoint_is_supported(operator, before):
    before["services"]["voice"]["entrypoint"] = ["python", "-m", "projetv0_voice.server"]
    assert isinstance(invoke(operator, before), str)


@pytest.mark.parametrize("override", [
    {"expected_preimage_sha256": "0" * 64}, {"expected_preimage_sha256": "A" * 64},
    {"attempt_sha256": "a" * 63}, {"candidate_sha256": "C" * 64},
    {"nonce_sha256": "raw-synthetic-nonce"}, {"attempt_sha256": 123},
    {"candidate_expiry": EXPIRY + timedelta(microseconds=1)},
    {"original_expiry": EXPIRY.replace(tzinfo=None)},
    {"candidate_expiry": EXPIRY.astimezone(timezone(timedelta(hours=1)))},
])
def test_preimage_hashes_and_original_utc_expiry_are_binding(operator, before, override):
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before, **override)


@pytest.mark.parametrize("kind", ["added", "removed", "private", "type", "image", "candidate-path"])
def test_environment_delta_is_exact(operator, before, kind):
    env = copy.deepcopy(before["services"]["voice"]["environment"])
    env["VOICE_IMAGE_DIGEST"] = REPO + NEW
    if kind == "added":
        env["NEW_SYNTHETIC_SETTING"] = "forbidden"
    elif kind == "removed":
        del env["SYNTHETIC_PRIVATE_SETTING"]
    elif kind == "private":
        env["SYNTHETIC_PRIVATE_SETTING"] = "changed"
    elif kind == "type":
        env["VOICE_QUALIFICATION_RUN_ID"] = 1
    elif kind == "image":
        env["VOICE_IMAGE_DIGEST"] = REPO + OLD
    else:
        env["VOICE_QUALIFICATION_CANDIDATE_PATH"] = "/different/candidate.json"
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before, new_environment=env)


def test_original_image_environment_must_match(operator, before):
    before["services"]["voice"]["environment"]["VOICE_IMAGE_DIGEST"] = REPO + NEW
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before)


def test_original_candidate_path_cannot_be_normalized_to_expected(operator, before):
    before["services"]["voice"]["environment"]["VOICE_QUALIFICATION_CANDIDATE_PATH"] = (
        "/synthetic/wrong.json"
    )
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before)


@pytest.mark.parametrize("key,value", [
    ("VOICE_QUALIFICATION_RUN_ID", "synthetic-not-a-uuid"),
    ("VOICE_QUALIFICATION_RUN_ID", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"),
    ("VOICE_AGENT_BUNDLE_SHA256", "B" * 64),
    ("VOICE_INFERENCE_PROFILE_SHA256", "f" * 63),
])
@pytest.mark.parametrize("side", ["old", "new"])
def test_old_and_new_identity_shapes_refuse(operator, before, key, value, side):
    if side == "old":
        before["services"]["voice"]["environment"][key] = value
        changes = {}
    else:
        env = copy.deepcopy(before["services"]["voice"]["environment"])
        env["VOICE_IMAGE_DIGEST"] = REPO + NEW
        env[key] = value
        changes = {"new_environment": env}
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before, **changes)


def test_same_canonical_run_id_is_allowed_without_claiming_authority(operator, before):
    env = copy.deepcopy(before["services"]["voice"]["environment"])
    env["VOICE_IMAGE_DIGEST"] = REPO + NEW
    selected = parse_selection(invoke(operator, before, new_environment=env))
    assert selected["services"]["voice"]["environment"] == env


@pytest.mark.parametrize("raw", ["[]", "null", "services: []", "services: {}", "{broken",
                                  "services: {}\nservices: {}\n"])
def test_malformed_nonmapping_or_duplicate_yaml_refuses(operator, before, raw):
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before, before_spec=raw)


def test_duplicate_nested_key_uses_real_unique_loader(operator, before):
    import yaml

    raw = yaml.safe_dump(before, sort_keys=False).replace(
        "    restart: 'no'", "    restart: 'no'\n    restart: 'no'"
    )
    assert raw.count("restart:") == 2
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before, before_spec=raw)


def test_competing_service_refuses(operator, before):
    before["services"]["sidecar"] = {"image": "synthetic:only"}
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before)


def test_single_service_with_other_name_refuses(operator, before):
    before["services"]["other"] = before["services"].pop("voice")
    # Avoid invoke's nominal fixture lookup; pass the real full-reference contract directly.
    import yaml

    raw = yaml.safe_dump(before, sort_keys=False)
    env = copy.deepcopy(before["services"]["other"]["environment"])
    env["VOICE_IMAGE_DIGEST"] = REPO + NEW
    with pytest.raises(operator.HoldRefused):
        operator.derive_held_compose(
            before_spec=raw, expected_preimage_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            original_expiry=EXPIRY, candidate_expiry=EXPIRY, new_environment=env,
            attempt_sha256="a" * 64, candidate_sha256="c" * 64, nonce_sha256="d" * 64,
        )


@pytest.mark.parametrize("side", ["old", "new"])
def test_bare_environment_image_digest_refuses(operator, before, side):
    if side == "old":
        before["services"]["voice"]["environment"]["VOICE_IMAGE_DIGEST"] = OLD
        changes = {}
    else:
        env = copy.deepcopy(before["services"]["voice"]["environment"])
        env["VOICE_IMAGE_DIGEST"] = NEW
        changes = {"new_environment": env}
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before, **changes)


@pytest.mark.parametrize("target", ["/run/sparra-material", "/run/sparra-material/control",
    "/run/sparra-material/control/release.json", "/run/sparra-material/material_handover_1465.py",
    "/run/sparra-material/other", "/run", "/run/sparra-v2", "/run/sparra-v2/nested",
    "relative", "/run/../run/data"])
def test_existing_hold_collision_or_ambiguous_target_refuses(operator, before, target):
    before["services"]["voice"]["volumes"].append(
        {"type": "bind", "source": "/synthetic/collision", "target": target, "read_only": True})
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before)


@pytest.mark.parametrize("kind", ["absent", "writable", "short", "relative-source", "named-volume"])
def test_candidate_bind_must_be_one_absolute_readonly_long_bind(operator, before, kind):
    mounts = before["services"]["voice"]["volumes"]
    if kind == "absent":
        del mounts[0]
    elif kind == "writable":
        mounts[0]["read_only"] = False
    elif kind == "short":
        mounts[0] = "/synthetic/original:/run/sparra-v2:ro"
    elif kind == "relative-source":
        mounts[0]["source"] = "relative"
    else:
        mounts[0]["type"] = "volume"
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before)


def test_refusal_is_silent_and_contains_no_input_details(operator, before, capsys):
    before["services"]["voice"]["image"] = "synthetic-private-sentinel"
    with pytest.raises(operator.HoldRefused) as failure:
        invoke(operator, before)
    assert str(failure.value) == ""
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("shared", ["candidate-mount", "voice-service", "mount-list"])
def test_yaml_alias_cannot_mutate_retained_extension(operator, before, shared):
    import yaml

    voice = before["services"]["voice"]
    if shared == "candidate-mount":
        before["x-retained"] = voice["volumes"][0]
    elif shared == "voice-service":
        before["x-retained"] = voice
    else:
        before["x-retained"] = voice["volumes"]
    raw = yaml.safe_dump(before, sort_keys=False)
    assert "&id" in raw and "*id" in raw, "synthetic shared-node YAML was not emitted"
    # Real loader sharing must be rejected, not restored together by reverse-delta validation.
    with pytest.raises(operator.HoldRefused):
        invoke(operator, before, before_spec=raw)
