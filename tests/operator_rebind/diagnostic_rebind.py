"""Local SQLite rebind kernel; deliberately no CLI, path opener or host authority.

The protected caller must exclusively own the already-open connection and establish
owner exclusion, filesystem identity, backup, carrier and qualification evidence.
No status here certifies those external facts or permits automatic recovery/retry.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, replace
from decimal import Decimal

from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPolicy, reported_floors_valid
from projetv0_voice.persistence import schema
from projetv0_voice.persistence.writer import _expected_schema_objects, _normalize_schema_sql


_SCHEMAS = {
    10: (5, schema.OWNED_BUDGET_SCHEMA_SQL),
    11: (5, schema.OWNED_AUDIO_BUDGET_SCHEMA_SQL),
    12: (6, schema.SIXTH_BUDGET_SCHEMA_SQL),
    13: (6, schema.SIXTH_AUDIO_BUDGET_SCHEMA_SQL),
}
_HASH = re.compile(r"[0-9a-f]{64}")
_IMAGE = re.compile(
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]*)?"
    r"@sha256:[0-9a-f]{64}"
)


@dataclass(frozen=True)
class Preimage:
    state_sha256: str
    schema_sha256: str
    schema_version: int


@dataclass(frozen=True)
class Outcome:
    status: str
    code: str
    old_state_sha256: str | None = None
    new_state_sha256: str | None = None


@dataclass(frozen=True, repr=False)
class _State:
    header: tuple
    objects: tuple
    temp_objects: tuple
    tables: dict

    @property
    def sha256(self):
        return _digest((self.header, self.objects, self.temp_objects, self.tables))

    @property
    def schema_sha256(self):
        return _digest((self.header, self.objects, self.temp_objects))


class Refusal(Exception):
    """Constant code only: public capture failures never include SQLite values."""


class _UnexpectedMutation(Exception):
    pass


def _digest(value):
    def typed(item):
        if isinstance(item, dict):
            return ["map", [[key, typed(item[key])] for key in sorted(item)]]
        if isinstance(item, (tuple, list)):
            return ["sequence", [typed(part) for part in item]]
        if item is None:
            return ["null"]
        if type(item) is bytes:
            return ["blob", item.hex()]
        if type(item) is int:
            return ["integer", str(item)]
        if type(item) is float:
            return ["real", item.hex()]
        if type(item) is str:
            return ["text", item]
        raise Refusal("unsupported_sqlite_value")
    raw = json.dumps(typed(value), ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _connection_shape(connection):
    if (not isinstance(connection, sqlite3.Connection)
            or connection.row_factory is not None or connection.text_factory is not str):
        raise Refusal("unsupported_connection")


def _snapshot(connection):
    # Main and temporary schemas are the only permitted database namespaces.
    temp = tuple(connection.execute("SELECT * FROM temp.sqlite_master ORDER BY type,name"))
    if {row[1] for row in connection.execute("PRAGMA database_list")} - {"main", "temp"}:
        raise Refusal("attached_database")
    objects = tuple(connection.execute("SELECT * FROM main.sqlite_master ORDER BY type,name"))
    header = tuple(connection.execute(f"PRAGMA main.{name}").fetchone()[0] for name in
                   ("user_version", "schema_version", "application_id", "encoding"))
    tables = {}
    for kind, name, _, _, _ in objects:
        if kind == "table":
            quoted = '"' + name.replace('"', '""') + '"'
            tables[name] = tuple(connection.execute(f"SELECT rowid,* FROM main.{quoted} ORDER BY rowid"))
    return _State(header, objects, temp, tables)


def _read_state(connection):
    _connection_shape(connection)
    if connection.in_transaction:
        raise Refusal("existing_transaction")
    try:
        connection.execute("BEGIN")
        state = _snapshot(connection)
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise Refusal("snapshot_unavailable") from None
    connection.execute("ROLLBACK")
    return state


def capture_preimage(connection: sqlite3.Connection) -> Preimage:
    """Read a coherent logical preimage; return only hashes and schema metadata."""
    state = _read_state(connection)
    return Preimage(state.sha256, state.schema_sha256, state.header[0])


def _installed_policy(connection, state):
    version = state.header[0]
    if version not in _SCHEMAS or state.temp_objects:
        raise Refusal("schema_mismatch")
    total, native_sql = _SCHEMAS[version]
    actual = {
        (kind, name): _normalize_schema_sql(sql)
        for kind, name, _, _, sql in state.objects if not name.startswith("sqlite_")
    }
    if actual != _expected_schema_objects(native_sql):
        raise Refusal("schema_mismatch")
    if version in (11, 13) and connection.execute(
            "SELECT singleton,contract_version FROM local_audio_contract").fetchall() != [(1, 2)]:
        raise Refusal("contract_marker")
    rows = state.tables["owned_pilot_budget"]
    if len(rows) != 1:
        raise Refusal("budget_row_count")
    row = rows[0]
    _, slot, raw, fingerprint, used, floor_or, floor_tel, suspended, paid = row
    if slot != "sparra-original-pilot" or suspended != 0 or paid is not None:
        raise Refusal("budget_not_quiescent")
    try:
        policy = OwnedPilotBudgetPolicy.model_validate_json(raw)
        floors = (Decimal(floor_or), Decimal(floor_tel))
    except (ValueError, TypeError, ArithmeticError):
        raise Refusal("invalid_stored_policy") from None
    if (policy.model_dump_json() != raw or policy.sha256 != fingerprint
            or policy.total_calls != total or type(used) is not int or not 0 <= used <= total
            or not reported_floors_valid(floors)):
        raise Refusal("invalid_stored_policy")
    busy_queries = (
        "SELECT 1 FROM call_leases WHERE state!='terminal' LIMIT 1",
        "SELECT 1 FROM outbox LIMIT 1",
        "SELECT 1 FROM sparra_content_fences WHERE lease_acked!=1 OR lease_settled!=1 LIMIT 1",
        "SELECT 1 FROM recording_archives WHERE state IN ('pending','archived') LIMIT 1",
    )
    if version in (11, 13):
        busy_queries += ("SELECT 1 FROM local_audio_terminal WHERE acked!=1 LIMIT 1",)
    if any(connection.execute(sql).fetchone() is not None for sql in busy_queries):
        raise Refusal("unsettled_local_state")
    return policy, row


def rebind(connection: sqlite3.Connection, preimage: Preimage, *,
           image_digest: str, qualified_profile_sha256: str) -> Outcome:
    """One attempt only; change exactly the stored policy's image/profile pair.

    A COMMIT exception is always uncertain, even when a rollback appears possible.
    Caller-owned existing transactions are refused without committing/rolling back.
    """
    try:
        _connection_shape(connection)
        if connection.in_transaction:
            return Outcome("refused", "existing_transaction")
        if (not isinstance(preimage, Preimage)
                or type(preimage.schema_version) is not int
                or not isinstance(preimage.state_sha256, str)
                or _HASH.fullmatch(preimage.state_sha256) is None
                or not isinstance(preimage.schema_sha256, str)
                or _HASH.fullmatch(preimage.schema_sha256) is None
                or not isinstance(image_digest, str) or _IMAGE.fullmatch(image_digest) is None
                or not isinstance(qualified_profile_sha256, str)
                or _HASH.fullmatch(qualified_profile_sha256) is None):
            return Outcome("refused", "invalid_request")
    except Exception:
        return Outcome("refused", "unsupported_connection")
    old_digest, new_digest, committing = preimage.state_sha256, None, False
    try:
        connection.execute("BEGIN IMMEDIATE")
        before = _snapshot(connection)
        if (before.sha256 != old_digest or before.schema_sha256 != preimage.schema_sha256
                or before.header[0] != preimage.schema_version):
            raise Refusal("preimage_mismatch")
        old, row = _installed_policy(connection, before)
        target = OwnedPilotBudgetPolicy.model_validate(old.model_dump() | dict(
            image_digest=image_digest, qualified_profile_sha256=qualified_profile_sha256))
        if target.sha256 == old.sha256:
            raise Refusal("unchanged_target")
        new_row = (row[0], row[1], target.model_dump_json(), target.sha256, *row[4:])
        expected = replace(before, tables=before.tables | {"owned_pilot_budget": (new_row,)})
        new_digest = expected.sha256
        changes = connection.total_changes
        cursor = connection.execute(
            "UPDATE owned_pilot_budget SET policy_json=?,policy_sha256=? "
            "WHERE rowid IS ? AND slot IS ? AND policy_json IS ? AND policy_sha256 IS ? "
            "AND used_calls IS ? AND openrouter_floor IS ? AND telnyx_floor IS ? "
            "AND suspended IS ? AND paid_call_control_id IS ?",
            (target.model_dump_json(), target.sha256, *row),
        )
        if cursor.rowcount != 1 or connection.total_changes != changes + 1:
            raise _UnexpectedMutation()
        if not connection.in_transaction or _snapshot(connection).sha256 != new_digest:
            raise _UnexpectedMutation()
        committing = True
        connection.execute("COMMIT")
        return Outcome("committed", "exact_policy_rebind", old_digest, new_digest)
    except BaseException as error:
        try:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
        except BaseException:
            return Outcome("commit_uncertain", "rollback_unconfirmed", old_digest, new_digest)
        if committing:
            return Outcome("commit_uncertain", "commit_unconfirmed", old_digest, new_digest)
        if isinstance(error, Refusal):
            return Outcome("refused", str(error), old_digest, new_digest)
        return Outcome("rolled_back", "transaction_not_applied", old_digest, new_digest)


def classify(connection: sqlite3.Connection, outcome: Outcome) -> str:
    """Separate read-only observation of exact logical old/new/other; never retry."""
    if (not isinstance(outcome, Outcome) or outcome.old_state_sha256 is None
            or outcome.new_state_sha256 is None):
        return "other"
    try:
        observed = _read_state(connection).sha256
    except Exception:
        return "other"
    if observed == outcome.old_state_sha256:
        return "exact_old"
    if observed == outcome.new_state_sha256:
        return "exact_new"
    return "other"
