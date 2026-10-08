"""Exact SQLite schemas for durable local Voice Cell authority."""

from __future__ import annotations

SCHEMA_VERSION = 6

V1_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS call_leases (
    call_control_id TEXT PRIMARY KEY NOT NULL,
    call_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'active', 'terminal')),
    token_hash BLOB NOT NULL CHECK (length(token_hash) = 32),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    closed_at TEXT,
    CHECK (
        (state = 'terminal' AND closed_at IS NOT NULL)
        OR (state IN ('pending', 'active') AND closed_at IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS webhook_receipts (
    event_id TEXT PRIMARY KEY NOT NULL,
    event_type TEXT NOT NULL,
    call_control_id TEXT,
    occurred_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    semantic_fingerprint_sha256 BLOB NOT NULL
        CHECK (length(semantic_fingerprint_sha256) = 32),
    provider_enrichment_fingerprint_sha256 BLOB
        CHECK (
            provider_enrichment_fingerprint_sha256 IS NULL
            OR length(provider_enrichment_fingerprint_sha256) = 32
        )
);

CREATE TABLE IF NOT EXISTS outbox (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    op_id TEXT NOT NULL UNIQUE,
    deployment_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('call.upsert', 'turn.upsert', 'recording.upsert')),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    call_id TEXT NOT NULL,
    turn_id TEXT,
    recording_id TEXT,
    crypto_version INTEGER NOT NULL CHECK (crypto_version = 1),
    key_version INTEGER NOT NULL CHECK (key_version > 0),
    nonce BLOB NOT NULL CHECK (length(nonce) = 12),
    ciphertext BLOB NOT NULL CHECK (length(ciphertext) >= 16),
    created_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TEXT NOT NULL,
    last_error_code TEXT
);

CREATE INDEX IF NOT EXISTS outbox_due_fifo_idx
    ON outbox (deployment_id, queue_id, next_attempt_at);

PRAGMA user_version = 1;
"""

QUALIFICATION_RUNS_SQL = """CREATE TABLE qualification_runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    consumed_at TEXT NOT NULL
)"""

V2_SCHEMA_SQL = V1_SCHEMA_SQL.replace(
    "PRAGMA user_version = 1;",
    f"{QUALIFICATION_RUNS_SQL};\n\nPRAGMA user_version = 2;",
)

CALL_LIFECYCLE_MIGRATION_SQL = "ALTER TABLE call_leases ADD COLUMN lifecycle_json TEXT"
V3_SCHEMA_SQL = V2_SCHEMA_SQL.replace(
    "closed_at TEXT,",
    "closed_at TEXT, lifecycle_json TEXT,",
    1,
).replace("PRAGMA user_version = 2;", "PRAGMA user_version = 3;")

SPARRA_CONTENT_SQL = """
CREATE TABLE sparra_turn_decisions (
    call_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    op_id TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    fingerprint BLOB NOT NULL CHECK (length(fingerprint) = 32),
    key_version INTEGER,
    nonce BLOB,
    ciphertext BLOB,
    lost INTEGER NOT NULL CHECK (lost IN (0, 1)),
    CHECK ((key_version IS NULL AND nonce IS NULL AND ciphertext IS NULL)
        OR (key_version > 0 AND length(nonce) = 12 AND length(ciphertext) >= 16)),
    PRIMARY KEY (call_id, turn_id)
);
CREATE TABLE sparra_publications (
    call_id TEXT PRIMARY KEY NOT NULL,
    op_id TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    key_version INTEGER NOT NULL CHECK (key_version > 0),
    nonce BLOB NOT NULL CHECK (length(nonce) = 12),
    ciphertext BLOB NOT NULL CHECK (length(ciphertext) >= 16)
);
CREATE TABLE sparra_content_fences (
    call_id TEXT PRIMARY KEY NOT NULL,
    cleaned_at TEXT NOT NULL,
    lease_token TEXT,
    lease_cleaned_at TEXT,
    lease_acked INTEGER NOT NULL DEFAULT 0 CHECK (lease_acked IN (0,1)),
    lease_settled INTEGER NOT NULL DEFAULT 0 CHECK (lease_settled IN (0,1))
);
"""
V4_SCHEMA_SQL = V3_SCHEMA_SQL.replace(
    "PRAGMA user_version = 3;", SPARRA_CONTENT_SQL + "\nPRAGMA user_version = 4;"
)

RECORDING_ARCHIVE_SQL = """
CREATE TABLE recording_archives (
    recording_id TEXT PRIMARY KEY NOT NULL,
    call_id TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    fingerprint BLOB NOT NULL CHECK (length(fingerprint) = 32),
    state TEXT NOT NULL CHECK (state IN
        ('pending','archived','acknowledged','unavailable','expired','erased')),
    observed_at TEXT NOT NULL,
    retention_until TEXT NOT NULL,
    context_key_version INTEGER NOT NULL CHECK (context_key_version > 0),
    context_nonce BLOB NOT NULL CHECK (length(context_nonce) = 12),
    context_ciphertext BLOB NOT NULL CHECK (length(context_ciphertext) >= 16),
    receipt_json TEXT,
    audio_nonce BLOB CHECK (audio_nonce IS NULL OR length(audio_nonce) = 12),
    receipt_op_id TEXT,
    CHECK ((receipt_json IS NULL AND audio_nonce IS NULL AND receipt_op_id IS NULL)
        OR (receipt_json IS NOT NULL AND audio_nonce IS NOT NULL AND receipt_op_id IS NOT NULL))
);
"""
V5_SCHEMA_SQL = V4_SCHEMA_SQL.replace(
    "PRAGMA user_version = 4;", RECORDING_ARCHIVE_SQL + "\nPRAGMA user_version = 5;"
)

QUALIFICATION_ALLOWANCE_MIGRATION_SQL = """
ALTER TABLE qualification_runs ADD COLUMN profile_sha256 BLOB
    CHECK (profile_sha256 IS NULL OR length(profile_sha256) = 32);
ALTER TABLE qualification_runs ADD COLUMN total_calls INTEGER NOT NULL DEFAULT 1
    CHECK (typeof(total_calls) = 'integer' AND total_calls BETWEEN 1 AND 10);
ALTER TABLE qualification_runs ADD COLUMN used_calls INTEGER NOT NULL DEFAULT 1
    CHECK (typeof(used_calls) = 'integer' AND used_calls BETWEEN 1 AND total_calls
        AND (profile_sha256 IS NOT NULL OR (total_calls = 1 AND used_calls = 1)));
"""
QUALIFICATION_ALLOWANCE_SQL = """CREATE TABLE qualification_runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    consumed_at TEXT NOT NULL,
    profile_sha256 BLOB
        CHECK (profile_sha256 IS NULL OR length(profile_sha256) = 32),
    total_calls INTEGER NOT NULL DEFAULT 1
        CHECK (typeof(total_calls) = 'integer' AND total_calls BETWEEN 1 AND 10),
    used_calls INTEGER NOT NULL DEFAULT 1
        CHECK (typeof(used_calls) = 'integer' AND used_calls BETWEEN 1 AND total_calls
            AND (profile_sha256 IS NOT NULL OR (total_calls = 1 AND used_calls = 1)))
)"""
SCHEMA_SQL = V5_SCHEMA_SQL.replace(QUALIFICATION_RUNS_SQL, QUALIFICATION_ALLOWANCE_SQL).replace(
    "PRAGMA user_version = 5;", "PRAGMA user_version = 6;"
)

LOCAL_AUDIO_SCHEMA_VERSION = 7
LOCAL_AUDIO_PIN_SQL = """
CREATE TABLE local_audio_contract (
    singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
    contract_version INTEGER NOT NULL CHECK (contract_version = 2)
);
CREATE TABLE local_audio_pin (
    call_id TEXT PRIMARY KEY NOT NULL,
    generation TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    recording_id TEXT,
    configuration_revision INTEGER NOT NULL CHECK (configuration_revision > 0),
    recording_policy TEXT NOT NULL CHECK (recording_policy IN ('off','local_30d')),
    audio_available INTEGER NOT NULL CHECK (audio_available IN (0,1)),
    admitted_at TEXT NOT NULL,
    retention_until TEXT NOT NULL,
    denied_at TEXT,
    CHECK ((audio_available = 1 AND recording_policy = 'local_30d' AND recording_id IS NOT NULL)
        OR (audio_available = 0 AND recording_id IS NULL))
);
INSERT INTO local_audio_contract(singleton,contract_version) VALUES(1,2);
"""
LOCAL_AUDIO_SCHEMA_SQL = SCHEMA_SQL.replace(
    "kind TEXT NOT NULL CHECK (kind IN ('call.upsert', 'turn.upsert', 'recording.upsert'))",
    "kind TEXT NOT NULL CHECK (kind IN ('call.upsert', 'turn.upsert', 'recording.upsert', "
    "'audio.chunk', 'audio.finish', 'audio.revoke'))",
).replace(
    "schema_version INTEGER NOT NULL CHECK (schema_version = 1)",
    "schema_version INTEGER NOT NULL CHECK (schema_version IN (1,2))",
).replace(
    "last_error_code TEXT\n",
    "last_error_code TEXT,\n    CHECK ((schema_version = 1 AND kind IN "
    "('call.upsert','turn.upsert','recording.upsert')) OR schema_version = 2)\n",
).replace(
    "PRAGMA user_version = 6;", LOCAL_AUDIO_PIN_SQL + "\nPRAGMA user_version = 7;"
)

# V7 remains the exact historical provenance for this single forward migration.
LOCAL_AUDIO_CHOICE_SCHEMA_VERSION = 8
_LOCAL_AUDIO_PIN_V7_TABLE = LOCAL_AUDIO_PIN_SQL[
    LOCAL_AUDIO_PIN_SQL.index("CREATE TABLE local_audio_pin"):
    LOCAL_AUDIO_PIN_SQL.index("INSERT INTO local_audio_contract")
]
_LOCAL_AUDIO_PIN_V8_TABLE = _LOCAL_AUDIO_PIN_V7_TABLE.replace(
    "    denied_at TEXT,\n",
    "    denied_at TEXT,\n"
    "    choice_state TEXT NOT NULL DEFAULT 'undecided' "
    "CHECK (choice_state IN ('undecided','accepted','off')),\n"
    "    choice_occurred_at TEXT,\n"
    "    CHECK ((choice_state='undecided' AND choice_occurred_at IS NULL) "
    "OR (choice_state='accepted' AND choice_occurred_at IS NOT NULL) OR choice_state='off'),\n",
)
LOCAL_AUDIO_CHOICE_SCHEMA_SQL = LOCAL_AUDIO_SCHEMA_SQL.replace(
    _LOCAL_AUDIO_PIN_V7_TABLE, _LOCAL_AUDIO_PIN_V8_TABLE
).replace("PRAGMA user_version = 7;", "PRAGMA user_version = 8;")
LOCAL_AUDIO_CHOICE_MIGRATION_SQL = (
    "ALTER TABLE local_audio_pin RENAME TO local_audio_pin_v7;\n"
    + _LOCAL_AUDIO_PIN_V8_TABLE
    + "INSERT INTO local_audio_pin(call_id,generation,workspace_id,deployment_id,recording_id,"
    "configuration_revision,recording_policy,audio_available,admitted_at,retention_until,"
    "denied_at) "
    "SELECT call_id,generation,workspace_id,deployment_id,recording_id,configuration_revision,"
    "recording_policy,audio_available,admitted_at,retention_until,denied_at "
    "FROM local_audio_pin_v7;\n"
    "DROP TABLE local_audio_pin_v7;\nPRAGMA user_version = 8;"
)

LOCAL_AUDIO_TERMINAL_SCHEMA_VERSION = 9
_LOCAL_AUDIO_PIN_V9_TABLE = _LOCAL_AUDIO_PIN_V8_TABLE.replace(
    "    choice_occurred_at TEXT,\n",
    "    choice_occurred_at TEXT,\n"
    "    committed_last_sequence INTEGER CHECK (committed_last_sequence BETWEEN 0 AND 599),\n"
    "    committed_total_samples INTEGER DEFAULT 0 "
    "CHECK (committed_total_samples BETWEEN 0 AND 4800000),\n"
    "    CHECK ((committed_total_samples IS NULL AND committed_last_sequence IS NULL) "
    "OR (committed_total_samples IS NOT NULL AND committed_total_samples=0 "
    "AND committed_last_sequence IS NULL) "
    "OR (committed_total_samples IS NOT NULL AND committed_total_samples>0 "
    "AND committed_last_sequence IS NOT NULL AND committed_total_samples "
    "BETWEEN committed_last_sequence+1 AND (committed_last_sequence+1)*8000)),\n",
)
LOCAL_AUDIO_TERMINAL_SLOTS_SQL = """
CREATE TABLE local_audio_terminal (
    call_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('audio.finish','audio.revoke')),
    op_id TEXT UNIQUE NOT NULL,
    deployment_id TEXT NOT NULL,
    fingerprint BLOB NOT NULL CHECK (length(fingerprint)=32),
    key_version INTEGER NOT NULL CHECK (key_version>0),
    nonce BLOB NOT NULL CHECK (length(nonce)=12),
    ciphertext BLOB NOT NULL CHECK (length(ciphertext)>=16),
    acked INTEGER NOT NULL DEFAULT 0 CHECK (acked IN (0,1)),
    PRIMARY KEY(call_id,kind)
);
"""
LOCAL_AUDIO_TERMINAL_SCHEMA_SQL = LOCAL_AUDIO_CHOICE_SCHEMA_SQL.replace(
    _LOCAL_AUDIO_PIN_V8_TABLE, _LOCAL_AUDIO_PIN_V9_TABLE
).replace("PRAGMA user_version = 8;", LOCAL_AUDIO_TERMINAL_SLOTS_SQL + "\nPRAGMA user_version = 9;")
LOCAL_AUDIO_TERMINAL_MIGRATION_SQL = (
    "ALTER TABLE local_audio_pin RENAME TO local_audio_pin_v8;\n"
    + _LOCAL_AUDIO_PIN_V9_TABLE
    + "INSERT INTO local_audio_pin(call_id,generation,workspace_id,deployment_id,recording_id,"
    "configuration_revision,recording_policy,audio_available,admitted_at,retention_until,denied_at,"
    "choice_state,choice_occurred_at,committed_last_sequence,committed_total_samples) "
    "SELECT call_id,generation,workspace_id,deployment_id,recording_id,"
    "configuration_revision,recording_policy,audio_available,admitted_at,retention_until,denied_at,"
    "choice_state,choice_occurred_at,NULL,NULL FROM local_audio_pin_v8;\n"
    "DROP TABLE local_audio_pin_v8;\n"
    + LOCAL_AUDIO_TERMINAL_SLOTS_SQL + "\nPRAGMA user_version = 9;"
)
