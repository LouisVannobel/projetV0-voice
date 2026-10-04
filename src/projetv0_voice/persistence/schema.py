"""Exact SQLite schemas for durable local Voice Cell authority."""

from __future__ import annotations

SCHEMA_VERSION = 5

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
SCHEMA_SQL = V4_SCHEMA_SQL.replace(
    "PRAGMA user_version = 4;", RECORDING_ARCHIVE_SQL + "\nPRAGMA user_version = 5;"
)
