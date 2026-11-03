# ADR 0005: An audit log of ids and hashes, hash-chained, with no document content

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

Auditors ask four questions: who asked what, which documents were shown to them, who changed which document, and who was refused. A RAG system also gets a fifth question that most systems do not face: "can we delete this document, as a privacy request requires?" If the audit log copies document text or question text, the deletion has to reach the log too, and the log then stops being an audit trail.

## Decision

**Record ids, hashes and counts. Never document text.** Each entry (`src/fastrag/observability/audit.py`, `AuditLog.record`) has:

- `ts`, `request_id` (from the `X-Request-ID` header or a generated id), `action`, `outcome` (`cancelled` when a streamed answer is abandoned)
- `tenant_id` and `user_id` of the principal (both `null` for unauthenticated attempts)
- `details`: for `ask`, `question_sha256` and `question_chars`, `shown_chunk_ids`, `shown_doc_ids`, `cited_doc_ids`, `abstained`, `usage`, and any `invalid_citations`, `injection_flagged_chunk_ids`, `question_redactions` (counts by kind, when the question sent to the LLM was redacted) or `ungrounded: true`, plus `streamed: true` for streamed calls. For `ingest`, `doc_id`, `version`, `status`, `chunks`, `acl_entries` (a count, not the tokens) and `redactions` by kind.

The question text is stored only when `FASTRAG_AUDIT_STORE_QUERY_TEXT=true`. The default is a SHA-256 fingerprint, which is enough to correlate repeated questions.

**Hash chain.** Each entry stores `prev_hash`, the hash of the entry before it, and its own `hash`, the SHA-256 of the canonical JSON of the rest of the entry. `verify_chain` walks the list from the genesis hash of 64 zeros and checks both links. `fastrag audit verify` runs it over the JSONL file.

**One chain across writers.** `JsonlAuditSink.append` holds an exclusive `flock` on the file while it repairs a torn final line, reads the last entry's hash, and appends. Workers and CLI calls on one host therefore extend one chain instead of forking it. A torn line (no trailing newline) is removed, because the write was never acknowledged. A complete last line that does not parse stops appends with a `ConfigurationError`.

**Durable writes.** Each append is flushed and fsynced before `append` returns.

**Bounded memory sink.** `MemoryAuditSink` (used when no file path is configured) keeps at most 100,000 entries and drops the oldest. Its `anchor()` is the `prev_hash` of the oldest retained entry, so the retained tail still verifies.

**Fail closed.** A failed audit write is logged, and it never replaces the original exception. If the call itself succeeded, the request fails, so a successful answer is never returned without its audit record.

**Recorded in all cases.** `RAGService._operation` writes the entry in its `finally` block, so denials, rate limits and errors are recorded as well as successes. Failed authentications are recorded in the API dependency with `tenant_id` set to null. Attempts past the auth-failure throttle are aggregated: one `throttled` entry per client per minute, carrying the number of attempts, so a credential-guessing flood stays visible without flooding the log.

**Who can read it.** `GET /v1/audit` returns the entries for the caller's tenant, to principals with the `read_audit` permission, which only `admin` has. Reads are themselves recorded as `read_audit`.

## Consequences

Good:

- **Right to be forgotten works without touching the log.** Deleting a document removes its text from the store and indexes. The log still holds the `doc_id`, which is a hash of the source id and the tenant. The log never held the text, so there is nothing to rewrite. `source_id` can still be inferred from `doc_id` if someone already knows the source id.
- **Edits are detectable.** Changing or removing a middle entry breaks the chain at that point.
- The log is small and easy to ship to a SIEM.

Limits (these matter in practice):

- **The hash is unkeyed.** Anyone who can write the file can recompute every hash after an edit. The chain detects accidental and naive tampering only.
- **Tail truncation is undetected.** Removing the last N entries leaves a valid chain. Anchor the head hash somewhere the writer cannot reach, for example in WORM storage, a signed log service or a ticket, and compare it on a schedule.
- **Memory sink drops history.** After 100,000 entries, the oldest are gone. Verification still passes for the retained tail, but the log no longer proves anything about the dropped part.
- **Unauthenticated attempts are not visible to tenants.** They have `tenant_id: null`, so they never appear in a tenant's `GET /v1/audit`. Only the file or the SIEM shows them. Throttled attempts are not written at all, only counted in metrics.
- **Same-host only.** The `flock` serializes writers on one host. It is POSIX-only, and it is not reliable on network filesystems. Give each host its own file. With several replicas on Postgres (ADR 0007), each host still keeps its own chain: the audit log is not stored in the database.
- **Readers do not lock.** A read during an append can see a partial last line. The next read succeeds.
- **Permissions.** The file is created 0600. A file that already exists keeps its mode, so check it at deployment.
- **Reads cost O(file).** `events()` reads the whole file on each call. Fine for a reference, not for a busy tenant.

## Alternatives considered

- **Store the question and answer in the audit entry.** Most useful for investigation, and it makes deletion requests and retention rules much harder. Available as an opt-in flag, off by default.
- **HMAC each entry with a key in a KMS.** Stops the forgery problem noted above, but needs key management, which this reference does not include. The entry shape would not need to change, so this can be added later.
- **Append to an external ledger service (for example, a managed append-only database).** Gives external anchoring. The `AuditSink` protocol (`observability/audit.py`) is the seam for it.
- **Log through the normal application logger only.** Logs are mutable, rotated and often sampled, so they do not meet the audit requirement. Logs are kept for debugging, and the audit log is kept for compliance.
