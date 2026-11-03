# ADR 0005: An audit log of ids and hashes, hash-chained, with no document content

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

Auditors ask four questions: who asked what, which documents were shown to them, who changed which document, and who was refused. A RAG system also gets a fifth question that most systems do not face: "can we delete this document, as a privacy request requires?" If the audit log copies document text or question text, the deletion has to reach the log too, and the log then stops being an audit trail.

## Decision

**Record ids, hashes and counts. Never document text.** Each entry (`src/fastrag/observability/audit.py:111-122`) has:

- `ts`, `request_id` (from the `X-Request-ID` header or a generated id), `action`, `outcome`
- `tenant_id` and `user_id` of the principal (both `null` for unauthenticated attempts)
- `details`: for `ask`, `question_sha256` and `question_chars`, `shown_chunk_ids`, `shown_doc_ids`, `cited_doc_ids`, `abstained`, `usage`, and any `invalid_citations`. For `ingest`, `doc_id`, `version`, `status`, `chunks`, `acl_entries` (a count, not the tokens) and `redactions` by kind.

The question text is stored only when `FASTRAG_AUDIT_STORE_QUERY_TEXT=true` (`service.py:329-336`). The default is a SHA-256 fingerprint (`audit.py:133-135`), which is enough to correlate repeated questions.

**Hash chain.** Each entry stores `prev_hash`, the hash of the entry before it, and its own `hash`, the SHA-256 of the canonical JSON of the rest of the entry (`audit.py:30-32`). `verify_chain` walks the list from the genesis hash of 64 zeros and checks both links (`audit.py:35-41`). `fastrag audit verify` runs it over the JSONL file (`cli.py:137-146`).

**Durable writes.** `JsonlAuditSink.append` flushes and fsyncs after each entry (`audit.py:80-87`). `AuditLog.record` holds a lock, so the chain is written in order within one process.

**Fail closed.** A failed audit write is logged, and it never replaces the original exception. If the call itself succeeded, the request fails, so a successful answer is never returned without its audit record.

**Recorded in all cases.** `_operation` writes the entry in its `finally` block, so denials, rate limits and errors are recorded as well as successes (`service.py:309-310`). Unauthenticated attempts are recorded in the API dependency (`api/app.py:99-106`), with `tenant_id` set to null.

**Who can read it.** `GET /v1/audit` returns the entries for the caller's tenant, to principals with the `read_audit` permission, which only `admin` has (`security/rbac.py:18-22`). Reading the audit log is itself not recorded: `audit_events` passes `audited=False` (`service.py:249-251`).

## Consequences

Good:

- **Right to be forgotten works without touching the log.** Deleting a document removes its text from the store and indexes. The log still holds the `doc_id`, which is a hash of the source id and the tenant. The log never held the text, so there is nothing to rewrite. Note that `source_id` can still be inferred from `doc_id` if someone already knows the source id.
- **Edits are detectable.** Changing or removing a middle entry breaks the chain at that point.
- The log is small and easy to ship to a SIEM.

Limits (these matter in practice):

- **The hash is unkeyed.** Anyone who can write the file can recompute every hash after an edit. The chain detects accidental and naive tampering only.
- **Tail truncation is undetected.** Removing the last N entries leaves a valid chain. Anchor the head hash somewhere the writer cannot reach, for example in WORM storage, a signed log service or a ticket, and compare it on a schedule.
- **Memory sink drops history.** `MemoryAuditSink` keeps at most 100,000 entries (`audit.py:53-54`). After that, `verify_chain` on its contents fails, because the first remaining entry no longer links to the genesis hash.
- **Unauthenticated attempts are not visible to tenants.** They have `tenant_id: null`, so they never appear in a tenant's `GET /v1/audit`. Only the file or the SIEM shows them.
- **Single writer.** The JSONL sink supports one writer. The chain is kept per process. Two processes appending to one file each start from the last hash they read at startup, so their entries fork the chain. Run one process per log.
- **Permissions.** The file is created with the default umask (`audit.py:83`). Set the directory and file modes explicitly.
- **Reads cost O(file).** `events()` reads the whole file on each call (`audit.py:125-127`). Fine for a reference, not for a busy tenant.

## Alternatives considered

- **Store the question and answer in the audit entry.** Most useful for investigation, and it makes deletion requests and retention rules much harder. Available as an opt-in flag, off by default.
- **HMAC each entry with a key in a KMS.** Stops the forgery problem noted above, but needs key management, which this reference does not include. The entry shape would not need to change, so this can be added later.
- **Append to an external ledger service (for example, a managed append-only database).** Gives external anchoring. The `AuditSink` protocol (`audit.py:44-49`) is the seam for it.
- **Log through the normal application logger only.** Logs are mutable, rotated and often sampled, so they do not meet the audit requirement. Logs are kept for debugging, and the audit log is kept for compliance.
