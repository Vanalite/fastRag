# fastRag documentation

**Last updated:** 2026-11-03

fastRag is a reference implementation of an enterprise RAG system: multi-tenant, access-controlled, audited, and evaluated. These docs explain the design decisions and show where each one lives in the code. Read them with the source open.

## Contents

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | Component diagram, the request lifecycle for `ask`, the ingest and delete lifecycle, the data model, and where each enterprise concern lives. |
| [security.md](security.md) | Threat model with mitigations and residual risks, and a production hardening checklist. |
| [operations.md](operations.md) | Configuration reference, endpoints, health and metrics, audit verification, key management, and scaling. |
| [adr/0001-single-source-of-truth.md](adr/0001-single-source-of-truth.md) | Why the document store is the system of record and indexes are rebuilt from it. |
| [adr/0002-access-control.md](adr/0002-access-control.md) | Principals, ACL tokens, deny-by-default, mandatory access filters, pre- versus post-filtering, and pgvector options. |
| [adr/0003-hybrid-retrieval.md](adr/0003-hybrid-retrieval.md) | Dense plus keyword retrieval, reciprocal rank fusion, and why the confidence gate uses raw signals. |
| [adr/0004-prompt-injection-defense.md](adr/0004-prompt-injection-defense.md) | The layered injection defense and what it cannot stop. |
| [adr/0005-audit-log.md](adr/0005-audit-log.md) | Audit entries with ids and hashes, not content, plus the hash chain and its limits. |
| [adr/0006-offline-first-defaults.md](adr/0006-offline-first-defaults.md) | Why the defaults run offline, where adapters plug in, and what the evaluation gate does and does not prove. |

The repository root `README.md` covers installation and quick start.

## Suggested reading order

Read the concepts first, then the design, then the code.

1. **The problem.** Read the top of [architecture.md](architecture.md). It names the one entry point (`RAGService`) and the sequence every call follows. The question is: what does a caller learn about documents it may not read?
2. **The moving parts.** Read the component diagram and the `ask` sequence in [architecture.md](architecture.md). Trace one question from the API to the audit log.
3. **Data and truth.** Read the data model in [architecture.md](architecture.md), then [ADR 0001](adr/0001-single-source-of-truth.md). Ask what happens to the indexes after a delete.
4. **Access control.** Read [ADR 0002](adr/0002-access-control.md), then the threat table in [security.md](security.md) for the isolation rows. Work through `src/fastrag/security/acl.py` and the `access` argument in `src/fastrag/storage/base.py`.
5. **Retrieval.** Read [ADR 0003](adr/0003-hybrid-retrieval.md). Run `fastrag search` on the sample corpus and look at the `dense=` and `bm25=` values for each result.
6. **Generation and its risks.** Read [ADR 0004](adr/0004-prompt-injection-defense.md), then the generation rows of the threat table. Read `src/fastrag/generation/prompt.py`, and try a document that contains an injection phrase.
7. **Accountability.** Read [ADR 0005](adr/0005-audit-log.md), then the audit section of [operations.md](operations.md). Run `fastrag audit verify`, then edit one line of the audit file and run it again.
8. **Running it.** Read [operations.md](operations.md): configuration, endpoints, keys, and scaling. Then run `fastrag eval`, which is the regression gate.
9. **Going to production.** Read the hardening checklist in [security.md](security.md) and the scaling notes in [operations.md](operations.md). Read [ADR 0006](adr/0006-offline-first-defaults.md) last. It explains why the offline numbers are not production numbers.

## Conventions

- Code references use repository paths, such as `src/fastrag/service.py`. Line numbers may drift, so search for the function name if a line number does not match.
- Where a capability is an extension point and not implemented, the docs say so explicitly. OIDC and JWKS, pgvector, a reranker, an async ingestion queue, streaming, Redis, and OpenTelemetry are all in that category.
- ADRs follow Status, Context, Decision, Consequences and Alternatives considered. A new decision gets the next number, and a superseded decision is marked as superseded rather than deleted.

## Keeping these docs accurate

- When a module changes behavior, update the document that describes it, and the ADR if the decision changed.
- Re-check the configuration table in [operations.md](operations.md) against `src/fastrag/config.py` whenever a setting is added or its default changes.
- Re-run `fastrag eval` after any retrieval change, and update the thresholds only if the change is an improvement you have explained.
