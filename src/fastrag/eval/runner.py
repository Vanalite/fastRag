"""Run the golden dataset against a fresh, in-memory ``RAGService``.

Dataset layout (see ``eval/golden``)::

    corpus.json       {"tenant_id": ..., "documents": [{"path": "x.md", "acl": [...]}]}
    corpus/*.md       the documents
    questions.jsonl   one {"id", "question", "relevant", "principal", "expect"} per line

Each question is asked *as its own principal*, so the dataset exercises access control
as well as ranking: a question about a finance-only document asked by an engineer must
abstain, and any retrieved source the principal could not see counts as an ACL leak.
Leaks are checked against the manifest ACLs, independently of the service's own filter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from fastrag.config import Settings
from fastrag.domain import EVERYONE, DocumentIn, Principal, Role, group_token, user_token
from fastrag.eval.metrics import dedupe, ndcg_at_k, recall_at_k, reciprocal_rank
from fastrag.ratelimit import RateLimiter

Expectation = Literal["answer", "abstain"]
_MEDIA_TYPES = {".md": "text/markdown", ".markdown": "text/markdown", ".html": "text/html"}


class DatasetError(ValueError):
    """The golden dataset is malformed or internally inconsistent."""


@dataclass(frozen=True)
class GoldenDoc:
    source_id: str  # the corpus file name, e.g. "pto-policy.md"
    text: str
    acl: frozenset[str]
    media_type: str = "text/markdown"


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    relevant: tuple[str, ...]
    user: str
    groups: frozenset[str]
    expect: Expectation

    def acl_tokens(self) -> frozenset[str]:
        """The tokens this question's principal holds, derived from the dataset alone."""
        return frozenset({EVERYONE, user_token(self.user)} | {group_token(g) for g in self.groups})

    def can_read(self, doc: GoldenDoc) -> bool:
        return not self.acl_tokens().isdisjoint(doc.acl)


@dataclass(frozen=True)
class Dataset:
    tenant_id: str
    docs: list[GoldenDoc]
    questions: list[GoldenQuestion]


@dataclass(frozen=True)
class Thresholds:
    min_recall: float = 0.0
    min_mrr: float = 0.0
    min_ndcg: float = 0.0
    min_abstain_accuracy: float = 0.0
    max_acl_leaks: int = 0


@dataclass(frozen=True)
class QuestionResult:
    id: str
    question: str
    expect: Expectation
    relevant: list[str]
    retrieved: list[str]  # de-duplicated source ids from search(), in rank order
    confident: bool
    abstained: bool
    cited: list[str]
    recall: float | None  # None for questions that should abstain
    reciprocal_rank: float | None
    ndcg: float | None
    leaked: list[str]  # retrieved / shown / cited sources the principal may not read

    @property
    def abstain_correct(self) -> bool:
        return self.abstained == (self.expect == "abstain")

    @property
    def ok(self) -> bool:
        return self.abstain_correct and not self.leaked and (self.recall in (None, 1.0))


@dataclass
class EvalReport:
    k: int
    embedder: str
    rows: list[QuestionResult] = field(default_factory=list)

    @property
    def answerable(self) -> list[QuestionResult]:
        return [r for r in self.rows if r.expect == "answer"]

    @property
    def recall_at_k(self) -> float:
        return _mean(r.recall for r in self.answerable)

    @property
    def mrr(self) -> float:
        return _mean(r.reciprocal_rank for r in self.answerable)

    @property
    def ndcg_at_k(self) -> float:
        return _mean(r.ndcg for r in self.answerable)

    @property
    def abstain_accuracy(self) -> float:
        return _mean(1.0 if r.abstain_correct else 0.0 for r in self.rows)

    @property
    def false_abstentions(self) -> int:
        return sum(1 for r in self.rows if r.expect == "answer" and r.abstained)

    @property
    def false_answers(self) -> int:
        return sum(1 for r in self.rows if r.expect == "abstain" and not r.abstained)

    @property
    def acl_leaks(self) -> int:
        """Number of questions where any source outside the principal's access was returned."""
        return sum(1 for r in self.rows if r.leaked)

    def metrics(self) -> dict[str, float | int]:
        return {
            "questions": len(self.rows),
            "answerable": len(self.answerable),
            f"recall@{self.k}": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            f"ndcg@{self.k}": round(self.ndcg_at_k, 4),
            "abstain_accuracy": round(self.abstain_accuracy, 4),
            "false_abstentions": self.false_abstentions,
            "false_answers": self.false_answers,
            "acl_leaks": self.acl_leaks,
        }

    def check(self, thresholds: Thresholds) -> list[str]:
        """Return one message per failed threshold; an empty list means the gate passes."""
        failures: list[str] = []
        for name, value, minimum in (
            (f"recall@{self.k}", self.recall_at_k, thresholds.min_recall),
            ("mrr", self.mrr, thresholds.min_mrr),
            (f"ndcg@{self.k}", self.ndcg_at_k, thresholds.min_ndcg),
            ("abstain_accuracy", self.abstain_accuracy, thresholds.min_abstain_accuracy),
        ):
            if value < minimum:
                failures.append(f"{name} {value:.3f} is below the minimum {minimum:.3f}")
        if self.acl_leaks > thresholds.max_acl_leaks:
            leaked = ", ".join(r.id for r in self.rows if r.leaked)
            failures.append(
                f"acl_leaks {self.acl_leaks} exceeds the maximum "
                f"{thresholds.max_acl_leaks} ({leaked})"
            )
        return failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "embedder": self.embedder,
            "metrics": self.metrics(),
            "questions": [
                {**asdict(r), "abstain_correct": r.abstain_correct, "ok": r.ok} for r in self.rows
            ],
        }


def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return fmean(values) if values else 0.0


# --- loading ---------------------------------------------------------------


def load_dataset(dataset_dir: str | Path) -> Dataset:
    """Load and validate ``corpus.json``, ``corpus/`` and ``questions.jsonl``.

    Validation catches dataset bugs that would otherwise show up as confusing metric
    drops: unknown relevant files, answerable questions with nothing relevant, and
    relevant documents the asking principal is not allowed to read.
    """
    root = Path(dataset_dir)
    try:
        manifest = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read {root / 'corpus.json'}: {exc}") from exc

    tenant_id = manifest.get("tenant_id", "northwind")
    docs: list[GoldenDoc] = []
    for entry in manifest.get("documents", []):
        path = root / "corpus" / entry["path"]
        acl = frozenset(entry.get("acl", []))
        if not acl:
            raise DatasetError(f"{entry['path']}: empty ACL (the document would be unreadable)")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetError(f"cannot read corpus file {path}: {exc}") from exc
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "text/plain")
        docs.append(GoldenDoc(entry["path"], text, acl, media_type))
    if not docs:
        raise DatasetError(f"{root / 'corpus.json'} lists no documents")
    by_id = {d.source_id: d for d in docs}

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    questions_path = root / "questions.jsonl"
    try:
        lines = questions_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"cannot read {questions_path}: {exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{questions_path.name}:{lineno}"
        try:
            raw = json.loads(line)
            principal = raw["principal"]
            q = GoldenQuestion(
                id=raw["id"],
                question=raw["question"],
                relevant=tuple(raw.get("relevant", [])),
                user=principal["user"],
                groups=frozenset(principal.get("groups", [])),
                expect=raw["expect"],
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DatasetError(f"{where}: malformed question ({exc})") from exc
        if q.id in seen:
            raise DatasetError(f"{where}: duplicate id {q.id!r}")
        seen.add(q.id)
        if q.expect not in ("answer", "abstain"):
            raise DatasetError(f"{where}: expect must be 'answer' or 'abstain', not {q.expect!r}")
        if q.expect == "answer" and not q.relevant:
            raise DatasetError(f"{where}: an answerable question needs at least one relevant doc")
        for source in q.relevant:
            if source not in by_id:
                raise DatasetError(f"{where}: relevant doc {source!r} is not in the corpus")
            if not q.can_read(by_id[source]):
                raise DatasetError(f"{where}: {q.user!r} cannot read relevant doc {source!r}")
        questions.append(q)
    if not questions:
        raise DatasetError(f"{questions_path} contains no questions")
    return Dataset(tenant_id=tenant_id, docs=docs, questions=questions)


# --- running ---------------------------------------------------------------


def run_eval(dataset_dir: str | Path, settings: Settings | None = None, k: int = 4) -> EvalReport:
    """Ingest the corpus into a fresh in-memory service and score every question.

    Storage, audit and API-key files are always forced off so an evaluation never touches
    a real database, and rate limiting is disabled because the harness is a single caller.
    """
    from fastrag.service import RAGService  # deferred: keeps `fastrag.eval.metrics` light

    dataset = load_dataset(dataset_dir)
    settings = (settings or Settings.for_testing()).model_copy(
        update={"database_path": ":memory:", "audit_log_path": None, "api_keys_file": None}
    )
    service = RAGService.from_settings(
        settings, limiter=RateLimiter(0, 0), user_limiter=RateLimiter(0, 0)
    )

    admin = Principal(dataset.tenant_id, "eval-admin", roles=frozenset({Role.ADMIN}))
    for doc in dataset.docs:
        service.ingest(
            admin, DocumentIn(doc.source_id, doc.text, acl=doc.acl, media_type=doc.media_type)
        )

    by_id = {d.source_id: d for d in dataset.docs}
    report = EvalReport(k=k, embedder=service.embedder.name)
    for q in dataset.questions:
        principal = Principal(dataset.tenant_id, q.user, groups=q.groups)
        search = service.search(principal, q.question, k)
        answer = service.ask(principal, q.question, k)

        retrieved = dedupe(s.chunk.source_id for s in search.chunks)
        shown = dedupe(s.chunk.source_id for s in answer.sources)
        cited = dedupe(c.source_id for c in answer.citations)
        leaked = [
            source
            for source in dedupe([*retrieved, *shown, *cited])
            if source not in by_id or not q.can_read(by_id[source])
        ]
        scored = q.expect == "answer"
        report.rows.append(
            QuestionResult(
                id=q.id,
                question=q.question,
                expect=q.expect,
                relevant=list(q.relevant),
                retrieved=retrieved,
                confident=search.confident,
                abstained=answer.abstained,
                cited=cited,
                recall=recall_at_k(retrieved, q.relevant, k) if scored else None,
                reciprocal_rank=reciprocal_rank(retrieved, q.relevant) if scored else None,
                ndcg=ndcg_at_k(retrieved, q.relevant, k) if scored else None,
                leaked=leaked,
            )
        )
    return report
