"""Retrieval-quality evaluation harness: a regression gate for CI.

``fastrag eval`` ingests the golden dataset (``eval/golden``) into a fresh in-memory
service, asks every question as its own principal and scores the results:

* recall@k, MRR and nDCG@k over the answerable questions (document-level, binary);
* abstention accuracy: did the service abstain exactly when it should (out-of-scope
  questions, and questions about documents the asker may not read)?
* ACL leaks: questions where any retrieved, shown or cited source is outside the
  asker's access per the dataset manifest. This must always be zero.

The default thresholds are calibrated a little *below* the scores measured with the
default hashing embedder and default retriever settings. They are a regression gate,
not a quality claim: passing means "no worse than when the thresholds were set", and
the absolute numbers on a ten-document corpus say little about production quality.
Re-measure and raise them when retrieval improves; never lower them to make CI green
without understanding the regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from fastrag.eval.metrics import dedupe, ndcg_at_k, recall_at_k, reciprocal_rank
from fastrag.eval.runner import (
    DatasetError,
    EvalReport,
    QuestionResult,
    Thresholds,
    load_dataset,
    run_eval,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DatasetError",
    "EvalReport",
    "QuestionResult",
    "Thresholds",
    "cli_main",
    "dedupe",
    "load_dataset",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_eval",
]

# Measured with embedder="hashing", k=4 and default retriever settings (2026-11):
# recall@4 1.000, MRR 1.000, nDCG@4 1.000, abstain accuracy 0.969 (31/32), ACL leaks 0.
# The one miss is a false answer: "cash runway" matches common words ("month",
# "company") in an unrelated policy. Raising min_keyword_coverage to 0.4 fixes it but
# makes two answerable questions abstain - the usual precision/recall trade-off.
DEFAULT_THRESHOLDS = Thresholds(
    min_recall=0.95,
    min_mrr=0.95,
    min_ndcg=0.95,
    min_abstain_accuracy=0.90,
    max_acl_leaks=0,
)


def _parser() -> argparse.ArgumentParser:
    d = DEFAULT_THRESHOLDS
    p = argparse.ArgumentParser(
        prog="fastrag eval",
        description="Run the golden retrieval evaluation and fail if quality regresses.",
    )
    p.add_argument(
        "--dataset", default="eval/golden", help="dataset directory (default: %(default)s)"
    )
    p.add_argument("-k", type=int, default=4, help="results per question (default: %(default)s)")
    p.add_argument("--embedder", default="hashing", help="embedder spec (default: %(default)s)")
    p.add_argument("--min-recall", type=float, default=d.min_recall, help="default: %(default)s")
    p.add_argument("--min-mrr", type=float, default=d.min_mrr, help="default: %(default)s")
    p.add_argument("--min-ndcg", type=float, default=d.min_ndcg, help="default: %(default)s")
    p.add_argument(
        "--min-abstain-accuracy",
        type=float,
        default=d.min_abstain_accuracy,
        help="default: %(default)s",
    )
    p.add_argument("--json", action="store_true", help="print a machine-readable JSON report")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="list every question, not just misses"
    )
    return p


def _describe(row: QuestionResult) -> str:
    problems = []
    if row.leaked:
        problems.append(f"ACL LEAK {row.leaked}")
    if not row.abstain_correct:
        problems.append(
            "abstained, expected an answer" if row.abstained else "answered, expected abstain"
        )
    if row.recall is not None and row.recall < 1.0:
        missing = [doc for doc in row.relevant if doc not in row.retrieved]
        problems.append(f"missed {missing}")
    return "; ".join(problems) or "ok"


def _format_table(
    report: EvalReport, thresholds: Thresholds, failures: list[str], verbose: bool
) -> str:
    rows = report.rows if verbose else [r for r in report.rows if not r.ok]
    lines = [
        f"Golden evaluation: {len(report.rows)} questions, k={report.k}, "
        f"embedder={report.embedder}",
        "",
    ]
    if rows:
        width = max(len(r.id) for r in rows)
        lines.append(f"{'question':<{width}}  expect   R@k    RR    result")
        for r in rows:
            recall = "  -  " if r.recall is None else f"{r.recall:.2f} "
            rr = "  -  " if r.reciprocal_rank is None else f"{r.reciprocal_rank:.2f} "
            lines.append(f"{r.id:<{width}}  {r.expect:<7}  {recall} {rr} {_describe(r)}")
    else:
        lines.append("No misses: every question was retrieved, answered or abstained as expected.")
    lines.append("")

    gates = {
        f"recall@{report.k}": (report.recall_at_k, f">= {thresholds.min_recall:.3f}"),
        "mrr": (report.mrr, f">= {thresholds.min_mrr:.3f}"),
        f"ndcg@{report.k}": (report.ndcg_at_k, f">= {thresholds.min_ndcg:.3f}"),
        "abstain_accuracy": (report.abstain_accuracy, f">= {thresholds.min_abstain_accuracy:.3f}"),
        "acl_leaks": (report.acl_leaks, f"<= {thresholds.max_acl_leaks}"),
    }
    lines.append(f"{'metric':<18} {'value':>7}  gate")
    for name, (value, gate) in gates.items():
        shown = f"{value:>7}" if isinstance(value, int) else f"{value:>7.3f}"
        lines.append(f"{name:<18} {shown}  {gate}")
    lines.append(
        f"{'':<18} false abstentions {report.false_abstentions}, "
        f"false answers {report.false_answers}"
    )
    lines.append("")
    if failures:
        lines.append("FAIL")
        lines.extend(f"  - {f}" for f in failures)
    else:
        lines.append("PASS")
    return "\n".join(lines)


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for ``fastrag eval``. Returns 0 if the gate passes, 1 if it fails, 2 on error."""
    from fastrag.config import Settings
    from fastrag.errors import FastRagError

    args = _parser().parse_args(argv)
    thresholds = Thresholds(
        min_recall=args.min_recall,
        min_mrr=args.min_mrr,
        min_ndcg=args.min_ndcg,
        min_abstain_accuracy=args.min_abstain_accuracy,
        max_acl_leaks=0,
    )
    try:
        report = run_eval(args.dataset, Settings.for_testing(embedder=args.embedder), k=args.k)
    except (DatasetError, FastRagError, ValueError, ImportError) as exc:
        # Bad dataset, invalid -k, unknown embedder spec or a missing optional dependency.
        print(f"fastrag eval: {exc}", file=sys.stderr)
        return 2

    failures = report.check(thresholds)
    if args.json:
        payload = {**report.to_dict(), "thresholds": asdict(thresholds), "failures": failures}
        payload["passed"] = not failures
        print(json.dumps(payload, indent=2))
    else:
        print(_format_table(report, thresholds, failures, args.verbose))
    return 1 if failures else 0
