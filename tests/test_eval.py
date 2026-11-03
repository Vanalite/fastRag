import json
import math
from pathlib import Path

import pytest

from fastrag.eval import (
    DEFAULT_THRESHOLDS,
    DatasetError,
    EvalReport,
    QuestionResult,
    Thresholds,
    cli_main,
    load_dataset,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    run_eval,
)

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden"


# --- metrics -----------------------------------------------------------------


def test_recall_at_k_counts_relevant_in_top_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_recall_dedupes_before_cutting_at_k():
    # Two chunks of "a" must not push "b" out of the top 2.
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, k=2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y", "a"], {"a", "b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "x", "a"], {"a"}) == 0.5  # duplicates don't cost a rank
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_at_k_hand_computed():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)
    # Relevant doc at rank 2 of 2: (1/log2 3) / (1/log2 2).
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))
    # Relevant at ranks 1 and 3, ideal has them at 1 and 2.
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(expected)
    # The ideal is capped at k: 3 relevant, k=1, a hit at rank 1 is perfect.
    assert ndcg_at_k(["a"], {"a", "b", "c"}, k=1) == 1.0
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


@pytest.mark.parametrize("fn", [lambda r, rel: recall_at_k(r, rel, 3), reciprocal_rank])
def test_metrics_reject_empty_relevant(fn):
    with pytest.raises(ValueError):
        fn(["a"], set())


def test_metrics_reject_bad_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=0)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, k=0)


# --- gate --------------------------------------------------------------------


def _row(**overrides) -> QuestionResult:
    base = dict(
        id="q",
        question="?",
        expect="answer",
        relevant=["a.md"],
        retrieved=["a.md"],
        confident=True,
        abstained=False,
        cited=["a.md"],
        recall=1.0,
        reciprocal_rank=1.0,
        ndcg=1.0,
        leaked=[],
    )
    return QuestionResult(**{**base, **overrides})


def test_check_fails_on_any_acl_leak_even_with_perfect_metrics():
    report = EvalReport(k=4, embedder="test", rows=[_row(), _row(id="leaky", leaked=["secret.md"])])
    failures = report.check(Thresholds())
    assert len(failures) == 1 and "acl_leaks" in failures[0] and "leaky" in failures[0]


def test_check_reports_each_metric_below_threshold():
    report = EvalReport(
        k=4, embedder="test", rows=[_row(recall=0.5, reciprocal_rank=0.5, ndcg=0.5)]
    )
    failures = report.check(Thresholds(min_recall=0.9, min_mrr=0.9, min_ndcg=0.9))
    assert len(failures) == 3


def test_dataset_validation_rejects_unreadable_relevant_doc(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "secret.md").write_text("# Secret\n\nRevenue was 3 million.")
    (tmp_path / "corpus.json").write_text(
        json.dumps(
            {"tenant_id": "t", "documents": [{"path": "secret.md", "acl": ["group:finance"]}]}
        )
    )
    question = {
        "id": "q1",
        "question": "What was revenue?",
        "relevant": ["secret.md"],
        "principal": {"user": "eve", "groups": ["engineering"]},
        "expect": "answer",
    }
    (tmp_path / "questions.jsonl").write_text(json.dumps(question) + "\n")
    with pytest.raises(DatasetError, match="cannot read relevant doc"):
        load_dataset(tmp_path)


# --- golden dataset, end to end ------------------------------------------------


def test_golden_dataset_shape():
    dataset = load_dataset(GOLDEN)
    assert len(dataset.docs) >= 10
    assert len(dataset.questions) >= 25
    expects = [q.expect for q in dataset.questions]
    assert expects.count("abstain") >= 4


def test_golden_eval_passes_the_gate_with_no_acl_leaks():
    report = run_eval(GOLDEN)
    assert report.acl_leaks == 0
    assert report.check(DEFAULT_THRESHOLDS) == []
    # The ACL cases must abstain rather than answer from documents the asker can't see.
    by_id = {r.id: r for r in report.rows}
    assert by_id["acl-board-engineer"].abstained
    assert "q3-board-summary.md" not in by_id["acl-board-engineer"].retrieved
    assert by_id["acl-board-finance"].retrieved[0] == "q3-board-summary.md"


def test_cli_json_output_and_exit_codes(capsys):
    assert cli_main(["--dataset", str(GOLDEN), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True and payload["metrics"]["acl_leaks"] == 0

    assert cli_main(["--dataset", str(GOLDEN), "--min-recall", "1.01"]) == 1
    assert "FAIL" in capsys.readouterr().out

    assert cli_main(["--dataset", str(GOLDEN / "missing")]) == 2
