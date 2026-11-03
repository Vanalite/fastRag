import pytest

from fastrag.errors import InvalidInput
from fastrag.ingest import FilesystemConnector, Redactor, parse
from fastrag.ingest.redact import _luhn_valid


def test_parse_markdown_title():
    parsed = parse("Intro\n\n# Travel Policy\n\nBody", "text/markdown")
    assert parsed.title == "Travel Policy"


def test_parse_html_strips_scripts_and_keeps_blocks():
    html = (
        "<html><head><title>Runbook</title><style>p{}</style></head>"
        "<body><h1>Step 1</h1><p>Restart &amp; check.</p><script>alert(1)</script></body></html>"
    )
    parsed = parse(html, "text/html; charset=utf-8")
    assert parsed.title == "Runbook"
    assert "alert" not in parsed.text and "p{}" not in parsed.text
    assert parsed.text == "Step 1\n\nRestart & check."


def test_parse_rejects_unknown_media_type():
    with pytest.raises(InvalidInput):
        parse("%PDF", "application/pdf")


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("mail jane.doe@example.com now", "EMAIL"),
        ("SSN 123-45-6789 on file", "SSN"),
        ("card 4111 1111 1111 1111 exp", "CARD"),
        ("call (555) 123-4567 today", "PHONE"),
    ],
)
def test_redaction(text, label):
    result = Redactor().redact(text)
    assert f"[REDACTED_{label}]" in result.text
    assert result.counts == {label: 1}


def test_redaction_leaves_non_pii_alone():
    text = "Order 1234 5678 9012 3456 shipped; error NW-4012; version 2.3.4; 2026-11-03"
    assert Redactor().redact(text).text == text  # card-like number fails the Luhn check


def test_luhn():
    assert _luhn_valid("4111111111111111")
    assert not _luhn_valid("4111111111111112")


def test_filesystem_connector(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nalpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.html").write_text("<p>beta</p>")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c.md").write_text("hidden")
    docs = list(FilesystemConnector(tmp_path, frozenset({"tenant:*"})).documents())
    assert [(d.source_id, d.media_type) for d in docs] == [
        ("a.md", "text/markdown"),
        ("sub/b.html", "text/html"),
    ]
