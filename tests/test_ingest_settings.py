"""Per-collection ingestion settings and the window chunking strategy (D25). No database."""

import dataclasses

import pytest
from typer.testing import CliRunner

from fastrag.cli import app
from fastrag.config import LlmProvider, get_settings
from fastrag.ingest.chunking import chunk_sections
from fastrag.ingest.parsers import Section
from fastrag.ingest.settings import IngestSettingsError, merge_settings, resolve_ingest, validate_collection_settings
from fastrag.providers.llm import ChatLLM
from fastrag.services import build_services


@pytest.fixture
def plain():
    """Services with no LLM configured."""
    svc = build_services(get_settings())
    return dataclasses.replace(svc, llm=None, llms={})


@pytest.fixture
def with_llms(plain):
    llms = {
        "default": ChatLLM("default", LlmProvider(url="http://big", model="big-model")),
        "cheap": ChatLLM("cheap", LlmProvider(url="http://cheap", model="small-model")),
    }
    return dataclasses.replace(plain, llm=llms["default"], llms=llms)


def words(n: int, word: str = "chữ") -> str:
    return " ".join([word] * n)


def count(text: str) -> int:
    return len(text.split())


def test_defaults_come_from_the_deployment(plain):
    s = plain.settings
    cfg = resolve_ingest({}, plain)
    assert (cfg.parser, cfg.strategy, cfg.ocr) == (s.parser, "structure", True)
    assert (cfg.target_tokens, cfg.max_tokens, cfg.overlap_tokens) == (
        s.chunk_target_tokens,
        s.chunk_max_tokens,
        s.chunk_overlap_tokens,
    )
    # No LLM configured: "auto" resolves to the deterministic header.
    assert cfg.context_mode == "header" and cfg.llm_name is None


def test_partial_overrides_and_fingerprint(plain):
    base = resolve_ingest({}, plain)
    small = resolve_ingest({"ingest": {"chunking": {"max_tokens": 200, "target_tokens": 150}}}, plain)
    assert (small.target_tokens, small.max_tokens, small.overlap_tokens) == (150, 200, base.overlap_tokens)
    assert small.fingerprint != base.fingerprint
    # Same effective settings, same fingerprint, however they are written.
    explicit = {"ingest": {"parser": base.parser, "chunking": {"target_tokens": base.target_tokens}}}
    assert resolve_ingest(explicit, plain).fingerprint == base.fingerprint
    # Unrelated settings (retrieval profiles) do not change it.
    assert resolve_ingest({"profiles": {"customer": {"top_k": 3}}}, plain).fingerprint == base.fingerprint


@pytest.mark.parametrize(
    ("ingest", "message"),
    [
        ({"chunking": {"target_tokens": 600, "max_tokens": 300}}, "must not exceed"),
        ({"chunking": {"target_tokens": 100, "overlap_tokens": 100}}, "below target_tokens"),
        ({"chunking": {"strategy": "semantic"}}, "ingest.chunking.strategy"),
        ({"chunking": {"max_tokens": 10}}, "ingest.chunking.max_tokens"),
        ({"chunk": {}}, "ingest.chunk"),
        ({"context": {"mode": "llm"}}, "no LLM is configured"),
        ({"context": {"llm": "cheap"}}, "no LLM provider named 'cheap'"),
    ],
)
def test_invalid_settings_are_rejected_on_save(plain, ingest, message):
    with pytest.raises(IngestSettingsError, match=message):
        validate_collection_settings("documents", {"ingest": ingest}, plain)


def test_listings_collections_take_no_ingest_settings(plain):
    with pytest.raises(IngestSettingsError, match="documents collections only"):
        validate_collection_settings("listings", {"ingest": {}}, plain)
    validate_collection_settings("listings", {"gazetteer": {}}, plain)


def test_missing_provider_falls_back_at_ingest_time(plain):
    # Saved while the provider existed, which was then removed from the deployment.
    cfg = resolve_ingest({"ingest": {"context": {"mode": "llm", "llm": "cheap"}}}, plain)
    assert cfg.context_mode == "header" and cfg.llm_name is None
    assert any("cheap" in w for w in cfg.warnings)


def test_llm_provider_choice(with_llms):
    auto = resolve_ingest({}, with_llms)
    assert (auto.context_mode, auto.llm_name, auto.llm_model) == ("llm", "default", "big-model")
    cheap = resolve_ingest({"ingest": {"context": {"llm": "cheap", "instructions": " Name the project. "}}}, with_llms)
    assert (cheap.llm_name, cheap.instructions) == ("cheap", "Name the project.")
    assert cheap.fingerprint != auto.fingerprint
    header = resolve_ingest({"ingest": {"context": {"mode": "header", "instructions": "x"}}}, with_llms)
    assert header.llm_name is None and header.instructions is None


def test_merge_settings():
    base = {"profiles": {"customer": {"top_k": 3}}, "ingest": {"chunking": {"max_tokens": 300, "target_tokens": 200}}}
    out = merge_settings(base, {"ingest": {"chunking": {"max_tokens": 400, "target_tokens": None}}})
    assert out == {"profiles": {"customer": {"top_k": 3}}, "ingest": {"chunking": {"max_tokens": 400}}}
    assert base["ingest"]["chunking"] == {"max_tokens": 300, "target_tokens": 200}


SECTIONS = [
    Section(words(30, "một"), ["Doc", "A"], 1),
    Section(words(30, "hai"), ["Doc", "B"], 1),
    Section(words(30, "ba"), ["Doc", "C"], 2),
]


def test_structure_strategy_never_crosses_headings():
    chunks = chunk_sections(SECTIONS, count, target=100, maximum=120, overlap=0)
    assert [c.heading for c in chunks] == [["Doc", "A"], ["Doc", "B"], ["Doc", "C"]]


def test_window_strategy_packs_across_headings():
    chunks = chunk_sections(SECTIONS, count, target=100, maximum=120, overlap=0, strategy="window")
    assert len(chunks) == 1
    only = chunks[0]
    assert only.heading == ["Doc", "A"]  # where the chunk starts
    assert "\nB\n" in only.text and "\nC\n" in only.text  # later headings stay as lines
    assert (only.page_start, only.page_end) == (1, 2)

    small = chunk_sections(SECTIONS, count, target=40, maximum=50, overlap=0, strategy="window")
    assert len(small) == 3
    assert all(c.tokens <= 50 for c in small)
    assert small[1].heading == ["Doc", "B"]


def test_unknown_strategy():
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_sections(SECTIONS, count, strategy="semantic")


def test_chunk_preview_command(tmp_path):
    doc = tmp_path / "faq.md"
    doc.write_text(
        "# Hỏi đáp\n\n## Phí quản lý\n" + words(200, "phí") + "\n\n## Gửi xe\n" + words(50, "xe") + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    small_set = '{"chunking": {"target_tokens": 60, "max_tokens": 80}}'
    small = runner.invoke(app, ["chunk-preview", str(doc), "--set", small_set])
    assert small.exit_code == 0, small.output
    assert "strategy=structure target=60 max=80" in small.output
    big = runner.invoke(app, ["chunk-preview", str(doc), "--json"])
    assert big.exit_code == 0, big.output
    assert '"title": "Hỏi đáp"' in big.output

    bad = runner.invoke(app, ["chunk-preview", str(doc), "--set", '{"chunking": {"max_tokens": 1}}'])
    assert bad.exit_code == 2 and "max_tokens" in bad.output
