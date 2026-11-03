"""A tiny Prometheus-compatible metrics registry (counters and histograms).

Kept dependency-free for the reference; ``prometheus_client`` is a drop-in upgrade.
Labels must stay low-cardinality: tenant ids are fine, user ids and queries are not.
"""

from __future__ import annotations

import bisect
import threading
from collections.abc import Sequence

Labels = tuple[tuple[str, str], ...]


def _labels(labels: dict[str, str] | None) -> Labels:
    return tuple(sorted((labels or {}).items()))


def _fmt(labels: Labels, extra: dict[str, str] | None = None) -> str:
    items = list(labels) + sorted((extra or {}).items())
    if not items:
        return ""
    body = ",".join(
        f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in items
    )
    return "{" + body + "}"


class Counter:
    def __init__(self, name: str, help_text: str):
        self.name, self.help = name, help_text
        self._values: dict[Labels, float] = {}
        self._lock = threading.Lock()

    def inc(self, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = _labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(_labels(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            lines += [f"{self.name}{_fmt(k)} {v}" for k, v in sorted(self._values.items())]
        return lines


class Histogram:
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

    def __init__(self, name: str, help_text: str, buckets: Sequence[float] = DEFAULT_BUCKETS):
        self.name, self.help = name, help_text
        self.buckets = tuple(sorted(buckets))
        self._series: dict[Labels, list[float]] = {}  # bucket counts..., sum, count
        self._lock = threading.Lock()

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _labels(labels)
        with self._lock:
            series = self._series.setdefault(key, [0.0] * (len(self.buckets) + 2))
            for i in range(bisect.bisect_left(self.buckets, value), len(self.buckets)):
                series[i] += 1
            series[-2] += value
            series[-1] += 1

    def count(self, labels: dict[str, str] | None = None) -> float:
        series = self._series.get(_labels(labels))
        return series[-1] if series else 0.0

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key, series in sorted(self._series.items()):
                for bound, n in zip(self.buckets, series, strict=False):
                    lines.append(f"{self.name}_bucket{_fmt(key, {'le': str(bound)})} {n}")
                lines.append(f"{self.name}_bucket{_fmt(key, {'le': '+Inf'})} {series[-1]}")
                lines.append(f"{self.name}_sum{_fmt(key)} {series[-2]}")
                lines.append(f"{self.name}_count{_fmt(key)} {series[-1]}")
        return lines


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Histogram] = {}
        self.requests = self.counter(
            "fastrag_requests_total", "Service calls by action and outcome"
        )
        self.latency = self.histogram(
            "fastrag_stage_seconds", "Latency of pipeline stages (retrieval, generation, ...)"
        )
        self.tokens = self.counter("fastrag_llm_tokens_total", "LLM tokens by tenant and kind")
        self.injection_flags = self.counter(
            "fastrag_injection_flags_total", "Retrieved chunks flagged as possible prompt injection"
        )
        self.redactions = self.counter("fastrag_pii_redactions_total", "PII redactions at ingest")
        self.invalid_citations = self.counter(
            "fastrag_invalid_citations_total", "Citations to sources that were not provided"
        )
        self.ungrounded = self.counter(
            "fastrag_ungrounded_answers_total", "Answers that cite none of their sources"
        )

    def counter(self, name: str, help_text: str) -> Counter:
        metric = Counter(name, help_text)
        self._metrics[name] = metric
        return metric

    def histogram(self, name: str, help_text: str, **kwargs) -> Histogram:
        metric = Histogram(name, help_text, **kwargs)
        self._metrics[name] = metric
        return metric

    def render(self) -> str:
        lines: list[str] = []
        for metric in self._metrics.values():
            lines += metric.render()
        return "\n".join(lines) + "\n"
