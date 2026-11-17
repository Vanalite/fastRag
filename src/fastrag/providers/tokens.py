"""Token counting for chunk budgets. Vietnamese uses more tokens per
character than English (D5), so budgets are in tokens, not characters."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

_WORDISH = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _heuristic(text: str) -> int:
    # Calibrated against XLM-R/BGE-M3: about 1.3 tokens per Vietnamese
    # syllable, 1 per punctuation mark, more for long Latin words.
    n = 0.0
    for tok in _WORDISH.findall(text):
        n += 1.0 if not tok[0].isalnum() else 1.3 + max(0, len(tok) - 7) / 4
    return int(n) + 1


@lru_cache(maxsize=2)
def get_counter(spec: str = "heuristic") -> Callable[[str], int]:
    if spec == "heuristic":
        return _heuristic
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(spec) if spec.endswith(".json") else Tokenizer.from_pretrained(spec)

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    return count
