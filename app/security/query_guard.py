"""
Lightweight prompt-injection and ambiguity heuristics.
Not a substitute for full safety review; suitable for demo/eval guardrails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QueryRisk(str, Enum):
    NONE = "none"
    AMBIGUOUS = "ambiguous"
    ADVERSARIAL = "adversarial"


@dataclass
class GuardResult:
    risk: QueryRisk
    reason: str
    safe_response: Optional[str] = None


_ADV_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I), "instruction_override"),
    (re.compile(r"\byou\s+are\s+now\s+dan\b", re.I), "jailbreak_persona"),
    (re.compile(r"system\s*:\s*you\s+are", re.I), "fake_system_message"),
    (re.compile(r"\bsay\s+hack(ed)?\b", re.I), "forced_output"),
]

# Very short vague prompts (eval-style)
_AMBIGUOUS_HINTS = [
    re.compile(r"^compare\s+them\.?$", re.I),
    re.compile(r"^tell\s+me\s+about\s+it\.?$", re.I),
    re.compile(r"^how\s+does\s+it\s+work\??$", re.I),
    re.compile(r"^why\s+is\s+that\??$", re.I),
    re.compile(r"^what'?s\s+the\s+difference\??$", re.I),
]


def assess_query(query: str) -> GuardResult:
    text = (query or "").strip()
    if not text:
        return GuardResult(QueryRisk.AMBIGUOUS, "empty_query", "Please provide a specific question.")

    for rx, tag in _ADV_PATTERNS:
        if rx.search(text):
            return GuardResult(
                QueryRisk.ADVERSARIAL,
                tag,
                "I can't comply with requests to ignore system instructions or adopt unrestricted personas. "
                "Ask a factual question and I will answer using the knowledge base.",
            )

    for rx in _AMBIGUOUS_HINTS:
        if rx.match(text):
            return GuardResult(
                QueryRisk.AMBIGUOUS,
                "underspecified_reference",
                "Your question is missing the subject. Specify what “it” or “them” refers to (e.g. Python vs Java).",
            )

    return GuardResult(QueryRisk.NONE, "ok")
