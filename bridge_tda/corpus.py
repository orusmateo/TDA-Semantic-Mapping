"""Corpus construction and relevance auditing.

The v2 notebooks labelled a document "Indigenous" or "AI safety" by which
arXiv query returned it. The queries were unquoted, so single words such as
"knowledge" were enough to pull a paper into Field A. Here a field label still
comes from the query, but every document is also checked against a lexicon
for its field, and the analysis can be restricted to documents that pass.

The lexicons are a priori, hand-written lists. They are not a classifier and
will miss documents that discuss a field without using these words; treat the
relevance flag as an audit, not ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

FIELD_A = "Field_A"  # Indigenous knowledge systems / TEK
FIELD_B = "Field_B"  # AI safety / algorithmic ethics

# Quoted phrases with explicit field prefixes. In the arXiv API, unquoted
# multi-word queries do not behave as phrase searches.
ARXIV_QUERIES = {
    FIELD_A: " OR ".join([
        'abs:"indigenous knowledge"',
        'abs:"traditional ecological knowledge"',
        'abs:"traditional knowledge"',
        'abs:"indigenous methodology"',
        'abs:"indigenous methodologies"',
        'abs:"decolonizing research"',
        'abs:"two-eyed seeing"',
        'abs:"indigenous data sovereignty"',
        'ti:indigenous',
    ]),
    FIELD_B: " OR ".join([
        'abs:"AI alignment"',
        'abs:"AI safety"',
        'abs:"AI ethics"',
        'abs:"artificial intelligence ethics"',
        'abs:"algorithmic bias"',
        'abs:"algorithmic fairness"',
        'abs:"AI fairness"',
        'abs:"machine learning safety"',
        'abs:"explainable AI"',
    ]),
}

# Non-ethnic senses of "indigenous" and "traditional knowledge" found by
# reading every Field A match in the 998-abstract corpus: algebraic geometry
# ("indigenous bundles"), botany ("indigenous crops", "indigenous and
# invasive grass"), domestic manufacture ("indigenously developed"),
# astrobiology ("indigenous technological species") and ML/management jargon
# ("traditional knowledge tracing"). The adverb "indigenously" is not matched.
_INDIGENOUS_NOT_BEFORE = (
    r"bundles?|semirings?|technological|technosignatures?|crops?|species|plants?|grass(?:es)?|"
    r"cactus|flora|fauna|morphology|unbalanced|web|(?:and|or) (?:invasive|introduced|exotic|non-native)"
)
_TK_NOT_BEFORE = r"tracing|management|graphs?|bases?|distillation"

_A_PATTERNS = [
    rf"\bindigen(?:ous|eity|i[sz]e\w*|i[sz]ation\w*|i[sz]ing)\b(?!\s+(?:{_INDIGENOUS_NOT_BEFORE})\b)",
    r"\baborigin\w*",
    r"\bfirst nations?\b",
    rf"\btraditional (?:ecological )?knowledge\b(?!\s+(?:{_TK_NOT_BEFORE})\b)",
    r"\btwo-eyed seeing\b",
    r"\bdecoloni[sz]\w*",
    r"\binuit\b",
    r"\bm[eé]tis\b",
    r"\bm[aā]ori\b",
    r"\bs[aá]mi\b",
    r"\bnative (?:american|hawaiian|alaskan|peoples?|nations?|communit(?:y|ies)|title|lands?)\b",
    r"\btribal (?:nations?|communit(?:y|ies)|lands?|sovereignty|governments?|peoples?)\b",
    r"\bethno(?:botan|ecolog)\w*",
    r"\banishinaab\w*",
    r"\bhaudenosaunee\b",
]
_A_CASE_SENSITIVE = [r"\bTEK\b"]

_B_PATTERNS = [
    r"\b(?:ai|artificial intelligence|machine learning|ml|algorithmic) (?:alignment|safety|ethics|fairness|bias|accountability|governance|transparency|risks?|harms?)\b",
    r"\b(?:value|model|llm|human) alignment\b",
    r"\baligned with human\b",
    r"\bfairness\b",
    r"\b(?:algorithmic|social|gender|racial|demographic) bias(?:es)?\b",
    r"\bdebias\w*",
    r"\bexplainab\w*",
    r"\binterpretab\w*",
    r"\bxai\b",
    r"\bresponsible (?:ai|artificial intelligence)\b",
    r"\btrustworth\w* (?:ai|artificial intelligence|machine learning)\b",
    r"\bethics of (?:ai|artificial intelligence|algorithms)\b",
    r"\bethical (?:ai|artificial intelligence|machine learning)\b",
    r"\bred[- ]teaming\b",
    r"\bjailbreak\w*",
]

# Broader Field B rule: an AI term and a safety/ethics/governance term in the
# same text. Added after reading 30 rejected Field B abstracts, about a
# quarter of which were plainly about AI risk, regulation or ethics.
_AI_TERM = r"\b(?:ai|artificial intelligence|machine learning|llms?|(?:large )?language models?|neural networks?|algorithm\w*)\b"
_B_CONTEXT = (r"\b(?:ethic\w*|safety|harm\w*|risks?|adversarial|robustness|catastroph\w*|regulat\w*|liabilit\w*|"
              r"governance|accountab\w*|pluralis\w*|moral\w*|vulnerab\w*|misuse|oversight)\b")

_A_RE = re.compile("|".join(_A_PATTERNS), re.IGNORECASE)
_A_CS_RE = re.compile("|".join(_A_CASE_SENSITIVE))
_B_RE = re.compile("|".join(_B_PATTERNS), re.IGNORECASE)
_AI_RE = re.compile(_AI_TERM, re.IGNORECASE)
_B_CONTEXT_RE = re.compile(_B_CONTEXT, re.IGNORECASE)


def mentions_field_a(text: str) -> bool:
    return bool(_A_RE.search(text) or _A_CS_RE.search(text))


def mentions_field_b(text: str) -> bool:
    return bool(_B_RE.search(text) or (_AI_RE.search(text) and _B_CONTEXT_RE.search(text)))


def preprocess(text: str) -> str:
    """Clean an abstract or document before embedding (same rules as v2)."""
    text = re.sub(r"^(In this (paper|work|study),?\s*(we|the authors)?\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\$[^$]+\$", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    return text


def audit_relevance(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Add mentions_a, mentions_b, on_topic and dual columns.

    on_topic: the document mentions its own field's lexicon.
    dual:     the document mentions both lexicons (a candidate natural bridge).
    """
    out = df.copy()
    full = (out["title"].fillna("") + ". " + out[text_col].fillna("")).tolist()
    out["mentions_a"] = [mentions_field_a(t) for t in full]
    out["mentions_b"] = [mentions_field_b(t) for t in full]
    out["on_topic"] = [
        (lab == FIELD_A and a) or (lab == FIELD_B and b)
        for lab, a, b in zip(out["label"], out["mentions_a"], out["mentions_b"])
    ]
    out["dual"] = out["mentions_a"] & out["mentions_b"]
    return out


@dataclass
class RelevanceSummary:
    n: int
    field_a: int
    field_b: int
    field_a_on_topic: int
    field_b_on_topic: int
    dual: int

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "field_a": self.field_a,
            "field_b": self.field_b,
            "field_a_on_topic": self.field_a_on_topic,
            "field_a_on_topic_rate": round(self.field_a_on_topic / max(self.field_a, 1), 3),
            "field_b_on_topic": self.field_b_on_topic,
            "field_b_on_topic_rate": round(self.field_b_on_topic / max(self.field_b, 1), 3),
            "dual": self.dual,
        }


def summarize_relevance(df: pd.DataFrame) -> RelevanceSummary:
    a = df["label"] == FIELD_A
    b = df["label"] == FIELD_B
    return RelevanceSummary(
        n=len(df),
        field_a=int(a.sum()),
        field_b=int(b.sum()),
        field_a_on_topic=int((a & df["on_topic"]).sum()),
        field_b_on_topic=int((b & df["on_topic"]).sum()),
        dual=int(df["dual"].sum()),
    )
