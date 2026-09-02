"""Local NLI entailment classification: does a retrieved evidence chunk support,
contradict, or say nothing useful about a claim? Runs entirely locally (no LLM
call, no rate limit) via a cross-encoder NLI model — the same
Chroma-candidates -> cross-encoder pattern rag/reranker.py uses for relevance
reranking, just a different pretrained head (entailment/contradiction/neutral
instead of a relevance score).

Real bug found during Phase 6 acceptance testing: filing chunks constantly say
"the Company" rather than the company's actual name (SEC 10-Ks establish "the
Company" = "Apple Inc." once, on the cover page, then use the pronoun for the
rest of the document) — so a chunk pulled from the middle of the filing loses
that antecedent entirely. Fed a claim phrased with the real name ("Apple's total
net sales were...") against a chunk phrased with the pronoun ("the Company's
total net sales were..."), cross-encoder/nli-deberta-v3-xsmall and even
nli-deberta-v3-base scored it "neutral" at high confidence — the model has no
way to know "the Company" means Apple. Verified directly: prefixing the evidence
with one context sentence naming the company (see _CONTEXT_PREFIX) flips these
from ~85%+ confidence neutral to ~98%+ confidence entailment, with no effect on
genuinely unrelated (not_enough_evidence) or contradicting pairs. Also switched
MODEL_NAME from nli-deberta-v3-xsmall to nli-deberta-v3-base: xsmall stayed
weak on filing-table-formatted evidence ("$ 416,161 6 % $ 391,035") even after
the prefix fix, while base classified it correctly on its own.

Second real bug, same root cause: SEC filing HTML tables (net-sales-by-category,
term-debt schedules, etc.) get flattened into prose by mcp_server/tools/get_filing.py's
_strip_html with stray internal spaces around currency/percent tokens — "$ 416,161"
and "6 %" instead of "$416,161" / "6%". Verified directly: swapping that exact
spaced fragment for a fully natural-language restatement of the identical fact
("...total net sales of $416,161 million...") flipped the model from ~92%-confidence
neutral to ~99.8%-confidence entailment with nothing else changed. _normalize_spacing
below collapses that stray spacing — a narrow, targeted fix (it only touches
whitespace directly adjacent to $/%, never reflows or rewords anything) rather than
a general text rewrite.
"""

import re
from functools import lru_cache

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

NOT_ENOUGH_EVIDENCE = "not_enough_evidence"

_CONTEXT_PREFIX = 'This text is from {company}\'s SEC filing. "The Company" refers to {company}. '

_DOLLAR_SPACE_RE = re.compile(r"\$\s+(?=\d)")
_PERCENT_SPACE_RE = re.compile(r"(?<=\d)\s+%")


def _normalize_spacing(text: str) -> str:
    text = _DOLLAR_SPACE_RE.sub("$", text)
    text = _PERCENT_SPACE_RE.sub("%", text)
    return text


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


@lru_cache(maxsize=1)
def _label_map() -> dict[int, str]:
    """Maps the model's own id2label config to our schema's three labels
    (supported | contradicted | not_enough_evidence) instead of assuming a fixed
    label order. cross-encoder/nli-* models are consistently trained on
    ['contradiction', 'entailment', 'neutral'], but reading it from the model's
    config avoids a silent mismatch if MODEL_NAME is ever swapped for one with a
    different label order.
    """
    id2label = _model().model.config.id2label
    mapping: dict[int, str] = {}
    for idx, label in id2label.items():
        lower = label.lower()
        if "entail" in lower:
            mapping[int(idx)] = "supported"
        elif "contradict" in lower:
            mapping[int(idx)] = "contradicted"
        else:
            mapping[int(idx)] = NOT_ENOUGH_EVIDENCE
    return mapping


def classify_entailment(claim: str, evidence: str | None, company: str | None = None) -> tuple[str, float]:
    """Classifies whether `evidence` supports, contradicts, or gives not enough
    information about `claim`. Returns (label, confidence): label is one of
    "supported" | "contradicted" | "not_enough_evidence" (spec.md's Claim schema),
    confidence is the model's softmax probability for that label.

    `company` should be the company the evidence chunk was retrieved for — it's
    prefixed onto the evidence so the model can resolve "the Company" (see the
    module docstring's real-bug writeup). Pass None to skip the prefix (e.g. for
    evidence that already names the company explicitly).

    No evidence chunk (nothing retrieved for this claim) is always
    not_enough_evidence — there's nothing to run the model against.
    """
    if not evidence or not evidence.strip():
        return NOT_ENOUGH_EVIDENCE, 1.0

    evidence = _normalize_spacing(evidence)
    premise = _CONTEXT_PREFIX.format(company=company) + evidence if company else evidence

    # NLI convention: (premise, hypothesis) — the evidence is the premise, the
    # claim being checked is the hypothesis. predict() returns shape (n_pairs,
    # n_labels); take row 0 (this call always passes exactly one pair) before
    # taking argmax, otherwise argmax flattens across both axes and indexes the
    # wrong dimension.
    row = _model().predict([(premise, claim)], apply_softmax=True)[0]
    label_map = _label_map()
    best_idx = int(row.argmax())
    return label_map[best_idx], float(row[best_idx])
