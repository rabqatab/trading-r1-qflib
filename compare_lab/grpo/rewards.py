"""GRPO reward functions (paper §5.2 + §8 format).

Three staged rewards (kept SEPARATE, not summed into one objective — the R0
lesson, §6): Stage I structure (XML section shape), Stage II evidence
(opinion + quote + source bullets), Stage III decision (asymmetric matrix vs the
deterministic volatility label — the verifiable core).

TRL usage: pass the `*_reward_func` callables to GRPOTrainer(reward_funcs=[...]).
Each takes (prompts, completions, **kwargs) and reads the per-example `label`
column from kwargs.
"""
from __future__ import annotations

import re

CLASSES: tuple[str, ...] = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")

# Asymmetric decision reward matrix M[pred][true] (§5.2 Stage III).
# false-bullish is penalised harder than false-bearish (capital preservation),
# and HOLD when action is needed is penalised (anti-HOLD bias).
#
# NOTE — source discrepancy: the paper-summary §5.2 *table* prints the two extreme
# corners as M[SS][SB]=-2.25 / M[SB][SS]=-2.00, but its design-principle ① text
# states the opposite (predicting StrongBuy when truth is StrongSell — long into a
# crash — must be the WORST, -2.25). We follow the stated economic principle: the
# two extreme corners are oriented so false-bullish is the harshest.
_M = [
    # true:  SS      S      H      B      SB
    [1.00,  0.75, -1.25, -2.00, -2.00],   # pred SS  (SS/SB = missed rally)
    [0.75,  1.00, -0.75, -1.50, -2.00],   # pred S
    [-1.50, -1.00,  1.00, -1.00, -1.50],  # pred H   (anti-HOLD, per principle ③)
    [-1.75, -1.25, -0.75,  1.00,  0.75],  # pred B
    [-2.25, -1.50, -1.25,  0.75,  1.00],  # pred SB  (SB/SS = long into crash, worst)
]
DECISION_MATRIX: dict[str, dict[str, float]] = {
    CLASSES[i]: {CLASSES[j]: _M[i][j] for j in range(5)} for i in range(5)
}

_DECISION = re.compile(r"\[\[\[\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)\s*\]\]\]",
                       re.IGNORECASE)
_HEADER = re.compile(r"^\s*(?:#{1,6}\s+(.*)|<(\w+)>\s*)$")
_BULLET = re.compile(r"^\s*[-*○•]\s+(.*)")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)")   # *quote*, not **bold**
_SOURCE = re.compile(r"`([^`]+)`")                         # `source`


# ---- Stage III: decision (verifiable core) --------------------------------

def parse_last_decision(text: str) -> str | None:
    m = _DECISION.findall(text or "")
    return m[-1].upper() if m else None


def decision_reward(text: str, label: str, lam: float = 1.0) -> float:
    d = parse_last_decision(text)
    if d is None or label not in DECISION_MATRIX:
        return -1.5 * lam
    return DECISION_MATRIX[d][label] * lam


# ---- section parsing (shared) ---------------------------------------------

def _sections(text: str) -> list[tuple[str, str]]:
    """Return [(name_lower, body)] for markdown headers and XML tags."""
    out: list[tuple[str, list[str]]] = []
    name, body = None, []
    for ln in (text or "").split("\n"):
        m = _HEADER.match(ln)
        if m:
            if name is not None:
                out.append((name, body))
            name = (m.group(1) or m.group(2) or "").strip().lower()
            body = []
        elif name is not None:
            body.append(ln)
    if name is not None:
        out.append((name, body))
    return [(n, "\n".join(b)) for n, b in out]


def _is_conclusion(name: str) -> bool:
    return "conclusion" in name or "decision" in name


def _analysis_sections(text: str) -> list[tuple[str, str]]:
    return [(n, b) for n, b in _sections(text)
            if n != "think" and not _is_conclusion(name=n)]


# ---- Stage I: structure ----------------------------------------------------

def _r_count_sections(s: int) -> float:
    if 5 <= s <= 7:
        return 1.0
    if s < 5:
        return max(0.3, s / 5 * 0.7)
    return max(0.3, 1 - 0.15 * (s - 7))


def _r_struct_section(body: str) -> float:
    if len(body.split()) < 50:
        return 0.2
    has_header = bool(re.search(r"^\s*#{1,6}\s", body, re.M))
    has_bullets = bool(_BULLET.search(body)) or bool(re.search(r"^\s*[-*○•]\s", body, re.M))
    has_bold = "**" in body
    has_table = "|" in body
    return 0.3 * has_header + 0.4 * has_bullets + 0.2 * has_bold + 0.1 * has_table


def structure_reward(text: str) -> float:
    analysis = _analysis_sections(text)
    s = len(analysis)
    if s == 0:
        return 0.0
    r_count = _r_count_sections(s)
    r_struct = sum(_r_struct_section(b) for _, b in analysis) / s
    return 0.6 * r_count + 0.4 * r_struct


# ---- Stage II: evidence ----------------------------------------------------

def _bullets(body: str) -> list[str]:
    return [m.group(1) for ln in body.split("\n") if (m := _BULLET.match(ln))]


def _r_opinion(bullet: str) -> float:
    quotes = _ITALIC.findall(bullet)
    sources = _SOURCE.findall(bullet)
    opinion = bullet.split("*", 1)[0] if "*" in bullet else bullet
    w = len(opinion.split())
    grounded = bool(quotes) and bool(sources)
    if grounded:
        if 15 <= w <= 90:
            return 1.0
        if w < 15:
            return w / 15
        return max(0.5, 1 - 0.02 * (w - 90))
    return min(0.3, w / 15 * 0.3)


def _r_bullet(bullet: str) -> float:
    q = bool(_ITALIC.search(bullet))
    s = bool(_SOURCE.search(bullet))
    return 0.4 * _r_opinion(bullet) + 0.35 * q + 0.25 * s


def _r_count_bullets(n: int) -> float:
    if 4 <= n <= 7:
        return 1.0
    if n < 4:
        return n / 4
    return max(0.3, 1 - 0.1 * (n - 7))


def _harmonic(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return len(scores) / sum(1 / max(s, 0.01) for s in scores)


def _r_evidence_section(body: str) -> float:
    bs = _bullets(body)
    if not bs:
        return 0.0
    return 0.3 * _r_count_bullets(len(bs)) + 0.7 * _harmonic([_r_bullet(b) for b in bs])


def evidence_reward(text: str) -> float:
    sections = _analysis_sections(text)
    sec_scores = [_r_evidence_section(b) for _, b in sections]
    sec_scores = [s for s in sec_scores if s > 0]
    return _harmonic(sec_scores)


# ---- TRL-compatible wrappers ----------------------------------------------

def _texts(completions):
    # completions may be list[str] or list[list[{"content":...}]]
    out = []
    for c in completions:
        out.append(c if isinstance(c, str) else c[0]["content"])
    return out


def decision_reward_func(prompts=None, completions=None, label=None, **kw):
    labels = label if label is not None else [None] * len(completions)
    return [decision_reward(t, lab) for t, lab in zip(_texts(completions), labels)]


def structure_reward_func(prompts=None, completions=None, **kw):
    return [structure_reward(t) for t in _texts(completions)]


def evidence_reward_func(prompts=None, completions=None, **kw):
    return [evidence_reward(t) for t in _texts(completions)]
