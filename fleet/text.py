"""Shared text utilities — the single source of truth for chain-of-thought
stripping.

Reasoning models (DeepSeek-R1, QwQ, o1-style, deepseek-v4-pro reasoning mode)
emit their internal chain-of-thought wrapped in ``<think>...</think>`` or
``<thinking>...</thinking>``. That reasoning must never reach the user, leak
into scoring, or get embedded in downstream judge/escalation/refinement
prompts. Strip it exactly ONCE at the candidate boundary so every consumer sees
the final answer only.

Before this module the strip regex was copy-pasted across five files and only
matched the ``<think>`` spelling — even though the project's own memory calls
out ``<thinking>``. Centralizing here fixes both the spelling gap and the
truncated-generation (unclosed-tag) case.
"""
from __future__ import annotations

import re

# Complete chain-of-thought blocks: <think>...</think> and
# <thinking>...</thinking>, case-insensitive, spanning newlines. The trailing
# \s* swallows the blank line reasoning models usually emit after the block.
_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>\s*",
    re.DOTALL | re.IGNORECASE,
)

# A dangling opening tag with no matching close — what a generation truncated
# mid-thought looks like. Everything from the unmatched <think>/<thinking> to
# the end of the string is internal reasoning we never saw the end of, so we
# drop it. Runs AFTER the block pass, so it only fires on genuinely unclosed
# tags.
_THINK_UNCLOSED_RE = re.compile(
    r"<think(?:ing)?>.*\Z",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Remove chain-of-thought blocks and return only the final answer.

    Handles:
    - closed ``<think>...</think>`` and ``<thinking>...</thinking>`` blocks
      (both spellings, case-insensitive),
    - an UNCLOSED leading tag from a truncated generation — text that opens
      ``<think>`` and never closes it drops everything from that tag onward
      (an all-reasoning truncation collapses to ``""``),
    - leading/trailing whitespace left behind by the removals.

    Internal whitespace is preserved so code indentation and multi-line
    answers survive untouched.

    KNOWN LIMITATION (conscious tradeoff, pinned by a test): the unclosed-tag
    pass deletes everything from an unmatched ``<think>``/``<thinking>`` to the
    end of the string. That is correct for a truncated generation, but it also
    corrupts a legitimate answer that mentions the literal token with no closing
    tag — e.g. a reasoning-model router answering "how does the <think> tag
    work?" loses everything after the word. We accept this: truncated reasoning
    leaking to the user is the higher-stakes failure, and tag-literal answers
    are rare. Do not silently "fix" this without revisiting the tradeoff.

    Idempotent: ``strip_thinking(strip_thinking(t)) == strip_thinking(t)``.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    return cleaned.strip()
