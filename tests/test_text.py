"""Tests for the shared chain-of-thought stripper (fleet/text.py).

Regression coverage for the bug where the strip regex only matched the
``<think>`` spelling (missing ``<thinking>``, which the project's own memory
calls out) and did not handle truncated generations that open a tag without
ever closing it.
"""
from fleet.text import strip_thinking


def test_strips_think_block():
    assert strip_thinking("<think>internal</think>final answer") == "final answer"


def test_strips_thinking_block_both_spellings():
    assert strip_thinking("<thinking>internal</thinking>final") == "final"
    assert strip_thinking("<think>internal</think>final") == "final"


def test_case_insensitive():
    assert strip_thinking("<THINK>x</THINK>ans") == "ans"
    assert strip_thinking("<Thinking>x</Thinking>ans") == "ans"


def test_multiline_block():
    text = "<think>line one\nline two\nline three</think>the answer"
    assert strip_thinking(text) == "the answer"


def test_multiple_blocks():
    text = "<think>a</think>part one <think>b</think>part two"
    assert strip_thinking(text) == "part one part two"


def test_unclosed_leading_tag_collapses_to_empty():
    """A truncated all-reasoning generation (opens <think>, never closes)
    has no final answer to surface → empty."""
    assert strip_thinking("<think>reasoning that got cut off mid") == ""
    assert strip_thinking("<thinking>still thinking and then truncated") == ""


def test_unclosed_tag_after_answer_drops_only_the_reasoning():
    assert strip_thinking("real answer <thinking>oops truncated") == "real answer"


def test_closed_then_unclosed():
    text = "<think>a</think>kept<think>b never closes"
    assert strip_thinking(text) == "kept"


def test_idempotent():
    raw = "<thinking>reason</thinking>answer"
    once = strip_thinking(raw)
    assert strip_thinking(once) == once
    assert once == "answer"


def test_preserves_internal_whitespace_and_indentation():
    """Code answers must survive — only leading/trailing whitespace is
    trimmed, never internal indentation/newlines."""
    text = "<think>plan</think>def foo():\n    return 1\n"
    assert strip_thinking(text) == "def foo():\n    return 1"


def test_empty_and_clean_inputs():
    assert strip_thinking("") == ""
    assert strip_thinking("   ") == ""
    assert strip_thinking("just a normal answer") == "just a normal answer"


def test_block_with_no_answer_after_is_empty():
    assert strip_thinking("<think>only reasoning here</think>") == ""
    assert strip_thinking("<think>only reasoning</think>   ") == ""


def test_unclosed_tag_eats_literal_mention_known_limitation():
    """KNOWN LIMITATION, deliberately pinned (see strip_thinking docstring):
    an answer that mentions the literal <think> token with no closing tag is
    treated as a truncated reasoning block, so everything from the token onward
    is dropped. This corrupts a legitimate "how does the <think> tag work?"
    answer — an accepted tradeoff (truncated-reasoning leakage is the
    higher-stakes failure). This test pins the behavior so any future change is
    a conscious decision, not a silent surprise."""
    assert strip_thinking("The <think> tag wraps reasoning.") == "The"
