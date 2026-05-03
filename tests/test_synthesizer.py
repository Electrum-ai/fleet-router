import pytest
from fleet.synthesizer import Synthesizer


def test_pick_code_valid():
    synth = Synthesizer()
    responses = {
        "deepseek": "def foo():\n    return 1",
        "glm": "def bar():\n    x = 1\n    return x + 2",
        "minimax": "invalid python here",
    }
    best = synth.pick(responses, task_tag="code")
    # Both deepseek and glm are valid; glm is longest
    assert best == "def bar():\n    x = 1\n    return x + 2"


def test_pick_code_no_valid():
    synth = Synthesizer()
    responses = {
        "deepseek": "broken syntax (",
        "glm": "also broken [ but longer now",
    }
    best = synth.pick(responses, task_tag="code")
    assert best == "also broken [ but longer now"


def test_pick_creative_diversity():
    synth = Synthesizer()
    responses = {
        "glm": "word word word word word word word",
        "minimax": "a much longer and more detailed creative response with many words",
    }
    best = synth.pick(responses, task_tag="creative")
    # minimax has higher lexical diversity (more unique words relative to length)
    assert "much longer" in best


def test_pick_creative_tie_breaker():
    synth = Synthesizer()
    responses = {
        "glm": "one two three four five six seven eight nine ten",
        "minimax": "one two three four five six seven eight nine ten",
    }
    best = synth.pick(responses, task_tag="creative")
    # Equal diversity; tie-break with longest (they're equal, so either is fine)
    assert best in responses.values()


def test_pick_summarize_shortest():
    synth = Synthesizer()
    responses = {
        "glm": "this is a reasonably short summary",
        "minimax": "a much longer and more detailed summary that goes on and on",
    }
    best = synth.pick(responses, task_tag="summarize")
    assert best == "this is a reasonably short summary"


def test_pick_general_weak_consensus_longest_wins():
    synth = Synthesizer()
    responses = {
        "glm": "short",
        "minimax": "this is a much longer response with many more characters",
    }
    best = synth.pick(responses, task_tag="general")
    # Consensus is weak (very different strings), longest should win
    assert best == "this is a much longer response with many more characters"


def test_pick_general_tie_returns_dict():
    synth = Synthesizer()
    responses = {
        "glm": "abc",
        "minimax": "def",
    }
    best = synth.pick(responses, task_tag="general")
    # Consensus is weak and there is a tie for longest length
    assert best == {"glm": "abc", "minimax": "def"}


def test_pick_math_consensus():
    synth = Synthesizer()
    responses = {
        "glm": "The answer is 42 because 6 times 7 equals 42.",
        "minimax": "The answer is 42 because 6 * 7 = 42.",
        "deepseek": "I think the answer is 99.",
    }
    best = synth.pick(responses, task_tag="math")
    # glm and minimax are similar (consensus), so one of them should win
    assert "42" in best


def test_all_models_failed():
    synth = Synthesizer()
    responses = {
        "glm": None,
        "minimax": None,
    }
    best = synth.pick(responses, task_tag="general")
    assert best == "(all models failed)"


def test_single_valid_response():
    synth = Synthesizer()
    responses = {
        "glm": None,
        "minimax": "only valid response",
    }
    best = synth.pick(responses, task_tag="general")
    assert best == "only valid response"


def test_empty_string_responses():
    synth = Synthesizer()
    responses = {
        "glm": "",
        "minimax": "  ",
        "deepseek": "valid content here",
    }
    best = synth.pick(responses, task_tag="general")
    # Empty strings and whitespace-only should be filtered out
    assert best == "valid content here"
