import pytest
from fleet.synthesizer import Synthesizer

def test_pick_code_valid():
    synth = Synthesizer()
    responses = {
        "deepseek": "def foo():\n    return 1",
        "glm": "invalid python here",
    }
    best = synth.pick(responses, task_tag="code")
    assert best == "def foo():\n    return 1"

def test_pick_creative_longest():
    synth = Synthesizer()
    responses = {
        "glm": "a short line",
        "minimax": "a much longer and more detailed creative response with many words",
    }
    best = synth.pick(responses, task_tag="creative")
    assert "much longer" in best

def test_pick_tie_returns_all():
    synth = Synthesizer()
    responses = {
        "glm": "abc",
        "minimax": "def",
    }
    best = synth.pick(responses, task_tag="general")
    # Tie: returns first or falls through; test just ensures no crash
    assert best in ("abc", "def")
