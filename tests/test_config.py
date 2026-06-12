import pytest
from pathlib import Path
from fleet.config import Config, load_config, ModelEntry

SAMPLE_YAML = """
ollama:
  base_url: http://localhost:11434
models:
  deepseek-v4-pro:
    tags: [code, reasoning, math]
    priority: 1
  glm-5.1:
    tags: [creative, chinese, long_context]
    priority: 2
thresholds:
  single_confidence: 0.8
  parallel_timeout: 60
  max_parallel: 3
classifier:
  embeddings_model: all-MiniLM-L6-v2
"""

def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SAMPLE_YAML)
    cfg = load_config(config_path)
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.ollama.api_key == ""
    assert "deepseek-v4-pro" in cfg.models
    assert cfg.models["deepseek-v4-pro"].tags == ["code", "reasoning", "math"]
    assert cfg.models["deepseek-v4-pro"].priority == 1
    assert "glm-5.1" in cfg.models
    assert cfg.models["glm-5.1"].tags == ["creative", "chinese", "long_context"]
    assert cfg.models["glm-5.1"].priority == 2
    assert cfg.thresholds.single_confidence == 0.8
    assert cfg.thresholds.parallel_timeout == 60
    assert cfg.thresholds.max_parallel == 3
    assert cfg.classifier.embeddings_model == "all-MiniLM-L6-v2"


def test_cloud_suffix_stripping(tmp_path):
    yaml_text = """
models:
  deepseek-v4-pro:cloud:
    tags: [code]
    priority: 5
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    assert "deepseek-v4-pro" in cfg.models
    assert "deepseek-v4-pro:cloud" not in cfg.models
    assert cfg.models["deepseek-v4-pro"].tags == ["code"]
    assert cfg.models["deepseek-v4-pro"].priority == 5


def test_missing_file_fallback():
    nonexistent = Path("/nonexistent/path/config.yaml")
    cfg = load_config(nonexistent)
    assert isinstance(cfg, Config)
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.models == {}
    # Max-quality defaults: every prompt fans out (single_confidence > 1.0).
    assert cfg.thresholds.single_confidence > 1.0
    assert cfg.thresholds.parallel_timeout == 60
    assert cfg.thresholds.max_parallel == 3
    assert cfg.classifier.embeddings_model == "all-MiniLM-L6-v2"


def test_model_class_parses(tmp_path):
    yaml_text = """
models:
  thinker:
    tags: [reasoning]
    priority: 1
    class: reasoning
  chatter:
    tags: [creative]
    priority: 2
    class: chat
  default-class:
    tags: [code]
    priority: 3
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    assert cfg.models["thinker"].model_class == "reasoning"
    assert cfg.models["chatter"].model_class == "chat"
    # Default when 'class' key is absent
    assert cfg.models["default-class"].model_class == "chat"


def test_model_class_invalid_falls_back_to_chat(tmp_path):
    yaml_text = """
models:
  weird:
    tags: [code]
    class: superintelligent
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    # Unknown class values fall back to "chat" rather than raising.
    assert cfg.models["weird"].model_class == "chat"


def test_timeouts_default_when_no_thresholds_block():
    """Bare Config() gets the documented per-class defaults."""
    cfg = Config()
    assert cfg.thresholds.timeouts == {"chat": 60, "reasoning": 240}


def test_timeouts_derived_from_parallel_timeout_when_absent(tmp_path):
    """A config that only sets parallel_timeout (no `timeouts` block) derives
    per-class budgets so reasoning models are never cut at the chat budget."""
    yaml_text = """
thresholds:
  parallel_timeout: 90
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    # chat == parallel_timeout; reasoning == max(parallel_timeout, 240).
    assert cfg.thresholds.timeouts == {"chat": 90, "reasoning": 240}


def test_timeouts_derived_keeps_large_parallel_timeout_for_reasoning(tmp_path):
    """If parallel_timeout already exceeds 240, reasoning derives to it."""
    yaml_text = """
thresholds:
  parallel_timeout: 300
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    assert cfg.thresholds.timeouts == {"chat": 300, "reasoning": 300}


def test_timeouts_explicit_parse_coerces_and_ignores_junk(tmp_path):
    """Explicit per-class budgets are coerced to ints; junk values are ignored
    and missing keys keep their defaults."""
    yaml_text = """
thresholds:
  parallel_timeout: 60
  timeouts:
    chat: 30
    reasoning: not-a-number
    code: "120"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    assert cfg.thresholds.timeouts["chat"] == 30      # overridden
    assert cfg.thresholds.timeouts["reasoning"] == 240  # junk ignored, default kept
    assert cfg.thresholds.timeouts["code"] == 120     # coerced from string


def test_ollama_api_key_parsed(tmp_path):
    yaml_text = """
ollama:
  base_url: http://localhost:11434
  api_key: sk-ollama-secret
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)
    assert cfg.ollama.api_key == "sk-ollama-secret"


def test_malformed_yaml_fallback(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ollama: [not_a_dict\n")
    cfg = load_config(config_path)
    assert isinstance(cfg, Config)
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.models == {}
    assert cfg.thresholds.single_confidence > 1.0
    assert cfg.classifier.embeddings_model == "all-MiniLM-L6-v2"


def test_max_quality_defaults_when_yaml_omits_optional_blocks(tmp_path):
    """A YAML file with only the basics still gets max-quality defaults
    for refinement / escalation / bandit / sampling — that's the policy."""
    yaml_text = """
ollama:
  base_url: http://localhost:11434
models:
  some-model:
    tags: [code]
    priority: 1
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)

    # All quality features ON by default.
    assert cfg.refinement.enabled is True
    assert cfg.escalation.enabled is True
    assert cfg.bandit.enabled is True

    # Sampling is aggressive across tags.
    assert cfg.sampling.samples_by_tag["math"] >= 5
    assert cfg.sampling.samples_by_tag["reasoning"] >= 3
    assert cfg.sampling.samples_by_tag["default"] >= 2

    # Every prompt fans out (single_confidence above the [0,1] band).
    assert cfg.thresholds.single_confidence > 1.0


def test_max_quality_defaults_on_bare_config():
    """Even Config() with no YAML at all gets max-quality defaults — the
    feature flags are silent no-ops when the wired models are empty."""
    cfg = Config()
    assert cfg.refinement.enabled is True
    assert cfg.refinement.critique_model == ""  # silent no-op until wired
    assert cfg.escalation.enabled is True
    assert cfg.escalation.model == ""           # silent no-op until wired
    assert cfg.bandit.enabled is True
    assert cfg.bandit.state_path == ""          # in-memory until wired
    assert cfg.thresholds.single_confidence > 1.0


def test_user_can_downshift_quality_in_yaml(tmp_path):
    """The whole point of max-quality-by-default is that you opt OUT, not in.
    Verify the YAML override path works."""
    yaml_text = """
thresholds:
  single_confidence: 0.7
sampling:
  samples_by_tag:
    default: 1
refinement:
  enabled: false
escalation:
  enabled: false
bandit:
  enabled: false
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)
    cfg = load_config(config_path)

    assert cfg.thresholds.single_confidence == 0.7
    assert cfg.sampling.samples_by_tag["default"] == 1
    assert cfg.refinement.enabled is False
    assert cfg.escalation.enabled is False
    assert cfg.bandit.enabled is False
