"""Load fleet/config.yaml into typed dataclasses."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".fleet" / "config.yaml"
BUNDLED_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class OllamaConfig:
    base_url: str = _DEFAULT_BASE_URL
    api_key: str = ""


@dataclass
class ModelEntry:
    tags: list[str] = field(default_factory=list)
    priority: int = 99
    # "chat" or "reasoning". Reserved for class-aware early-exit policies in
    # streaming dispatch — a fast chat model should not cancel a reasoning
    # model mid-thought, since the first tokens of a reasoning model are
    # internal chain-of-thought, not the final answer.
    model_class: str = "chat"
    # Provider name in the ProviderPool. This project ships only "ollama";
    # the field exists so future Ollama-compatible backends (vLLM,
    # LM Studio) can register additional providers without schema changes.
    provider: str = "ollama"
    # Optional override of the provider-side model identifier. Defaults to
    # the registry key. Use this when the canonical short name differs from
    # the Ollama tag (e.g. key "deepseek" → "deepseek-v4-pro:cloud").
    api_model: str = ""


@dataclass
class ThresholdConfig:
    # Max-quality default: 1.01 means classification confidence can never
    # clear the bar, so every prompt goes to parallel mode (multi-model
    # ensemble + verifier-driven synthesis). Set to 0.8 to opt back into
    # the speed-vs-quality split at the classifier.
    single_confidence: float = 1.01
    # Legacy session-wide budget. Retained as a backward-compatible source
    # for `timeouts` when a config predates the per-class field.
    parallel_timeout: int = 60
    max_parallel: int = 3
    # Per-model-class wall-clock budget (seconds) for one generation. A
    # reasoning model (deepseek-v4-pro) routinely thinks for minutes; cutting
    # it off at the chat budget returns None and degrades the round to its
    # fast-chat survivors — and with a one-model reasoning pool, to
    # "(all models failed)". The dispatcher resolves a model's class and
    # applies its budget as a PER-REQUEST aiohttp ClientTimeout, which fully
    # replaces (not caps) the provider's session default. The shared session
    # is therefore sized to the chat budget; the reasoning class still gets
    # its full per-request budget.
    timeouts: dict[str, int] = field(
        default_factory=lambda: {"chat": 60, "reasoning": 240}
    )


@dataclass
class ClassifierConfig:
    embeddings_model: str = _DEFAULT_EMBEDDINGS_MODEL
    # "keyword" | "llm" — opt into LLM-based classification.
    mode: str = "keyword"
    # Model to use when mode=="llm". Empty falls back to keyword.
    llm_model: str = ""


@dataclass
class SynthesisConfig:
    """Controls which selection strategy turns N candidates into 1 answer."""
    # "verifier" (executable + LLM judge) or "heuristic" (length/AST/diversity).
    mode: str = "verifier"
    # Model to use as LLM-judge (registry key from `models`). Empty = no judge,
    # JudgeVerifier not registered, falls through to HeuristicVerifier.
    judge_model: str = ""
    # Verifier-set winner.score below this triggers calibrated abstention.
    abstention_threshold: float = 0.4
    # Swap-order consistency for the LLM judge: run the judge twice (given
    # order + reversed) and average per-candidate scores to cancel position
    # bias. ON by default (quality-first); OFF reverts to a single pass.
    judge_swap_order: bool = True
    # CodeVerifier only: opt into running candidate code in a subprocess.
    # OFF by default — running LLM-generated code is a real RCE vector.
    code_execute: bool = False
    code_execute_timeout: int = 5
    # Operator-supplied sandbox command TEMPLATE used to run candidate code.
    # Execution happens ONLY when code_execute is True AND this is non-empty.
    # Placeholders {python}/{file}/{dir} are substituted at run time, e.g.
    # "firejail --net=none --private={dir} {python} {file}". Empty (default)
    # means code is NEVER executed — the AST denylist is an advisory filter,
    # not a sandbox, so CodeVerifier falls back to AST-only static scoring.
    code_execute_sandbox: str = ""


@dataclass
class SamplingConfig:
    """Self-consistency: how many independent samples per model per tag.

    Math/reasoning benefit hugely from N>1 (Wang et al., +18pp on GSM8K).
    Max-quality defaults push samples up across the board — even tags
    without an executable verifier benefit from giving the LLM judge
    multiple drafts to rank.
    """
    samples_by_tag: dict[str, int] = field(
        default_factory=lambda: {
            "math": 7,         # majority vote sweet spot per Wang+
            "reasoning": 5,
            "code": 3,         # execution verifier is the strong signal
            "default": 3,      # everything else: 3 drafts for the judge
        }
    )
    temperature: float = 0.7


@dataclass
class RefinementConfig:
    """Draft → critique → revise loop. ON by default — ~5-20pp gain on
    most tasks. Becomes a silent no-op when `critique_model` is empty
    (preserves backward compatibility for users with custom configs that
    pre-date this field)."""
    enabled: bool = True
    critique_model: str = ""  # empty = no refinement (silent no-op)


@dataclass
class EscalationConfig:
    """When the verifier abstains or scores low, escalate to a stronger
    model with all candidates as context for arbitration. ON by default —
    becomes a silent no-op when `model` is empty."""
    enabled: bool = True
    model: str = ""
    # Score below which we escalate even if the verifier didn't abstain.
    score_threshold: float = 0.6


@dataclass
class RetrievalConfig:
    """Optional context augmentation for tags that benefit from grounding."""
    enabled: bool = False
    # Comma-list of tags that should be augmented (e.g. "general,reasoning").
    tags: list[str] = field(default_factory=list)
    # Provider type: "noop" | "websearch" (websearch needs SERP_API_KEY).
    provider: str = "noop"
    # Max characters of retrieved context to prepend.
    max_chars: int = 4000


@dataclass
class BanditConfig:
    """Outcome-driven Thompson-sampling bandit for (tag, model) selection.
    ON by default — runs in-memory if `state_path` is empty (no learning
    across restarts). The bundled YAML wires a persistent path."""
    enabled: bool = True
    # Where to persist Beta posteriors. Empty = in-memory only.
    state_path: str = ""
    # Exponential forgetting in (0, 1]. 1.0 = no decay (learned posteriors
    # accumulate forever, exact historical behavior). <1.0 shrinks the
    # learned excess over the prior toward the prior before each new
    # observation, so a model that was good then silently changed under the
    # same tag (Ollama cloud upgrades) recovers faster.
    decay: float = 1.0
    # Cold-start priority seeding strength. 0.0 = uniform Beta(1,1) for every
    # fresh arm (random-shuffle cold start). >0.0 seeds an unseen arm with a
    # mild priority-biased prior (alpha = 1 + strength/priority, beta = 1) so
    # the human priority ordering dominates the zero-evidence cold start; a
    # few real observations quickly wash it out. Default 0.5 = mild.
    priority_prior_strength: float = 0.5


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    models: dict[str, ModelEntry] = field(default_factory=dict)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)


def clean_model_key(key: str) -> str:
    """Strip the :cloud suffix so YAML keys like `glm-5.1:cloud` and CLI args
    like `--model glm-5.1` resolve to the same registry entry."""
    return str(key).removesuffix(":cloud")


# Backwards-compatible alias.
_clean_model_key = clean_model_key


def _validate_base_url(url: str) -> str:
    """Reject schemes other than http/https. Warn (don't reject) on plaintext
    http to non-local hosts so misconfigurations are visible in logs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        logger.warning(
            "ollama.base_url scheme %r is not http/https; using default %s",
            parsed.scheme, _DEFAULT_BASE_URL,
        )
        return _DEFAULT_BASE_URL
    if not parsed.hostname:
        logger.warning("ollama.base_url has no hostname; using default %s", _DEFAULT_BASE_URL)
        return _DEFAULT_BASE_URL
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
        logger.warning(
            "ollama.base_url uses plaintext http to non-local host %s; consider https",
            parsed.hostname,
        )
    return url


def _coerce_tags(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(t) for t in raw if t is not None]
    if isinstance(raw, str):
        return [raw]
    return []


def _coerce_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _coerce_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def load_config(path: Path | str | None = None) -> Config:
    """Resolve a config file: explicit path > ~/.fleet/config.yaml > bundled
    fleet/config.yaml > built-in defaults."""
    config_path: Path | None = None
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            # Explicit path missing: honor user intent and return defaults
            # rather than silently falling back to the bundled file.
            logger.warning("config %s not found; using built-in defaults", config_path)
            return Config()
    elif DEFAULT_CONFIG_PATH.exists():
        config_path = DEFAULT_CONFIG_PATH
    elif BUNDLED_CONFIG_PATH.exists():
        # Bootstrap fresh installs with the model list that ships in the package.
        config_path = BUNDLED_CONFIG_PATH

    if config_path is None:
        return Config()

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        logger.warning("malformed YAML in %s (%s); using built-in defaults", config_path, exc)
        return Config()
    if not isinstance(raw, dict):
        logger.warning(
            "config root in %s is not a mapping; using built-in defaults", config_path
        )
        return Config()

    ollama_raw = raw.get("ollama", {})
    if not isinstance(ollama_raw, dict):
        ollama_raw = {}
    base_url = ollama_raw.get("base_url", _DEFAULT_BASE_URL)
    if not isinstance(base_url, str):
        base_url = _DEFAULT_BASE_URL
    api_key = ollama_raw.get("api_key")
    if not isinstance(api_key, str):
        api_key = ""
    ollama = OllamaConfig(
        base_url=_validate_base_url(base_url),
        api_key=api_key,
    )

    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict):
        models_raw = {}
    models: dict[str, ModelEntry] = {}
    for key, val in models_raw.items():
        if not isinstance(val, dict):
            continue
        raw_class = val.get("class", "chat")
        model_class = raw_class if raw_class in ("chat", "reasoning") else "chat"
        provider = str(val.get("provider", "ollama")) or "ollama"
        api_model = val.get("api_model")
        models[clean_model_key(key)] = ModelEntry(
            tags=_coerce_tags(val.get("tags")),
            priority=_coerce_int(val.get("priority"), 99),
            model_class=model_class,
            provider=provider,
            api_model=str(api_model) if api_model else "",
        )

    thresh_raw = raw.get("thresholds", {})
    if not isinstance(thresh_raw, dict):
        thresh_raw = {}
    parallel_timeout = _coerce_int(thresh_raw.get("parallel_timeout"), 60)
    timeouts_raw = thresh_raw.get("timeouts")
    if isinstance(timeouts_raw, dict):
        # Explicit per-class budgets. Seed from the ThresholdConfig dataclass
        # default (not a duplicated literal — so the two can't drift) so a
        # partial mapping (e.g. {"reasoning": 300}) keeps chat at its default,
        # then overlay coerced ints. Junk values are ignored (default kept).
        timeouts = dict(ThresholdConfig().timeouts)
        for k, v in timeouts_raw.items():
            try:
                timeouts[str(k)] = max(1, int(v))
            except (TypeError, ValueError):
                continue
    else:
        # Backward-compat: no `timeouts` block. Derive from parallel_timeout so
        # existing configs don't suddenly cut reasoning models off at the chat
        # budget — reasoning gets at least 240s.
        timeouts = {
            "chat": parallel_timeout,
            "reasoning": max(parallel_timeout, 240),
        }
    thresholds = ThresholdConfig(
        single_confidence=_coerce_float(thresh_raw.get("single_confidence"), 1.01),
        parallel_timeout=parallel_timeout,
        max_parallel=_coerce_int(thresh_raw.get("max_parallel"), 3),
        timeouts=timeouts,
    )

    clf_raw = raw.get("classifier", {})
    if not isinstance(clf_raw, dict):
        clf_raw = {}
    embeddings_model = clf_raw.get("embeddings_model", _DEFAULT_EMBEDDINGS_MODEL)
    if not isinstance(embeddings_model, str):
        embeddings_model = _DEFAULT_EMBEDDINGS_MODEL
    clf_mode = str(clf_raw.get("mode", "keyword"))
    if clf_mode not in ("keyword", "llm"):
        clf_mode = "keyword"
    classifier = ClassifierConfig(
        embeddings_model=embeddings_model,
        mode=clf_mode,
        llm_model=str(clf_raw.get("llm_model", "")),
    )

    syn_raw = raw.get("synthesis", {}) if isinstance(raw.get("synthesis"), dict) else {}
    syn_mode = str(syn_raw.get("mode", "verifier"))
    if syn_mode not in ("verifier", "heuristic"):
        syn_mode = "verifier"
    synthesis = SynthesisConfig(
        mode=syn_mode,
        judge_model=str(syn_raw.get("judge_model", "")),
        abstention_threshold=_coerce_float(syn_raw.get("abstention_threshold"), 0.4),
        judge_swap_order=bool(syn_raw.get("judge_swap_order", True)),
        code_execute=bool(syn_raw.get("code_execute", False)),
        code_execute_timeout=_coerce_int(syn_raw.get("code_execute_timeout"), 5),
        code_execute_sandbox=str(syn_raw.get("code_execute_sandbox", "") or ""),
    )

    samp_raw = raw.get("sampling", {}) if isinstance(raw.get("sampling"), dict) else {}
    samp_by_tag_raw = samp_raw.get("samples_by_tag", {})
    samp_by_tag: dict[str, int] = {
        "math": 7, "reasoning": 5, "code": 3, "default": 3,
    }
    if isinstance(samp_by_tag_raw, dict):
        for k, v in samp_by_tag_raw.items():
            samp_by_tag[str(k)] = max(1, _coerce_int(v, 1))
    sampling = SamplingConfig(
        samples_by_tag=samp_by_tag,
        temperature=_coerce_float(samp_raw.get("temperature"), 0.7),
    )

    ref_raw = raw.get("refinement", {}) if isinstance(raw.get("refinement"), dict) else {}
    refinement = RefinementConfig(
        enabled=bool(ref_raw.get("enabled", True)),
        critique_model=str(ref_raw.get("critique_model", "")),
    )

    esc_raw = raw.get("escalation", {}) if isinstance(raw.get("escalation"), dict) else {}
    escalation = EscalationConfig(
        enabled=bool(esc_raw.get("enabled", True)),
        model=str(esc_raw.get("model", "")),
        score_threshold=_coerce_float(esc_raw.get("score_threshold"), 0.6),
    )

    ret_raw = raw.get("retrieval", {}) if isinstance(raw.get("retrieval"), dict) else {}
    ret_tags_raw = ret_raw.get("tags", [])
    ret_tags = [str(t) for t in ret_tags_raw] if isinstance(ret_tags_raw, list) else []
    retrieval = RetrievalConfig(
        enabled=bool(ret_raw.get("enabled", False)),
        tags=ret_tags,
        provider=str(ret_raw.get("provider", "noop")),
        max_chars=_coerce_int(ret_raw.get("max_chars"), 4000),
    )

    bandit_raw = raw.get("bandit", {}) if isinstance(raw.get("bandit"), dict) else {}
    # Decay clamps to (0, 1]; anything outside (junk, 0, negatives, >1) falls
    # back to 1.0 = no decay, the safe historical default.
    decay = _coerce_float(bandit_raw.get("decay"), 1.0)
    if not (0.0 < decay <= 1.0):
        decay = 1.0
    # Seeding strength clamps to >= 0.0 (0.0 disables seeding).
    strength = max(0.0, _coerce_float(bandit_raw.get("priority_prior_strength"), 0.5))
    bandit = BanditConfig(
        enabled=bool(bandit_raw.get("enabled", True)),
        state_path=str(bandit_raw.get("state_path", "")),
        decay=decay,
        priority_prior_strength=strength,
    )

    return Config(
        ollama=ollama,
        models=models,
        thresholds=thresholds,
        classifier=classifier,
        synthesis=synthesis,
        sampling=sampling,
        refinement=refinement,
        escalation=escalation,
        retrieval=retrieval,
        bandit=bandit,
    )
