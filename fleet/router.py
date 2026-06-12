"""Main orchestrator: classify → decide → dispatch → verify → (escalate/refine).

Three modes interplay:
- synthesis.mode = "verifier" (default) routes through tag-specific verifiers
  with calibrated abstention. mode = "heuristic" uses the legacy length/AST
  picker.
- sampling.samples_by_tag enables self-consistency (multi-sample voting) on
  tags that benefit from it (math, reasoning).
- escalation + refinement are opt-in post-synthesis passes.
"""
from __future__ import annotations

import logging
from typing import Optional

from fleet.bandit import ThompsonBandit
from fleet.classifier import TaskClassifier
from fleet.config import Config
from fleet.dispatcher import EnsembleDispatcher
from fleet.events import EventBus, ModelDispatched, PromptClassified, ResponseSynthesized
from fleet.registry import ModelRegistry
from fleet.synthesizer import Synthesizer
from fleet.text import strip_thinking
from fleet.verifiers.base import VerificationResult
from fleet.verifiers.code import CodeVerifier
from fleet.verifiers.judge import JudgeVerifier
from fleet.verifiers.math import MathVerifier
from fleet.verifiers.registry import VerifierRegistry
from fleet.verifiers.synthesizer import VerifierSynthesizer

logger = logging.getLogger(__name__)

ERROR_MODEL_FAILED = "(model failed)"
ERROR_ALL_MODELS_FAILED = "(all models failed)"
ERROR_NO_MODEL = "(no model available)"
ERROR_NO_MODELS = "(no models available)"

_JUDGE_TAGS = ("reasoning", "creative", "summarize", "translate", "general")


class FleetRouter:
    """Route prompts to the best model(s) and return the best response."""

    def __init__(
        self,
        config: Config | None = None,
        events: Optional[EventBus] = None,
    ):
        self._config = config or Config()
        self._classifier = TaskClassifier(self._config.classifier.embeddings_model)
        self._registry = ModelRegistry(self._config)
        self._dispatcher = EnsembleDispatcher(self._config)
        self._synthesizer = Synthesizer()  # heuristic fallback path
        self._verifier_synth = self._build_verifier_synth()
        self._events = events or EventBus()
        self._bandit: Optional[ThompsonBandit] = None
        if self._config.bandit.enabled:
            self._bandit = ThompsonBandit(
                state_path=self._config.bandit.state_path or None,
            )

    def _build_verifier_synth(self) -> VerifierSynthesizer:
        registry = VerifierRegistry()
        registry.register(CodeVerifier(
            execute=self._config.synthesis.code_execute,
            execute_timeout=self._config.synthesis.code_execute_timeout,
        ))
        registry.register(MathVerifier())

        judge_key = self._config.synthesis.judge_model
        if judge_key:
            entry = self._config.models.get(judge_key)
            provider_name = entry.provider if entry else "ollama"
            api_model = entry.api_model if entry and entry.api_model else judge_key
            provider = self._dispatcher._pool.get(provider_name)
            if provider is not None:
                for tag in _JUDGE_TAGS:
                    registry.register(JudgeVerifier(provider, api_model, tag=tag))
            else:
                logger.warning(
                    "judge provider %r not in pool; skipping JudgeVerifier",
                    provider_name,
                )

        return VerifierSynthesizer(
            registry,
            abstention_threshold=self._config.synthesis.abstention_threshold,
        )

    def refresh(self) -> None:
        """Eagerly refresh the model registry."""
        self._registry.refresh()

    async def aclose(self) -> None:
        """Close the underlying provider pool's aiohttp sessions. Without
        this, short-lived callers (the CLI, eval harness) leak a session
        per run — aiohttp logs a noisy `Unclosed client session` warning
        at interpreter shutdown."""
        await self._dispatcher.aclose()

    async def ask(
        self,
        prompt: str,
        force_parallel: bool = False,
        force_model: str | None = None,
        system: str | None = None,
    ) -> str | dict[str, str]:
        if force_model:
            responses = await self._dispatcher.run(prompt, [force_model], system=system)
            result = responses.get(force_model)
            if result is None:
                return f"{ERROR_MODEL_FAILED}: {force_model}"
            cleaned = strip_thinking(result)
            # A non-empty raw result that strips to "" was ALL chain-of-thought
            # (e.g. a truncated <think> block). `cleaned or result` would leak
            # that raw reasoning to the user — the exact bug this guards against.
            # Surface the failure sentinel instead. A genuinely empty model
            # response ("") is a valid answer and passes straight through.
            if cleaned or not result:
                return cleaned
            return ERROR_ALL_MODELS_FAILED

        tag, confidence = self._classifier.classify(prompt)
        self._events.emit(PromptClassified(tag=tag, confidence=confidence, prompt=prompt))

        if force_parallel or confidence < self._config.thresholds.single_confidence:
            return await self._parallel(prompt, tag, system=system)
        return await self._single(prompt, tag, system=system)

    async def _single(
        self, prompt: str, tag: str, system: str | None = None
    ) -> str | dict[str, str]:
        primary = self._registry.get_best_for_tag(tag)
        if not primary:
            return f"{ERROR_NO_MODEL} for tag: {tag}"

        responses = await self._dispatcher.run(prompt, [primary], system=system)
        result = responses.get(primary)
        if result is not None:
            cleaned = strip_thinking(result)
            # All-chain-of-thought primary (non-empty raw, empty after strip):
            # do NOT leak raw reasoning — fall to the failure sentinel. A
            # genuinely empty "" response is a valid answer and passes through.
            if cleaned or not result:
                return cleaned
            return ERROR_ALL_MODELS_FAILED

        fallbacks = [
            m for m in self._registry.all_available() if m != primary
        ]
        if not fallbacks:
            return ERROR_ALL_MODELS_FAILED
        fb_responses = await self._dispatcher.run(prompt, fallbacks, system=system)
        for model in fallbacks:
            fb = fb_responses.get(model)
            if fb is not None:
                cleaned = strip_thinking(fb)
                # Skip a fallback whose entire output was chain-of-thought
                # (would strip to "") and try the next model rather than
                # returning raw reasoning. A genuine "" response is acceptable.
                if cleaned or not fb:
                    return cleaned
        return ERROR_ALL_MODELS_FAILED

    async def _parallel(
        self, prompt: str, tag: str, system: str | None = None
    ) -> str | dict[str, str]:
        max_parallel = self._config.thresholds.max_parallel
        # Build the candidate pool first; the bandit (if enabled) re-ranks
        # the FULL pool so it can explore beyond the priority-sorted head.
        pool = self._registry.all_models_for_tag(tag) or self._registry.all_available()
        models = self._select_models(tag, pool, max_parallel)
        if not models:
            return ERROR_NO_MODELS

        samples_n = self._sample_count(tag)
        self._events.emit(ModelDispatched(models=list(models), tag=tag, samples=samples_n))

        # Heuristic fast path keeps backward compatibility with code that
        # mocks `_synthesizer.pick` directly.
        if self._config.synthesis.mode == "heuristic" and samples_n == 1:
            responses = await self._dispatcher.run(prompt, models, system=system)
            chosen = self._synthesizer.pick(responses, task_tag=tag)
            self._events.emit(ResponseSynthesized(tag=tag, mode="heuristic"))
            return chosen

        # Verifier path: multi-sample dispatch → verifier → optional escalation/refinement.
        samples_per_model = await self._dispatcher.run_multi(
            prompt, models, samples=samples_n, system=system,
            temperature=self._config.sampling.temperature,
        )
        result = await self._verifier_synth.pick(prompt, samples_per_model, task_tag=tag)
        self._events.emit(ResponseSynthesized(
            tag=tag, mode="verifier",
            winner_model=result.winner.model if result.winner else None,
            winner_score=result.winner.score if result.winner else None,
            abstain=result.abstain,
        ))
        # Feed verifier scores back into the bandit's posteriors. Each sampled
        # candidate is an independent observation — with samples_per_model=5
        # the bandit gets 5× more signal per dispatch.
        self._update_bandit(tag, result)

        # Disagreement escalation: when verifier abstains OR winner score is
        # weak, ask a stronger model to arbitrate using all candidates as context.
        if self._should_escalate(result):
            escalated = await self._escalate(prompt, result, system=system)
            if escalated is not None:
                return escalated

        if result.abstain:
            return self._format_abstention(result, tag)

        winner_text = result.winner_text or ERROR_ALL_MODELS_FAILED

        # Refinement: critique → revise pass on the winning answer.
        if self._config.refinement.enabled and result.winner is not None:
            refined = await self._refine(
                prompt, winner_text,
                winner_model=result.winner.model, system=system,
            )
            if refined:
                return refined

        return winner_text

    def _select_models(
        self, tag: str, pool: list[str], top_n: int
    ) -> list[str]:
        """Bandit-aware model selection. With bandit enabled, Thompson-rank
        the entire pool so the bandit can explore tail candidates. Without
        bandit, take the top-N by priority (pool is already priority-sorted)."""
        if not pool:
            return []
        if self._bandit is not None:
            return self._bandit.rank(tag, pool)[:top_n]
        return pool[:top_n]

    def _update_bandit(self, tag: str, result: VerificationResult) -> None:
        """Push verifier scores into the bandit posteriors. Skipped when:
        - bandit disabled, OR
        - the verifier produced no scored candidates, OR
        - the verifier marked its scores unreliable (judge crashed,
          returned empty, output unparseable, or only one candidate).
          Updating from those would poison posteriors with all-0.5 noise
          and prevent the bandit from ever discriminating between models."""
        if self._bandit is None or not result.all_scored or not result.scores_reliable:
            return
        for c in result.all_scored:
            self._bandit.update(tag, c.model, c.score)

    def _sample_count(self, tag: str) -> int:
        by_tag = self._config.sampling.samples_by_tag
        n = by_tag.get(tag, by_tag.get("default", 1))
        return max(1, int(n))

    def _should_escalate(self, result) -> bool:
        if not self._config.escalation.enabled or not self._config.escalation.model:
            return False
        if not result.all_scored:
            return False
        if result.abstain:
            return True
        return (
            result.winner is not None
            and result.winner.score < self._config.escalation.score_threshold
        )

    async def _escalate(
        self, prompt: str, result, system: str | None = None
    ) -> str | None:
        configured_model = self._config.escalation.model
        if not configured_model:
            return None
        candidate_models = {c.model for c in result.all_scored}
        model = self._pick_arbiter(configured_model, candidate_models)
        if model is None:
            logger.info(
                "escalation skipped: configured model %r is the only candidate "
                "and no other model is available — self-judging would bias",
                configured_model,
            )
            return None
        # Show the top 3 candidates by score for arbitration.
        top = sorted(result.all_scored, key=lambda c: -c.score)[:3]
        candidates_block = "\n\n".join(
            f"--- Candidate {chr(65+i)} (model={c.model}, score={c.score:.2f}) ---\n{c.text}"
            for i, c in enumerate(top)
        )
        escalate_prompt = (
            "Multiple LLM candidates produced divergent answers. Synthesize the "
            "single best answer — pick the strongest, fix its errors, or write "
            "a fresh one that supersedes them.\n\n"
            f"USER PROMPT:\n{prompt}\n\n"
            f"CANDIDATES:\n{candidates_block}\n\n"
            "BEST ANSWER:"
        )
        responses = await self._dispatcher.run(escalate_prompt, [model], system=system)
        escalated = responses.get(model)
        if not escalated:
            return None
        # Strip the arbiter's own chain-of-thought before it reaches the user.
        return strip_thinking(escalated) or None

    def _pick_arbiter(
        self, configured: str, candidates: set[str]
    ) -> Optional[str]:
        """Pick a model to act as judge/critic/escalator that ISN'T already
        a candidate. LLM judges have a well-documented self-preference
        bias: when asked to rank answers including their own, they
        consistently overrate themselves. If the configured arbiter is
        in the candidate set, swap to the next-priority available model;
        only return the configured model when no neutral alternative
        exists (or when it isn't a candidate at all)."""
        if configured not in candidates:
            return configured
        for alt in self._registry.all_available():
            if alt not in candidates:
                return alt
        return None

    def _format_abstention(self, result, tag: str) -> str:
        """Calibrated 'I don't know' — surfaces the candidates so the user
        can judge for themselves rather than seeing a confident wrong answer."""
        if not result.all_scored:
            return f"(no answer for tag={tag}): {result.rationale}"
        top = sorted(result.all_scored, key=lambda c: -c.score)[:3]
        candidates_summary = "\n\n".join(
            f"--- {c.model}#{c.sample_idx} (score={c.score:.2f}) ---\n{c.text[:1000]}"
            for c in top
        )
        return (
            f"(uncertain — {result.rationale})\n\n"
            f"Top candidates considered:\n\n{candidates_summary}"
        )

    async def _refine(
        self, prompt: str, draft: str, winner_model: Optional[str] = None,
        system: str | None = None,
    ) -> str | None:
        configured = self._config.refinement.critique_model
        if not configured:
            return None
        # Critic should not be the same model that produced the draft —
        # self-critique routinely returns "looks good" because the model
        # rationalizes its own output. Swap to a different available model.
        candidates = {winner_model} if winner_model else set()
        critique_model = self._pick_arbiter(configured, candidates)
        if critique_model is None:
            logger.info(
                "refinement skipped: configured critic %r wrote the draft "
                "and no other model is available", configured,
            )
            return None
        critique_prompt = (
            "Find errors, omissions, ambiguity, and weaknesses in this answer. "
            "Be specific and concrete. If the answer is excellent, say 'no critique needed'.\n\n"
            f"USER ASKED:\n{prompt}\n\nANSWER:\n{draft}\n\nCRITIQUE:"
        )
        critique_resp = await self._dispatcher.run(
            critique_prompt, [critique_model], system=system
        )
        critique = critique_resp.get(critique_model)
        if not critique or "no critique needed" in critique.lower():
            return None
        revise_prompt = (
            "Rewrite the answer addressing every point in the critique. "
            "Preserve what was correct; fix what was wrong.\n\n"
            f"USER ASKED:\n{prompt}\n\nORIGINAL:\n{draft}\n\n"
            f"CRITIQUE:\n{critique}\n\nREVISED ANSWER:"
        )
        revise_resp = await self._dispatcher.run(
            revise_prompt, [critique_model], system=system
        )
        revised = revise_resp.get(critique_model)
        if not revised:
            return None
        # Strip the critic's chain-of-thought from the revised answer.
        return strip_thinking(revised) or None
