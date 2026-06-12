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
from fleet.llm_classifier import LLMClassifier
from fleet.registry import ModelRegistry
from fleet.retrieval import RetrievalProvider, build_retrieval_provider
from fleet.synthesizer import Synthesizer
from fleet.text import strip_thinking
from fleet.verifiers.base import Candidate, VerificationResult
from fleet.verifiers.code import CodeVerifier
from fleet.verifiers.heuristic import HeuristicVerifier
from fleet.verifiers.judge import JudgeVerifier, _candidate_label
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
        # Keyword classifier is always built: it is the LLM classifier's
        # fallback AND the path callers take when LLM mode isn't configured.
        self._classifier = TaskClassifier(self._config.classifier.embeddings_model)
        self._registry = ModelRegistry(self._config)
        self._dispatcher = EnsembleDispatcher(self._config)
        self._synthesizer = Synthesizer()  # heuristic fallback path
        self._verifier_synth = self._build_verifier_synth()
        self._llm_classifier = self._build_llm_classifier()
        self._retrieval = self._build_retrieval()
        self._events = events or EventBus()
        self._bandit: Optional[ThompsonBandit] = None
        if self._config.bandit.enabled:
            self._bandit = ThompsonBandit(
                state_path=self._config.bandit.state_path or None,
                decay=self._config.bandit.decay,
                prior_provider=self._make_prior_provider(),
            )

    def _build_llm_classifier(self) -> Optional[LLMClassifier]:
        """Build an LLMClassifier when classifier.mode == 'llm' and a
        llm_model is configured. Degrades to None (keyword) when the mode
        isn't 'llm', the model name is empty, or the provider isn't in the
        pool — every case logs why and leaves the keyword path intact."""
        clf = self._config.classifier
        if clf.mode != "llm":
            return None
        if not clf.llm_model:
            logger.warning(
                "classifier.mode='llm' but classifier.llm_model is empty; "
                "falling back to keyword classification"
            )
            return None
        # Resolve the provider the same way _make_judge does: honor the
        # model entry's provider, defaulting to ollama.
        entry = self._config.models.get(clf.llm_model)
        provider_name = entry.provider if entry else "ollama"
        api_model = entry.api_model if entry and entry.api_model else clf.llm_model
        provider = self._dispatcher._pool.get(provider_name)
        if provider is None:
            logger.warning(
                "classifier provider %r for llm_model %r not in pool; "
                "falling back to keyword classification",
                provider_name, clf.llm_model,
            )
            return None
        return LLMClassifier(provider, api_model, fallback=self._classifier)

    def _build_retrieval(self) -> Optional[RetrievalProvider]:
        """Build the configured retrieval provider when retrieval.enabled,
        else None (no construction, no call)."""
        if not self._config.retrieval.enabled:
            return None
        if not self._config.retrieval.tags:
            logger.warning(
                "retrieval enabled but no tags configured; augmentation is a no-op"
            )
        name = self._config.retrieval.provider
        if name not in ("noop", "websearch"):
            logger.warning(
                "unknown retrieval provider %r; falling back to noop (no augmentation)",
                name,
            )
        return build_retrieval_provider(name)

    async def _classify(self, prompt: str) -> tuple[str, float]:
        """Classify via the LLM classifier when wired (it falls back to
        keyword internally on any failure), else the sync keyword path."""
        if self._llm_classifier is not None:
            return await self._llm_classifier.classify(prompt)
        return self._classifier.classify(prompt)

    def _build_verifier_synth(self) -> VerifierSynthesizer:
        registry = VerifierRegistry()
        registry.register(CodeVerifier(
            execute=self._config.synthesis.code_execute,
            execute_timeout=self._config.synthesis.code_execute_timeout,
            sandbox=self._config.synthesis.code_execute_sandbox,
        ))
        registry.register(MathVerifier())

        judge_key = self._config.synthesis.judge_model
        if judge_key:
            judges = [
                self._make_judge(judge_key, tag, with_neutral=True)
                for tag in _JUDGE_TAGS
            ]
            if all(j is not None for j in judges):
                for j in judges:
                    registry.register(j)
            else:
                logger.warning(
                    "judge provider for %r not in pool; skipping JudgeVerifier",
                    judge_key,
                )

        return VerifierSynthesizer(
            registry,
            abstention_threshold=self._config.synthesis.abstention_threshold,
            abstention_thresholds=self._config.synthesis.abstention_thresholds,
        )

    def _make_judge(
        self,
        model_key: str,
        tag: str,
        *,
        with_neutral: bool = False,
        swap_order: Optional[bool] = None,
    ) -> Optional[JudgeVerifier]:
        """Construct a JudgeVerifier bound to `model_key` for `tag`, or None
        when that model's provider isn't in the pool. Shared by the registry
        build and by the neutral-judge swap that defeats verify-step
        self-preference (a judge grading its own escalated/refined answer).

        `with_neutral=True` (PRIMARY scoring path only) wires a
        `neutral_factory` so that, at scoring time, a judge whose own model is
        among the candidates delegates to a neutral judge bound to a
        non-candidate model. The neutral judge is built with
        `with_neutral=False`, which guarantees termination: it carries no
        factory, and `_pick_arbiter` only ever returns a model that is NOT a
        candidate, so it could never re-trigger the self-preference swap.

        `swap_order` (position-bias de-biasing) defaults to config ONLY on the
        primary path (`with_neutral=True`); verify-step callers
        (`_score_neutral`) get a single pass, leaving the Stage-2 verify
        behavior untouched. The neutral judge composes swap-order: it is built
        with the same `swap_order` as the judge it stands in for."""
        entry = self._config.models.get(model_key)
        provider_name = entry.provider if entry else "ollama"
        api_model = entry.api_model if entry and entry.api_model else model_key
        provider = self._dispatcher._pool.get(provider_name)
        if provider is None:
            return None

        if swap_order is None:
            swap_order = (
                self._config.synthesis.judge_swap_order if with_neutral else False
            )

        neutral_factory = None
        if with_neutral:

            def neutral_factory(
                candidate_models: set[str],
                judge_key: str = model_key,
                judge_tag: str = tag,
                swap: bool = swap_order,
            ) -> Optional[JudgeVerifier]:
                # Reuse the production self-preference guard: pick an available
                # model that is NOT a candidate. Returns the configured key when
                # the judge isn't a candidate (no swap) and None when no neutral
                # model exists — both mean "don't swap".
                alt = self._pick_arbiter(judge_key, candidate_models)
                if alt is None or alt == judge_key:
                    return None
                # The neutral judge composes swap-order but carries NO factory
                # (with_neutral defaults False) → recursion cannot occur, since
                # _pick_arbiter only ever returns a non-candidate model.
                return self._make_judge(alt, judge_tag, swap_order=swap)

        return JudgeVerifier(
            provider,
            api_model,
            tag=tag,
            model_key=model_key,
            neutral_factory=neutral_factory,
            swap_order=swap_order,
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

        tag, confidence = await self._classify(prompt)
        self._events.emit(PromptClassified(tag=tag, confidence=confidence, prompt=prompt))

        # Retrieval augmentation: classification used the ORIGINAL prompt; only
        # the dispatched prompt is grounded with retrieved context. Failures are
        # non-fatal (providers swallow their own errors and return ""), and an
        # empty context leaves the prompt byte-for-byte unchanged.
        dispatch_prompt = await self._augment(prompt, tag)

        if force_parallel or confidence < self._config.thresholds.single_confidence:
            return await self._parallel(dispatch_prompt, tag, system=system)
        return await self._single(dispatch_prompt, tag, system=system)

    async def _augment(self, prompt: str, tag: str) -> str:
        """Prepend retrieved context to `prompt` when retrieval is enabled and
        `tag` is configured for augmentation. Returns the prompt unchanged when
        retrieval is off, the tag isn't configured, or the context is empty."""
        if self._retrieval is None or tag not in self._config.retrieval.tags:
            return prompt
        context = await self._retrieval.retrieve(
            prompt, self._config.retrieval.max_chars
        )
        if not context:
            return prompt
        logger.debug("retrieval augmented prompt for tag=%r (%d chars)", tag, len(context))
        return f"{context}\n\n{prompt}"

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
        # Feed verifier scores back into the bandit's posteriors. The N
        # candidates a model emits in one round are correlated samples of the
        # SAME generation, not N independent Bernoulli trials, so they collapse
        # to ONE aggregated observation per model (their mean score). More
        # samples sharpen that observation's score ESTIMATE; they do NOT
        # multiply the observation count.
        self._update_bandit(tag, result)

        # The synthesis event is emitted ONCE, reflecting the FINAL returned
        # outcome — AFTER any escalation/refinement post-pass. The calibration
        # collector reads the LAST event, so it MUST match what _parallel
        # actually returns: a verifier abstention that escalation then rescues
        # has to be recorded as ANSWERED with the escalated score, not as an
        # abstention. Emitting before the post-passes (the old behavior) biased
        # `fleet --calibrate` by undercounting coverage on exactly the tags
        # where escalation does the most work.
        def _emit_final(
            *, winner_model: Optional[str], winner_score: Optional[float],
            abstain: bool,
        ) -> None:
            self._events.emit(ResponseSynthesized(
                tag=tag, mode="verifier",
                winner_model=winner_model, winner_score=winner_score,
                abstain=abstain,
            ))

        # Disagreement escalation: when verifier abstains OR winner score is
        # weak, ask a stronger model to arbitrate using all candidates as context.
        if self._should_escalate(result):
            escalated = await self._escalate(prompt, result, tag, system=system)
            if escalated is not None:
                esc_text, esc_score, esc_model = escalated
                _emit_final(
                    winner_model=esc_model, winner_score=esc_score,
                    abstain=False,
                )
                return esc_text

        if result.abstain:
            _emit_final(winner_model=None, winner_score=None, abstain=True)
            return self._format_abstention(result, tag)

        winner_text = result.winner_text or ERROR_ALL_MODELS_FAILED

        # Refinement: critique → revise pass on the winning answer.
        if self._config.refinement.enabled and result.winner is not None:
            refined = await self._refine(
                prompt, winner_text, tag, result,
                winner_model=result.winner.model, system=system,
            )
            if refined:
                ref_text, ref_score = refined
                _emit_final(
                    winner_model=result.winner.model, winner_score=ref_score,
                    abstain=False,
                )
                return ref_text

        _emit_final(
            winner_model=result.winner.model if result.winner else None,
            winner_score=result.winner.score if result.winner else None,
            abstain=False,
        )
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

    def _make_prior_provider(self):
        """Build a (tag, model) -> (alpha, beta) prior seeder from configured
        model priorities, or None when seeding is disabled. Used by the bandit
        ONLY when an arm is first created, so a fresh state file respects the
        human priority ordering instead of random-shuffling all arms.

        Mapping: alpha = 1 + strength/priority, beta = 1. Priority 1 (highest)
        gets the highest prior mean; larger priority numbers tend toward
        uniform. Total prior mass stays low (alpha < 2) so a few real
        observations dominate quickly. Unknown models fall back to the
        ModelEntry default priority (99) ≈ uniform."""
        strength = self._config.bandit.priority_prior_strength
        if strength <= 0.0:
            return None
        models = self._config.models

        def provider(tag: str, model: str) -> tuple[float, float]:
            entry = models.get(model)
            priority = entry.priority if entry else 99
            if priority < 1:
                priority = 1  # guard against div-by-zero / negative configs
            return 1.0 + strength / priority, 1.0

        return provider

    def _update_bandit(self, tag: str, result: VerificationResult) -> None:
        """Push verifier scores into the bandit posteriors. Skipped when:
        - bandit disabled, OR
        - the verifier produced no scored candidates, OR
        - the verifier marked its scores unreliable (judge crashed,
          returned empty, output unparseable, or only one candidate).
          Updating from those would poison posteriors with all-0.5 noise
          and prevent the bandit from ever discriminating between models.

        Candidates are aggregated to ONE observation per model per round: the
        N samples a model emits are correlated draws from the same generation,
        so feeding each as an independent Bernoulli trial would make posteriors
        overconfident by ~sqrt(N) and let one lucky round lock in exploitation.
        We average a model's candidate scores and apply a single update."""
        if self._bandit is None or not result.all_scored or not result.scores_reliable:
            return
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for c in result.all_scored:
            sums[c.model] = sums.get(c.model, 0.0) + c.score
            counts[c.model] = counts.get(c.model, 0) + 1
        for model, total in sums.items():
            self._bandit.update(tag, model, total / counts[model])

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
        self, prompt: str, result, tag: str, system: str | None = None
    ) -> Optional[tuple[str, float, str]]:
        """Returns ``(escalated_answer, esc_score, arbiter_model)`` on a verified
        rescue, else None. The score + model are threaded out so ``_parallel``
        can emit a FINAL synthesis event reflecting the escalated outcome."""
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
            f"--- Candidate {_candidate_label(i)} (model={c.model}, score={c.score:.2f}) ---\n{c.text}"
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
        escalated_clean = strip_thinking(escalated) or None
        if escalated_clean is None:
            return None
        # Closed-loop verification: never return an unverified arbiter answer.
        # Score it with the SAME tag verifier against the original candidates;
        # accept only when it clears the abstention threshold AND verifies at
        # least as well as the best original. Otherwise return None so the
        # caller falls through to the abstention path.
        verified, esc_score = await self._escalation_verified(
            prompt, tag, escalated_clean, model, result
        )
        if not verified:
            logger.info(
                "escalated answer from %r did not verify; abstaining instead",
                model,
            )
            return None
        return escalated_clean, esc_score, model

    async def _escalation_verified(
        self, prompt: str, tag: str, answer: str, arbiter_model: str,
        result: VerificationResult,
    ) -> tuple[bool, Optional[float]]:
        """Returns ``(verified, esc_score)``. ``verified`` is True iff `answer`
        scores >= the abstention threshold AND >= the best original candidate
        under the tag verifier; ``esc_score`` is the arbiter answer's verified
        score (None when it couldn't be scored). The score is threaded out so
        the caller can record the FINAL escalated outcome in the synthesis
        event (calibration must see the rescued score, not the pre-escalation
        abstention). The arbiter answer is tagged with sample_idx=-1 so it's
        identifiable after re-scoring.

        Neutral-judge discipline (self-preference guard): if the tag verifier is
        the LLM judge and the judge model IS the arbiter that wrote `answer`, we
        swap in a neutral judge — a judge grading its own output over-rates it
        (the same bias `_pick_arbiter` guards at production time). When no
        neutral model exists, we conservatively return False (abstain) rather
        than trust a self-graded score.
        """
        verify_cands = [
            Candidate(model=arbiter_model, sample_idx=-1, text=answer)
        ] + list(result.all_scored)
        vres, ok = await self._score_neutral(prompt, verify_cands, tag, arbiter_model)
        if not ok:
            return False, None
        esc_score = next(
            (c.score for c in vres.all_scored if c.sample_idx == -1), None
        )
        if esc_score is None:
            return False, None
        # NOTE: esc_score comes from a re-score of the pool that INCLUDES the
        # arbiter's own answer (a mild self-confirmation advantage), while
        # best_original is the pre-escalation score the originals earned WITHOUT
        # the arbiter in the set — the two aren't on identical footing. We
        # accept that mild asymmetry: requiring esc_score >= best_original is
        # already a conservative bar, and re-scoring the originals in isolation
        # would discard the arbiter's relative ranking signal entirely.
        best_original = max((c.score for c in result.all_scored), default=0.0)
        # Use the SAME per-tag cut-point the synthesizer would apply to this
        # tag — an escalated answer must clear the bar for the tag being
        # escalated, not a global default that ignores the tag's score scale.
        threshold = self._verifier_synth.abstention_threshold_for(tag)
        verified = esc_score >= threshold and esc_score >= best_original
        return verified, esc_score

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

    def _is_heuristic_tag(self, tag: str) -> bool:
        """True when the tag is scored by the HeuristicVerifier — whose
        pairwise-similarity consensus is SYMMETRIC and therefore order-dependent
        for a 2-candidate compare. Refinement accepts on such tags must not
        trust an insertion-order 'winner'."""
        return isinstance(self._verifier_synth.verifier_for(tag), HeuristicVerifier)

    def _judge_model_for_tag(self, tag: str) -> Optional[str]:
        """Registry key of the judge model iff the tag is actually scored by a
        JudgeVerifier (else None). Used to detect verify-step self-preference."""
        if isinstance(self._verifier_synth.verifier_for(tag), JudgeVerifier):
            return self._config.synthesis.judge_model or None
        return None

    async def _score_neutral(
        self, prompt: str, candidates: list[Candidate], tag: str,
        producer_model: str,
    ) -> tuple[Optional[VerificationResult], bool]:
        """Score `candidates` with the tag verifier, but when that verifier is
        the LLM judge AND the judge model is `producer_model` (the model whose
        answer is under test), swap in a neutral judge bound to a different
        available model. Returns (result, ok); ok=False means the judge equals
        the producer and no neutral model exists — the caller must fall back
        conservatively rather than trust a self-graded score."""
        judge_key = self._judge_model_for_tag(tag)
        if judge_key is not None and judge_key == producer_model:
            neutral = self._pick_arbiter(judge_key, {producer_model})
            if neutral is None:
                return None, False
            neutral_judge = self._make_judge(neutral, tag)
            if neutral_judge is None:
                return None, False
            return await neutral_judge.aggregate(prompt, candidates), True
        result = await self._verifier_synth.score_candidates(prompt, candidates, tag)
        return result, True

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
        self, prompt: str, draft: str, tag: str, result: VerificationResult,
        winner_model: Optional[str] = None, system: str | None = None,
    ) -> Optional[tuple[str, float]]:
        """Returns ``(revised_answer, revised_score)`` when an accepted rewrite
        improves on the draft, else None. The score is threaded out so
        ``_parallel`` records the post-refinement score in the FINAL event."""
        # Don't let an unverified rewrite overwrite a strong numeric majority:
        # when the math vote already agrees strongly, skip refinement entirely.
        # A verified arithmetic answer shouldn't be "improved" by prose.
        if (
            tag == "math"
            and result.winner is not None
            and result.winner.score >= 0.6
        ):
            logger.info(
                "refinement skipped for math: majority agreement %.2f is strong",
                result.winner.score,
            )
            return None
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
        revised_clean = strip_thinking(revised) or None
        if revised_clean is None:
            return None
        # Closed-loop verification: only accept the rewrite if the tag verifier
        # judges it better than the original winner. On tie or worse, keep the
        # original (return None) — a single unverified rewrite must never
        # silently overwrite a verified answer.
        improved, revised_score = await self._refinement_improves(
            prompt, tag, revised_clean, draft, result,
            producer_model=critique_model,
        )
        if improved:
            return revised_clean, revised_score
        return None

    async def _refinement_improves(
        self, prompt: str, tag: str, revised: str, original: str,
        result: VerificationResult, producer_model: Optional[str] = None,
    ) -> tuple[bool, Optional[float]]:
        """Returns ``(improves, revised_score)``. ``improves`` is True iff
        `revised` is judged better than `original` by the tag verifier;
        ``revised_score`` is the revised answer's verified score (None when not
        accepted / not scorable), threaded out so ``_parallel`` can record the
        post-refinement score in the FINAL synthesis event. Synthetic, DISTINCT
        model names keep the heuristic/judge
        verifier from collapsing the two texts into one model bucket; sentinel
        sample_idx values (-1 revised, -2 original) make them identifiable
        after re-scoring.

        Two regimes, because verifiers differ in how trustworthy their pairwise
        signal is:

        - Heuristic-backed tags (general/reasoning with no judge configured):
          the verifier's consensus score is pairwise-SYMMETRIC, so a 2-candidate
          compare always ties and the 'winner' is decided by insertion order —
          pure noise. We re-score with the order SWAPPED and accept only when
          `revised` wins BOTH orderings, i.e. the tag's non-symmetric tiebreak
          (length / brevity / diversity) genuinely prefers it independent of
          order. A symmetric tie therefore resolves to KEEPING THE ORIGINAL.

        - Discriminative tags (judge / executable / math): the verifier gives a
          real asymmetric score, so a single strict `revised > original` is
          enough. For judge tags we additionally apply the neutral-judge guard
          (`_score_neutral`) so the critic can't grade its own rewrite.
        """
        revised_cand = Candidate(model="__refined__", sample_idx=-1, text=revised)
        original_cand = Candidate(model="__original__", sample_idx=-2, text=original)

        if self._is_heuristic_tag(tag):
            fwd = await self._verifier_synth.score_candidates(
                prompt, [revised_cand, original_cand], tag
            )
            rev = self._score_for(fwd, -1)
            orig = self._score_for(fwd, -2)
            if not (rev > orig):
                return False, None
            # Swap order: an order-dependent symmetric tie flips here, so only a
            # genuine non-symmetric preference survives both passes.
            swp = await self._verifier_synth.score_candidates(
                prompt, [original_cand, revised_cand], tag
            )
            rev_sw = self._score_for(swp, -1)
            orig_sw = self._score_for(swp, -2)
            if rev_sw > orig_sw:
                return True, rev
            return False, None

        cands = [revised_cand, original_cand]
        if tag == "math":
            # Include the original sample pool so the majority vote is
            # meaningful — two lone candidates can't out-vote a 7-sample round.
            cands.extend(result.all_scored)
        vres, ok = await self._score_neutral(prompt, cands, tag, producer_model or "")
        if not ok:
            # Judge == critic and no neutral judge available → don't trust a
            # self-graded score; keep the original.
            return False, None
        rev_score = self._score_for(vres, -1)
        if rev_score > self._score_for(vres, -2):
            return True, rev_score
        return False, None

    @staticmethod
    def _score_for(vres: VerificationResult, sample_idx: int) -> float:
        return next(
            (c.score for c in vres.all_scored if c.sample_idx == sample_idx), 0.0
        )
