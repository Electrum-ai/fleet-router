"""Main orchestrator: classify → decide → dispatch → synthesize."""
from __future__ import annotations

from typing import Optional

from fleet.classifier import TaskClassifier
from fleet.config import Config
from fleet.dispatcher import EnsembleDispatcher
from fleet.registry import ModelRegistry
from fleet.synthesizer import Synthesizer


class FleetRouter:
    """Route prompts to the best model(s) and return the best response."""

    def __init__(self, config: Config | None = None):
        self._config = config or Config()
        self._classifier = TaskClassifier(self._config.classifier.embeddings_model)
        self._registry = ModelRegistry(self._config)
        self._dispatcher = EnsembleDispatcher(self._config)
        self._synthesizer = Synthesizer()

    async def ask(
        self,
        prompt: str,
        force_parallel: bool = False,
        force_model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> str | dict[str, str]:
        """Process a prompt and return the best response."""
        # Override: explicit model
        if force_model:
            responses = await self._dispatcher.run(prompt, [force_model], system=system)
            return responses.get(force_model) or "(model failed)"

        # Classify
        tag, confidence = self._classifier.classify(prompt)

        # Decide single vs parallel
        if force_parallel or confidence < self._config.thresholds.single_confidence:
            return await self._parallel(prompt, tag, system=system)
        return await self._single(prompt, tag, system=system)

    async def _single(
        self, prompt: str, tag: str, system: Optional[str] = None
    ) -> str | dict[str, str]:
        model = self._registry.get_best_for_tag(tag)
        if not model:
            return f"(no model available for tag: {tag})"

        responses = await self._dispatcher.run(prompt, [model], system=system)
        result = responses.get(model)
        if result is None:
            # Fallback: try any available model
            for fallback in self._registry.all_available():
                fb_responses = await self._dispatcher.run(prompt, [fallback], system=system)
                result = fb_responses.get(fallback)
                if result:
                    break
        return result or "(all models failed)"

    async def _parallel(
        self, prompt: str, tag: str, system: Optional[str] = None
    ) -> str | dict[str, str]:
        max_parallel = self._config.thresholds.max_parallel
        models = self._registry.models_for_tag(tag, top_n=max_parallel)
        if not models:
            models = self._registry.all_available()[:max_parallel]
        if not models:
            return "(no models available)"

        responses = await self._dispatcher.run(prompt, models, system=system)
        return self._synthesizer.pick(responses, task_tag=tag)
