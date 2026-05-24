# Contributing to Fleet Router

Thanks for your interest in contributing. This project values quality over speed — every change should make the router measurably better at picking the best answer.

## How to contribute

1. **Open an issue first** for anything beyond a typo or trivial fix. Describe the problem or feature, include a minimal reproduction if it's a bug, and wait for a maintainer's okay before investing time in code.
2. **Fork and branch** — `git checkout -b feature/your-thing`.
3. **Write tests** — The test suite is the safety net. If you change routing logic, add or update tests in `tests/`. If you change the eval harness, add a fixture in `evals/fixtures/`.
4. **Run the full test suite** locally before opening a PR:
   ```bash
   pytest tests/
   ```
5. **Open a PR** with a clear description of what changed and why. Link to the issue you opened in step 1.

## What we look for in PRs

- **Correctness first.** A faster or cheaper router that picks worse answers is not a win.
- **Backwards compatibility.** The public API (`FleetRouter.ask`, CLI flags, config schema) should not break without a strong justification and a deprecation path.
- **Test coverage.** New features need tests. Bug fixes need regression tests.
- **Focused scope.** One PR per concern. Don't bundle unrelated changes.

## Eval-first workflow

If your change affects how the router picks answers, run the eval harness before and after:

```bash
# Save baseline before changes
fleet --eval evals/fixtures/hard/ --save-baseline before.json

# Make your changes
# ...

# Run eval against baseline
fleet --eval evals/fixtures/hard/ --baseline before.json
```

Exit code `3` means regression — that's a blocker.

## Questions?

Open a [discussion](https://github.com/Electrum-ai/fleet-router/discussions) if you want to talk through an idea before writing code.
