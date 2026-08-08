"""Phase 0 of the Stage 3 (clip selection) rebuild: the measurement harness.

Mirrors eval/'s precedent from the framing rebuild — see that package's own
docstrings for the standing rule this follows: every fix ships with a metric
that moved, or it is not a fix. The existing Stage 3 test suites
(tests/test_viral_clip_finder.py, tests/test_deepseek_worker.py) pin plumbing
(schema shape, retries, chunking thresholds) against mocked LLM responses —
none of them exercise actual selection judgment. This package is what closes
that gap: real transcript fixtures, hand-labeled ground truth, and metrics
that answer "did this change make selection better," not "did the code run."
"""
