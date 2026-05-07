# Aditya Kapoor — Macro Placement Submission

Original placer attempts for the Partcl/HRT Macro Placement Challenge.

## Status

Work in progress. Best result so far: **avg 1.4634** on 3-benchmark subset
(ibm01/07/14) — see `run_3bench.py`. Full 17-benchmark score is pending.

## Files

- `placer_v4.py` — current best placer. Analytical global placement with:
  - LSE wirelength on pin offsets
  - Multi-scale Gaussian density loss with Coulomb (1/r) coupling at coarse scales
  - Pin-density congestion proxy
  - Soft-macro density floor (precomputed)
  - Boundary penalty + Adam optimizer
- `placer.py` — v1 baseline (analytical + radial-search legalize + SA refine)
- `placer_v2.py`, `placer_v3.py` — intermediate experiments
- `placer_multi.py`, `placer_multi_v2.py` — multi-start wrappers
- `incremental_proxy.py` — fast incremental TILOS proxy (for SA inner loops)
- `run_3bench.py` — fast 3-benchmark test harness for hypothesis iteration
- `sweep_3bench.py` — hyperparameter sweep harness

## Running

```bash
uv run python submissions/aditya/run_3bench.py        # 3-bench subset (~25s)
uv run python submissions/aditya/sweep_3bench.py      # hyperparameter sweep
```

## Notes on attribution

All code in this directory is original work. References to published
techniques (LSE wirelength, ePlace electrostatic potentials, RUDY congestion)
are documented in inline comments where used.

Research notes (private, not committed) live in `.claude/research/`.
