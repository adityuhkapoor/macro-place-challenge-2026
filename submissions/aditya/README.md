# Aditya Kapoor — Macro Placement Submission

Original placer attempts for the Partcl/HRT Macro Placement Challenge.

## Status

Work in progress.

| Variant | 3-bench (ibm01/07/14) | All 17 IBM |
|--|--|--|
| v4 single-shot (current default) | 1.4063 | **1.4783** |
| v4 + LNS 30 episodes | 1.4008 | (estimated ~1.47) |
| v4 + LNS 60 episodes | 1.3991 | (not run) |
| v4 + K=3 multi-start (jitter=0.04) | 1.3965 | (not run) |
| RePlAce baseline | 1.3348 | 1.4578 |

Single-shot v4 beats RePlAce on 12/17 benchmarks; ibm02 and ibm10 are the
weakest. Multi-start gives -0.7% on the 3-bench but ~5-10× slower per
benchmark — full-17 verification pending.

## Files

- `placer_v4.py` — current best placer. Analytical global placement with:
  - WA (weighted-average) wirelength on pin offsets, gamma=0.01·canvas_scale
  - High-fanout net filter (degree > 30 dropped)
  - Multi-scale density loss with Coulomb (1/r) coupling at coarse scales (≤16)
  - Pin-density congestion proxy at 32×32 with low cong_w=0.05
  - Soft-macro density floor (precomputed)
  - Boundary penalty + Adam (lr_frac=0.003)
  - Post-process SA disabled (`swap_iters=0` — was net-negative)
- `placer_multi_v2.py` — K-multi-start wrapper (uses given init + perturbed seeds)
- `lns_refine.py` — Large Neighborhood Search post-process (subset SA via
  incremental proxy; episodes-validated against TILOS)
- `incremental_proxy.py` — fast in-loop proxy evaluator (~2000 moves/sec)
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
