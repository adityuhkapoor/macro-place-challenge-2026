"""
Multi-start wrapper for AdityaPlacerV4.

Runs K parallel placements with different seeds + perturbed init,
returns the best by TILOS proxy cost.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from placer_v4 import AdityaPlacerV4


class AdityaPlacerMultiV2:
    """Multi-start: K AdityaPlacerV4 runs with different seeds, return best."""

    def __init__(self, k: int = 3, seeds=None, base_seed: int = 42,
                 jitter_scale: float = 0.04,
                 include_given: bool = True,  # always include unjittered initial.plc
                 use_sobol: bool = False,
                 verbose: bool = False, **placer_kwargs):
        self.k = k
        self.seeds = seeds if seeds is not None else [base_seed + i for i in range(k)]
        self.jitter_scale = jitter_scale  # fraction of canvas
        self.include_given = include_given
        self.use_sobol = use_sobol
        self.verbose = verbose
        self.placer_kwargs = placer_kwargs

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        from macro_place.loader import load_benchmark_from_dir  # noqa
        # Load plc once (for cost evaluation)
        bench_path = f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}"
        _, plc = load_benchmark_from_dir(bench_path)

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        n_hard = benchmark.num_hard_macros

        best_pos = None
        best_cost = float("inf")
        best_seed = None

        orig_positions = benchmark.macro_positions.clone()
        movable_mask = benchmark.get_movable_mask()[:n_hard]

        # Build the list of starting position tensors
        starts = []  # list of (label, positions_tensor, seed)
        if self.include_given:
            starts.append(("given", orig_positions.clone(), self.seeds[0]))

        sobol = None
        if self.use_sobol:
            sobol = torch.quasirandom.SobolEngine(dimension=2 * n_hard, scramble=True,
                                                   seed=self.seeds[0])

        for i, seed in enumerate(self.seeds):
            if self.use_sobol and sobol is not None:
                # Sobol points in [0,1]^(2N), shift to [-jitter_scale, +jitter_scale]
                samples = sobol.draw(1).reshape(n_hard, 2)
                jitter = (samples - 0.5) * 2 * self.jitter_scale
            else:
                rng = np.random.RandomState(seed)
                jitter = torch.from_numpy(
                    rng.uniform(-self.jitter_scale, self.jitter_scale,
                                size=(n_hard, 2))
                ).float()
            jitter[:, 0] *= cw
            jitter[:, 1] *= ch
            jitter[~movable_mask] = 0.0

            perturbed = orig_positions.clone()
            perturbed[:n_hard] = orig_positions[:n_hard] + jitter
            perturbed[:n_hard, 0].clamp_(0, cw)
            perturbed[:n_hard, 1].clamp_(0, ch)
            starts.append((f"seed{seed}", perturbed, seed))

        for label, start_pos, seed in starts:
            benchmark.macro_positions = start_pos

            placer = AdityaPlacerV4(seed=seed, **self.placer_kwargs)
            pos = placer.place(benchmark)
            cost = compute_proxy_cost(pos, benchmark, plc)["proxy_cost"]
            if self.verbose:
                print(f"  [multi] {label} cost={cost:.4f}")
            if cost < best_cost:
                best_cost = cost
                best_pos = pos.clone()
                best_seed = label

        # Restore original positions on benchmark object
        benchmark.macro_positions = orig_positions

        if self.verbose:
            print(f"  [multi] best={best_seed} cost={best_cost:.4f}")
        return best_pos


if __name__ == "__main__":
    from run_3bench import run_3bench
    placer = AdityaPlacerMultiV2(k=3, jitter_scale=0.04)
    print("=== AdityaPlacerMultiV2 (k=3, jitter=0.04) ===")
    run_3bench(placer)
