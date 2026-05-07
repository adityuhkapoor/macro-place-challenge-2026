"""
Aditya Multi-start — K=4 analytical seeds, picked by TILOS proxy.

Per v-x-zhang's K=4 result: ensemble lifts 1.2792 → 1.2621 (~1.3% better).
Williyami's principle: "every late-stage local move has exhausted its reach…
the remaining headroom is diversity in the starting basin."

Pipeline:
  1. Run K AdityaPlacer instances with different random seeds.
  2. Score each via the TRUE TILOS proxy (load PlacementCost in this benchmark dir).
  3. Return the best by proxy.

The seeds also vary the random initial-position jitter, so each gets a
genuinely different starting basin.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from placer import AdityaPlacer  # noqa: E402


def _load_plc(name: str):
    """Load PlacementCost for the benchmark in this evaluation context."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    ibm_root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm_root.exists():
        _, plc = load_benchmark_from_dir(str(ibm_root))
        return plc
    ng45_map = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45_map.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


class AdityaPlacerMulti:
    """K-way multi-start, ranked by TRUE TILOS proxy."""

    def __init__(self, num_seeds: int = 4, base_seed: int = 42, verbose: bool = False):
        self.num_seeds = max(1, num_seeds)
        self.base_seed = base_seed
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark.name)
        if plc is None:
            # Fall back to single-shot
            return AdityaPlacer(seed=self.base_seed).place(benchmark)

        best_pos = None
        best_proxy = float("inf")

        for k in range(self.num_seeds):
            seed = self.base_seed + 17 * k
            t0 = time.time()
            placer = AdityaPlacer(seed=seed, verbose=False)
            pos = placer.place(benchmark)
            costs = compute_proxy_cost(pos, benchmark, plc)
            elapsed = time.time() - t0
            if self.verbose:
                print(f"  [seed {seed}] proxy={costs['proxy_cost']:.4f}  "
                      f"wl={costs['wirelength_cost']:.3f}  "
                      f"den={costs['density_cost']:.3f}  "
                      f"cong={costs['congestion_cost']:.3f}  "
                      f"ovl={costs['overlap_count']}  ({elapsed:.1f}s)")
            # Reject if any overlap
            if costs.get("overlap_count", 0) > 0:
                continue
            if costs["proxy_cost"] < best_proxy:
                best_proxy = costs["proxy_cost"]
                best_pos = pos.clone()

        if best_pos is None:
            # All seeds had overlaps — return last attempt
            return pos
        if self.verbose:
            print(f"  [multi] best proxy={best_proxy:.4f}")
        return best_pos
