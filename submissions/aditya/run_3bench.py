"""
Fast 3-benchmark test harness for hypothesis iteration.
Picks ibm01 (246 macros, dense small), ibm07 (335, medium), ibm14 (460, large).

Usage:
    uv run python submissions/aditya/run_3bench.py [placer_path]
"""

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

BENCHES = ["ibm01", "ibm07", "ibm14"]
# RePlAce ref (from README)
REPLACE = {"ibm01": 0.9976, "ibm07": 1.4633, "ibm14": 1.5436}


def run_3bench(placer):
    results = {}
    total = 0.0
    for name in BENCHES:
        bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
        t0 = time.time()
        pos = placer.place(bm)
        place_t = time.time() - t0
        costs = compute_proxy_cost(pos, bm, plc)
        total += costs["proxy_cost"]
        diff = (costs["proxy_cost"] - REPLACE[name]) / REPLACE[name] * 100
        flag = "✓" if costs["proxy_cost"] < REPLACE[name] else " "
        print(f"  {flag} {name}: {costs['proxy_cost']:.4f}  "
              f"(wl={costs['wirelength_cost']:.3f} "
              f"den={costs['density_cost']:.3f} "
              f"cong={costs['congestion_cost']:.3f}) "
              f"vs RePlAce {REPLACE[name]:.4f} ({diff:+.1f}%)  [{place_t:.1f}s]")
        results[name] = costs
    avg = total / len(BENCHES)
    avg_replace = sum(REPLACE.values()) / len(REPLACE)
    print(f"\n  AVG: {avg:.4f}  vs RePlAce {avg_replace:.4f}  ({(avg-avg_replace)/avg_replace*100:+.1f}%)")
    return results, avg


if __name__ == "__main__":
    from placer_v4 import AdityaPlacerV4
    placer = AdityaPlacerV4()
    print(f"=== AdityaPlacerV4 ===")
    run_3bench(placer)
