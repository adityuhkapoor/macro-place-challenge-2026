"""
Hypothesis sweep on 3-bench. Add configs to CONFIGS, run.
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
from placer_v4 import AdityaPlacerV4

BENCHES = ["ibm01", "ibm07", "ibm14"]
REPLACE = {"ibm01": 0.9976, "ibm07": 1.4633, "ibm14": 1.5436}
AVG_REPLACE = sum(REPLACE.values()) / len(REPLACE)


def run_one(name: str, kwargs: dict):
    placer = AdityaPlacerV4(**kwargs)
    total = 0.0
    total_t = 0.0
    rows = []
    for bn in BENCHES:
        bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bn}")
        t0 = time.time()
        pos = placer.place(bm)
        t = time.time() - t0
        c = compute_proxy_cost(pos, bm, plc)
        total += c["proxy_cost"]
        total_t += t
        rows.append((bn, c["proxy_cost"], c["wirelength_cost"], c["density_cost"], c["congestion_cost"], t))
    avg = total / len(BENCHES)
    diff = (avg - AVG_REPLACE) / AVG_REPLACE * 100
    print(f"=== {name} ===  AVG: {avg:.4f}  ({diff:+.1f}%)  [{total_t:.1f}s]")
    for bn, p, w, d, c_, tt in rows:
        rd = (p - REPLACE[bn]) / REPLACE[bn] * 100
        print(f"  {bn}: {p:.4f}  (wl={w:.3f} den={d:.3f} cong={c_:.3f})  ({rd:+.1f}%)  [{tt:.1f}s]")
    return avg


if __name__ == "__main__":
    # Each entry: (name, kwargs)
    # SGLD escape-phase noise sweep (cheap saddle escape)
    BASE = {}
    configs = [
        ("baseline_no_sgld", BASE),
        ("sgld_001", {**BASE, "sgld_noise": 0.001}),
        ("sgld_005", {**BASE, "sgld_noise": 0.005}),
        ("sgld_01", {**BASE, "sgld_noise": 0.01}),
        ("sgld_02", {**BASE, "sgld_noise": 0.02}),
        ("sgld_05", {**BASE, "sgld_noise": 0.05}),
        # Different windows
        ("sgld_01_early", {**BASE, "sgld_noise": 0.01, "sgld_start_frac": 0.2, "sgld_end_frac": 0.7}),
        ("sgld_01_late", {**BASE, "sgld_noise": 0.01, "sgld_start_frac": 0.7, "sgld_end_frac": 0.95}),
    ]

    results = {}
    for name, kw in configs:
        try:
            results[name] = run_one(name, kw)
        except Exception as e:
            print(f"=== {name} ===  FAILED: {e}")
            results[name] = float("inf")

    print("\n========== SUMMARY ==========")
    sorted_r = sorted(results.items(), key=lambda x: x[1])
    for name, avg in sorted_r:
        diff = (avg - AVG_REPLACE) / AVG_REPLACE * 100 if avg != float("inf") else 0.0
        print(f"  {avg:.4f}  ({diff:+.1f}%)  {name}")
