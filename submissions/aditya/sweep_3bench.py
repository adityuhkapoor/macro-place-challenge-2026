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
    # NEW BASELINE: swap_iters=0
    BASE = {"swap_iters": 0}
    configs = [
        ("baseline_swap0", BASE),
        # Boundary weight
        ("bd_50", {**BASE, "bd_w": 50.0}),
        ("bd_200", {**BASE, "bd_w": 200.0}),
        ("bd_500", {**BASE, "bd_w": 500.0}),
        # Density weight
        ("den_3", {**BASE, "den_w": 3.0}),
        ("den_8", {**BASE, "den_w": 8.0}),
        ("den_10", {**BASE, "den_w": 10.0}),
        ("den_15", {**BASE, "den_w": 15.0}),
        # Cong weight
        ("cong_05", {**BASE, "cong_w": 0.5}),
        ("cong_15", {**BASE, "cong_w": 1.5}),
        ("cong_2", {**BASE, "cong_w": 2.0}),
        # ov ramp
        ("ov_wide", {**BASE, "ov_start": 5.0, "ov_end": 5000.0}),
        ("ov_narrow", {**BASE, "ov_start": 50.0, "ov_end": 1000.0}),
        ("ov_aggressive", {**BASE, "ov_start": 10.0, "ov_end": 10000.0}),
        # More iters - did badly with swap, retest with swap=0
        ("iters_1200", {**BASE, "global_iters": 1200}),
        ("iters_500", {**BASE, "global_iters": 500}),
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
