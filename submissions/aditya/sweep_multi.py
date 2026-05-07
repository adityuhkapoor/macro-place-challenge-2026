"""
Atomic multi-start test on 3-bench. Compare K=1 (no multi) vs K=3, K=5
with various jitter and Sobol toggle.
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
from placer_multi_v2 import AdityaPlacerMultiV2

BENCHES = ["ibm01", "ibm07", "ibm14"]
REPLACE = {"ibm01": 0.9976, "ibm07": 1.4633, "ibm14": 1.5436}
AVG_REPLACE = sum(REPLACE.values()) / len(REPLACE)


def run(name: str, placer):
    total = 0.0
    total_t = 0.0
    for bn in BENCHES:
        bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bn}")
        t0 = time.time()
        pos = placer.place(bm)
        t = time.time() - t0
        c = compute_proxy_cost(pos, bm, plc)
        total += c["proxy_cost"]
        total_t += t
    avg = total / len(BENCHES)
    diff = (avg - AVG_REPLACE) / AVG_REPLACE * 100
    print(f"=== {name} ===  AVG: {avg:.4f}  ({diff:+.1f}%)  [{total_t:.1f}s]")
    return avg


if __name__ == "__main__":
    configs = [
        ("k1_solo", AdityaPlacerV4()),
        ("k3_jitter04", AdityaPlacerMultiV2(k=3, jitter_scale=0.04, include_given=True)),
        ("k3_jitter10", AdityaPlacerMultiV2(k=3, jitter_scale=0.10, include_given=True)),
        ("k3_jitter02", AdityaPlacerMultiV2(k=3, jitter_scale=0.02, include_given=True)),
        ("k3_sobol", AdityaPlacerMultiV2(k=3, jitter_scale=0.04, use_sobol=True, include_given=True)),
        ("k5_jitter04", AdityaPlacerMultiV2(k=5, jitter_scale=0.04, include_given=True)),
    ]
    results = {}
    for name, p in configs:
        try:
            results[name] = run(name, p)
        except Exception as e:
            print(f"=== {name} ===  FAILED: {e}")
            results[name] = float("inf")

    print("\n========== SUMMARY ==========")
    for name, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {v:.4f}  {name}")
