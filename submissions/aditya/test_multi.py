"""Atomic multi-start test on 3-bench. LNS off to isolate basin-sampling effect.

Hypothesis: PT/basin-hop fail because gradient placer's basin from given init
is a true local optimum to single-macro perturbation. If multi-start (different
jittered inits + full gradient placer) finds different per-seed costs, the
landscape is multimodal and worth exploring. If all seeds give ~same cost, the
basin really is structurally optimal in the given init's neighborhood.
"""

import sys, time
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


def run_3bench(placer, label):
    print(f"\n=== {label} ===")
    total = 0.0
    t_total = 0.0
    for name in BENCHES:
        bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
        t0 = time.time()
        pos = placer.place(bm)
        t = time.time() - t0
        t_total += t
        c = compute_proxy_cost(pos, bm, plc)["proxy_cost"]
        total += c
        print(f"  {name}: {c:.4f}  [{t:.0f}s]")
    avg = total / len(BENCHES)
    print(f"  AVG: {avg:.4f}  [total {t_total:.0f}s]")
    return avg


if __name__ == "__main__":
    # Baseline: single-start, LNS off (isolates gradient placer)
    p_base = AdityaPlacerV4(lns_episodes=0)
    a_base = run_3bench(p_base, "baseline single-start LNS=0")

    # Multi-start k=3, jitter=0.04 — does jittered init give diversity?
    p_m3 = AdityaPlacerMultiV2(k=3, jitter_scale=0.04, verbose=True, lns_episodes=0)
    a_m3 = run_3bench(p_m3, "multi-start k=3 j=0.04 LNS=0")

    # Multi-start k=5, larger jitter — broader basin sampling
    p_m5l = AdityaPlacerMultiV2(k=5, jitter_scale=0.10, verbose=True, lns_episodes=0)
    a_m5l = run_3bench(p_m5l, "multi-start k=5 j=0.10 LNS=0")

    print(f"\n=== SUMMARY ===")
    print(f"  baseline single:  {a_base:.4f}")
    print(f"  multi k=3 j=0.04: {a_m3:.4f}  ({(a_m3-a_base)/a_base*100:+.2f}%)")
    print(f"  multi k=5 j=0.10: {a_m5l:.4f}  ({(a_m5l-a_base)/a_base*100:+.2f}%)")
