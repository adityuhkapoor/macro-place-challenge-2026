"""Atomic spectral-init test on 3-bench (LNS off — isolate init effect)."""

import sys, time
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


def run_cfg(label, **kwargs):
    print(f"\n=== {label} ===")
    placer = AdityaPlacerV4(lns_episodes=0, **kwargs)
    total = 0.0
    t0 = time.time()
    for name in BENCHES:
        bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
        pos = placer.place(bm)
        c = compute_proxy_cost(pos, bm, plc)["proxy_cost"]
        total += c
        print(f"  {name}: {c:.4f}")
    avg = total / len(BENCHES)
    print(f"  AVG: {avg:.4f}   [{time.time()-t0:.0f}s]")
    return avg


if __name__ == "__main__":
    base = run_cfg("baseline given init", init_mode="given")
    sp1 = run_cfg("spectral aw=5  jit=0.02", init_mode="spectral",
                  spectral_anchor_w=5.0, spectral_jitter_frac=0.02)
    sp2 = run_cfg("spectral aw=1  jit=0.02", init_mode="spectral",
                  spectral_anchor_w=1.0, spectral_jitter_frac=0.02)
    sp3 = run_cfg("spectral aw=10 jit=0.02", init_mode="spectral",
                  spectral_anchor_w=10.0, spectral_jitter_frac=0.02)
    sp4 = run_cfg("spectral aw=5  jit=0.0",  init_mode="spectral",
                  spectral_anchor_w=5.0, spectral_jitter_frac=0.0)
    print(f"\n=== SUMMARY ===")
    print(f"  baseline given:     {base:.4f}")
    print(f"  spectral aw=5  j=2: {sp1:.4f}  ({(sp1-base)/base*100:+.2f}%)")
    print(f"  spectral aw=1  j=2: {sp2:.4f}  ({(sp2-base)/base*100:+.2f}%)")
    print(f"  spectral aw=10 j=2: {sp3:.4f}  ({(sp3-base)/base*100:+.2f}%)")
    print(f"  spectral aw=5  j=0: {sp4:.4f}  ({(sp4-base)/base*100:+.2f}%)")
