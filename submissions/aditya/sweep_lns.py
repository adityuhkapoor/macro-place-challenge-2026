"""
Atomic LNS sweep on ibm01. Each config varies ONE parameter from a fixed BASE.
Fast: only ibm01 (~10s placer + ~30s LNS = 40s/config).

Hypotheses to test (one variable at a time):
  H1: episodes count (8 vs 30 vs 60) — does more episodes give linear gain?
  H2: subset size N (5 vs 10 vs 20) — sweet spot?
  H3: sa_steps per episode (400 vs 800 vs 2000)
  H4: greedy vs SA acceptance (we saw greedy slightly better)
  H5: step size scale (0.005 vs 0.02 vs 0.05 of canvas)
  H6: subset selection strategy (worst-only vs shaw-only vs mix)
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
from lns_refine import lns_refine

BENCH = "ibm01"


def run(name: str, lns_kwargs: dict):
    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{BENCH}")
    placer = AdityaPlacerV4()
    pos = placer.place(bm)
    base = compute_proxy_cost(pos, bm, plc)["proxy_cost"]
    if lns_kwargs is None:
        print(f"=== {name} === proxy={base:.4f} (baseline)")
        return base
    t0 = time.time()
    refined = lns_refine(pos, bm, plc, verbose=False, **lns_kwargs)
    elapsed = time.time() - t0
    new = compute_proxy_cost(refined, bm, plc)["proxy_cost"]
    diff = (new - base) / base * 100
    print(f"=== {name} === pre={base:.4f} post={new:.4f} ({diff:+.2f}%) [{elapsed:.0f}s LNS]")
    return new


if __name__ == "__main__":
    # BASE: greedy, N=10, 800 sa_steps, 8 episodes, step_frac=0.02
    BASE = {
        "sa_mode": "greedy", "subset_size": 10, "sa_steps": 800,
        "n_episodes": 8, "step_init_frac": 0.02, "step_end_frac": 0.0002,
        "time_budget": 120, "seed": 0,
    }
    print(f"BASE: {BASE}\n")

    run("baseline_no_lns", None)

    # Push episodes further, combine best knobs
    run("ep_100", {**BASE, "n_episodes": 100, "time_budget": 600})
    run("ep_200", {**BASE, "n_episodes": 200, "time_budget": 1200})
    # Combine: 60 ep + N=20
    run("ep60_N20", {**BASE, "n_episodes": 60, "time_budget": 600, "subset_size": 20})
    # Combine: 100 ep + N=20 + steps_2000
    run("ep100_N20_st2k", {
        **BASE, "n_episodes": 100, "time_budget": 1200,
        "subset_size": 20, "sa_steps": 2000,
    })
    # SA mode revisit at high episode count
    run("ep_60_sa", {**BASE, "n_episodes": 60, "time_budget": 600, "sa_mode": "sa"})
