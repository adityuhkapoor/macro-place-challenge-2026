"""Atomic LNS test: run placer_v4 on ibm01, then layer LNS, compare."""

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


def run(name: str, lns_kwargs=None):
    bm, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    placer = AdityaPlacerV4()
    t0 = time.time()
    pos = placer.place(bm)
    t_place = time.time() - t0
    base_cost = compute_proxy_cost(pos, bm, plc)["proxy_cost"]
    print(f"=== {name}: pre-LNS ===  proxy={base_cost:.4f}  [{t_place:.1f}s]")

    if lns_kwargs is not None:
        t1 = time.time()
        refined = lns_refine(pos, bm, plc, verbose=True, **lns_kwargs)
        t_lns = time.time() - t1
        new_cost = compute_proxy_cost(refined, bm, plc)["proxy_cost"]
        improvement = (new_cost - base_cost) / base_cost * 100
        print(f"=== {name}: post-LNS ===  proxy={new_cost:.4f}  ({improvement:+.2f}%)  [{t_lns:.1f}s]")
        return base_cost, new_cost
    return base_cost, base_cost


if __name__ == "__main__":
    # Test 1: greedy hill-climb (strict improvement only on internal proxy)
    run("greedy_strict", lns_kwargs={
        "time_budget": 60, "n_episodes": 8, "subset_size": 10, "sa_steps": 800,
        "sa_mode": "greedy", "step_init_frac": 0.02, "step_end_frac": 0.0002,
    })
    # Test 2: SA but with smaller initial step
    run("sa_small_step", lns_kwargs={
        "time_budget": 60, "n_episodes": 8, "subset_size": 10, "sa_steps": 800,
        "sa_mode": "sa", "step_init_frac": 0.01, "step_end_frac": 0.0001,
    })
