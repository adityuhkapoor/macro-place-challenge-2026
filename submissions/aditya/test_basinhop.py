"""Atomic basin hopping test on ibm01."""

import sys, time
from pathlib import Path
import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placer_v4 import AdityaPlacerV4
from lns_refine import basin_hop


def run(name, **bh_kwargs):
    bm, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    placer = AdityaPlacerV4()
    pos = placer.place(bm)
    base = compute_proxy_cost(pos, bm, plc)["proxy_cost"]
    if not bh_kwargs:
        print(f"=== {name} === proxy={base:.4f} (baseline)")
        return base
    t0 = time.time()
    refined = basin_hop(pos, bm, plc, verbose=True, **bh_kwargs)
    t = time.time() - t0
    new = compute_proxy_cost(refined, bm, plc)["proxy_cost"]
    diff = (new - base) / base * 100
    print(f"=== {name} === pre={base:.4f} post={new:.4f} ({diff:+.2f}%) [{t:.0f}s]")
    return new


if __name__ == "__main__":
    run("baseline_no_bh")
    run("bh_100ep_p10", n_episodes=100, perturb_frac=0.10, relax_steps=20, time_budget=120)
    run("bh_100ep_p05", n_episodes=100, perturb_frac=0.05, relax_steps=20, time_budget=120)
    run("bh_100ep_p20", n_episodes=100, perturb_frac=0.20, relax_steps=20, time_budget=120)
    run("bh_200ep", n_episodes=200, perturb_frac=0.10, relax_steps=20, time_budget=240)
