"""Atomic parallel-tempering test on 3-bench.

Compares plain LNS (current default) vs PT-LNS at roughly equal compute.
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
from lns_refine import parallel_tempering, lns_refine

BENCHES = ["ibm01", "ibm07", "ibm14"]


def get_baseline_positions(name):
    """Run v4 once with LNS off to get the post-legalize starting point."""
    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    placer = AdityaPlacerV4(lns_episodes=0)
    pos = placer.place(bm)
    return bm, plc, pos


def run_pt(name, **pt_kwargs):
    bm, plc, pos = get_baseline_positions(name)
    base = compute_proxy_cost(pos, bm, plc)["proxy_cost"]
    t0 = time.time()
    new = parallel_tempering(pos, bm, plc, verbose=True, **pt_kwargs)
    t = time.time() - t0
    new_cost = compute_proxy_cost(new, bm, plc)["proxy_cost"]
    diff = (new_cost - base) / base * 100
    print(f"  {name}: pre={base:.4f} post={new_cost:.4f} ({diff:+.2f}%) [{t:.0f}s]")
    return base, new_cost


def run_lns(name, **lns_kwargs):
    bm, plc, pos = get_baseline_positions(name)
    base = compute_proxy_cost(pos, bm, plc)["proxy_cost"]
    t0 = time.time()
    new = lns_refine(pos, bm, plc, verbose=False, **lns_kwargs)
    t = time.time() - t0
    new_cost = compute_proxy_cost(new, bm, plc)["proxy_cost"]
    diff = (new_cost - base) / base * 100
    print(f"  {name}: pre={base:.4f} post={new_cost:.4f} ({diff:+.2f}%) [{t:.0f}s]")
    return base, new_cost


def avg_over(label, run_fn, **kwargs):
    print(f"\n=== {label} ===")
    bases, news = [], []
    for n in BENCHES:
        b, nv = run_fn(n, **kwargs)
        bases.append(b); news.append(nv)
    avg_b = sum(bases) / len(bases)
    avg_n = sum(news) / len(news)
    print(f"  AVG: pre={avg_b:.4f} post={avg_n:.4f} ({(avg_n-avg_b)/avg_b*100:+.2f}%)")
    return avg_n


if __name__ == "__main__":
    # Baseline: plain LNS at current default (30 ep, 800 steps each = ~24K steps)
    a_lns30 = avg_over("LNS 30ep / 800steps  (current default)",
                       run_lns, n_episodes=30, sa_steps=800,
                       subset_size=12, time_budget=180)

    # PT 4 chains × 6 rounds × 1000 steps = 24K steps total — matched compute
    a_pt = avg_over("PT 4ch x 6rd x 1000s  (~matched compute)",
                    run_pt, n_chains=4, n_swap_rounds=6,
                    sa_steps_per_round=1000,
                    T_min_frac=0.001, T_max_frac=0.05,
                    time_budget=180)

    # PT 4 chains x larger T spread (more exploration)
    a_pt_wide = avg_over("PT 4ch wide T  (T_max=0.10)",
                         run_pt, n_chains=4, n_swap_rounds=6,
                         sa_steps_per_round=1000,
                         T_min_frac=0.001, T_max_frac=0.10,
                         time_budget=180)

    # PT 6 chains finer ladder
    a_pt6 = avg_over("PT 6ch x 4rd x 1000s",
                     run_pt, n_chains=6, n_swap_rounds=4,
                     sa_steps_per_round=1000,
                     T_min_frac=0.001, T_max_frac=0.05,
                     time_budget=240)

    print(f"\n=== SUMMARY ===")
    print(f"  LNS 30ep:           {a_lns30:.4f}")
    print(f"  PT 4ch tight:       {a_pt:.4f}  ({(a_pt-a_lns30)/a_lns30*100:+.2f}%)")
    print(f"  PT 4ch wide:        {a_pt_wide:.4f}  ({(a_pt_wide-a_lns30)/a_lns30*100:+.2f}%)")
    print(f"  PT 6ch:             {a_pt6:.4f}  ({(a_pt6-a_lns30)/a_lns30*100:+.2f}%)")
