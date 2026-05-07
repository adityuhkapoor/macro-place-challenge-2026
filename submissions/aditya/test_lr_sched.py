"""Atomic test: cosine LR schedule vs constant."""

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
AVG_REPLACE = sum(REPLACE.values()) / len(REPLACE)


def run(name, **kw):
    placer = AdityaPlacerV4(**kw)
    total = 0.0; total_t = 0.0
    for bn in BENCHES:
        bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bn}")
        t0 = time.time()
        pos = placer.place(bm)
        t = time.time() - t0
        c = compute_proxy_cost(pos, bm, plc)
        total += c["proxy_cost"]; total_t += t
    avg = total / len(BENCHES)
    diff = (avg - AVG_REPLACE) / AVG_REPLACE * 100
    print(f"=== {name} ===  AVG: {avg:.4f}  ({diff:+.1f}%)  [{total_t:.1f}s]")
    return avg


if __name__ == "__main__":
    run("baseline_const", lr_schedule="constant")
    run("cosine_end_01", lr_schedule="cosine", lr_end_frac=0.1)
    run("cosine_end_03", lr_schedule="cosine", lr_end_frac=0.3)
    run("cosine_high_lr", lr_schedule="cosine", lr_frac=0.005, lr_end_frac=0.1)
    run("cosine_high_lr_03", lr_schedule="cosine", lr_frac=0.005, lr_end_frac=0.3)
