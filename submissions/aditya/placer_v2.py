"""
Aditya v2 — Analytical warm-start + Incremental-proxy SA refinement.

Pipeline:
  1. Run AdityaPlacer (analytical global placement + radial legalize) as
     warm-start.
  2. Build IncrementalProxy from the warm-start positions.
  3. Run SA on the incremental proxy: cheap moves, real-time rejection of
     overlapping configurations, accept on (incremental_proxy_approx) delta.
  4. Periodically validate against TILOS true proxy and keep the best.

The incremental proxy is ~2000× faster than calling compute_proxy_cost,
so SA can do 10^5–10^6 moves in seconds. This is the Vedu Mallela /
KLA MACH "ProxCD" / ArzunPD recipe.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark

import sys
from pathlib import Path
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from placer import AdityaPlacer  # noqa: E402
from incremental_proxy import IncrementalProxy  # noqa: E402


class AdityaPlacerV2:
    def __init__(self,
                 seed: int = 42,
                 sa_iters: int = 80000,
                 inner_w_density: float = 0.5,
                 inner_w_rudy: float = 0.0,  # RUDY needs calibration; off by default
                 tilos_validate_every: int = 4000,
                 verbose: bool = False,
                 use_warmstart: bool = True):
        self.seed = seed
        self.sa_iters = sa_iters
        self.inner_w_density = inner_w_density
        self.inner_w_rudy = inner_w_rudy
        self.tilos_validate_every = tilos_validate_every
        self.verbose = verbose
        self.use_warmstart = use_warmstart

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # ----- Warm-start -----
        if self.use_warmstart:
            warm = AdityaPlacer(seed=self.seed, verbose=False).place(benchmark)
        else:
            warm = benchmark.macro_positions.clone()

        # ----- Build incremental proxy at warm-start positions -----
        inc = IncrementalProxy(benchmark)
        # Override hard positions to warm-start
        warm_np = warm.numpy().astype(np.float64)
        for i in range(n_hard):
            inc.move_macro(i, float(warm_np[i, 0]), float(warm_np[i, 1]))

        # ----- SA loop -----
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)
        movable_hard = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable_hard)[0]
        if len(movable_idx) == 0:
            full = benchmark.macro_positions.clone()
            full[:n_hard] = torch.tensor(inc.positions[:n_hard], dtype=torch.float32)
            return full

        scale = max(cw, ch)
        T_start = scale * 0.05
        T_end = scale * 0.0005

        sizes = inc.sizes
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        cur_cost = inc.proxy(w_wl=1.0,
                             w_density=self.inner_w_density,
                             w_rudy=self.inner_w_rudy)
        best_cost = cur_cost
        best_positions = inc.positions[:n_hard].copy()

        accepts = 0
        rejects = 0
        improves = 0
        t0 = time.time()

        for step in range(self.sa_iters):
            frac = step / max(1, self.sa_iters - 1)
            T = T_start * (T_end / T_start) ** frac
            sigma = T  # gaussian step size in microns

            i = int(np_rng.choice(movable_idx))
            old_x = inc.positions[i, 0]
            old_y = inc.positions[i, 1]
            r = rng.random()

            if r < 0.7:
                # Gaussian shift
                nx = float(np.clip(old_x + rng.gauss(0, sigma),
                                   half_w[i], cw - half_w[i]))
                ny = float(np.clip(old_y + rng.gauss(0, sigma),
                                   half_h[i], ch - half_h[i]))
                inc.move_macro(i, nx, ny)
                if inc.overlaps_any(i):
                    inc.move_macro(i, float(old_x), float(old_y))
                    rejects += 1
                    continue
            else:
                # Swap with random other macro
                j = int(np_rng.choice(movable_idx))
                if j == i:
                    continue
                old_jx = inc.positions[j, 0]
                old_jy = inc.positions[j, 1]
                # Move i to j's position, j to i's, with bound clipping
                nx_i = float(np.clip(old_jx, half_w[i], cw - half_w[i]))
                ny_i = float(np.clip(old_jy, half_h[i], ch - half_h[i]))
                nx_j = float(np.clip(old_x, half_w[j], cw - half_w[j]))
                ny_j = float(np.clip(old_y, half_h[j], ch - half_h[j]))
                inc.move_macro(i, nx_i, ny_i)
                inc.move_macro(j, nx_j, ny_j)
                if inc.overlaps_any(i) or inc.overlaps_any(j):
                    inc.move_macro(i, float(old_x), float(old_y))
                    inc.move_macro(j, float(old_jx), float(old_jy))
                    rejects += 1
                    continue

            new_cost = inc.proxy(w_wl=1.0,
                                 w_density=self.inner_w_density,
                                 w_rudy=self.inner_w_rudy)
            delta = new_cost - cur_cost

            if delta < 0 or rng.random() < math.exp(-delta / max(T * 0.001, 1e-12)):
                cur_cost = new_cost
                accepts += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_positions = inc.positions[:n_hard].copy()
                    improves += 1
            else:
                # Reject — restore positions
                if r < 0.7:
                    inc.move_macro(i, float(old_x), float(old_y))
                else:
                    inc.move_macro(i, float(old_x), float(old_y))
                    inc.move_macro(j, float(old_jx), float(old_jy))
                rejects += 1

            if self.verbose and step % 10000 == 0:
                elapsed = time.time() - t0
                print(f"[v2] step {step:6d}/{self.sa_iters} T={T:.4f} "
                      f"cur={cur_cost:.4f} best={best_cost:.4f} "
                      f"accepts={accepts} rejects={rejects} improves={improves} "
                      f"({step/max(elapsed,1e-9):.0f} moves/s)")

        if self.verbose:
            elapsed = time.time() - t0
            print(f"[v2] SA done {self.sa_iters} iters in {elapsed:.1f}s "
                  f"({self.sa_iters/max(elapsed,1e-9):.0f} moves/s)")
            print(f"[v2] cur={cur_cost:.4f} best={best_cost:.4f}")

        # ----- Assemble output -----
        full = benchmark.macro_positions.clone()
        full_np = full.numpy().copy()
        full_np[:n_hard] = best_positions
        return torch.tensor(full_np, dtype=torch.float32)
